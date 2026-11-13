"""Select up to 10 concise, quality-filtered RAG workflows per task category.

Reuses the enriched-dataset analysis outputs:
  - enriched_dataset_analysis/workflows_base.parquet   (descriptions, SysML models, structure features)
  - enriched_dataset_analysis/embeddings/*             (enriched-description embeddings, L2-normalised)
  - enriched_dataset_analysis/workflow_assignments.csv (cluster_id and category per workflow)

Both strategies require saved normalization/kernel validation and fresh static quality checks.
Complexity levels are metadata, not quotas. The semantic variant omits the structural-diversity
tie-break; neither variant relaxes admission criteria to fill a category.

Run from the project root:
    python -m Analysis.rag_selection --strategy complexity
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Analysis.enriched_pipeline import Config, Paths, load_embeddings
from Analysis.rag_quality import QualityLimits, assess_quality

N_PER_CATEGORY = 10
COMPLEXITY_LEVELS = ("simple", "medium", "complex")
COMPLEXITY_WEIGHTS = {
    "num_actions": 0.08,
    "num_decisions": 0.18,
    "num_forks": 0.15,
    "num_joins": 0.15,
    "control_flow_edges": 0.10,
    "conditional_branches": 0.12,
    "parallel_branches": 0.08,
    "max_nested_parallelism_depth": 0.10,
    "has_cycles": 0.02,
    "repeated_action_occurrences": 0.02,
}
def load_inputs(cfg: Config, paths: Paths) -> pd.DataFrame:
    emb, rows = load_embeddings(cfg, paths)
    rows = rows.reset_index(drop=True)
    assign = pd.read_csv(paths.assignments, keep_default_na=False, low_memory=False,
                         usecols=["origin", "cluster_id", "category"], dtype=str)
    rows = rows.merge(assign, on="origin", how="left", validate="one_to_one")
    if (rows["category"].fillna("") == "").any():
        raise SystemExit("Some workflows have no category; run `python -m Analysis.enriched_report` first.")
    rows["emb_row"] = np.arange(len(rows))
    rows.attrs["emb"] = emb
    return rows


def max_similarity_to(emb: np.ndarray, ids: np.ndarray, ref: np.ndarray, block: int = 2048) -> np.ndarray:
    """Maximum cosine similarity of emb[ids] to any row of ref (block-wise, no N x N matrix)."""
    out = np.full(len(ids), -1.0, dtype=np.float32)
    if len(ref) == 0:
        return out
    for start in range(0, len(ids), block):
        sims = emb[ids[start:start + block]] @ ref.T
        out[start:start + block] = sims.max(axis=1)
    return out


def partition_masks(rows: pd.DataFrame, emb: np.ndarray, cfg: Config, exclude_ids: set[str],
                    leak_threshold: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (is_heldout, is_leak_neighbour, is_train_candidate)."""
    heldout = rows["origin"].str.contains("::test::", regex=False) | rows["origin"].isin(exclude_ids)
    held_emb = emb[rows.loc[heldout, "emb_row"].values]
    leak = pd.Series(False, index=rows.index)
    train_idx = rows.index[~heldout]
    sims = max_similarity_to(emb, rows.loc[train_idx, "emb_row"].values, held_emb)
    leak.loc[train_idx] = sims >= leak_threshold
    return heldout, leak, ~heldout & ~leak


def eligible_pool(cat_rows: pd.DataFrame) -> pd.DataFrame:
    ok = (
        cat_rows["used_in_clustering"].astype(bool)
        & ~cat_rows["missing_description"].astype(bool)
        & ~cat_rows["too_short"].astype(bool)
        & ~cat_rows["duplicate_description"].astype(bool)
        & (cat_rows["sysml_model"].fillna("").str.strip() != "")
    )
    pool = cat_rows[ok].copy()
    # Keep one workflow per identical SysML model text.
    return pool.drop_duplicates(subset="sysml_model", keep="first")


def is_linear(row) -> bool:
    return int(row["num_decisions"]) + int(row["num_forks"]) + int(row["num_joins"]) == 0


def select_category(cat: str, cat_rows: pd.DataFrame, emb: np.ndarray,
                    seed: int, dup_threshold: float, max_per_category: int = N_PER_CATEGORY,
                    ) -> tuple[list[dict], dict, pd.DataFrame, np.ndarray]:
    return select_category_complexity(
        cat, cat_rows, emb, seed, dup_threshold, max_per_category, structural_diversity=False,
    )


def assign_complexity_levels(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rank-normalize features, calculate a weighted score, and use corpus tertiles."""
    data = rows.copy()
    normalized = {}
    for name in COMPLEXITY_WEIGHTS:
        values = pd.to_numeric(data[name], errors="coerce").fillna(0)
        normalized[name] = values.rank(method="average", pct=True)
        data[f"complexity_rank_{name}"] = normalized[name]
    data["complexity_score"] = sum(
        normalized[name] * weight for name, weight in COMPLEXITY_WEIGHTS.items()
    )
    q1, q2 = (float(data.complexity_score.quantile(q)) for q in (1 / 3, 2 / 3))
    data["complexity_level"] = np.select(
        [data.complexity_score <= q1, data.complexity_score <= q2],
        ["simple", "medium"],
        default="complex",
    )
    return data, {
        "method": "empirical percentile-rank normalization, then weighted sum",
        "weights": COMPLEXITY_WEIGHTS,
        "thresholds": {"simple_max": q1 if len(data) else None, "medium_max": q2 if len(data) else None},
        "quantiles": [1 / 3, 2 / 3],
        "normalized_features": {
            name: f"complexity_rank_{name}" for name in COMPLEXITY_WEIGHTS
        },
    }


def filter_candidates(rows: pd.DataFrame, limits: QualityLimits) -> pd.DataFrame:
    """Audit train rows before ranking. Missing parser evidence always fails closed."""
    records = []
    for row in rows.sort_values("origin").to_dict("records"):
        reasons = []
        if not bool(row["used_in_clustering"]) or bool(row["missing_description"]) or bool(row["too_short"]):
            reasons.append("quality_missing_or_short_description")
        if bool(row["duplicate_description"]):
            reasons.append("duplicate_description")
        if row.get("normalization_status") != "valid":
            reasons.append("quality_normalization_failed_or_missing")
        parser = pd.to_numeric(row.get("parser_valid"), errors="coerce")
        if parser != 1:
            reasons.append("invalid_syntax" if parser == 0 else "parser_unassessed")
        model = str(row.get("normalized_sysml") or "")
        source = str(row["sysml_model"])
        if not model.strip():
            reasons.append("quality_missing_normalized_model")
        if pd.to_numeric(row.get("structural_valid"), errors="coerce") != 1:
            reasons.append("quality_cached_structure_failed_or_missing")
        metadata = {
            "description_chars": len(row["description"]), "quality_score": 0.0,
            "action_description_coverage": 0.0, "mean_action_token_overlap": 0.0,
            "quality_details": [],
        }
        if metadata["description_chars"] > limits.max_description_chars:
            reasons.append("excessive_description_length")
        if not reasons or reasons == ["excessive_description_length"]:
            metadata = assess_quality(model, row["description"], limits, source)
            reasons.extend(metadata.pop("quality_reasons"))
        row.update(metadata)
        row["quality_reasons"] = sorted(set(reasons))
        row["quality_pass"] = not reasons
        row["original_sysml_model"] = source
        row["normalized_sysml_model"] = model
        row["sysml_model"] = model
        records.append(row)
    result = pd.DataFrame(records, columns=list(dict.fromkeys(
        list(rows.columns) + [
            "description_chars", "quality_score", "action_description_coverage",
            "mean_action_token_overlap", "quality_details", "quality_reasons", "quality_pass",
            "original_sysml_model", "normalized_sysml_model",
        ] + list(COMPLEXITY_WEIGHTS) + [
            "max_execution_path_depth", "max_incoming_edges", "duplicate_action_declarations",
            "has_parallelism", "has_nested_parallelism",
        ]
    )))
    ranked = result[result.quality_pass.eq(True)].sort_values(
        ["quality_score", "description_chars", "origin"], ascending=[False, True, True],
    )
    duplicates = ranked.index[ranked.sysml_model.duplicated()]
    for index in duplicates:
        result.at[index, "quality_pass"] = False
        result.at[index, "quality_reasons"] = ["duplicate_model"]
    return result


def projection(pool: pd.DataFrame, selected: pd.DataFrame, emb: np.ndarray, seed: int):
    """PCA coordinates, including zero/singleton categories without fitting invalid PCA."""
    if len(pool) < 2:
        return np.zeros((len(pool), 2)), np.zeros((len(selected), 2))
    pca = PCA(n_components=min(2, emb.shape[1]), random_state=seed).fit(emb[pool["emb_row"].values])
    xy = pca.transform(emb[pool["emb_row"].values])
    sxy = pca.transform(emb[selected["emb_row"].values]) if len(selected) else np.empty((0, xy.shape[1]))
    return np.pad(xy, ((0, 0), (0, 2 - xy.shape[1]))), np.pad(sxy, ((0, 0), (0, 2 - sxy.shape[1])))


def select_category_complexity(
    cat: str, cat_rows: pd.DataFrame, emb: np.ndarray, seed: int, dup_threshold: float,
    max_per_category: int = N_PER_CATEGORY, structural_diversity: bool = True,
) -> tuple[list[dict], dict, pd.DataFrame, np.ndarray]:
    """Rank admitted candidates by quality, conciseness, then structural/semantic diversity."""
    if not 1 <= max_per_category <= N_PER_CATEGORY:
        raise ValueError(f"max_per_category must be between 1 and {N_PER_CATEGORY}")
    # Fail closed: this entry point never selects rows lacking admission evidence.
    pool = cat_rows[cat_rows["quality_pass"].eq(True)].sort_values("origin").copy()
    info = {"category": cat, "n_in_category": len(cat_rows), "n_eligible_pool": len(pool)}
    if pool.empty:
        info.update(n_selected=0, n_near_duplicates=0, n_category_limit=0)
        pool["subcluster_id"] = pd.Series(dtype=int)
        return [], info, pool, np.array([])

    X = emb[pool["emb_row"].values]
    n_clusters = min(max_per_category, len(pool), len(np.unique(X, axis=0)))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    labels = km.labels_
    centroids = km.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, np.finfo(float).eps)
    pool["subcluster_id"] = labels
    pool["centroid_similarity"] = np.sum(X * centroids[labels], axis=1)
    pool["quality_band"] = np.floor(pool["quality_score"] * 20 + 1e-9).astype(int)
    pool["length_band"] = pool["description_chars"] // 250
    pool["action_count_band"] = (pool["num_actions"] - 1) // 5
    ranked = pool.sort_values(
        ["quality_band", "length_band", "action_count_band", "centroid_similarity", "origin"],
        ascending=[False, True, True, False, True],
    )
    ranks = {origin: index for index, origin in enumerate(ranked.origin, start=1)}
    chosen: list[dict] = []
    chosen_vectors: list[np.ndarray] = []
    chosen_origins: set[str] = set()
    covered: set[str] = set()
    selected_clusters: set[int] = set()
    duplicates: set[str] = set()

    def constructs(row) -> set[str]:
        return {
            label for label, present in (
                ("sequence", is_linear(row)),
                ("conditional", row["num_decisions"] > 0),
                ("parallel", row["num_forks"] > 0),
                ("synchronization", row["num_joins"] > 0),
                ("loop", bool(row.get("has_cycles", False))),
                ("repeated_occurrence", row.get("repeated_action_occurrences", 0) > 0),
                ("nested_parallelism", row.get("max_nested_parallelism_depth", 0) > 1),
            ) if present
        }

    while len(chosen) < max_per_category:
        available = []
        for _, row in ranked.iterrows():
            if row.origin in chosen_origins or row.origin in duplicates:
                continue
            vector = emb[int(row.emb_row)]
            if chosen_vectors and float(np.max(np.stack(chosen_vectors) @ vector)) >= dup_threshold:
                duplicates.add(row.origin)
                continue
            gain = len(constructs(row) - covered) if structural_diversity else 0
            key = (
                -int(row.quality_band), int(row.length_band), int(row.action_count_band),
                -gain, -int(row.subcluster_id not in selected_clusters),
                -float(row.centroid_similarity), int(row.description_chars), str(row.origin),
            )
            available.append((key, row))
        if not available:
            break
        _, row = min(available, key=lambda item: item[0])
        cluster = int(row.subcluster_id)
        item = row.to_dict()
        item.update(
            category=cat, workflow_type="linear" if is_linear(row) else "non-linear",
            candidate_rank=ranks[row.origin],
            selection_reason="quality_conciseness_diversity" if structural_diversity else "quality_conciseness_centroid",
            subcluster_size=int((labels == cluster).sum()),
        )
        chosen.append(item)
        chosen_origins.add(row.origin)
        chosen_vectors.append(emb[int(row.emb_row)])
        covered.update(constructs(row))
        selected_clusters.add(cluster)
    # Attribute every unselected admitted candidate without relaxing any constraint.
    remaining = pool[~pool.origin.isin(chosen_origins)]
    for _, row in remaining.iterrows():
        if chosen_vectors and float(np.max(np.stack(chosen_vectors) @ emb[int(row.emb_row)])) >= dup_threshold:
            duplicates.add(row.origin)
    pool["selection_rejection"] = [
        "" if origin in chosen_origins else "near_duplicate" if origin in duplicates else "category_limit"
        for origin in pool.origin
    ]
    info.update(
        n_selected=len(chosen), n_near_duplicates=len(duplicates),
        n_category_limit=len(pool) - len(chosen) - len(duplicates),
    )
    return chosen, info, pool, centroids


def plot_selection(selected: pd.DataFrame, pools: dict, emb: np.ndarray, out_stem: Path, seed: int) -> None:
    cats = list(pools)
    ncols = 4
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, cat in zip(axes.ravel(), cats):
        pool = pools[cat]
        sel = selected[selected["category"] == cat]
        xy, sxy = projection(pool, sel, emb, seed)
        shown = rng.choice(len(pool), size=min(len(pool), 1500), replace=False)
        ax.scatter(xy[shown, 0], xy[shown, 1], c=pool.iloc[shown]["subcluster_id"],
                   cmap="tab10", s=5, alpha=0.35, linewidths=0)
        for marker, kind in (("o", "linear"), ("^", "non-linear")):
            m = (sel["workflow_type"] == kind).values
            ax.scatter(sxy[m, 0], sxy[m, 1], c=sel["subcluster_id"].values[m], cmap="tab10", vmin=0, vmax=9,
                       marker=marker, s=70, edgecolors="black", linewidths=1.2, label=kind, zorder=3)
        ax.set_title(f"{cat}\n(n={len(sel)} selected of {len(pool)} eligible)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="best")
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    fig.suptitle("RAG corpus selection in embedding space (PCA per category; colour = semantic sub-cluster)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category(cat: str, sel: pd.DataFrame, pool: pd.DataFrame, emb: np.ndarray, out_stem: Path, seed: int) -> None:
    """One paper-ready figure per category, including empty and singleton categories."""
    xy, sxy = projection(pool, sel, emb, seed)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(int(c) % 10) for c in pool["subcluster_id"]], s=8, alpha=0.3, linewidths=0)
    if "complexity_level" in sel:
        for marker, level in (("o", "simple"), ("^", "medium"), ("s", "complex")):
            m = (sel["complexity_level"] == level).values
            ax.scatter(sxy[m, 0], sxy[m, 1], c=[cmap(int(c) % 10) for c in sel["subcluster_id"].values[m]],
                       marker=marker, s=120, edgecolors="black", linewidths=1.3,
                       label=f"selected, {level}", zorder=3)
    else:
        for marker, kind in (("o", "linear"), ("^", "non-linear")):
            m = (sel["workflow_type"] == kind).values
            ax.scatter(sxy[m, 0], sxy[m, 1], c=[cmap(int(c) % 10) for c in sel["subcluster_id"].values[m]],
                       marker=marker, s=120, edgecolors="black", linewidths=1.3,
                       label=f"selected, {kind}", zorder=3)
    ax.set_title(f"{cat}\n{len(sel)} selected workflows among {len(pool)} eligible (colour = semantic sub-cluster)", fontsize=9)
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="enriched_dataset_analysis")
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--strategy", choices=("complexity", "semantic"), default="complexity",
                    help="Quality-first selection, with structural diversity or semantic tie-breaking.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-threshold", type=float, default=0.90,
                    help="Selected workflows must have pairwise cosine below this value.")
    ap.add_argument("--leak-threshold", type=float, default=0.95,
                    help="Candidates with cosine >= this to any held-out workflow are excluded.")
    ap.add_argument("--exclude-ids", help="Text file, one origin per line, of additional held-out (test) workflows.")
    ap.add_argument("--max-per-category", type=int, default=N_PER_CATEGORY)
    ap.add_argument("--max-actions", type=int, default=15)
    ap.add_argument("--max-decisions", type=int, default=5)
    ap.add_argument("--max-forks", type=int, default=2)
    ap.add_argument("--max-joins", type=int, default=2)
    ap.add_argument("--max-description-chars", type=int, default=1500)
    ap.add_argument("--min-action-coverage", type=float, default=0.8)
    ap.add_argument("--min-action-token-overlap", type=float, default=0.3)
    args = ap.parse_args(argv)
    try:
        limits = QualityLimits(
            max_actions=args.max_actions, max_decisions=args.max_decisions,
            max_forks=args.max_forks, max_joins=args.max_joins,
            max_description_chars=args.max_description_chars,
            min_action_coverage=args.min_action_coverage,
            min_action_token_overlap=args.min_action_token_overlap,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if not 1 <= args.max_per_category <= N_PER_CATEGORY:
        ap.error(f"--max-per-category must be between 1 and {N_PER_CATEGORY}")
    if not 0 < args.dup_threshold <= 1 or not 0 < args.leak_threshold <= 1:
        ap.error("cosine thresholds must be in (0, 1]")

    cfg = Config(output_dir=args.output_dir, model=args.model, seed=args.seed)
    paths = Paths(cfg)
    out_dir = paths.root / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_inputs(cfg, paths)
    emb = rows.attrs["emb"]
    exclude = set(Path(args.exclude_ids).read_text().split()) if args.exclude_ids else set()
    heldout, leak, train = partition_masks(rows, emb, cfg, exclude, args.leak_threshold)
    print(f"Workflows: {len(rows)} | held-out (test): {int(heldout.sum())} | "
          f"train near-duplicates of held-out removed: {int(leak.sum())} | train candidates: {int(train.sum())}")

    selected, summaries, pools, audits = [], [], {}, []
    result_file = out_dir / "normalization_results.csv"
    if not result_file.exists():
        raise SystemExit("Normalized validation results are missing; run `python -m Analysis.normalize_rag_pool` first.")
    validated = pd.read_csv(result_file, keep_default_na=False, low_memory=False)
    validation_columns = [
        "origin", "normalization_status", "parser_valid", "parser_error", "transformations",
        "structural_valid", "structural_status", "structural_errors", "intentional_exception",
        "normalized_sysml",
    ]
    if missing_columns := set(validation_columns) - set(validated.columns):
        raise SystemExit(f"Missing normalization results columns: {', '.join(sorted(missing_columns))}")
    # Restrict to training rows before examining model text. Held-out models never enter quality scoring.
    training_rows = rows[train].copy()
    training_rows.attrs = {}
    candidates = training_rows.merge(
        validated[validation_columns], on="origin", how="left", validate="one_to_one",
    ).fillna({"normalized_sysml": "", "normalization_status": ""})
    print("Applying hard quality and length filters to training candidates...", flush=True)
    assessed_parts = []
    for cat in sorted(rows["category"].unique()):
        part = filter_candidates(candidates[candidates.category == cat], limits)
        assessed_parts.append(part)
        print(f"[{cat}] assessed={len(part)} qualifying={int(part.quality_pass.sum())}", flush=True)
    assessed = pd.concat(assessed_parts, ignore_index=True)
    selection_rows, complexity_info = assign_complexity_levels(assessed[assessed.quality_pass.eq(True)].copy())
    parser_ok = pd.to_numeric(candidates.parser_valid, errors="coerce").eq(1)
    structure_ok = pd.to_numeric(candidates.structural_valid, errors="coerce").eq(1)
    structural_counts = {
        "parser_valid_candidates": int(parser_ok.sum()),
        "parser_invalid_candidates": int(pd.to_numeric(candidates.parser_valid, errors="coerce").eq(0).sum()),
        "parser_unassessed_candidates": int(pd.to_numeric(candidates.parser_valid, errors="coerce").isna().sum()),
        "cached_structural_valid_candidates": int((parser_ok & structure_ok).sum()),
        "quality_qualifying_candidates": len(selection_rows),
        "parser_evidence": "saved kernel results; no new kernel run",
    }

    for cat in sorted(rows["category"].unique()):
        cat_rows = selection_rows[
            (selection_rows["category"] == cat)
            & (selection_rows["origin"].str.contains("::test::", regex=False) == False)
            & ~selection_rows["origin"].isin(exclude)
        ]
        if args.strategy == "complexity":
            picks, info, pool, _ = select_category_complexity(
                cat, cat_rows, emb, args.seed, args.dup_threshold, args.max_per_category,
            )
        else:
            picks, info, pool, _ = select_category(
                cat, cat_rows, emb, args.seed, args.dup_threshold, args.max_per_category,
            )
        audit = assessed[assessed.category == cat].copy()
        category_audits = []
        selection_reasons = pool.set_index("origin").selection_rejection.to_dict() if len(pool) else {}
        for row in audit.to_dict("records"):
            reasons = row["quality_reasons"] or (
                [selection_reasons[row["origin"]]] if selection_reasons.get(row["origin"]) else []
            )
            category_audits.append({
                "origin": row["origin"], "dataset": row["dataset"], "category": cat,
                "quality_pass": row["quality_pass"], "selected": not reasons,
                "rejection_reasons": json.dumps(reasons),
                "quality_details": json.dumps(row["quality_details"]),
                "description_chars": row["description_chars"],
                "quality_score": row["quality_score"],
                "action_description_coverage": row["action_description_coverage"],
                "mean_action_token_overlap": row["mean_action_token_overlap"],
                "parser_valid": row["parser_valid"],
                **{name: row.get(name) for name in COMPLEXITY_WEIGHTS},
            })
        audits.extend(category_audits)
        reason_lists = [json.loads(row["rejection_reasons"]) for row in category_audits]
        counts = Counter(reason for reasons in reason_lists for reason in reasons)
        for reason in ("invalid_syntax", "parser_unassessed", "excessive_description_length",
                       "excessive_actions", "excessive_decisions", "excessive_forks", "excessive_joins"):
            counts.setdefault(reason, 0)
        info["rejection_reasons"] = json.dumps(dict(sorted(counts.items())))
        for name, count in counts.items():
            info[f"rejected_{name}"] = count
        info["n_in_category"] = int((rows["category"] == cat).sum())
        info["n_train_candidates"] = len(audit)
        info["n_rejected"] = len(audit) - len(picks)
        info["n_filter_rejected"] = len(audit) - len(pool)
        info["rejected_excessive_complexity"] = sum(
            bool(set(reasons) & {"excessive_actions", "excessive_decisions", "excessive_forks", "excessive_joins"})
            for reasons in reason_lists
        )
        info["rejected_quality_issues"] = sum(
            any(reason.startswith("quality_") for reason in reasons) for reasons in reason_lists
        )
        selected += picks
        pools[cat] = pool
        if picks:
            sel_emb = emb[[p["emb_row"] for p in picks]]
            pair = sel_emb @ sel_emb.T
            if len(picks) > 1:
                offdiag = pair[~np.eye(len(picks), dtype=bool)]
                info["max_pairwise_cosine"] = round(float(offdiag.max()), 4)
                info["mean_pairwise_cosine"] = round(float(offdiag.mean()), 4)
        info["n_selected"] = len(picks)
        info["n_linear"] = sum(p["workflow_type"] == "linear" for p in picks)
        info["n_non_linear"] = sum(p["workflow_type"] == "non-linear" for p in picks)
        for level in COMPLEXITY_LEVELS:
            info[f"n_{level}"] = sum(p["complexity_level"] == level for p in picks)
        for ds in ("sopbench", "worfbench", "sapsam"):
            info[f"n_{ds}"] = sum(p["dataset"] == ds for p in picks)
        summaries.append(info)
        print(f"[{cat}] candidates={len(audit)} qualifying={len(pool)} selected={len(picks)} "
              f"rejected={info['n_rejected']} reasons={dict(sorted(counts.items()))}", flush=True)

    sel = pd.DataFrame(selected) if selected else selection_rows.iloc[:0].copy()
    cols = ["origin", "dataset", "source", "category", "subcluster_id", "cluster_id", "selection_reason",
            "candidate_rank", "centroid_similarity", "subcluster_size", "workflow_type", "num_actions",
            "num_decisions", "num_forks", "num_joins", "description", "sysml_model"]
    additional = [
        "normalization_status", "parser_valid", "parser_error", "structural_valid",
        "structural_status", "structural_errors", "intentional_exception", "transformations",
        "original_sysml_model", "normalized_sysml_model", "complexity_score", "complexity_level",
        "control_flow_edges", "conditional_branches", "parallel_branches",
        "max_nested_parallelism_depth", "has_parallelism", "has_nested_parallelism",
        "has_cycles", "repeated_action_occurrences",
        "max_execution_path_depth", "max_incoming_edges",
        "duplicate_action_declarations",
        "quality_pass", "quality_score", "description_chars", "action_description_coverage",
        "mean_action_token_overlap",
    ] + [f"complexity_rank_{name}" for name in COMPLEXITY_WEIGHTS]
    cols += [c for c in additional if c in sel.columns]
    sel = sel.reindex(columns=cols).rename(columns={"cluster_id": "global_cluster_id", "description": "enriched_description"})
    for flag in ("parser_valid", "structural_valid"):
        sel[flag] = pd.to_numeric(sel[flag], errors="raise").astype(int)
    sel.to_csv(out_dir / "rag_selected_workflows.csv", index=False)
    pd.DataFrame(summaries).fillna(0).to_csv(out_dir / "rag_selection_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(out_dir / "rag_candidate_audit.csv", index=False)
    reason_rows = [
        {"category": item["category"], "reason": reason, "count": count}
        for item in summaries for reason, count in json.loads(item["rejection_reasons"]).items()
    ]
    pd.DataFrame(reason_rows, columns=["category", "reason", "count"]).to_csv(
        out_dir / "rag_rejection_reasons.csv", index=False,
    )

    plot_input = sel.merge(rows[["origin", "emb_row"]], on="origin")
    plot_selection(plot_input, pools, emb, out_dir / "rag_selection_embedding_space", args.seed)

    per_cat = out_dir / "per_category"
    per_cat.mkdir(exist_ok=True)
    for cat, pool in pools.items():
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in cat).strip("_")
        plot_category(cat, plot_input[plot_input["category"] == cat], pool, emb, per_cat / slug, args.seed)

    config = {
        "strategy": args.strategy,
        "seed": args.seed, "n_per_category": args.max_per_category, "embedding_model": cfg.model,
        "category_cap_only": True, "quality_limits": asdict(limits),
        "quality_ranking": "quality bands of 0.05, description bands of 250 chars, action bands of 5, "
                           "construct novelty, cluster novelty, centroid similarity, exact length, origin",
        "semantic_alignment": "deterministic lexical proxy; not proof of natural-language equivalence",
        "selection_dup_threshold": args.dup_threshold, "heldout_leak_threshold": args.leak_threshold,
        "train_partition_rule": "exclude origins containing '::test::' and any --exclude-ids; drop candidates "
                                "with cosine >= leak threshold to any held-out workflow",
        "n_heldout": int(heldout.sum()), "n_leak_removed": int(leak.sum()),
        "baseline_corpus": "rag/baseline/rag_selected_workflows.csv",
        "validation": structural_counts,
    }
    if complexity_info:
        config["complexity"] = complexity_info
    (out_dir / "rag_selection_config.json").write_text(json.dumps(config, indent=2))

    print("\nSelected per category (linear / non-linear):")
    for s in summaries:
        print(f"  {s['category']}: {s['n_selected']}  ({s['n_linear']} linear / {s['n_non_linear']} non-linear)"
              f"{'  [structural replacement]' if s.get('structural_replacement') else ''}"
              f"{'  NOTE: ' + s['note'] if s.get('note') else ''}")
    total = len(sel)
    print(f"Total: {total} | linear={int((sel.workflow_type == 'linear').sum())} "
          f"non-linear={int((sel.workflow_type == 'non-linear').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
