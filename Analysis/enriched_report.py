"""Category assignment, distributions, figures and report for the enriched-dataset analysis.

Reads the saved cluster assignments and the (reviewable) cluster -> category mapping, so it can be
re-run without recomputing embeddings:

    python -m Analysis.enriched_report [--output-dir DIR] [--model minilm] [--init-mapping]

`--init-mapping` writes a template ``tables/category_mapping.csv`` (one row per cluster with the
c-TF-IDF keywords) if none exists. Edit that file to rename/merge categories, then re-run.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Analysis.enriched_pipeline import Config, Paths, load_embeddings

UNCLASSIFIED = "Generic Process Flows (Domain-Agnostic)"
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948", "#9C755F",
           "#FF9DA7", "#86BC86", "#D37295", "#A0CBE8", "#FFBE7D", "#BAB0AC", "#8CD17D", "#B6992D"]
GREY = "#9E9E9E"
DATASET_ORDER = ["sopbench", "worfbench", "sapsam"]
DATASET_LABEL = {"sopbench": "SOPBench", "worfbench": "WorFBench", "sapsam": "SAP-SAM"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def init_mapping(paths: Paths) -> None:
    target = paths.tables / "category_mapping.csv"
    if target.exists():
        print(f"Mapping already exists, not overwritten: {target}")
        return
    kw = pd.read_csv(paths.tables / "cluster_keywords.csv")
    pd.DataFrame({
        "cluster_id": kw["cluster"], "category": "TODO", "n_workflows": kw["n_workflows"],
        "supporting_keywords": kw["ctfidf_keywords"].str.split(", ").str[:12].str.join(", "),
        "explanation": "", "possible_overlap_with": "", "label_status": "proposed_not_validated",
    }).to_csv(target, index=False)
    print(f"Template written: {target}")


def normalized_entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts))) if len(counts) > 1 else 0.0


def category_colors(order: list[str]) -> dict[str, str]:
    colors, i = {}, 0
    for name in order:
        if name == UNCLASSIFIED:
            colors[name] = GREY
        else:
            colors[name] = PALETTE[i % len(PALETTE)]; i += 1
    return colors


def save_figure(fig, base: Path, formats=("png", "pdf", "svg")) -> None:
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def pie_chart(dist: pd.DataFrame, colors: dict, out_base: Path) -> None:
    """One slice per category; percentages on slices (outside with leader line when small)."""
    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0.12, 0.04, 0.76, 0.66])
    values = dist["n_workflows"].values
    pct = dist["percent"].values
    wedges, _ = ax.pie(values, colors=[colors[c] for c in dist["category"]], startangle=90,
                       counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 1.0})
    for w, p in zip(wedges, pct):
        ang = np.deg2rad((w.theta2 + w.theta1) / 2)
        x, y = np.cos(ang), np.sin(ang)
        label = f"{p:.1f}%"
        if p >= 4.0:
            ax.text(0.66 * x, 0.66 * y, label, ha="center", va="center", fontsize=9,
                    color="white" if w.get_facecolor()[:3] != (1, 1, 1) else "black", fontweight="bold")
        else:
            ax.annotate(label, xy=(0.98 * x, 0.98 * y), xytext=(1.22 * x, 1.22 * y), fontsize=8.5,
                        ha="left" if x >= 0 else "right", va="center",
                        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.6})
    ax.set_aspect("equal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in dist["category"]]
    fig.legend(handles, dist["category"], loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.4, handlelength=1.2)
    save_figure(fig, out_base)


def bar_chart(dist: pd.DataFrame, colors: dict, out_base: Path) -> None:
    d = dist.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(d) + 1.2))
    bars = ax.barh([textwrap.fill(c, 42) for c in d["category"]], d["n_workflows"],
                   color=[colors[c] for c in d["category"]])
    for b, n, p in zip(bars, d["n_workflows"], d["percent"]):
        ax.text(b.get_width() + d["n_workflows"].max() * 0.01, b.get_y() + b.get_height() / 2,
                f"{n:,} ({p:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, d["n_workflows"].max() * 1.22)
    ax.set_xlabel("Number of workflows")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_figure(fig, out_base, formats=("png", "pdf"))


def stacked_by_dataset(by_ds_pct: pd.DataFrame, colors: dict, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bottoms = np.zeros(len(by_ds_pct.columns))
    for cat in by_ds_pct.index:
        vals = by_ds_pct.loc[cat].values
        ax.bar([DATASET_LABEL.get(c, c) for c in by_ds_pct.columns], vals, bottom=bottoms,
               color=colors[cat], label=cat, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v >= 5:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        bottoms += vals
    ax.set_ylabel("Share of dataset workflows (%)")
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7.5)
    save_figure(fig, out_base, formats=("png", "pdf"))


def embedding_plots(cfg: Config, paths: Paths, assign: pd.DataFrame, colors: dict, sample_n: int = 20000) -> dict:
    from sklearn.decomposition import PCA

    emb, rows = load_embeddings(cfg, paths)
    cats = assign.loc[rows.index, "category"].values
    clusters = assign.loc[rows.index, "cluster_id"].values
    rng = np.random.RandomState(cfg.seed)
    idx = rng.choice(len(emb), size=min(sample_n, len(emb)), replace=False)
    pca = PCA(n_components=2, random_state=cfg.seed).fit(emb)
    pts = pca.transform(emb[idx])
    info = {"pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "plotted_points": int(len(idx))}

    def scatter(labels, palette, name, legend_title):
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for lab in sorted(set(labels[idx]), key=str):
            m = labels[idx] == lab
            ax.scatter(pts[m, 0], pts[m, 1], s=3, alpha=0.45, color=palette[lab], label=str(lab), linewidths=0)
        ax.set_xlabel(f"PCA 1 ({100 * pca.explained_variance_ratio_[0]:.1f}% var.)")
        ax.set_ylabel(f"PCA 2 ({100 * pca.explained_variance_ratio_[1]:.1f}% var.)")
        leg = ax.legend(markerscale=4, fontsize=7, frameon=False, loc="upper center",
                        bbox_to_anchor=(0.5, -0.12), ncol=2, title=legend_title)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        save_figure(fig, paths.figures / name, formats=("png",))

    scatter(cats, colors, "embedding_by_category", "Category (2-D PCA is illustrative only)")
    cl_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(set(clusters)))}
    scatter(clusters, cl_colors, "embedding_by_cluster", "Cluster")
    return info


def cluster_size_chart(assign: pd.DataFrame, paths: Paths) -> None:
    sizes = assign[assign["cluster_id"] >= 0]["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(sizes.index.astype(str), sizes.values, color="#4E79A7")
    for i, v in enumerate(sizes.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Workflows")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_figure(fig, paths.figures / "cluster_size_distribution", formats=("png",))


def length_chart(assign: pd.DataFrame, paths: Paths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    data = [np.log10(assign.loc[assign["dataset"] == d, "n_words"].clip(lower=1)) for d in DATASET_ORDER]
    axes[0].violinplot(data, showmedians=True)
    axes[0].set_xticks(range(1, 4)); axes[0].set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
    axes[0].set_ylabel("log10(words per enriched description)")
    both = assign[(assign["original_prompt"].fillna("").str.strip() != "")]
    if len(both):
        for d in ("sopbench", "worfbench"):
            sub = both[both["dataset"] == d]
            axes[1].scatter(sub["original_prompt"].str.split().str.len().clip(lower=1), sub["n_words"].clip(lower=1),
                            s=2, alpha=0.25, label=DATASET_LABEL[d])
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("Original prompt (words)"); axes[1].set_ylabel("Enriched description (words)")
        axes[1].legend(markerscale=4, frameon=False, fontsize=7)
        mn = min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]); mx = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
        axes[1].plot([mn, mx], [mn, mx], color="#888888", lw=0.6, ls="--")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.figures / "description_length_distribution", formats=("png", "pdf"))


def generate(cfg: Config, paths: Paths) -> None:
    assign = pd.read_csv(paths.assignments, keep_default_na=False, dtype={"category": str, "category_status": str})
    for col in ("n_words", "cluster_id"):
        assign[col] = pd.to_numeric(assign[col])
    mapping = pd.read_csv(paths.tables / "category_mapping.csv", keep_default_na=False)
    if (mapping["category"].isin(["TODO", ""])).any():
        raise SystemExit("category_mapping.csv still contains TODO/empty categories. Fill them in first.")
    cat_of = dict(zip(mapping["cluster_id"], mapping["category"]))
    status_of = dict(zip(mapping["cluster_id"], mapping["label_status"]))
    assign["category"] = assign["cluster_id"].map(cat_of).fillna(UNCLASSIFIED)
    assign["category_status"] = assign["cluster_id"].map(status_of).fillna("not_clustered")
    assign.loc[assign["cluster_id"] < 0, "category"] = UNCLASSIFIED
    assign.to_csv(paths.assignments, index=False)

    analysed = assign[assign["cluster_id"] >= 0]
    dist = (analysed.groupby("category").size().rename("n_workflows").reset_index()
            .sort_values("n_workflows", ascending=False))
    dist["percent"] = 100 * dist["n_workflows"] / dist["n_workflows"].sum()
    # keep "Generic Process Flows (Domain-Agnostic)" last regardless of size
    dist = pd.concat([dist[dist["category"] != UNCLASSIFIED], dist[dist["category"] == UNCLASSIFIED]], ignore_index=True)
    ct = pd.crosstab(analysed["category"], analysed["dataset"]).reindex(columns=DATASET_ORDER, fill_value=0)
    dist = dist.merge(ct.add_prefix("n_").reset_index(), on="category")
    for d in DATASET_ORDER:
        dist[f"pct_within_{d}"] = 100 * dist[f"n_{d}"] / ct[d].sum()
    dist["cluster_ids"] = dist["category"].map(
        lambda c: ", ".join(map(str, sorted(k for k, v in cat_of.items() if v == c))))
    dist.to_csv(paths.tables / "category_distribution.csv", index=False)

    by_ds = ct.loc[dist["category"]]
    by_ds_pct = 100 * by_ds / by_ds.sum()
    by_ds_pct.to_csv(paths.tables / "category_by_dataset.csv")

    colors = category_colors(dist["category"].tolist())
    pie_chart(dist, colors, paths.figures / "task_category_distribution")
    bar_chart(dist, colors, paths.figures / "task_category_bar_chart")
    stacked_by_dataset(by_ds_pct, colors, paths.figures / "category_distribution_by_dataset")
    cluster_size_chart(assign, paths)
    length_chart(assign, paths)
    emb_info = embedding_plots(cfg, paths, assign, colors)

    # Diversity measures (descriptive, not a quality score)
    diversity = {"normalized_entropy_overall": normalized_entropy(dist["n_workflows"].values.astype(float))}
    for d in DATASET_ORDER:
        diversity[f"normalized_entropy_{d}"] = normalized_entropy(ct[d].values.astype(float))
    diversity["categories_above_5pct"] = int((dist["percent"] >= 5).sum())
    diversity["smallest_category_pct"] = float(dist["percent"].min())

    # Per-category structure summary (structural diversity, kept separate from thematic diversity)
    structure = analysed.groupby("category").agg(
        mean_actions=("num_actions", "mean"), mean_decisions=("num_decisions", "mean"),
        pct_branching=("has_branching", "mean"), pct_parallel=("has_parallelism", "mean"),
        mean_words=("n_words", "mean")).reindex(dist["category"])
    structure.to_csv(paths.tables / "category_structure_summary.csv")

    # Manual-validation sample (to be filled in by a human; nothing is claimed as validated here)
    rng_seed = cfg.seed
    samples = []
    for cat, g in analysed.groupby("category"):
        samples.append(g.sample(n=min(cfg.n_validation_per_category, len(g)), random_state=rng_seed))
    val = pd.concat(samples)[["origin", "dataset", "cluster_id", "category", "description"]].copy()
    val["description"] = val["description"].str[:1200]
    val["manual_category_correct (yes/no/partial)"] = ""
    val["manual_notes"] = ""
    val.to_csv(paths.tables / "category_validation_samples.csv", index=False)

    summary = {
        "loaded_workflows": int(len(assign)), "analysed_in_clustering": int(len(analysed)),
        "not_clustered": int(len(assign) - len(analysed)),
        "k_clusters": int(analysed["cluster_id"].nunique()), "n_categories": int(len(dist)),
        "category_percentages_sum": float(dist["percent"].sum()),
        "diversity": diversity, "embedding_projection": emb_info,
        "label_validation": "NONE - labels are analyst-proposed; category_validation_samples.csv is blank for manual review",
    }
    clustering_summary = paths.root / "clustering_summary.json"
    if clustering_summary.exists():
        summary["clustering"] = json.loads(clustering_summary.read_text())
    (paths.root / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(paths, dist, by_ds_pct, summary, mapping)
    print(dist[["category", "n_workflows", "percent"]].to_string(index=False))
    print(json.dumps(diversity, indent=2))


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_markdown(paths: Paths, dist, by_ds_pct, summary, mapping) -> None:
    stats = pd.read_csv(paths.tables / "dataset_statistics.csv").set_index("dataset")
    meta = json.loads(paths.emb_meta.read_text())
    ksel_path = paths.tables / f"k_selection_{paths.model_tag}.csv"
    ksel = pd.read_csv(ksel_path) if ksel_path.exists() else None
    cmp_path = paths.tables / "embedding_model_comparison.json"
    cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else None
    cs = summary.get("clustering", {})
    bias = cs.get("bias", {})
    comp = stats.loc[["sopbench", "worfbench", "sapsam", "ALL"],
                     ["workflows", "percent_of_total", "unique_origins", "missing_descriptions", "too_short_descriptions",
                      "unique_descriptions", "duplicate_descriptions_beyond_first", "n_words_mean", "n_words_median",
                      "n_words_std", "n_words_min", "n_words_max", "n_sentences_mean"]].reset_index()
    struct = stats.loc[["sopbench", "worfbench", "sapsam", "ALL"],
                       ["num_actions_mean", "num_decisions_mean", "num_forks_mean", "num_joins_mean",
                        "pct_with_branching", "pct_with_parallelism", "pct_with_iteration"]].reset_index()
    d = summary["diversity"]
    top = dist.iloc[0]
    L = []
    L += ["# Enriched SysML v2 dataset: semantic characterisation", "",
          "> All numbers below were produced by `Analysis/enriched_pipeline.py` and `Analysis/enriched_report.py`. "
          "Category labels are **analyst-proposed from cluster keywords and examples; they are not manually validated** "
          "(see `tables/category_validation_samples.csv`).", ""]
    L += ["## 1. Dataset composition", "", md_table(comp), "",
          f"Loaded workflows: **{summary['loaded_workflows']:,}**; used in semantic clustering: **{summary['analysed_in_clustering']:,}** "
          f"(excluded: {summary['not_clustered']} with a description shorter than the minimum word count). "
          "No record was deleted; exact duplicate descriptions are retained and flagged (`duplicate_descriptions_beyond_first`).", ""]
    L += ["## 2. Description characteristics", "",
          "Enriched descriptions are narrative process descriptions (`detailed_description` column). "
          f"Token counts (model tokenizer): median {meta['token_count']['median']:.0f}, p95 {meta['token_count']['p95']:.0f}, max {meta['token_count']['max']:.0f}. "
          f"{meta['documents_exceeding_window']} descriptions ({meta['truncation_rate_if_not_chunked_percent']:.2f}%) exceed the model window of "
          f"{meta['window_tokens']} tokens and would have been truncated; they were split into overlapping chunks instead (see below).", "",
          "Structural statistics (regex-based on the SysML text; see limitations):", "", md_table(struct, "{:.2f}"), ""]
    L += ["## 3. Embedding methodology", "",
          f"Model `{meta['model']}` on device `{meta['device']}`; unit-normalised embeddings; long-text strategy `{meta['long_text_strategy']}` "
          f"({meta['chunk_overlap_tokens']}-token overlap; {meta['n_chunks']:,} chunks for {meta['n_documents']:,} descriptions); pooling: {meta['pooling']}. "
          "Only whitespace/markup normalisation was applied; wording, negation and terminology are untouched.", ""]
    if cmp:
        L += [f"MiniLM vs MPNet on a shared random sample of {cmp['sample_size']:,} workflows (K={cmp['k']}): "
              f"ARI={cmp['adjusted_rand_index']:.3f}, NMI={cmp['normalized_mutual_information']:.3f}, "
              f"best-match purity MiniLM→MPNet={cmp['minilm_to_mpnet_best_match_purity']:.3f}; "
              f"silhouette (cosine) MiniLM={cmp['minilm']['silhouette_cosine']:.3f} vs MPNet={cmp['mpnet']['silhouette_cosine']:.3f}; "
              f"Davies–Bouldin MiniLM={cmp['minilm']['davies_bouldin']:.2f} vs MPNet={cmp['mpnet']['davies_bouldin']:.2f}.", ""]
    L += ["## 4. Clustering methodology and choice of K", "",
          f"K-Means directly on the embeddings (cosine ≈ Euclidean on unit vectors), n_init=10, final K={summary['k_clusters']}. "
          "Candidate K values were compared with silhouette (cosine, fixed 10k sample), Davies–Bouldin, size balance, "
          "seed-to-seed stability (ARI) and dataset NMI:", ""]
    if ksel is not None:
        show = ksel[["k", "silhouette_mean", "davies_bouldin_mean", "stability_ari_mean", "nmi_with_dataset",
                     "smallest_cluster_pct", "largest_cluster_pct"]]
        L += [md_table(show, "{:.3f}"), ""]
    L += ["Metrics were used as guidance only; the final K was chosen for interpretability (distinct, coherent keyword profiles).", ""]
    L += ["## 5. Interpreted categories", "",
          "`tables/category_mapping.csv` maps `cluster_id → category` (editable). Supporting keywords, representative and random examples are in "
          "`tables/cluster_keywords.csv` and `tables/cluster_representative_examples.csv`.", "",
          md_table(mapping[["cluster_id", "category", "n_workflows", "supporting_keywords", "explanation"]].sort_values(["category", "cluster_id"])), ""]
    L += ["## 6. Category distribution", "",
          md_table(dist[["category", "n_workflows", "percent", "n_sopbench", "n_worfbench", "n_sapsam",
                         "pct_within_sopbench", "pct_within_worfbench", "pct_within_sapsam"]], "{:.1f}"), "",
          f"Percentages sum to {summary['category_percentages_sum']:.2f}% (each workflow has exactly one primary category = the category of its K-Means cluster). "
          "Overall percentages are dominated by SAP-SAM (75% of workflows); use the per-dataset columns/figure for source-specific conclusions.", "",
          "Figures: `figures/task_category_distribution.{png,pdf,svg}`, `task_category_bar_chart`, `category_distribution_by_dataset`, "
          "`embedding_by_category`, `embedding_by_cluster`, `cluster_size_distribution`, `description_length_distribution`.", ""]
    L += ["## 7. Source bias and diversity", "",
          f"- NMI(cluster, dataset) = {bias.get('nmi_cluster_vs_dataset', float('nan')):.3f}; Cramér's V = {bias.get('cramers_v_cluster_vs_dataset', float('nan')):.3f}; "
          f"NMI(cluster, sub-source) = {bias.get('nmi_cluster_vs_subsource', float('nan')):.3f}.",
          f"- Share of descriptions beginning with the template phrase \"The process begins\": {100 * bias.get('share_starting_with_the_process_begins', float('nan')):.1f}%.",
          f"- Near-duplicates (cosine ≥ {cs.get('near_duplicate_threshold')}): {cs.get('near_duplicates_total')} workflows (`tables/nearest_neighbor_similarity_by_dataset.csv`).",
          f"- Thematic diversity (normalised entropy of category distribution): overall {d['normalized_entropy_overall']:.3f}; "
          + ", ".join(f"{DATASET_LABEL[k]} {d[f'normalized_entropy_{k}']:.3f}" for k in DATASET_ORDER) + ".",
          f"- Largest category: {top['category']} ({top['percent']:.1f}%); smallest: {d['smallest_category_pct']:.1f}%.",
          "- Thematic (categories), structural (`tables/category_structure_summary.csv`) and source diversity are reported separately.", ""]
    L += ["## 8. Limitations", "",
          "- Categories are unsupervised themes labelled by an analyst; no manual validation has been performed yet.",
          "- Enriched descriptions are LLM-generated narratives with a recurring template, which can add stylistic signal to the embeddings.",
          "- The datasets are business/process and tool-use workflows; categories describe the *domain of the process*, not necessarily software-engineering activity types.",
          "- Structural features come from regular-expression counts on generated SysML, not a full parser; `has_iteration` relies on explicit `first A then B` edges and is a lower bound.",
          "- 2-D PCA plots are illustrative and do not prove category separability.", ""]
    (paths.root / "analysis_report.md").write_text("\n".join(L))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--model", default="minilm")
    p.add_argument("--init-mapping", action="store_true")
    args = p.parse_args(argv)
    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.model = args.model
    paths = Paths(cfg)
    if args.init_mapping:
        init_mapping(paths)
        return 0
    generate(cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
