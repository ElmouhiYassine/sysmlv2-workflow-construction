"""Extract an immutable, diverse WorFBench/SAP-SAM corpus; run from the repository root.

python -m rag_experiments.extract_corpus
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

import numpy as np
import pandas as pd

from Analysis.analyze_dataset_embeddings import get_dataset_from_origin, get_source_from_origin
from Analysis.enriched_pipeline import COLUMN_MAP, Config, Paths, load_embeddings, normalize_description
from Analysis.normalize_rag_pool import _validate_batch
from Analysis.rag_quality import QualityLimits, assess_quality
from Analysis.rag_selection import assign_complexity_levels, partition_masks
from Analysis.sysml_workflow import UnsupportedSyntax, normalize_workflow, parse_workflow
import validate_sysml_datasets as validator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rag_experiments" / "data"
SOURCES = ("worfbench", "sapsam")
VERSION = 1


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def connect_state(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS quality "
        "(cache_key TEXT PRIMARY KEY, assessment TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS parser "
        "(cache_key TEXT PRIMARY KEY, valid INTEGER NOT NULL, error TEXT NOT NULL)"
    )
    return db


def prepare_inputs(args) -> tuple[list[dict], np.ndarray, dict, dict]:
    cfg = Config(output_dir=str(ROOT / "enriched_dataset_analysis"))
    paths = Paths(cfg)
    emb, cached = load_embeddings(cfg, paths)
    cached = cached.reset_index(drop=True)
    cached["emb_row"] = np.arange(len(cached))
    cached = cached.set_index("origin")
    assignments = pd.read_csv(paths.assignments, keep_default_na=False, dtype=str).set_index(
        "origin"
    )
    if not cached.index.is_unique or not assignments.index.is_unique:
        raise ValueError("Cached embedding/assignment identifiers are not unique")
    records, hashes, seen = [], {}, set()
    counts = Counter()
    train_scenarios, test_scenarios = set(), set()
    for source in SOURCES:
        path = ROOT / "enriched dataset" / cfg.input_files[source]
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
        rows = validator.read_rows(path)
        counts[f"input_{source}"] = len(rows)
        for row in rows:
            missing = set(COLUMN_MAP.values()) - set(row)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            origin = row[COLUMN_MAP["origin"]]
            if get_dataset_from_origin(origin) != source:
                raise ValueError(f"{path}: unexpected source identifier {origin}")
            if origin in seen:
                raise ValueError(f"Duplicate original workflow ID: {origin}")
            seen.add(origin)
            if source == "worfbench":
                scenario = get_source_from_origin(origin)
                (test_scenarios if "::test::" in origin else train_scenarios).add(scenario)
            if "::test::" in origin or origin in args.exclude_ids:
                counts["heldout_excluded"] += 1
                continue
            description = row[COLUMN_MAP["description"]]
            if len(description.split()) < cfg.min_words:
                counts["missing_or_short_description_excluded"] += 1
                continue
            if origin not in cached.index:
                raise ValueError(f"{origin}: no cached description embedding; regenerate existing cache first")
            cache = cached.loc[origin]
            if normalize_description(description) != cache.description:
                raise ValueError(f"{origin}: cached embedding description differs from source")
            scenario = get_source_from_origin(origin) if source == "worfbench" else None
            if origin not in assignments.index or not assignments.loc[origin, "category"]:
                raise ValueError(f"{origin}: missing category-analysis assignment")
            category = assignments.loc[origin, "category"]
            records.append({
                "id": origin, "origin": source, "category": category, "domain": None,
                "scenario": scenario, "description": description,
                "sysml": row[COLUMN_MAP["sysml"]],
                "original_metadata": {
                    key: value for key, value in row.items()
                    if key not in {COLUMN_MAP["description"], COLUMN_MAP["sysml"]}
                },
                "analysis_category": category,
                "emb_row": int(cache.emb_row),
            })
    # Leakage uses only held-out description embeddings, never held-out gold model structure.
    indexed = cached.reset_index()
    allowed = indexed.origin.map(get_dataset_from_origin).isin(SOURCES)
    heldout, leak, _ = partition_masks(
        indexed[allowed], emb, cfg, args.exclude_ids, args.leak_threshold
    )
    leak_ids = set(indexed.loc[leak.index[leak], "origin"])
    records = [row for row in records if row["id"] not in leak_ids]
    counts["leak_neighbours_excluded"] = len(leak_ids)
    counts["heldout_description_references"] = int(heldout.sum())
    counts["training_candidates"] = len(records)
    counts["test_only_worfbench_scenarios"] = sorted(test_scenarios - train_scenarios)
    hashes[str(paths.emb_file.relative_to(ROOT))] = file_hash(paths.emb_file)
    hashes[str(paths.emb_index.relative_to(ROOT))] = file_hash(paths.emb_index)
    hashes[str(paths.assignments.relative_to(ROOT))] = file_hash(paths.assignments)
    return records, emb, dict(counts), hashes


def assess_original(row: dict, limits: QualityLimits) -> dict:
    """Canonicalize only an in-memory assessment view; never replace the original pair."""
    try:
        original = parse_workflow(row["sysml"])
        if original.duplicate_declarations or original.duplicate_control_nodes or original.duplicate_agent_declarations:
            return {"quality_reasons": ["duplicate_original_declarations"]}
        if any(typ is None for action in original.actions.values() for _, typ in action.outputs):
            return {"quality_reasons": ["untyped_original_output"]}
        canonical, _, _ = normalize_workflow(row["sysml"])
        result = assess_quality(canonical, row["description"], limits, row["sysml"])
        return result
    except UnsupportedSyntax as exc:
        return {"quality_reasons": ["unsupported_or_inconsistent_original"], "quality_details": [str(exc)]}


def quality_pool(records, db, limits, implementation_hash):
    qualified, audit = [], []
    reasons = Counter()
    for index, row in enumerate(sorted(records, key=lambda x: x["id"]), start=1):
        key = digest(json.dumps([
            row["id"], row["sysml"], row["description"], asdict(limits), implementation_hash,
        ], sort_keys=True))
        saved = db.execute("SELECT assessment FROM quality WHERE cache_key=?", (key,)).fetchone()
        if saved:
            result = json.loads(saved[0])
        else:
            result = assess_original(row, limits)
            db.execute("INSERT INTO quality VALUES (?, ?)", (key, json.dumps(result)))
        failures = result["quality_reasons"]
        audit.append({
            "id": row["id"], "origin": row["origin"], "category": row["category"],
            "analysis_category": row["analysis_category"], "reasons": failures,
        })
        if failures:
            reasons.update(failures)
        else:
            qualified.append({**row, "structure": {
                name: value for name, value in result.items()
                if name not in {"quality_reasons", "quality_details", "structural_errors", "intentional_exception"}
            }})
        if index % 1000 == 0 or index == len(records):
            db.commit()
            print(f"Quality {index}/{len(records)} qualified={len(qualified)}", flush=True)
    return qualified, audit, dict(sorted(reasons.items()))


def text_fingerprint(text: str) -> str:
    return digest(" ".join(text.lower().split()))


def structural_fingerprint(code: str) -> str:
    """Identifier-independent labelled graph; conservative extra duplicate check."""
    wf = parse_workflow(code)
    labels = {"start": "start", "done": "done"}
    labels.update({
        name: "action:" + " ".join((action.description or "").lower().split())
        for name, action in wf.actions.items()
    })
    labels.update({name: node.kind for name, node in wf.nodes.items()})
    for _ in range(3):
        labels = {
            name: digest(json.dumps([
                label,
                sorted(labels[b] for a, b in wf.edges() if a == name),
                sorted(labels[a] for a, b in wf.edges() if b == name),
            ])) for name, label in labels.items()
        }
    return digest(json.dumps(sorted(labels.values())))


def deduplicate_pool(pool):
    """Keep the best quality representative, not the first input file's representative."""
    fingerprints, kept, rejected = set(), [], []
    for row in sorted(pool, key=lambda x: (-x["structure"]["quality_score"], len(x["description"]), x["id"])):
        keys = (
            ("description", text_fingerprint(row["description"])),
            ("sysml", digest(row["sysml"])),
            ("labelled_graph", structural_fingerprint(row["sysml"])),
        )
        if any(key in fingerprints for key in keys):
            rejected.append(row["id"])
        else:
            kept.append({**row, "_graph_fingerprint": keys[2][1]})
            fingerprints.update(keys)
    return sorted(kept, key=lambda x: x["id"]), rejected


def annotate_strata(pool):
    features = pd.DataFrame([row["structure"] for row in pool])
    levels, complexity_config = assign_complexity_levels(features)
    for row, (_, level) in zip(pool, levels.iterrows()):
        features = row["structure"]
        features["complexity_level"] = level.complexity_level
        features["complexity_score"] = float(level.complexity_score)
        actions = features["num_actions"]
        features["size"] = "small" if actions <= 5 else "medium" if actions <= 15 else "large"
        features["structure_family"] = (
            "combined" if features["num_decisions"] and features["num_forks"]
            else "nested_parallel" if features["has_nested_parallelism"]
            else "parallel" if features["has_parallelism"]
            else "loop" if features["has_cycles"]
            else "conditional" if features["num_decisions"] else "sequential"
        )
        row["sampling_category"] = row["scenario"] or row["category"]
        row["sampling_category_provenance"] = (
            "original_worfbench_source" if row["scenario"]
            else "existing_embedding_cluster_label" if row["analysis_category"] else "missing"
        )
    return complexity_config


def diverse_select(pool, emb, count, threshold, seed):
    """Water-fill sources/categories then size/structure/complexity cells; never relax filters."""
    rng = random.Random(seed)
    ordered = list(pool)
    rng.shuffle(ordered)
    buckets = defaultdict(list)
    for row in ordered:
        s = row["structure"]
        cell = (row["origin"], row["sampling_category"], s["size"], s["structure_family"], s["complexity_level"])
        buckets[cell].append(row)
    # Quality is the admission gate; prioritize it within each diversity stratum.
    for cell in buckets:
        buckets[cell].sort(key=lambda row: -row["structure"]["quality_score"])
    positions = Counter()
    source_counts, category_counts, cell_counts = Counter(), Counter(), Counter()
    picked, vectors = [], []
    descriptions, models, graphs = set(), set(), set()
    near_rejected, exact_rejected = set(), set()
    while buckets and len(picked) < count:
        cell = min(buckets, key=lambda c: (
            source_counts[c[0]], category_counts[c[:2]], cell_counts[c],
            positions[c], str(c),
        ))
        row = buckets[cell][positions[cell]]
        positions[cell] += 1
        if positions[cell] == len(buckets[cell]):
            del buckets[cell]
        desc, code = text_fingerprint(row["description"]), digest(row["sysml"])
        graph = row.get("_graph_fingerprint") or structural_fingerprint(row["sysml"])
        if desc in descriptions or code in models or graph in graphs:
            exact_rejected.add(row["id"])
            continue
        vector = emb[row["emb_row"]]
        if vectors and float(np.max(np.asarray(vectors) @ vector)) >= threshold:
            near_rejected.add(row["id"])
            continue
        picked.append(row)
        vectors.append(vector)
        descriptions.add(desc)
        models.add(code)
        graphs.add(graph)
        source_counts[cell[0]] += 1
        category_counts[cell[:2]] += 1
        cell_counts[cell] += 1
    return picked, {
        "near_duplicate_rejections": sorted(near_rejected),
        "exact_duplicate_rejections": sorted(exact_rejected),
    }


def parser_context():
    kernel_dir = validator.find_sysml_kernel_dir(validator.KERNEL_DIR)
    kernel_spec = json.loads((kernel_dir / "kernel.json").read_text(encoding="utf-8"))
    jar_arguments = [
        Path(argument)
        for argument in kernel_spec.get("argv", [])
        if str(argument).lower().endswith(".jar")
    ]
    kernel_jars = {
        str(path.resolve()): file_hash(path)
        for path in sorted(kernel_dir.glob("*.jar"))
    }
    if jar_arguments:
        kernel_jar = jar_arguments[0]
        if not kernel_jar.is_absolute():
            kernel_jar = kernel_dir / kernel_jar
        kernel_jars[str(kernel_jar.resolve())] = file_hash(kernel_jar)

    return digest(json.dumps({
        "mosaico": file_hash(validator.MOSAICO_PATH),
        "validator": file_hash(Path(validator.__file__)),
        "batch_helper": file_hash(ROOT / "Analysis" / "normalize_rag_pool.py"),
        "kernelspec": file_hash(kernel_dir / "kernel.json"),
        "kernel_jars": kernel_jars,
    }, sort_keys=True))


def parser_key(row, context):
    return digest(context + row["sysml"])


def validate_pending(selected, db, context, batch_size):
    pending = [
        row for row in selected
        if db.execute("SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)).fetchone() is None
    ]
    if not pending:
        return
    km = kc = None
    try:
        km, kc = validator.start_sysml_kernel()
        validator.load_mosaico_library(kc, validator.MOSAICO_PATH)
        for offset in range(0, len(pending), batch_size):
            rows = pending[offset:offset + batch_size]
            items = [{
                "origin": row["id"], "normalized_sysml": row["sysml"],
                "package": parse_workflow(row["sysml"]).package,
            } for row in rows]
            # Helper aliases package names only in the kernel request, never in the corpus.
            _validate_batch(kc, items)
            for row, item in zip(rows, items):
                db.execute(
                    "INSERT OR REPLACE INTO parser VALUES (?, ?, ?)",
                    (parser_key(row, context), item["parser_valid"], item["parser_error"]),
                )
            db.commit()
            print(
                f"Kernel {min(offset + batch_size, len(pending))}/{len(pending)} "
                f"valid={sum(item['parser_valid'] for item in items)}/{len(items)} in batch",
                flush=True,
            )
    finally:
        if kc is not None:
            kc.stop_channels()
        if km is not None:
            km.shutdown_kernel(now=True)


def counts(values):
    return dict(sorted(Counter("missing" if value is None else str(value) for value in values).items()))


def statistics(selected, input_counts, pool, quality_reasons, duplicates, parser_invalid, threshold, emb):
    structure = [row["structure"] for row in selected]
    vectors = emb[[row["emb_row"] for row in selected]]
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)
    maximum = float(sims.max()) if len(selected) > 1 else None
    if maximum is not None and maximum >= threshold:
        raise RuntimeError(f"Final pairwise cosine {maximum} violates threshold {threshold}")
    return {
        "total_selected": len(selected), "eligible_before_parser": len(pool),
        "eligible_distributions": {
            "sources": counts(row["origin"] for row in pool),
            "sizes": counts(row["structure"]["size"] for row in pool),
            "complexity_levels": counts(row["structure"]["complexity_level"] for row in pool),
            "structure_families": counts(row["structure"]["structure_family"] for row in pool),
        },
        "input_counts": input_counts, "by_source": counts(row["origin"] for row in selected),
        "categories": counts(row["category"] for row in selected),
        "original_scenarios": counts(row["scenario"] for row in selected),
        "original_domains": counts(row["domain"] for row in selected),
        "analysis_categories": counts(row["analysis_category"] for row in selected),
        "sampling_categories": [
            {"source": source, "category": category, "count": n}
            for (source, category), n in sorted(Counter(
                (r["origin"], r["sampling_category"] or "missing") for r in selected
            ).items())
        ],
        "action_counts": counts(s["num_actions"] for s in structure),
        "sizes": counts(s["size"] for s in structure),
        "complexity_levels": counts(s["complexity_level"] for s in structure),
        "structure_families": counts(s["structure_family"] for s in structure),
        "contains": {
            "decisions": sum(s["num_decisions"] > 0 for s in structure),
            "forks": sum(s["num_forks"] > 0 for s in structure),
            "joins": sum(s["num_joins"] > 0 for s in structure),
            "nested_parallelism": sum(s["has_nested_parallelism"] for s in structure),
            "combined_decisions_and_parallelism": sum(bool(s["num_decisions"] and s["num_forks"]) for s in structure),
            "cycles": sum(s["has_cycles"] for s in structure),
            "repeated_action_occurrences": sum(s["repeated_action_occurrences"] > 0 for s in structure),
        },
        "quality_rejection_reasons": quality_reasons,
        "duplicate_exclusions": {key: len(value) for key, value in duplicates.items()},
        "parser_invalid_candidates_excluded": len(parser_invalid),
        "selected_parser_valid": len(selected),
        "selected_parser_validation_success_rate": 1.0 if selected else None,
        "maximum_pairwise_description_cosine": maximum,
        "near_duplicate_threshold": threshold,
        "unique_ids": len(set(row["id"] for row in selected)),
        "unique_descriptions": len(set(text_fingerprint(row["description"]) for row in selected)),
        "unique_sysml_texts": len(set(row["sysml"] for row in selected)),
        "domain_metadata_note": "Neither input CSV provides original domain labels; null is preserved. "
                                "Existing analysis categories are separately labelled derived metadata.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--leak-threshold", type=float, default=0.95)
    ap.add_argument("--exclude-ids", type=Path)
    ap.add_argument("--kernel-batch-size", type=int, default=40)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--max-decisions", type=int, default=20)
    ap.add_argument("--max-forks", type=int, default=10)
    ap.add_argument("--max-joins", type=int, default=10)
    ap.add_argument("--max-description-chars", type=int, default=12000)
    args = ap.parse_args(argv)
    if args.count < 2 or args.kernel_batch_size < 1:
        ap.error("count must be at least two and kernel batch size must be positive")
    if not 0 < args.dup_threshold <= 1 or not 0 < args.leak_threshold <= 1:
        ap.error("cosine thresholds must be in (0, 1]")
    args.exclude_ids = set(args.exclude_ids.read_text().split()) if args.exclude_ids else set()
    limits = QualityLimits(**{
        name: getattr(args, name) for name in (
            "max_actions", "max_decisions", "max_forks", "max_joins", "max_description_chars"
        )
    })
    OUT.mkdir(parents=True, exist_ok=True)
    implementation = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in [
            Path(__file__), ROOT / "Analysis" / "rag_quality.py",
            ROOT / "Analysis" / "sysml_workflow.py", ROOT / "Analysis" / "rag_selection.py",
        ]
    }
    implementation_hash = digest(json.dumps(implementation, sort_keys=True))
    records, emb, input_counts, hashes = prepare_inputs(args)
    db = connect_state(OUT / "extraction_state.sqlite")
    try:
        pool, audit, quality_reasons = quality_pool(records, db, limits, implementation_hash)
        pool, exact_rejected = deduplicate_pool(pool)
        if not pool:
            write_json(OUT / "rag_corpus_statistics.json", {
                "total_selected": 0, "shortage": args.count, "quality_rejection_reasons": quality_reasons,
            })
            raise RuntimeError("No quality-qualified candidates; criteria were not relaxed")
        complexity_config = annotate_strata(pool)
        context = parser_context()
        parser_invalid = {
            row["id"] for row in pool
            if (saved := db.execute(
                "SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)
            ).fetchone()) is not None and saved[0] == 0
        }
        near_rejected, selection_exact_rejected = set(), set()
        while True:
            eligible = [
                row for row in pool
                if not (saved := db.execute(
                    "SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)
                ).fetchone()) or saved[0] == 1
            ]
            selected, duplicate_info = diverse_select(
                eligible, emb, args.count, args.dup_threshold, args.seed,
            )
            near_rejected = set(duplicate_info["near_duplicate_rejections"])
            selection_exact_rejected = set(duplicate_info["exact_duplicate_rejections"])
            print(f"Selected provisionally {len(selected)}/{args.count}; validating original SysML", flush=True)
            validate_pending(selected, db, context, args.kernel_batch_size)
            invalid = {
                row["id"] for row in selected if db.execute(
                    "SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)
                ).fetchone()[0] != 1
            }
            parser_invalid.update(invalid)
            if not invalid:
                break
            print(f"Replacing {len(invalid)} parser-invalid candidates without relaxing filters", flush=True)
        again, _ = diverse_select(eligible, emb, args.count, args.dup_threshold, args.seed)
        if [row["id"] for row in selected] != [row["id"] for row in again]:
            raise RuntimeError("Seed-42 deterministic replay failed")
        if not set(row["origin"] for row in selected) == set(SOURCES):
            raise RuntimeError("Both required sources are not represented")
        selected_ids = {row["id"] for row in selected}
        near_rejected -= selected_ids
        selection_exact_rejected -= selected_ids
        duplicates = {
            "exact_pool_duplicates": exact_rejected,
            "near_duplicate_candidates": sorted(near_rejected),
            "exact_selection_duplicates": sorted(selection_exact_rejected),
        }
        stats = statistics(selected, input_counts, pool, quality_reasons, duplicates, parser_invalid, args.dup_threshold, emb)
        stats["requested"] = args.count
        stats["shortage"] = max(0, args.count - len(selected))
        stats["reproducible_replay_verified"] = True
        stats["unavailable_test_only_worfbench_scenarios"] = input_counts["test_only_worfbench_scenarios"]
        output = OUT / "rag_corpus_2000.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row in selected:
                record = {key: value for key, value in row.items() if key != "emb_row" and not key.startswith("_")}
                record["parser_valid"] = True
                record["sysml_sha256"] = digest(row["sysml"])
                record["description_sha256"] = digest(row["description"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        # Verify serialization preserves both text columns byte-for-byte as Python strings.
        original = {row["id"]: row for row in records}
        exported = [json.loads(line) for line in output.read_text().splitlines()]
        for row in exported:
            if any(row[key] != original[row["id"]][key] for key in ("sysml", "description")):
                raise RuntimeError(f"Source text was altered for {row['id']}")
        write_json(OUT / "rag_corpus_statistics.json", stats)
        with (OUT / "category_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source", "category", "count"])
            writer.writeheader()
            writer.writerows(stats["sampling_categories"])
        write_json(OUT / "rag_corpus_manifest.json", {
            "version": VERSION, "seed": args.seed, "requested": args.count, "selected": len(selected),
            "source_files_sha256": hashes, "implementation_sha256": implementation,
            "corpus_sha256": file_hash(output), "selected_ids": [row["id"] for row in selected],
            "sources": SOURCES, "quality_limits": asdict(limits), "complexity": complexity_config,
            "description_column": COLUMN_MAP["description"], "sysml_column": COLUMN_MAP["sysml"],
            "source_text_preservation_verified": True, "reproducible_replay_verified": True,
            "near_duplicate_threshold": args.dup_threshold, "leak_threshold": args.leak_threshold,
            "explicit_excluded_ids": sorted(args.exclude_ids),
            "parser_context_sha256": context,
            "parser_criterion": "Existing check_sysml: no ERROR: lines and no Jupyter errors; warnings acceptable",
            "parser_validation": "Original code in sequential shared-kernel batches; temporary package aliases only in requests",
            "sampling": "Water-fill sources, native scenario/derived analysis category, size, structure and complexity cells; "
                        "seeded quality ranking within cells; no filter relaxation or duplicate fallback",
            "metadata_provenance": "category and analysis_category use the existing workflow_assignments.csv "
                                   "category-analysis labels for both sources. scenario separately preserves "
                                   "the original WorFBench source label; SAP-SAM scenario and all domain fields are null.",
            "quality_limitation": "Lexical alignment and conservative static checks are proxies, not proof of prose/graph equivalence. "
                                  "Normalization is only an in-memory inspection view; outputs always contain unmodified source code.",
        })
        with (OUT / "candidate_audit.jsonl").open("w", encoding="utf-8") as stream:
            for row in audit:
                row["selected"] = row["id"] in selected_ids
                if row["id"] in exact_rejected:
                    row["reasons"].append("exact_or_labelled_graph_duplicate")
                if row["id"] in near_rejected and not row["selected"]:
                    row["reasons"].append("near_duplicate")
                if row["id"] in parser_invalid:
                    row["reasons"].append("parser_invalid")
                if not row["selected"] and not row["reasons"]:
                    row["reasons"].append("not_selected_by_diversity_budget")
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"\nCorpus: {output}", flush=True)
        if stats["shortage"]:
            print(f"SHORTAGE: {stats['shortage']}; selection criteria were not relaxed", file=sys.stderr)
            return 2
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
