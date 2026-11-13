"""Create separate proportional, quality-filtered sequential/complex RAG sets."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import random
import re
import sqlite3

import numpy as np
import pandas as pd

from Analysis.rag_quality import QualityLimits
from Analysis.sysml_workflow import UnsupportedSyntax, extract_complexity_features, parse_workflow
from rag_experiments.extract_corpus import (
    OUT, ROOT, assess_original, connect_state, digest, file_hash,
    parser_context, parser_key, prepare_inputs, structural_fingerprint,
    text_fingerprint, validate_pending, write_json,
)


def set_kind(features):
    if features["num_actions"] > 8 or features["num_actions"] < 1:
        return None
    if not any(features[k] for k in ("num_decisions", "num_forks", "num_joins")) and not features["has_cycles"]:
        return "sequential"
    if 1 <= features["num_decisions"] <= 2 and 1 <= features["num_forks"] <= 2:
        return "complex"
    return None


def meaningful_names(workflow):
    return all(
        len(name) > 1 and not re.fullmatch(r"(?:action|step|node|task)\d*", name, re.I)
        for name in workflow.actions
    )


def proportional_select(pool, embeddings, count=500, seed=42, threshold=0.90):
    """Largest proportional deficit first; soft targets yield to quality and duplicates."""
    groups = {}
    rng = random.Random(seed)
    for row in sorted(pool, key=lambda r: r["id"]):
        groups.setdefault(row["category"], []).append(row)
    for rows in groups.values():
        rng.shuffle(rows)
        rows.sort(key=lambda r: -r["structure"]["quality_score"])
    targets = {cat: len(rows) / len(pool) * count for cat, rows in groups.items()} if pool else {}
    actual, positions = Counter(), Counter()
    selected, vectors, near_duplicates = [], [], set()
    fingerprints = set()
    while groups and len(selected) < count:
        cat = min(groups, key=lambda c: (-(targets[c] - actual[c]), c))
        row = groups[cat][positions[cat]]
        positions[cat] += 1
        if positions[cat] == len(groups[cat]):
            del groups[cat]
        keys = {
            ("description", text_fingerprint(row["description"])),
            ("sysml", digest(row["sysml"])),
            ("graph", row["_graph"]),
        }
        vector = embeddings[row["emb_row"]]
        if keys & fingerprints or (
            vectors and float(np.max(np.asarray(vectors) @ vector)) >= threshold
        ):
            near_duplicates.add(row["id"])
            continue
        selected.append(row)
        vectors.append(vector)
        fingerprints.update(keys)
        actual[cat] += 1
    return selected, targets, near_duplicates


def main():
    args = argparse.Namespace(exclude_ids=set(), leak_threshold=0.95)
    records, embeddings, input_counts, hashes = prepare_inputs(args)
    limits = QualityLimits(
        max_actions=8, max_decisions=2, max_forks=2, max_joins=2,
        max_description_chars=1500,
    )
    pools = {"sequential": [], "complex": []}
    rejections = Counter()
    audit = []
    for row in records:
        try:
            features = extract_complexity_features(parse_workflow(row["sysml"]), row["sysml"])
            kind = set_kind(features)
        except UnsupportedSyntax:
            rejections["unsupported_original_syntax"] += 1
            continue
        if kind is None:
            rejections["outside_structural_limits"] += 1
            continue
        # Simple sequences cannot include undeclared implicit branching or synchronization;
        # shared quality checks enforce one successor per action and connected termination.
        result = assess_original(row, limits)
        if result["quality_reasons"]:
            rejections.update(result["quality_reasons"])
            audit.append({"origin": row["id"], "set": kind, "reasons": result["quality_reasons"]})
            continue
        if not meaningful_names(parse_workflow(row["sysml"])):
            rejections["nonmeaningful_action_identifier"] += 1
            continue
        row["structure"] = result
        row["_graph"] = structural_fingerprint(row["sysml"])
        pools[kind].append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    context = parser_context()
    db = connect_state(OUT / "stratified_1000_state.sqlite")
    existing_path = OUT / "extraction_state.sqlite"
    prior = sqlite3.connect(f"file:{existing_path}?mode=ro", uri=True) if existing_path.exists() else None
    try:
        # Reuse only hash-matching original-text kernel evidence; leave the previous state untouched.
        for pool in pools.values():
            for row in pool:
                key = parser_key(row, context)
                if prior:
                    saved = prior.execute("SELECT valid, error FROM parser WHERE cache_key=?", (key,)).fetchone()
                    if saved:
                        db.execute("INSERT OR IGNORE INTO parser VALUES (?, ?, ?)", (key, *saved))
        db.commit()
        summaries, distributions = {}, []
        for kind, quality_pool in pools.items():
            invalid = set()
            while True:
                eligible = []
                for row in quality_pool:
                    saved = db.execute("SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)).fetchone()
                    if saved == (0,):
                        invalid.add(row["id"])
                    else:
                        eligible.append(row)
                selected, targets, duplicates = proportional_select(eligible, embeddings)
                print(f"[{kind}] quality eligible={len(eligible)} selected={len(selected)}/500", flush=True)
                validate_pending(selected, db, context, 40)
                failed = {
                    row["id"] for row in selected if db.execute(
                        "SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)
                    ).fetchone() != (1,)
                }
                if not failed:
                    break
                invalid.update(failed)
            replay, _, _ = proportional_select(eligible, embeddings)
            if [r["id"] for r in selected] != [r["id"] for r in replay]:
                raise RuntimeError("Deterministic replay failed")
            allocation = Counter(r["category"] for r in eligible)
            actual = Counter(r["category"] for r in selected)
            # Include existing categories with no qualifying workflows explicitly.
            for cat in sorted({r["category"] for r in records}):
                distributions.append({
                    "set": kind, "category": cat, "eligible": allocation[cat],
                    "target": targets.get(cat, 0), "actual": actual[cat],
                    "difference": actual[cat] - targets.get(cat, 0),
                })
            vectors = embeddings[[r["emb_row"] for r in selected]]
            pair = vectors @ vectors.T
            np.fill_diagonal(pair, -1)
            maximum = float(pair.max()) if len(selected) > 1 else None
            if maximum is not None and maximum >= 0.90:
                raise RuntimeError("Near-duplicate remains in final set")
            path = OUT / f"rag_{kind}_500.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                fields = [
                    "origin", "category", "user_prompt", "sysml_code", "description",
                    "num_actions", "num_decisions", "num_forks", "num_joins", "parser_valid",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in selected:
                    writer.writerow({
                        "origin": row["id"], "category": row["category"],
                        "user_prompt": row["original_metadata"]["user_prompt"],
                        "sysml_code": row["sysml"], "description": row["description"],
                        **{key: row["structure"][key] for key in fields if key.startswith("num_")},
                        "parser_valid": True,
                    })
            csv.field_size_limit(2**31 - 1)
            with path.open(encoding="utf-8", newline="") as stream:
                exported = list(csv.DictReader(stream))
            if len(exported) != len(selected):
                raise RuntimeError("CSV row count mismatch")
            for out, row in zip(exported, selected):
                if out["sysml_code"] != row["sysml"] or out["user_prompt"] != row["original_metadata"]["user_prompt"]:
                    raise RuntimeError("Original fields changed")
                if set_kind(row["structure"]) != kind:
                    raise RuntimeError("Structural constraint violated")
            summaries[kind] = {
                "selected": len(selected), "eligible": len(eligible), "shortage": 500 - len(selected),
                "sources": dict(Counter(r["origin"] for r in selected)),
                "categories": dict(sorted(actual.items())),
                "counts": {
                    name: dict(sorted(Counter(r["structure"][name] for r in selected).items()))
                    for name in ("num_actions", "num_decisions", "num_forks", "num_joins")
                },
                "parser_valid_selected": len(selected),
                "parser_invalid_excluded": len(invalid),
                "near_or_exact_duplicate_exclusions": len(duplicates),
                "maximum_pairwise_cosine": maximum,
                "seed_replay_verified": True, "csv_sha256": file_hash(path),
            }
            print(f"[{kind}] final={len(selected)} shortage={500-len(selected)} path={path}", flush=True)
        pd.DataFrame(distributions).to_csv(OUT / "rag_stratified_category_allocations.csv", index=False)
        write_json(OUT / "rag_stratified_1000_summary.json", {
            "seed": 42, "requested_per_set": 500, "sets": summaries,
            "input_counts": input_counts, "input_sha256": hashes,
            "quality_limits": limits.__dict__, "quality_rejections": dict(rejections),
            "script_sha256": file_hash(ROOT / "rag_experiments" / "extract_stratified_1000.py"),
            "parser_context": context,
            "description": "Original detailed_description retained separately; user_prompt preserved including empty SAP-SAM prompts",
            "allocation": "eligible category proportion * 500, separately per set; largest soft deficit first",
            "eligibility_denominator": "Static quality-qualified original workflows after structural/length/name filters "
                                       "and known parser failures, before selected-set deduplication. "
                                       "No duplicate filtering is relaxed to meet allocations.",
            "validation_scope": "Final selected original SysML checked by shared kernel; eligible pools pass static quality checks, "
                                "known parser failures excluded, unselected candidates not all kernel-assessed",
        })
        write_json(OUT / "rag_stratified_quality_rejections.json", audit)
        print(json.dumps(summaries, indent=2))
    finally:
        if prior:
            prior.close()
        db.close()


if __name__ == "__main__":
    main()
