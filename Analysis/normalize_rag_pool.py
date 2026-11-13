"""Normalize and parse-validate the eligible train-side RAG pool, resumably.

Run: ``python -m Analysis.normalize_rag_pool``. The SQLite journal is durable across
interruptions; successful and failed origins are not reprocessed on restart.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from Analysis.enriched_pipeline import Config, Paths
from Analysis.rag_selection import eligible_pool, load_inputs, partition_masks
from Analysis.sysml_workflow import (
    UnsupportedSyntax,
    extract_complexity_features,
    normalize_workflow,
    parse_workflow,
    tokenize,
    validate_workflow_structure,
)
import validate_sysml_datasets as validator


FEATURE_COLUMNS = [
    "num_actions", "num_decisions", "num_forks", "num_joins",
    "control_flow_edges", "conditional_branches", "parallel_branches",
    "max_nested_parallelism_depth", "has_parallelism", "has_nested_parallelism",
    "has_cycles", "repeated_action_occurrences",
    "max_execution_path_depth", "max_incoming_edges",
    "duplicate_action_declarations",
]
MAX_BATCH_SOURCE_CHARS = 150_000


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            origin TEXT PRIMARY KEY,
            normalization_status TEXT NOT NULL,
            parser_valid INTEGER,
            parser_error TEXT NOT NULL,
            transformations TEXT NOT NULL,
            normalized_sysml TEXT NOT NULL,
            features TEXT NOT NULL
        )
    """)
    return db


def _export(db: sqlite3.Connection, destination: Path, original_models: pd.Series) -> None:
    columns = [
        "origin", "normalization_status", "parser_valid", "parser_error", "transformations",
        "structural_valid", "structural_status", "structural_errors",
        "intentional_exception", "normalized_sysml", "original_sysml_model", *FEATURE_COLUMNS,
    ]
    cursor = db.execute(
        "SELECT origin, normalization_status, parser_valid, parser_error, "
        "transformations, normalized_sysml, features FROM results ORDER BY origin"
    )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for origin, status, valid, error, transforms, model, features in cursor:
            record = {
                "origin": origin,
                "normalization_status": status,
                "parser_valid": valid,
                "parser_error": error,
                "transformations": transforms,
                "structural_valid": "",
                "structural_status": "not_assessed",
                "structural_errors": "[]",
                "intentional_exception": "",
                "normalized_sysml": model,
                "original_sysml_model": original_models.get(origin, ""),
            }
            if valid == 1:
                structure = validate_workflow_structure(
                    parse_workflow(model), record["original_sysml_model"]
                )
                record.update({
                    "structural_valid": int(structure["structural_valid"]),
                    "structural_status": structure["structural_status"],
                    "structural_errors": json.dumps(
                        structure["structural_errors"], ensure_ascii=False
                    ),
                    "intentional_exception": structure["intentional_exception"],
                })
            record.update(json.loads(features))
            writer.writerow(record)


_RESERVED_PACKAGE_NAMES = {
    "action", "agent", "decide", "def", "description", "done", "else", "first",
    "fork", "if", "import", "join", "merge", "out", "package", "part", "private",
    "specializes", "start", "then",
}
_BATCH_COUNTER = itertools.count()


def _batch_code(items: list[dict], attempt: int) -> str | None:
    """Join independent models using unique package scopes for a safe parser batch."""
    if sum(len(item["normalized_sysml"]) for item in items) > MAX_BATCH_SOURCE_CHARS:
        return None
    imports = []
    package_bodies = []
    for item in items:
        package = item["package"]
        if package in _RESERVED_PACKAGE_NAMES:
            return None
        tokens = tokenize(item["normalized_sysml"])
        if any(tokens[i][1] == package and tokens[i + 1][1] == "::"
               for i in range(len(tokens) - 1)):
            return None
        alias = "Validation_" + hashlib.sha1(
            f"{item['origin']}:{attempt}".encode("utf-8")
        ).hexdigest()[:12]
        code_lines = []
        for line in item["normalized_sysml"].splitlines():
            stripped = line.strip()
            if stripped.startswith("private import ") or stripped.startswith("import "):
                imports.append(stripped)
            else:
                code_lines.append(line)
        body = "\n".join(code_lines)
        body, replacements = re.subn(
            rf"(?m)^(\s*package\s+){re.escape(package)}(?=\s*\{{)",
            rf"\g<1>{alias}",
            body,
            count=1,
        )
        if replacements != 1:
            return None
        package_bodies.append(body)
    ordered_imports = []
    for required in ("private import MOSAICO::*;", "private import ScalarValues::*;"):
        if required in imports and required not in ordered_imports:
            ordered_imports.append(required)
    ordered_imports.extend(sorted(set(imports) - set(ordered_imports)))
    return "\n".join(ordered_imports + package_bodies)


def _validate_batch(kc, items: list[dict]) -> None:
    """Validate a batch, bisecting failures until errors are attributable to one workflow."""
    if not items:
        return
    attempt = next(_BATCH_COUNTER)
    code = _batch_code(items, attempt) if len(items) > 1 else items[0]["normalized_sysml"]
    if code is None:
        midpoint = len(items) // 2
        _validate_batch(kc, items[:midpoint])
        _validate_batch(kc, items[midpoint:])
        return
    result = validator.check_sysml(kc, code)
    if result["valid"]:
        for item in items:
            item["parser_valid"] = 1
            item["parser_error"] = ""
            item["normalization_status"] = "valid"
        return
    if len(items) > 1:
        midpoint = len(items) // 2
        _validate_batch(kc, items[:midpoint])
        _validate_batch(kc, items[midpoint:])
        return
    item = items[0]
    item["parser_valid"] = 0
    item["parser_error"] = "\n".join(result["errors"] + result["jupyter_errors"])
    item["normalization_status"] = "parser_invalid"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="enriched_dataset_analysis")
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-ids")
    ap.add_argument("--leak-threshold", type=float, default=0.95)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--kernel-batch-size", type=int, default=50,
                    help="Sequentially validate this many independent package models per kernel request.")
    ap.add_argument("--kernel-restart-every", type=int, default=2500,
                    help="Restart the shared kernel after this many pending rows.")
    ap.add_argument("--limit", type=int, help="Process at most N pending rows (smoke tests only).")
    ap.add_argument("--retry-failures", action="store_true",
                    help="Reprocess previously flagged or parser-invalid rows.")
    args = ap.parse_args(argv)
    if args.kernel_batch_size < 1 or args.progress_every < 1 or args.kernel_restart_every < 1:
        ap.error("--kernel-batch-size, --kernel-restart-every, and --progress-every "
                 "must be positive integers")

    cfg = Config(output_dir=args.output_dir, model=args.model, seed=args.seed)
    paths = Paths(cfg)
    out = paths.root / "rag"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_inputs(cfg, paths)
    emb = rows.attrs["emb"]
    excludes = set(Path(args.exclude_ids).read_text().split()) if args.exclude_ids else set()
    heldout, leak, train = partition_masks(rows, emb, cfg, excludes, args.leak_threshold)

    candidates = []
    for category in sorted(rows["category"].unique()):
        group = rows[train & (rows["category"] == category)]
        selected = eligible_pool(group)
        selected.attrs = {}
        candidates.append(selected)
    # Match eligible_pool's per-category de-duplication boundary. A model may legitimately
    # belong to two categories, so keep both origins rather than dropping one globally.
    pool = pd.concat(candidates, ignore_index=False)
    pool = pool.sort_values("origin")

    db_path = out / "normalization_state_v2.sqlite"
    result_path = out / "normalization_results.csv"
    db = _connect(db_path)
    completed = {x[0] for x in db.execute("SELECT origin FROM results")}
    pending = pool[~pool.origin.isin(completed)]
    if args.retry_failures:
        retry = {
            origin for origin, status in db.execute(
                "SELECT origin, normalization_status FROM results"
            ) if status != "valid"
        }
        pending = pd.concat([pending, pool[pool.origin.isin(retry)]])
        pending = pending.drop_duplicates("origin", keep="last")
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(
        f"Eligible train pool={len(pool)} | held-out={int(heldout.sum())} | "
        f"leak-excluded={int(leak.sum())} | already processed={len(completed)} | "
        f"remaining this run={len(pending)}",
        flush=True,
    )

    km = kc = None
    try:
        if len(pending):
            km, kc = validator.start_sysml_kernel()
            validator.load_mosaico_library(kc, validator.MOSAICO_PATH)
        staged = []
        next_kernel_restart = args.kernel_restart_every
        for count, row in enumerate(pending.itertuples(), start=1):
            try:
                normalized, transformations, parsed = normalize_workflow(row.sysml_model)
                features = extract_complexity_features(parsed, row.sysml_model)
                transforms = json.dumps(transformations, ensure_ascii=False)
            except (UnsupportedSyntax, ValueError) as exc:
                status, parser_valid, err = "normalization_failed", None, str(exc)
                normalized, transforms = "", "[]"
                features = {}
                package = ""
            else:
                status, parser_valid, err = "awaiting_parser", None, ""
                package = parsed.package
            staged.append({
                "origin": row.origin,
                "normalization_status": status,
                "parser_valid": parser_valid,
                "parser_error": err,
                "transformations": transforms,
                "normalized_sysml": normalized,
                "features": json.dumps(features, ensure_ascii=False),
                "package": package,
            })
            flush = len(staged) >= args.kernel_batch_size or count == len(pending)
            if not flush:
                continue
            to_validate = [item for item in staged if item["normalization_status"] == "awaiting_parser"]
            _validate_batch(kc, to_validate)
            for item in staged:
                db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item["origin"], item["normalization_status"], item["parser_valid"],
                     item["parser_error"], item["transformations"], item["normalized_sysml"],
                     item["features"]),
                )
            if count % args.progress_every == 0 or count == len(pending):
                db.commit()
                done = count + len(completed)
                n_valid = db.execute(
                    "SELECT COUNT(*) FROM results WHERE parser_valid=1"
                ).fetchone()[0]
                n_invalid = db.execute(
                    "SELECT COUNT(*) FROM results WHERE parser_valid=0"
                ).fetchone()[0]
                n_flagged = db.execute(
                    "SELECT COUNT(*) FROM results WHERE parser_valid IS NULL"
                ).fetchone()[0]
                dataset_name = {
                    "sapsam": "SAP-SAM", "sopbench": "SOPBench", "worfbench": "WorFBench",
                }.get(row.dataset, row.dataset)
                print(
                    f"[{dataset_name}] {done} / {len(pool)} checked; "
                    f"valid={n_valid} invalid={n_invalid} flagged={n_flagged}",
                    flush=True,
                )
            staged.clear()
            if count >= next_kernel_restart and count < len(pending):
                kc.stop_channels()
                km.shutdown_kernel(now=True)
                km, kc = validator.start_sysml_kernel()
                validator.load_mosaico_library(kc, validator.MOSAICO_PATH)
                while next_kernel_restart <= count:
                    next_kernel_restart += args.kernel_restart_every
        db.commit()
    finally:
        if kc is not None:
            kc.stop_channels()
        if km is not None:
            km.shutdown_kernel(now=True)
        db.close()

    db = sqlite3.connect(db_path)
    _export(db, result_path, pool.set_index("origin")["sysml_model"])
    db.close()
    print(f"Saved resumable state: {db_path}\nSaved results: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
