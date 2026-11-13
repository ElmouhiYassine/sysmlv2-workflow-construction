"""Verify the frozen corpus against its exact sources, manifest, embeddings and parser journal."""
import json
import sqlite3

import numpy as np

from Analysis.enriched_pipeline import Config, Paths, load_embeddings
from rag_experiments.extract_corpus import (
    OUT, ROOT, SOURCES, digest, file_hash, parser_context, parser_key,
    structural_fingerprint, text_fingerprint,
)
import validate_sysml_datasets as validator


def main():
    manifest = json.loads((OUT / "rag_corpus_manifest.json").read_text())
    stats = json.loads((OUT / "rag_corpus_statistics.json").read_text())
    path = OUT / "rag_corpus_2000.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 2000 or len({r["id"] for r in records}) != 2000:
        raise ValueError("Expected 2,000 distinct IDs")
    if {r["origin"] for r in records} != set(SOURCES):
        raise ValueError("Corpus contains an unexpected/missing source")
    if any("::test::" in r["id"] or r["origin"] == "sopbench" for r in records):
        raise ValueError("Test/SOPBench workflow leaked into corpus")
    if [r["id"] for r in records] != manifest["selected_ids"]:
        raise ValueError("Manifest ordering mismatch")
    if file_hash(path) != manifest["corpus_sha256"]:
        raise ValueError("Corpus checksum mismatch")
    for relative, expected in {
        **manifest["source_files_sha256"], **manifest["implementation_sha256"],
    }.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Input/implementation checksum mismatch: {relative}")
    cfg = Config(output_dir=str(ROOT / "enriched_dataset_analysis"))
    import pandas as pd
    assignments = pd.read_csv(
        Paths(cfg).assignments, usecols=["origin", "category"], keep_default_na=False, dtype=str,
    ).set_index("origin")
    originals = {}
    for source in SOURCES:
        originals.update({
            row["origin"]: row for row in validator.read_rows(
                ROOT / "enriched dataset" / cfg.input_files[source]
            )
        })
    for row in records:
        original = originals[row["id"]]
        if row["description"] != original["detailed_description"] or row["sysml"] != original["sysml_model"]:
            raise ValueError(f"Source text was altered: {row['id']}")
        if row["description_sha256"] != digest(row["description"]) or row["sysml_sha256"] != digest(row["sysml"]):
            raise ValueError(f"Text checksum mismatch: {row['id']}")
        if row["category"] != assignments.loc[row["id"], "category"] or row["category"] != row["analysis_category"]:
            raise ValueError(f"Category-analysis assignment mismatch: {row['id']}")
        if row["domain"] is not None or (row["origin"] == "sapsam" and row["scenario"] is not None):
            raise ValueError(f"Unexpected domain/scenario metadata: {row['id']}")
    if len({text_fingerprint(r["description"]) for r in records}) != len(records):
        raise ValueError("Duplicate description")
    if len({r["sysml"] for r in records}) != len(records):
        raise ValueError("Duplicate SysML")
    if len({structural_fingerprint(r["sysml"]) for r in records}) != len(records):
        raise ValueError("Duplicate labelled graph")
    emb, rows = load_embeddings(cfg, Paths(cfg))
    positions = {origin: i for i, origin in enumerate(rows.origin)}
    vectors = emb[[positions[r["id"]] for r in records]]
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1)
    maximum = float(similarities.max())
    if maximum >= manifest["near_duplicate_threshold"]:
        raise ValueError("Near-duplicate pair remains")
    context = parser_context()
    if context != manifest["parser_context_sha256"]:
        raise ValueError("Kernel/library context changed")
    db = sqlite3.connect(f"file:{OUT / 'extraction_state.sqlite'}?mode=ro", uri=True)
    try:
        for row in records:
            result = db.execute(
                "SELECT valid FROM parser WHERE cache_key=?", (parser_key(row, context),)
            ).fetchone()
            if result != (1,) or row["parser_valid"] is not True:
                raise ValueError(f"Missing original-text kernel validity evidence: {row['id']}")
    finally:
        db.close()
    if stats["total_selected"] != len(records) or stats["shortage"] != 0:
        raise ValueError("Statistics mismatch")
    print(f"Verified {len(records)} source-preserved, parser-valid workflows; "
          f"maximum pairwise cosine={maximum:.6f}; sources={stats['by_source']}")
    print(path)


if __name__ == "__main__":
    main()
