import json

import numpy as np

from Analysis.rag_quality import QualityLimits
from rag_experiments.extract_corpus import (
    annotate_strata,
    assess_original,
    connect_state,
    deduplicate_pool,
    diverse_select,
    digest,
    parser_key,
    statistics,
)


def example(index, source="worfbench", scenario="toolbench"):
    model = f"""package P{index} {{ action def W {{
        action receiveInvoice {{ description = "Receive invoice {index}."; }}
        action archiveInvoice {{ description = "Archive invoice {index}."; }}
        first start then receiveInvoice;
        first receiveInvoice then archiveInvoice;
        first archiveInvoice then done;
    }} }}"""
    row = {
        "id": f"{source}::{scenario}::train::{index}", "origin": source,
        "category": "Existing derived label",
        "scenario": scenario if source == "worfbench" else None, "domain": None,
        "analysis_category": "Existing derived label",
        "description": f"Receive invoice {index} first and then archive invoice {index} to complete the process.",
        "sysml": model, "emb_row": index,
    }
    return row


def prepared(rows):
    for row in rows:
        result = assess_original(row, QualityLimits())
        assert not result["quality_reasons"]
        row["structure"] = result
    annotate_strata(rows)
    return rows


def test_original_text_preserved_while_assessing_legacy_syntax():
    row = example(0)
    before = json.dumps(row)
    result = assess_original(row, QualityLimits())
    assert not result["quality_reasons"]
    assert json.dumps(row) == before
    assert "agent =" not in row["sysml"]
    assert "::> description" not in row["sysml"]


def test_balances_sources_native_scenarios_sizes_and_replays_seed():
    rows = prepared([
        example(i, "worfbench" if i < 12 else "sapsam", "toolbench" if i % 2 else "wikihow")
        for i in range(24)
    ])
    # Distinct text, diverse embeddings; no assumption that structural quotas are mandatory.
    selected, _ = diverse_select(rows, np.eye(24), 12, 0.9, 42)
    again, _ = diverse_select(rows, np.eye(24), 12, 0.9, 42)
    assert [r["id"] for r in selected] == [r["id"] for r in again]
    assert sum(r["origin"] == "worfbench" for r in selected) == 6
    assert {r["scenario"] for r in selected if r["origin"] == "worfbench"} == {"toolbench", "wikihow"}
    assert all(r["category"] == "Existing derived label" for r in selected)
    assert all(r["scenario"] is None for r in selected if r["origin"] == "sapsam")


def test_no_near_duplicate_fallback_and_shortage_is_visible():
    rows = prepared([example(0), example(1, "sapsam")])
    selected, rejected = diverse_select(rows, np.ones((2, 1)), 2, 0.9, 42)
    assert len(selected) == 1
    assert len(rejected["near_duplicate_rejections"]) == 1


def test_exact_duplicate_ignores_package_rename():
    rows = prepared([example(0), example(1)])
    rows[1]["description"] = rows[0]["description"] + " "
    rows[1]["sysml"] = rows[0]["sysml"].replace("package P0", "package P1")
    kept, rejected = deduplicate_pool(rows)
    assert len(kept) == len(rejected) == 1


def test_invalid_or_disconnected_original_not_repaired():
    row = example(0)
    row["sysml"] = row["sysml"].replace("first receiveInvoice then archiveInvoice;", "")
    result = assess_original(row, QualityLimits())
    assert result["quality_reasons"]
    assert "first receiveInvoice then archiveInvoice;" not in row["sysml"]


def test_parser_state_is_bound_to_exact_original_and_kernel_context(tmp_path):
    row = example(0)
    db = connect_state(tmp_path / "state.sqlite")
    key = parser_key(row, "kernel1")
    db.execute("INSERT INTO parser VALUES (?, ?, ?)", (key, 1, ""))
    db.commit()
    assert db.execute("SELECT valid FROM parser WHERE cache_key=?", (key,)).fetchone() == (1,)
    assert parser_key(row, "kernel2") != key
    assert parser_key({**row, "sysml": row["sysml"] + "\n"}, "kernel1") != key
    db.close()
    assert digest(row["description"]) == digest(row["description"])


def test_statistics_verify_all_pairs_and_null_original_domains():
    rows = prepared([example(0), example(1, "sapsam")])
    report = statistics(rows, {}, rows, {}, {}, set(), 0.9, np.eye(2))
    assert report["total_selected"] == 2
    assert report["maximum_pairwise_description_cosine"] == 0
    assert report["original_domains"] == {"missing": 2}
    assert report["by_source"] == {"sapsam": 1, "worfbench": 1}
