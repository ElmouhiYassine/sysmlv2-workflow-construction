import json
import re

import numpy as np
import pandas as pd
import pytest

from Analysis.rag_quality import QualityLimits, assess_quality
from Analysis.rag_selection import (
    assign_complexity_levels,
    filter_candidates,
    main,
    partition_masks,
    plot_category,
    select_category,
    select_category_complexity,
)
from Analysis.enriched_pipeline import Config
from Analysis.sysml_workflow import normalize_workflow


DESCRIPTION = "Receive the invoice, validate the invoice, and then archive the invoice."


def model(actions=("receiveInvoice", "validateInvoice", "archiveInvoice"), flow=None):
    declarations = "\n".join(
        f'action {name} {{ description = "{name}"; out result : Boolean; }}'
        for name in actions
    )
    if flow is None:
        flow = "\n".join(
            f"first {left} then {right};"
            for left, right in zip(("start", *actions), (*actions, "done"))
        )
    return normalize_workflow(f"package P {{ action def W {{ {declarations} {flow} }} }}")[0]


def candidate(origin="sapsam::1::1", text=None, description=DESCRIPTION, **overrides):
    text = text or model().replace("package P {", "package P_" + re.sub(r"\W", "_", origin) + " {")
    row = {
        "origin": origin, "dataset": origin.split("::")[0], "source": "source",
        "category": "category", "cluster_id": "0", "emb_row": 0,
        "used_in_clustering": True, "missing_description": False, "too_short": False,
        "duplicate_description": False, "sysml_model": text,
        "normalized_sysml": text, "normalization_status": "valid",
        "parser_valid": 1, "parser_error": "", "structural_valid": 1,
        "structural_status": "complete", "structural_errors": "[]",
        "intentional_exception": "", "transformations": "[]", "description": description,
    }
    row.update(overrides)
    return row


def admitted(rows):
    return assign_complexity_levels(filter_candidates(pd.DataFrame(rows), QualityLimits()))[0]


@pytest.mark.parametrize("origin", ["sapsam::1::1", "sopbench::train::1", "worfbench::train::1"])
def test_valid_concise_candidate_passes_across_sources(origin):
    data = filter_candidates(pd.DataFrame([candidate(origin)]), QualityLimits())
    assert data.iloc[0].quality_pass
    assert data.iloc[0].original_sysml_model == candidate(origin)["sysml_model"]


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"parser_valid": 0}, "invalid_syntax"),
        ({"parser_valid": ""}, "parser_unassessed"),
        ({"structural_valid": 0}, "quality_cached_structure_failed_or_missing"),
        ({"normalization_status": "normalization_failed"}, "quality_normalization_failed_or_missing"),
        ({"description": DESCRIPTION + " " * 1500}, "excessive_description_length"),
        ({"description": "Someone should do something unspecified and then perhaps finish somewhere."},
         "quality_description_action_mismatch"),
        ({"description": "TODO: Receive the invoice, validate the invoice, and archive the invoice."},
         "quality_ambiguous_description"),
    ],
)
def test_hard_filter_rejections(overrides, reason):
    data = filter_candidates(pd.DataFrame([candidate(**overrides)]), QualityLimits())
    assert not data.iloc[0].quality_pass
    assert reason in data.iloc[0].quality_reasons


def test_configurable_action_and_description_boundaries():
    assert not assess_quality(model(), DESCRIPTION, QualityLimits())["quality_reasons"]
    result = assess_quality(model(), DESCRIPTION, QualityLimits(max_actions=2))
    assert "excessive_actions" in result["quality_reasons"]
    assert not assess_quality(
        model(), DESCRIPTION, QualityLimits(max_actions=3, max_description_chars=len(DESCRIPTION))
    )["quality_reasons"]
    assert "excessive_description_length" in assess_quality(
        model(), DESCRIPTION, QualityLimits(max_description_chars=len(DESCRIPTION) - 1)
    )["quality_reasons"]


@pytest.mark.parametrize("field", ["max_decisions", "max_forks", "max_joins"])
def test_negative_limits_are_rejected(field):
    with pytest.raises(ValueError):
        QualityLimits(**{field: -1})


def test_fresh_structure_rejects_stale_cached_success_and_start_collision():
    text = model(flow="""
        first start then receiveInvoice;
        first start then validateInvoice;
        first validateInvoice then archiveInvoice;
        first archiveInvoice then done;
    """)
    result = filter_candidates(pd.DataFrame([candidate(text=text)]), QualityLimits()).iloc[0]
    assert not result.quality_pass
    assert "quality_ambiguous_entry" in result.quality_reasons
    assert "quality_structurally_incomplete" in result.quality_reasons


@pytest.mark.parametrize(
    "old,new",
    [
        ("agent = solutionAgent;", "agent = absent;"),
        ("out result : Boolean;", "out result;"),
        ("first receiveInvoice then validateInvoice;", "first receiveInvoice then absent;"),
    ],
)
def test_bad_declarations_and_references_are_excluded(old, new):
    result = assess_quality(model().replace(old, new, 1), DESCRIPTION, QualityLimits())
    assert "quality_invalid_declarations_or_references" in result["quality_reasons"]


def test_undeclared_guard_output_is_rejected():
    text = model(flow="""
        decide choice;
        first start then receiveInvoice;
        first receiveInvoice then choice;
        first choice;
            if receiveInvoice.missing then validateInvoice;
            else archiveInvoice;
        first validateInvoice then done;
        first archiveInvoice then done;
    """.replace("receiveInvoice.missing", "receiveInvoice.result"))
    result = assess_quality(text.replace("receiveInvoice.result", "receiveInvoice.missing"), DESCRIPTION, QualityLimits())
    assert "quality_invalid_declarations_or_references" in result["quality_reasons"]


def test_parallel_and_loop_examples_remain_eligible_when_clear():
    parallel = model(flow="""
        fork f;
        join j;
        first start then f;
        first f then receiveInvoice;
        first f then validateInvoice;
        first receiveInvoice then j;
        first validateInvoice then j;
        first j then archiveInvoice;
        first archiveInvoice then done;
    """)
    result = assess_quality(
        parallel, "Receive the invoice and validate the invoice in parallel, then archive the invoice.",
        QualityLimits(),
    )
    assert result["quality_reasons"] == []
    assert "excessive_forks" in assess_quality(
        parallel, DESCRIPTION, QualityLimits(max_forks=0)
    )["quality_reasons"]
    assert "excessive_joins" in assess_quality(
        parallel, DESCRIPTION, QualityLimits(max_joins=0)
    )["quality_reasons"]
    loop = model(flow="""
        decide d;
        first start then receiveInvoice;
        first receiveInvoice then validateInvoice;
        first validateInvoice then d;
        first d;
            if validateInvoice.result then archiveInvoice;
            else receiveInvoice;
        first archiveInvoice then done;
    """)
    result = assess_quality(
        loop, "Receive the invoice and validate the invoice. If valid, archive the invoice; otherwise repeat.",
        QualityLimits(),
    )
    assert result["quality_reasons"] == []
    assert result["has_cycles"] and result["repeated_action_occurrences"] == 2
    assert "excessive_decisions" in assess_quality(
        loop, DESCRIPTION, QualityLimits(max_decisions=0)
    )["quality_reasons"]


def test_conditional_merge_cannot_be_used_as_a_synchronization_join():
    text = model(flow="""
        decide d;
        join j;
        first start then receiveInvoice;
        first receiveInvoice then d;
        first d;
            if receiveInvoice.result then validateInvoice;
            else archiveInvoice;
        first validateInvoice then j;
        first archiveInvoice then j;
        first j then done;
    """)
    result = assess_quality(text, DESCRIPTION, QualityLimits())
    assert "quality_join_without_matching_fork" in result["quality_reasons"]


def test_nested_parallelism_with_two_forks_and_joins_is_preserved():
    text = model(
        actions=("receiveInvoice", "validateInvoice", "archiveInvoice", "sendInvoice"),
        flow="""
        fork outer;
        fork inner;
        join innerJoin;
        join outerJoin;
        first start then outer;
        first outer then receiveInvoice;
        first outer then inner;
        first inner then validateInvoice;
        first inner then archiveInvoice;
        first validateInvoice then innerJoin;
        first archiveInvoice then innerJoin;
        first innerJoin then outerJoin;
        first receiveInvoice then outerJoin;
        first outerJoin then sendInvoice;
        first sendInvoice then done;
        """,
    )
    result = assess_quality(
        text, "Receive the invoice in parallel with validation and archiving of the invoice. "
              "Validate the invoice and archive the invoice in parallel; synchronize, then send the invoice.",
        QualityLimits(),
    )
    assert result["quality_reasons"] == []
    assert result["has_nested_parallelism"]
    assert result["num_forks"] == result["num_joins"] == 2


def test_guard_output_must_be_available_on_every_path_to_decision():
    text = model(flow="""
        decide d;
        first start then receiveInvoice;
        first receiveInvoice then d;
        first d;
            if validateInvoice.result then validateInvoice;
            else archiveInvoice;
        first validateInvoice then done;
        first archiveInvoice then done;
    """)
    result = assess_quality(text, DESCRIPTION, QualityLimits())
    assert "quality_guard_output_not_available_on_all_paths" in result["quality_reasons"]


def test_description_cannot_claim_nonexistent_parallelism():
    result = assess_quality(
        model(), "Receive the invoice, validate the invoice and archive the invoice in parallel.",
        QualityLimits(),
    )
    assert "quality_description_claims_absent_parallelism" in result["quality_reasons"]


def test_exact_duplicate_keeps_best_alignment_not_first_id():
    text = model()
    rows = [
        candidate("a", text=text, description="Receive the invoice, validate something, then archive something after checking."),
        candidate("b", text=text),
    ]
    result = filter_candidates(pd.DataFrame(rows), QualityLimits()).set_index("origin")
    assert result.loc["b", "quality_pass"]
    assert not result.loc["a", "quality_pass"]
    assert result.loc["a", "quality_reasons"] == ["duplicate_model"]


@pytest.mark.parametrize("selector", [select_category, select_category_complexity])
@pytest.mark.parametrize("size", [0, 1, 3, 12])
def test_categories_are_caps_not_quotas_and_order_is_deterministic(selector, size):
    rows = [candidate(f"sapsam::1::{i}", emb_row=i) for i in range(max(size, 1))]
    data = admitted(rows).iloc[:size].copy()
    emb = np.eye(max(size, 1))
    selected, info, _, _ = selector("category", data, emb, 42, 0.9)
    again, _, _, _ = selector("category", data.iloc[::-1], emb, 42, 0.9)
    assert len(selected) == min(size, 10)
    assert [r["origin"] for r in selected] == [r["origin"] for r in again]
    assert info["n_selected"] == len(selected)


def test_no_duplicate_or_rejected_candidate_fallback_to_fill_category():
    data = admitted([
        candidate("a", emb_row=0),
        candidate("b", text=model(actions=("receiveInvoice",)), emb_row=1),
        candidate("c", text=model(actions=("archiveInvoice",)), description="x" * 1600, emb_row=2),
    ])
    selected, info, _, _ = select_category_complexity("category", data, np.ones((3, 1)), 42, 0.9)
    assert len(selected) == 1
    assert info["n_near_duplicates"] == 1
    assert all(r["origin"] != "c" for r in selected)


def test_quality_then_conciseness_precede_structural_coverage():
    rows = [
        candidate("high", emb_row=0),
        candidate("low", emb_row=1),
        candidate("long", emb_row=2),
    ]
    # Ranking uses already-admitted evidence; distinct names avoid exact-model deduplication.
    data = admitted([rows[0]])
    data = pd.concat([data] * 3, ignore_index=True)
    data["origin"] = ["high", "low", "long"]
    data["emb_row"] = [0, 1, 2]
    data["quality_score"] = [1.0, 0.8, 1.0]
    data["description_chars"] = [100, 100, 600]
    data["num_forks"] = [0, 2, 2]
    selected, _, _, _ = select_category_complexity("category", data, np.eye(3), 42, 0.9, 1)
    assert selected[0]["origin"] == "high"


def test_construct_diversity_breaks_equally_good_concise_ties():
    data = admitted([candidate()])
    data = pd.concat([data] * 3, ignore_index=True)
    data["origin"] = ["a", "b", "c"]
    data["emb_row"] = [0, 1, 2]
    data["num_decisions"] = [0, 0, 1]
    selected, _, _, _ = select_category_complexity("category", data, np.eye(3), 42, 0.9, 2)
    assert {r["origin"] for r in selected} == {"a", "c"}


def test_partition_uses_only_description_embeddings_for_leak_exclusions():
    rows = pd.DataFrame({
        "origin": ["sopbench::test::1", "sopbench::train::1", "sapsam::1::1", "excluded"],
        "emb_row": range(4),
    })
    emb = np.array([[1., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    held, leak, train = partition_masks(rows, emb, Config(), {"excluded"}, 0.95)
    assert held.tolist() == [True, False, False, True]
    assert leak.tolist() == [False, True, False, False]
    assert train.tolist() == [False, False, True, False]


@pytest.mark.parametrize("size", [0, 1])
def test_small_category_figures(tmp_path, size):
    data = admitted([candidate()]).iloc[:size]
    chosen, _, pool, _ = select_category_complexity("category", data, np.eye(1), 42, 0.9)
    selected = pd.DataFrame(chosen) if chosen else pool.assign(workflow_type=pd.Series(dtype=str))
    plot_category("category", selected, pool, np.eye(1), tmp_path / "category", 42)
    assert (tmp_path / "category.png").exists()


def test_main_exports_header_only_corpus_when_all_categories_empty(tmp_path, monkeypatch):
    import Analysis.rag_selection as selection

    row = candidate(description="x" * 1600)
    rows = pd.DataFrame([row]).drop(columns=[
        "normalized_sysml", "normalization_status", "parser_valid", "parser_error",
        "structural_valid", "structural_status", "structural_errors", "intentional_exception",
        "transformations",
    ])
    rows.attrs["emb"] = np.eye(1)
    monkeypatch.setattr(selection, "load_inputs", lambda *_: rows)
    root = tmp_path / "rag"
    root.mkdir()
    pd.DataFrame([row]).to_csv(root / "normalization_results.csv", index=False)
    assert main(["--output-dir", str(tmp_path)]) == 0
    result = pd.read_csv(root / "rag_selected_workflows.csv")
    assert result.empty
    assert {"origin", "enriched_description", "sysml_model", "parser_valid", "structural_valid"} <= set(result)
    audit = pd.read_csv(root / "rag_candidate_audit.csv")
    assert "excessive_description_length" in json.loads(audit.iloc[0].rejection_reasons)
    config = json.loads((root / "rag_selection_config.json").read_text())
    assert config["category_cap_only"]
    assert config["complexity"]["thresholds"] == {"simple_max": None, "medium_max": None}


@pytest.mark.parametrize("strategy", ["semantic", "complexity"])
def test_main_hard_filters_reports_and_heldout_gold_exclusion(tmp_path, monkeypatch, strategy):
    import Analysis.rag_selection as selection

    examples = [
        candidate("sapsam::1::good", emb_row=0),
        candidate("sapsam::1::long", description=DESCRIPTION * 100, category="empty", emb_row=1),
        candidate("sapsam::1::invalid", parser_valid=0, category="empty", emb_row=2),
        candidate("sopbench::test::gold", text="GOLD MUST NOT BE INSPECTED", emb_row=3),
    ]
    fields = [
        "normalized_sysml", "normalization_status", "parser_valid", "parser_error",
        "structural_valid", "structural_status", "structural_errors", "intentional_exception",
        "transformations",
    ]
    rows = pd.DataFrame(examples).drop(columns=fields)
    rows.attrs["emb"] = np.eye(4)
    monkeypatch.setattr(selection, "load_inputs", lambda *_: rows)
    original_assessor = selection.assess_quality

    def inspect_only_training(model, description, limits, source):
        assert "GOLD" not in model and "GOLD" not in source
        return original_assessor(model, description, limits, source)

    monkeypatch.setattr(selection, "assess_quality", inspect_only_training)
    root = tmp_path / "rag"
    root.mkdir()
    pd.DataFrame(examples).to_csv(root / "normalization_results.csv", index=False)
    assert main(["--output-dir", str(tmp_path), "--strategy", strategy]) == 0
    selected = pd.read_csv(root / "rag_selected_workflows.csv")
    assert selected.origin.tolist() == ["sapsam::1::good"]
    summary = pd.read_csv(root / "rag_selection_summary.csv").set_index("category")
    assert summary.loc["empty", "n_selected"] == 0
    assert summary.loc["empty", "n_rejected"] == 2
    assert summary.loc["empty", "rejected_invalid_syntax"] == 1
    assert summary.loc["empty", "rejected_excessive_description_length"] == 1
    audit = pd.read_csv(root / "rag_candidate_audit.csv")
    assert len(audit) == 3
    assert "sopbench::test::gold" not in set(audit.origin)
    assert (root / "per_category" / "empty.png").exists()
