import numpy as np

from rag_experiments.extract_stratified_1000 import meaningful_names, proportional_select, set_kind
from Analysis.sysml_workflow import parse_workflow


def features(**overrides):
    result = dict(num_actions=5, num_decisions=0, num_forks=0, num_joins=0, has_cycles=False)
    result.update(overrides)
    return result


def test_exact_structural_constraints():
    assert set_kind(features()) == "sequential"
    assert set_kind(features(has_cycles=True)) is None
    assert set_kind(features(num_actions=9)) is None
    assert set_kind(features(num_decisions=1, num_forks=1, num_joins=1)) == "complex"
    assert set_kind(features(num_decisions=2, num_forks=2, num_actions=8)) == "complex"
    assert set_kind(features(num_decisions=3, num_forks=1)) is None
    assert set_kind(features(num_decisions=1, num_forks=3)) is None
    assert set_kind(features(num_decisions=1)) is None
    assert set_kind(features(num_forks=1)) is None


def candidate(index, category):
    return {
        "id": str(index), "category": category, "description": f"Example {index}",
        "sysml": f"Model {index}", "_graph": str(index), "emb_row": index,
        "structure": {"quality_score": 1.0},
    }


def test_proportional_targets_and_deterministic_replay():
    rows = [candidate(i, "a" if i < 30 else "b") for i in range(40)]
    selected, targets, _ = proportional_select(rows, np.eye(40), count=20)
    again, _, _ = proportional_select(rows, np.eye(40), count=20)
    assert targets == {"a": 15, "b": 5}
    assert sum(r["category"] == "a" for r in selected) == 15
    assert [r["id"] for r in selected] == [r["id"] for r in again]


def test_small_pool_and_duplicates_never_relax_constraints():
    rows = [candidate(0, "a"), candidate(1, "b")]
    selected, targets, rejected = proportional_select(rows, np.ones((2, 1)), count=500)
    assert targets == {"a": 250, "b": 250}
    assert len(selected) == len(rejected) == 1
    assert proportional_select([], np.eye(1)) == ([], {}, set())


def test_placeholder_names_are_not_meaningful():
    bad = parse_workflow('package P { action def W { action action1; } }')
    good = parse_workflow('package P { action def W { action validateRequest; } }')
    assert not meaningful_names(bad)
    assert meaningful_names(good)
