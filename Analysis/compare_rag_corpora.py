"""Compare the preserved semantic baseline with the quality-filtered RAG corpus."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from Analysis.sysml_workflow import UnsupportedSyntax, extract_complexity_features, parse_workflow
from Analysis.rag_selection import COMPLEXITY_WEIGHTS


def _add_reference_levels(old: pd.DataFrame, candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    thresholds = config["complexity"]["thresholds"]
    if candidates.empty:
        return old.assign(complexity_score=np.nan, complexity_level="unclassified")
    weights = COMPLEXITY_WEIGHTS
    scores = []
    for _, row in old.iterrows():
        score = 0.0
        for name, weight in weights.items():
            if candidates[name].dtype == object:
                values = candidates[name].replace({"True": 1, "False": 0, "true": 1, "false": 0})
            else:
                values = candidates[name]
            values = pd.to_numeric(values, errors="coerce").dropna()
            raw_value = row[name]
            if isinstance(raw_value, str) and raw_value.lower() in ("true", "false"):
                raw_value = int(raw_value.lower() == "true")
            value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if len(values) and pd.notna(value):
                less = int((values < value).sum())
                equal = int((values == value).sum())
                rank = (less + (equal + 1) / 2) / len(values)
            else:
                rank = 0.0
            score += rank * weight
        scores.append(score)
    old = old.copy()
    old["complexity_score"] = scores
    q1 = thresholds["simple_max"]
    q2 = thresholds["medium_max"]
    old["complexity_level"] = np.select(
        [old.complexity_score <= q1, old.complexity_score <= q2],
        ["simple", "medium"], default="complex",
    )
    return old


def main(output_dir: str = "enriched_dataset_analysis") -> int:
    root = Path(output_dir) / "rag"
    baseline_path = root / "baseline" / "rag_selected_workflows.csv"
    selected_path = root / "rag_selected_workflows.csv"
    results_path = root / "normalization_results.csv"
    config_path = root / "rag_selection_config.json"
    for path in (baseline_path, selected_path, results_path, config_path):
        if not path.exists():
            raise SystemExit(f"Missing required report input: {path}")

    baseline = pd.read_csv(baseline_path, keep_default_na=False, low_memory=False)
    selected = pd.read_csv(selected_path, keep_default_na=False, low_memory=False)
    results = pd.read_csv(results_path, keep_default_na=False, low_memory=False)
    workflow_index_path = Path(output_dir) / "workflows_base.parquet"
    workflow_index = pd.read_parquet(workflow_index_path, columns=["origin", "dataset"])
    config = json.loads(config_path.read_text())
    if "complexity_level" not in selected:
        raise SystemExit("The quality-filtered corpus has not been generated.")

    baseline_features = []
    for row in baseline.itertuples():
        try:
            wf = parse_workflow(row.sysml_model)
            baseline_features.append(extract_complexity_features(wf, row.sysml_model))
        except UnsupportedSyntax:
            baseline_features.append({name: 0 for name in COMPLEXITY_WEIGHTS})
            baseline_features[-1].update({
                "control_flow_edges": 0, "conditional_branches": 0, "parallel_branches": 0,
                "max_nested_parallelism_depth": 0, "has_cycles": False,
                "repeated_action_occurrences": 0, "max_execution_path_depth": None,
            })
    feature_frame = pd.DataFrame(baseline_features)
    baseline = baseline.reset_index(drop=True)
    for feature in feature_frame.columns:
        baseline[feature] = feature_frame[feature].values
    valid_candidates = results[
        pd.to_numeric(results.parser_valid, errors="coerce").eq(1)
        & pd.to_numeric(results.structural_valid, errors="coerce").eq(1)
    ]
    audit_path = root / "rag_candidate_audit.csv"
    audit = pd.read_csv(audit_path, keep_default_na=False) if audit_path.exists() else None
    if audit is not None:
        valid_candidates = audit[audit.quality_pass.eq(True)]
    baseline = _add_reference_levels(baseline, valid_candidates, config)

    def record(name: str, data: pd.DataFrame, is_new: bool) -> dict:
        item = {
            "corpus": name,
            "workflows": len(data),
            "simple": int((data.complexity_level == "simple").sum()),
            "medium": int((data.complexity_level == "medium").sum()),
            "complex": int((data.complexity_level == "complex").sum()),
        }
        for col in (
            "num_actions", "num_decisions", "num_forks", "num_joins",
            "control_flow_edges", "parallel_branches", "max_nested_parallelism_depth",
        ):
            values = pd.to_numeric(data[col], errors="coerce")
            item[f"mean_{col}"] = round(values.mean(), 3)
            item[f"median_{col}"] = round(values.median(), 3)
            item[f"max_{col}"] = values.max()
        complex_rows = data[data.complexity_level == "complex"]
        for col in (
            "num_decisions", "num_forks", "num_joins",
            "max_nested_parallelism_depth", "parallel_branches",
        ):
            values = pd.to_numeric(complex_rows[col], errors="coerce")
            item[f"complex_mean_{col}"] = round(values.mean(), 3)
            item[f"complex_max_{col}"] = values.max()
        complex_checks = {
            "with_decisions": pd.to_numeric(complex_rows.num_decisions, errors="coerce").gt(0),
            "with_multiple_forks": pd.to_numeric(complex_rows.num_forks, errors="coerce").ge(2),
            "with_multiple_joins": pd.to_numeric(complex_rows.num_joins, errors="coerce").ge(2),
            "with_nested_parallelism": pd.to_numeric(
                complex_rows.max_nested_parallelism_depth, errors="coerce"
            ).gt(1),
        }
        for label, mask in complex_checks.items():
            item[f"complex_{label}"] = int(mask.sum())
            item[f"complex_{label}_rate"] = round(float(mask.mean()), 4) if len(complex_rows) else 0
        if is_new:
            item["normalization_success"] = int((data.normalization_status == "valid").sum())
            item["normalization_success_rate"] = round(item["normalization_success"] / len(data), 4) if len(data) else 0
            item["parser_valid"] = int(pd.to_numeric(data.parser_valid, errors="coerce").eq(1).sum())
            item["parser_valid_rate"] = round(item["parser_valid"] / len(data), 4) if len(data) else 0
            item["structural_valid"] = int(pd.to_numeric(data.structural_valid, errors="coerce").eq(1).sum())
            item["structural_valid_rate"] = round(item["structural_valid"] / len(data), 4) if len(data) else 0
            item["parser_and_structural_valid"] = int(
                (
                    pd.to_numeric(data.parser_valid, errors="coerce").eq(1)
                    & pd.to_numeric(data.structural_valid, errors="coerce").eq(1)
                ).sum()
            )
        else:
            item["normalization_success"] = "baseline_not_normalized"
            item["normalization_success_rate"] = "n/a"
            item["parser_valid"] = "not_revalidated_in_this_run"
            item["parser_valid_rate"] = "n/a"
            item["structural_valid"] = "not_revalidated_in_this_run"
            item["structural_valid_rate"] = "n/a"
            item["parser_and_structural_valid"] = "n/a"
        return item

    if "normalization_status" not in selected:
        selected["normalization_status"] = "valid"
    if "parser_valid" not in selected:
        selected["parser_valid"] = 1
    comparison = pd.DataFrame([
        record("semantic_baseline", baseline, False),
        record("quality_filtered", selected, True),
    ])
    comparison.to_csv(root / "rag_corpus_comparison.csv", index=False)

    parser_ok = pd.to_numeric(results.parser_valid, errors="coerce").eq(1)
    structure_ok = pd.to_numeric(results.structural_valid, errors="coerce").eq(1)
    excluded = results[~(parser_ok & structure_ok)][
        [
            "origin", "normalization_status", "parser_valid", "parser_error",
            "structural_valid", "structural_status", "structural_errors", "intentional_exception",
        ]
    ].copy()
    reason_counts = excluded.structural_status.value_counts().to_dict()
    excluded.to_csv(root / "rag_excluded_samples.csv", index=False)
    if audit is not None:
        excluded = audit[~audit.selected.eq(True)].copy()
        excluded.to_csv(root / "rag_excluded_samples.csv", index=False)
    structural_issue_counts = Counter()
    for error_list in results.loc[
        results.structural_status == "incomplete", "structural_errors"
    ]:
        for error in json.loads(error_list):
            if error.startswith("declared actions have no executable control-flow edges"):
                structural_issue_counts["declared actions but zero control-flow edges"] += 1
            elif error.startswith("start has no outgoing entry connection"):
                structural_issue_counts["missing start entry connection"] += 1
            elif error.startswith("actions unreachable from start"):
                structural_issue_counts["unreachable declared actions"] += 1
            elif error.startswith("no termination is reachable from start"):
                structural_issue_counts["no reachable termination"] += 1
            elif error.startswith("reachable actions have no path to done"):
                structural_issue_counts["reachable actions without a path to termination"] += 1
    by_dataset = results.merge(workflow_index, on="origin", how="left", validate="one_to_one")
    by_dataset["parser_and_structural_valid"] = (
        pd.to_numeric(by_dataset.parser_valid, errors="coerce").eq(1)
        & pd.to_numeric(by_dataset.structural_valid, errors="coerce").eq(1)
    )
    dataset_summary = (
        by_dataset.groupby("dataset", dropna=False)
        .agg(
            eligible_train_candidates=("origin", "size"),
            normalization_succeeded=("normalization_status", lambda s: int(s.eq("valid").sum())),
            parser_valid=("parser_valid", lambda s: int(pd.to_numeric(s, errors="coerce").eq(1).sum())),
            parser_invalid=("parser_valid", lambda s: int(pd.to_numeric(s, errors="coerce").eq(0).sum())),
            structural_valid=("structural_valid", lambda s: int(pd.to_numeric(s, errors="coerce").eq(1).sum())),
            structural_incomplete=("structural_status", lambda s: int(s.eq("incomplete").sum())),
            intentional_structural_exceptions=("structural_status", lambda s: int(s.eq("intentional_exception").sum())),
            normalization_failed=("normalization_status", lambda s: int(s.eq("normalization_failed").sum())),
            parser_and_structural_valid=("parser_and_structural_valid", "sum"),
        )
        .reset_index()
    )
    dataset_summary["parser_validity_rate"] = (
        dataset_summary["parser_valid"] / dataset_summary["eligible_train_candidates"]
    )
    dataset_summary["structural_validity_rate"] = (
        dataset_summary["structural_valid"] / dataset_summary["eligible_train_candidates"]
    )
    dataset_summary["structural_valid_among_parser_valid_rate"] = (
        dataset_summary["structural_valid"] / dataset_summary["parser_valid"].replace(0, np.nan)
    ).fillna(0)
    dataset_summary.to_csv(root / "rag_normalization_by_dataset.csv", index=False)
    columns = list(comparison.columns)
    markdown_table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record_row in comparison.to_dict("records"):
        markdown_table.append(
            "| " + " | ".join(str(record_row[col]).replace("|", "\\|") for col in columns) + " |"
        )
    dataset_table = [
        "| " + " | ".join(dataset_summary.columns) + " |",
        "| " + " | ".join("---" for _ in dataset_summary.columns) + " |",
    ]
    for record_row in dataset_summary.to_dict("records"):
        dataset_table.append(
            "| " + " | ".join(str(record_row[col]) for col in dataset_summary.columns) + " |"
        )
    md = [
        "# RAG corpus comparison",
        "",
        *markdown_table,
        "",
        "## Train-side normalization by source dataset",
        "",
        *dataset_table,
        "",
        f"**Saved normalized train candidates:** {len(results)}; "
        f"**parser-valid:** {int(parser_ok.sum())}; "
        f"**structurally complete:** {int(structure_ok.sum())}; "
        f"**both valid:** {int((parser_ok & structure_ok).sum())}; "
        f"**excluded from selection (all admission/ranking reasons):** {len(excluded)}.",
        "",
        f"**Parser validity rate:** {parser_ok.mean():.2%}; "
        f"**structural completeness rate among parser-valid candidates:** "
        f"{int((parser_ok & structure_ok).sum()) / max(1, int(parser_ok.sum())):.2%}.",
        "",
        f"**Exclusion reasons:** {json.dumps(reason_counts, sort_keys=True)}.",
        "",
        f"**Structural defect counts (checks may overlap):** "
        f"{json.dumps(dict(structural_issue_counts), sort_keys=True)}.",
        "",
        "Baseline complexity levels are scored against the quality-qualified candidate-pool empirical "
        "feature distributions and the thresholds in `rag_selection_config.json`.",
        "Structural failures are not repaired by adding control flow. Explicit comments of the "
        "form `// RAG-STRUCTURAL-EXCEPTION: reason` are recorded separately and remain excluded.",
        "Full excluded-origin details are in `rag_excluded_samples.csv`.",
        "Quality/length/ranking rejection counts are in `rag_rejection_reasons.csv`; per-category "
        "candidate and selection counts are in `rag_selection_summary.csv`. Quality scores use "
        "lexical alignment proxies, not a guarantee of natural-language equivalence.",
    ]
    if audit is not None:
        md.extend([
            "",
            f"**Audited training candidates:** {len(audit)}; "
            f"**quality-qualified:** {int(audit.quality_pass.eq(True).sum())}; "
            f"**selected:** {int(audit.selected.eq(True).sum())}. "
            "These counts include candidates without saved normalization/parser evidence, "
            "which fail closed.",
        ])
    (root / "rag_corpus_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
