"""Conservative, deterministic RAG admission checks, not a substitute for the SysML kernel."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from Analysis.sysml_workflow import (
    BASE_AGENT_TYPES,
    UnsupportedSyntax,
    Workflow,
    extract_complexity_features,
    normalize_workflow,
    parse_workflow,
    tokenize,
    validate_workflow_structure,
)


@dataclass(frozen=True)
class QualityLimits:
    max_actions: int = 15
    max_decisions: int = 5
    max_forks: int = 2
    max_joins: int = 2
    max_description_chars: int = 1500
    min_action_coverage: float = 0.8
    min_action_token_overlap: float = 0.3

    def __post_init__(self):
        for name in ("max_actions", "max_description_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_decisions", "max_forks", "max_joins"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("min_action_coverage", "min_action_token_overlap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


def _reachable(root: str, graph: dict[str, set[str]], stop: str | None = None) -> set[str]:
    seen, todo = set(), [root]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        if node != stop:
            todo.extend(graph.get(node, ()))
    return seen


def control_flow_issues(wf: Workflow) -> list[str]:
    """Reject suspicious concurrency/branching rather than repairing or guessing its intent."""
    issues = []
    graph: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for a, b in wf.edges():
        graph[a].add(b)
        reverse[b].add(a)
    declared = set(wf.actions) | set(wf.nodes)
    if declared & {"start", "done"}:
        issues.append("reserved_node_identifier")
    if len(graph["start"]) != 1 or reverse["start"]:
        issues.append("ambiguous_entry")
    if graph["done"]:
        issues.append("outgoing_termination")
    if any(a == b for a, b in wf.edges()):
        issues.append("self_loop")
    if declared - _reachable("start", graph):
        issues.append("disconnected_nodes")
    if declared - _reachable("done", reverse):
        issues.append("nodes_without_termination_path")
    for action in wf.actions:
        if len(graph[action]) != 1:
            issues.append("ambiguous_action_successors")
    for name, node in wf.nodes.items():
        if node.kind == "decide":
            block = wf.decisions.get(name)
            if not block or len(block["branches"]) + (block["else"] is not None) < 2:
                issues.append("incomplete_decision")
                continue
            if any(item[0] == "edge" and item[1] == name for item in wf.flow):
                issues.append("unguarded_decision_edge")
            guards = [re.sub(r"\s+", "", g).lower() for g, _ in block["branches"]]
            if len(set(guards)) != len(guards):
                issues.append("duplicate_decision_guard")
            targets = {target for _, target in block["branches"]}
            if block["else"] is not None:
                targets.add(block["else"])
            if len(targets) < 2:
                issues.append("redundant_decision")
            # Without else, certify only an explicit Boolean complement.
            if block["else"] is None:
                complements = set()
                for guard in guards:
                    match = re.fullmatch(r"([a-z_]\w*\.[a-z_]\w*)(?:(==|!=)(true|false))?", guard)
                    if match:
                        ref, op, value = match.groups()
                        truth = (value != "false") if op != "!=" else (value == "false")
                        complements.add((ref, truth))
                if not any((ref, not truth) in complements for ref, truth in complements):
                    issues.append("nonexhaustive_decision")
            for guard, _ in block["branches"]:
                tokens = tokenize(guard)
                i = 0
                while i < len(tokens):
                    kind, value = tokens[i]
                    if kind != "id" or value.lower() in {"true", "false", "and", "or", "not"}:
                        i += 1
                        continue
                    if i + 2 < len(tokens) and tokens[i + 1][1] == ".":
                        if value not in wf.actions:
                            issues.append("unresolved_guard_reference")
                        elif name in _reachable("start", graph, stop=value) - {value}:
                            issues.append("guard_output_not_available_on_all_paths")
                        i += 3
                    else:
                        issues.append("unresolved_guard_identifier")
                        i += 1
        elif node.kind == "fork" and len(graph[name]) < 2:
            issues.append("degenerate_fork")
        elif node.kind in {"join", "merge"}:
            if len(reverse[name]) < 2 or len(graph[name]) != 1:
                issues.append(f"degenerate_{node.kind}")

    matched_joins = set()
    joins = {n for n, node in wf.nodes.items() if node.kind == "join"}
    for fork, node in wf.nodes.items():
        if node.kind != "fork" or len(graph[fork]) < 2:
            continue
        branches = sorted(graph[fork])
        matches = []
        for join in sorted(joins):
            regions = [_reachable(branch, graph, stop=join) - {join} for branch in branches]
            if any(join not in _reachable(branch, graph) for branch in branches):
                continue
            if any("done" in region or fork in region for region in regions):
                continue
            if any(regions[i] & regions[j] for i in range(len(regions)) for j in range(i)):
                continue
            matches.append(join)
        if len(matches) != 1:
            issues.append("unbalanced_or_ambiguous_parallelism")
        else:
            matched_joins.add(matches[0])
    if joins - matched_joins:
        issues.append("join_without_matching_fork")
    return sorted(set(issues))


_GENERIC = {
    "action", "step", "workflow", "process", "result", "output", "true", "false",
    "begin", "start", "end", "done", "finish", "complete", "completion", "initiate",
    "perform", "execute", "execution", "required", "return", "returns", "boolean",
}


def _words(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z]+", text.lower())
    return {
        word[:-1] if word.endswith("s") and not word.endswith("ss") else word
        for word in words
        if len(word) > 2 and word not in ENGLISH_STOP_WORDS and word not in _GENERIC
    }


def assess_quality(model: str, description: str, limits: QualityLimits, source: str = "") -> dict:
    """Return all applicable hard-filter reasons plus auditable lexical-alignment metadata.

    Prose alignment is deliberately a lexical proxy, not a claim of semantic equivalence.
    No evaluation model, embedding model, network call or LLM is used here.
    """
    reasons, details = [], []
    result = {
        "description_chars": len(description),
        "quality_score": 0.0,
        "action_description_coverage": 0.0,
        "mean_action_token_overlap": 0.0,
    }
    if len(description) > limits.max_description_chars:
        reasons.append("excessive_description_length")
    if len(re.findall(r"\b\w+\b", description)) < 8:
        reasons.append("quality_missing_or_short_description")
    if re.search(
        r"\b(?:TODO|TBD|lorem ipsum|as an AI|cannot determine|not specified|not provided|unclear)\b",
        description, re.I,
    ):
        reasons.append("quality_ambiguous_description")
    try:
        wf = parse_workflow(model)
    except UnsupportedSyntax as exc:
        return {**result, "quality_reasons": reasons + ["unsupported_sysml"],
                "quality_details": [str(exc)]}

    features = extract_complexity_features(wf, source or model)
    result.update(features)
    for feature, maximum in (
        ("num_actions", limits.max_actions), ("num_decisions", limits.max_decisions),
        ("num_forks", limits.max_forks), ("num_joins", limits.max_joins),
    ):
        if features[feature] > maximum:
            reasons.append(f"excessive_{feature.removeprefix('num_')}")
    # Expensive graph/normalization checks are unnecessary for already oversized examples.
    if any(reason.startswith("excessive_") for reason in reasons):
        return {**result, "quality_reasons": reasons, "quality_details": details}

    structure = validate_workflow_structure(wf, source or model)
    result.update(structure)
    if not structure["structural_valid"]:
        reasons.append("quality_structurally_incomplete")
        details.extend(structure["structural_errors"])
    if not wf.actions:
        reasons.append("quality_no_actions")
    try:
        normalized, _, _ = normalize_workflow(model)
        if tokenize(normalized) != tokenize(model):
            reasons.append("quality_noncanonical_or_inconsistent_declarations")
    except UnsupportedSyntax as exc:
        reasons.append("quality_invalid_declarations_or_references")
        details.append(str(exc))
    for action in wf.actions.values():
        outputs = [name for name, _ in action.outputs]
        if len(outputs) != len(set(outputs)) or any(not typ for _, typ in action.outputs):
            reasons.append("quality_duplicate_or_untyped_output")
    agent_types = dict(wf.agent_types)
    for _, typ in wf.agent_insts:
        visited = set()
        while typ in agent_types and typ not in visited:
            visited.add(typ)
            typ = agent_types[typ]
        if typ not in BASE_AGENT_TYPES:
            reasons.append("quality_unresolved_or_cyclic_agent_type")
    reasons.extend(f"quality_{issue}" for issue in control_flow_issues(wf))

    narrative = _words(description)
    overlaps = []
    for action in wf.actions.values():
        action_text = action.description or '""'
        try:
            action_text = json.loads(action_text)
        except (ValueError, TypeError):
            reasons.append("quality_unreadable_action_description")
        if not action_text.strip():
            reasons.append("quality_empty_action_description")
        words = _words(action.name + " " + action_text)
        if words:
            overlaps.append(len(narrative & words) / len(words))
    if overlaps:
        coverage = sum(value >= limits.min_action_token_overlap for value in overlaps) / len(overlaps)
        result["action_description_coverage"] = coverage
        result["mean_action_token_overlap"] = sum(overlaps) / len(overlaps)
        result["quality_score"] = round(0.7 * coverage + 0.3 * result["mean_action_token_overlap"], 6)
        if coverage < limits.min_action_coverage:
            reasons.append("quality_description_action_mismatch")
    else:
        reasons.append("quality_no_informative_action_descriptions")
    if features["has_parallelism"] and not re.search(
        r"\b(?:parallel|concurrent\w*|simultaneous\w*|independent\w*|at the same time)\b", description, re.I
    ):
        reasons.append("quality_parallelism_not_described")
    if not features["has_parallelism"] and re.search(
        r"\b(?:in parallel|simultaneously|concurrently)\b", description, re.I
    ) and not re.search(r"\b(?:not|never)\s+(?:in parallel|simultaneously|concurrently)\b", description, re.I):
        reasons.append("quality_description_claims_absent_parallelism")
    if features["has_cycles"] and not re.search(
        r"\b(?:repeat\w*|loop\w*|again|until|retry|revisit\w*|return to|back to|iterat\w*)\b",
        description, re.I,
    ):
        reasons.append("quality_loop_not_described")
    return {**result, "quality_reasons": sorted(set(reasons)), "quality_details": details}
