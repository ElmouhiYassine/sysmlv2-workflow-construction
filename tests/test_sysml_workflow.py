import pytest

from Analysis.sysml_workflow import (
    UnsupportedSyntax,
    _topology,
    extract_complexity_features,
    normalize_workflow,
    parse_workflow,
    validate_workflow_structure,
)


def _node(name: str, output: str = "") -> str:
    return (
        f'action {name} {{ description = "{name}"; '
        f'{output} }}'
    )


def test_missing_agents_get_one_default_and_descriptions_are_normalized():
    source = f"""package P {{
        action def W {{
            {_node("a")}
            first start then a;
            first a then done;
        }}
    }}"""
    result, transforms, _ = normalize_workflow(source)
    assert result.count("solutionAgent : SolutionAgent;") == 1
    assert result.count("agent = solutionAgent;") == 1
    assert "action a : Step" in result
    assert "::> description = \"a\";" in result
    assert any("default solutionAgent" in item for item in transforms)


def test_specialized_agent_and_existing_assignment_are_preserved():
    source = """private import ScalarValues::*;
    private import MOSAICO::*;
    private import MOSAICO::*;
    package P { action def W {
        part def Specialist specializes ReferenceAgent {}
        specialist : Specialist;
        action a { description = "Keep."; :>> agent = specialist; }
        first start then a;
        first a then done;
    }}"""
    result, transforms, _ = normalize_workflow(source)
    assert result.startswith("private import MOSAICO::*;\nprivate import ScalarValues::*;")
    assert "part def Specialist specializes ReferenceAgent" in result
    assert "specialist : Specialist;" in result
    assert "agent = specialist;" in result
    assert "solutionAgent : SolutionAgent;" not in result
    assert any("removed duplicates" in item for item in transforms)


def test_identical_agent_declarations_are_deduplicated_deterministically():
    source = """package P { action def W {
        part def Specialist specializes SolutionAgent {}
        part def Specialist specializes SolutionAgent {}
        specialist : Specialist;
        specialist : Specialist;
        action a { description = "A"; agent = specialist; }
        first start then a;
        first a then done;
    }}"""
    result, transforms, _ = normalize_workflow(source)
    assert result.count("part def Specialist") == 1
    assert result.count("specialist : Specialist;") == 1
    assert any("identical duplicate agent" in item for item in transforms)


def test_identical_control_node_declarations_are_emitted_once():
    source = f"""package P {{ action def W {{
        {_node("a")}
        fork parallel;
        fork parallel;
        first start then parallel;
        first parallel then a;
        first a then done;
    }} }}"""
    result, transforms, _ = normalize_workflow(source)
    assert result.count("fork parallel;") == 1
    assert any("duplicate control-node" in item for item in transforms)


def test_partial_agent_assignments_reuse_the_only_declared_agent():
    source = f"""package P {{ action def W {{
        expert : SolutionAgent;
        {_node("a").replace("}", "agent = expert; }")}
        {_node("b")}
        first start then a;
        first a then b;
        first b then done;
    }} }}"""
    result, _, _ = normalize_workflow(source)
    assert result.count("agent = expert;") == 2
    assert "solutionAgent : SolutionAgent;" not in result


def test_legacy_agent_role_assignment_becomes_step_agent_assignment():
    source = """package P { action def W {
        part def ClaimsAgent specializes SolutionAgent {}
        claimsAgent : ClaimsAgent;
        action checkClaim : Step {
            description = "Check the claim.";
            claimsAgent = claimsAgent;
        }
        first start then checkClaim;
        first checkClaim then done;
    }}"""
    result, transforms, _ = normalize_workflow(source)
    assert "agent = claimsAgent;" in result
    assert "claimsAgent = claimsAgent;" not in result
    assert any("legacy agent-role" in item for item in transforms)


def test_untyped_output_infers_boolean_when_used_as_guard():
    source = f"""package P {{ action def W {{
        {_node("check", "out result;")}
        {_node("accept")}
        decide decision;
        first start then check;
        first check then decision;
        first decision;
        if check.result == true then accept;
        else done;
        first accept then done;
    }} }}"""
    result, transforms, _ = normalize_workflow(source)
    assert "out result : Boolean;" in result
    assert any("inferred Boolean" in item for item in transforms)


def test_untyped_output_without_reliable_evidence_is_flagged():
    source = f"""package P {{ action def W {{
        {_node("a", "out value;")}
        first start then a;
        first a then done;
    }} }}"""
    with pytest.raises(UnsupportedSyntax, match="cannot infer type"):
        normalize_workflow(source)


def test_nested_forks_and_joins_keep_their_control_topology():
    source = f"""package P {{ action def W {{
        {_node("a")}
        {_node("b")}
        {_node("c")}
        {_node("d")}
        fork outer;
        fork inner;
        join innerJoin;
        join outerJoin;
        first start then outer;
        first outer then a;
        first outer then inner;
        first inner then b;
        first inner then c;
        first b then innerJoin;
        first c then innerJoin;
        first innerJoin then d;
        first d then outerJoin;
        first a then outerJoin;
        first outerJoin then done;
    }} }}"""
    original = parse_workflow(source)
    normalized, _, parsed = normalize_workflow(source)
    assert _topology(original) == _topology(parsed)
    features = extract_complexity_features(parsed, source)
    assert features["num_forks"] == 2
    assert features["num_joins"] == 2
    assert features["max_nested_parallelism_depth"] >= 2
    assert "fork outer;" in normalized and "fork inner;" in normalized


def test_worfbench_fork_references_remain_parallel_branches():
    source = """package P { action def W {
        first start;
        then fork;
        then action left;
        then action right;
        action left { description = "left"; }
        then joinNode;
        action right { description = "right"; }
        then joinNode;
        join joinNode;
        then done;
    }}"""
    original = parse_workflow(source)
    normalized, _, parsed = normalize_workflow(source)
    assert set(original.edges()) == set(parsed.edges())
    fork = next(name for name, node in parsed.nodes.items() if node.kind == "fork")
    assert (fork, "left") in parsed.edges()
    assert (fork, "right") in parsed.edges()
    assert "first start;" in normalized


def test_multiple_incoming_edges_are_not_repeated_action_occurrences():
    source = f"""package P {{ action def W {{
        {_node("a")}
        {_node("b")}
        {_node("c")}
        first start then a;
        first a then c;
        first b then c;
        first c then done;
    }} }}"""
    wf = parse_workflow(source)
    features = extract_complexity_features(wf, source)
    assert features["repeated_action_occurrences"] == 0
    assert features["max_incoming_edges"] == 2


def test_cycle_detection_disables_unreliable_maximum_path_depth():
    source = f"""package P {{ action def W {{
        {_node("a")}
        first start then a;
        first a then a;
        first a then done;
    }} }}"""
    features = extract_complexity_features(parse_workflow(source), source)
    assert features["has_cycles"] is True
    assert features["repeated_action_occurrences"] == 1
    assert features["max_execution_path_depth"] is None


def test_structure_rejects_declared_actions_without_control_flow_edges():
    wf = parse_workflow(f"""package P {{ action def W {{
        {_node("a")}
    }} }}""")

    result = validate_workflow_structure(wf)

    assert result["structural_valid"] is False
    assert result["structural_status"] == "incomplete"
    assert any("no executable control-flow edges" in error for error in result["structural_errors"])
    assert any("no outgoing entry" in error for error in result["structural_errors"])


def test_structure_detects_unreachable_actions_and_missing_termination():
    wf = parse_workflow(f"""package P {{ action def W {{
        {_node("reachable")}
        {_node("orphan")}
        first start then reachable;
    }} }}""")

    result = validate_workflow_structure(wf)

    assert result["structural_valid"] is False
    assert any("orphan" in error and "unreachable" in error for error in result["structural_errors"])
    assert any("no termination is reachable" in error for error in result["structural_errors"])
    assert any("reachable actions have no path to done" in error for error in result["structural_errors"])


def test_explicit_structural_exception_is_recorded_but_never_passes():
    source = f"""// RAG-STRUCTURAL-EXCEPTION: external termination is intentional
    package P {{ action def W {{
        {_node("a")}
        first start then a;
        first a then done;
    }} }}"""
    wf = parse_workflow(source)

    result = validate_workflow_structure(wf, source)

    assert result["structural_status"] == "intentional_exception"
    assert result["intentional_exception"] == "external termination is intentional"
    assert result["structural_valid"] is False
