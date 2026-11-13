import json
import ast
import csv
import os
import statistics
from datasets import load_dataset


LOGIC_NODES = {"or", "and", "xor", "not"}


def load_sopbench_hf(split="train"):
    ds = load_dataset("Zekunli/SOPBench")
    print("Available splits:", ds.keys())
    data = ds[split]
    print(f"Loaded {len(data)} samples from {split}")
    return data


def parse_maybe_python_string(x):
    if not isinstance(x, str):
        return x

    x = x.strip()

    try:
        return json.loads(x)
    except Exception:
        pass

    try:
        return ast.literal_eval(x)
    except Exception as exc:
        raise ValueError("Could not parse string as JSON or Python literal.") from exc


def extract_tool_descriptions(sample):
    raw_tools = parse_maybe_python_string(sample.get("tools", "[]"))
    desc_map = {}

    if not isinstance(raw_tools, list):
        return desc_map

    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {})
        if not isinstance(fn, dict):
            continue

        name = fn.get("name")
        desc = fn.get("description")
        if name:
            desc_map[name] = desc

    return desc_map


def parse_node(node):
    if isinstance(node, list) and len(node) >= 1:
        name = node[0] if isinstance(node[0], str) else str(node[0])
        args = node[1] if len(node) > 1 and isinstance(node[1], dict) else {}
    elif isinstance(node, str):
        name = node
        args = {}
    else:
        name = str(node)
        args = {}

    kind = "operator" if name.lower() in LOGIC_NODES else "action"

    return {
        "name": name,
        "kind": kind,
        "arguments": args
    }


def extract_graph_description(sample):
    parts = []

    domain = sample.get("domain")
    goal = sample.get("user_goal")
    instruction = sample.get("user_instruction")
    prompt = sample.get("user_prompt")

    if domain:
        parts.append(f"Domain: {domain}.")
    if goal:
        parts.append(f"User goal: {goal}.")
    if instruction:
        parts.append(f"Instruction: {instruction}")
    if prompt:
        parts.append(f"Prompt: {prompt}")

    return " ".join(parts)


def extract_sopbench_raw(sample):
    raw_graph = parse_maybe_python_string(sample["directed_action_graph"])
    tool_desc = extract_tool_descriptions(sample)

    raw_nodes = raw_graph.get("nodes", [])
    raw_connections = raw_graph.get("connections", [])

    nodes = []
    for idx, raw_node in enumerate(raw_nodes):
        parsed = parse_node(raw_node)

        name = parsed["name"]
        description = tool_desc.get(name)

        nodes.append({
            "id": idx,
            "name": name,
            "kind": parsed["kind"],
            "arguments": parsed["arguments"],
            "description": description
        })

    edges = []
    for conn in raw_connections:
        if isinstance(conn, (list, tuple)) and len(conn) == 2:
            src, dst = conn
            edges.append([src, dst])

    return {
        "nodes": nodes,
        "edges": edges,
        "graph_description": extract_graph_description(sample)
    }
def transform_and_nodes_to_fork_join(graph):
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    incoming = {}
    outgoing = {}

    for src, dst in edges:
        incoming.setdefault(dst, []).append(src)
        outgoing.setdefault(src, []).append(dst)

    new_nodes = [
        n for n in graph["nodes"]
        if not (n["kind"] == "operator" and n["name"].lower() == "and")
    ]

    new_edges = []

    for src, dst in edges:
        src_node = nodes_by_id[src]
        dst_node = nodes_by_id[dst]

        if src_node["kind"] == "operator" and src_node["name"].lower() == "and":
            continue

        if dst_node["kind"] == "operator" and dst_node["name"].lower() == "and":
            continue

        new_edges.append([src, dst])

    for and_id, and_node in nodes_by_id.items():
        if and_node["kind"] != "operator" or and_node["name"].lower() != "and":
            continue

        dependencies = incoming.get(and_id, [])
        parents = outgoing.get(and_id, [])

        for dep in dependencies:
            for parent in parents:
                new_edges.append([dep, parent])

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "graph_description": graph.get("graph_description", "")
    }

def transform_or_nodes_to_fallback(graph):
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    outgoing = {}
    incoming = {}

    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    def dedup(seq):
        seen = set()
        out = []
        for item in seq:
            t = tuple(item) if isinstance(item, list) else item
            if t not in seen:
                seen.add(t)
                out.append(item)
        return out

    or_ids = {
        nid for nid, node in nodes_by_id.items()
        if node["kind"] == "operator" and node["name"].lower() == "or"
    }

    new_nodes = [
        node.copy()
        for node in graph["nodes"]
        if node["id"] not in or_ids
    ]

    new_edges = []
    decisions = []

    next_id = max(nodes_by_id.keys()) + 1 if nodes_by_id else 0

    # Keep normal edges that do not touch OR nodes.
    for src, dst in edges:
        if src in or_ids or dst in or_ids:
            continue
        new_edges.append([src, dst])

    for or_id in sorted(or_ids):
        options = [x for x in incoming.get(or_id, []) if x not in or_ids]
        continuations = [x for x in outgoing.get(or_id, []) if x not in or_ids]

        if not options:
            continue

        cloned_options = []

        for option in options:
            original = nodes_by_id[option]

            # Clone the option for this OR context.
            cloned_id = next_id
            next_id += 1

            cloned_node = original.copy()
            cloned_node["id"] = cloned_id
            cloned_node["name"] = f"{original['name']}_or_{or_id}"

            new_nodes.append(cloned_node)
            cloned_options.append(cloned_id)

        for i, option in enumerate(cloned_options):
            decision_id = next_id
            next_id += 1

            option_name = next(n["name"] for n in new_nodes if n["id"] == option)

            new_nodes.append({
                "id": decision_id,
                "name": f"DecisionAfter_{option}",
                "kind": "decision",
                "arguments": {},
                "description": f"Decision after trying fallback option {option_name} from OR node {or_id}."
            })

            new_edges.append([option, decision_id])

            for cont in continuations:
                new_edges.append([decision_id, cont])

            else_target = None
            if i + 1 < len(cloned_options):
                else_target = cloned_options[i + 1]
                new_edges.append([decision_id, else_target])

            decisions.append({
                "decision_id": decision_id,
                "decision_name": f"DecisionAfter_{option}",
                "source_or_id": or_id,
                "tested_action_id": option,
                "tested_action_name": option_name,
                "success_targets": continuations,
                "else_target": else_target,
                "is_last_fallback": else_target is None,
            })

    new_edges = dedup(new_edges)

    valid_ids = {n["id"] for n in new_nodes}
    new_edges = [
        [src, dst]
        for src, dst in new_edges
        if src in valid_ids and dst in valid_ids
    ]
    used_ids = set()
    for src, dst in new_edges:
        used_ids.add(src)
        used_ids.add(dst)

    for d in decisions:
        used_ids.add(d["decision_id"])
        used_ids.add(d["tested_action_id"])
        if d.get("else_target") is not None:
            used_ids.add(d["else_target"])
        for x in d.get("success_targets", []):
            used_ids.add(x)

    main_action_id = graph.get("main_action_id", 0)
    used_ids.add(main_action_id)

    new_nodes = [n for n in new_nodes if n["id"] in used_ids]

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "decisions": decisions,
        "main_action_id": graph.get("main_action_id", 0),
        "graph_description": graph.get("graph_description", "")
    }

def print_extraction(graph):
    print("\n=== GRAPH DESCRIPTION ===")
    print(graph["graph_description"])

    print("\n=== NODES ===")
    for n in graph["nodes"]:
        print(
            f'{n["id"]}: {n["name"]} | kind={n["kind"]} | '
            f'args={n["arguments"]} | desc={n["description"]}'
        )

    print("\n=== EDGES ===")
    for src, dst in graph["edges"]:
        print(f"{src} -> {dst}")


def print_transformed(transformed):
    print("\n=== TRANSFORMED NODES ===")
    for n in transformed["nodes"]:
        print(f'{n["id"]} {n["name"]} {n["kind"]}')

    print("\n=== TRANSFORMED EDGES ===")
    for src, dst in transformed["edges"]:
        print(f"{src} -> {dst}")

    print("\n=== DECISIONS ===")
    for d in transformed["decisions"]:
        print(f'\n{d["decision_name"]} (from OR node {d["source_or_id"]})')
        print(f'  test: {d["tested_action_name"]}.success')
        print(f'  success -> {d["success_targets"]}')
        print(f'  else -> {d["else_target"]}')
import re


def sanitize_name(name: str) -> str:
    """
    Convert arbitrary node names into SysML-safe identifiers.
    """
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name


def build_sysml_from_dependency_graph(
    graph: dict,
    package_name: str = "SOPBenchPackage",
    action_def_name: str = "Workflow",
    agent_part_name: str = "WorkflowAgent",
    agent_instance_name: str = "workflowAgent"
) -> str:

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    # IMPORTANT:
    # SOPBench direction:
    #
    #     parent -> dependency
    #
    # Therefore outgoing[node] = things that node depends on.
    dependencies = {nid: [] for nid in nodes_by_id}
    parents = {nid: [] for nid in nodes_by_id}

    for parent, child in edges:
        dependencies[parent].append(child)
        parents[child].append(parent)

    # ---------------------------------------------------------
    # SysML-safe names
    # ---------------------------------------------------------

    node_name = {}
    used_names = set()

    for node_id in sorted(nodes_by_id):
        name = sanitize_name(nodes_by_id[node_id]["name"])

        if name in used_names:
            name = f"{name}_{node_id}"

        used_names.add(name)
        node_name[node_id] = name

    # ---------------------------------------------------------
    # Output structures
    # ---------------------------------------------------------

    declarations = []
    flows = []

    emitted_flows = set()

    fork_counter = 0
    join_counter = 0
    decision_counter = 0

    def emit(statement):
        if statement not in emitted_flows:
            emitted_flows.add(statement)
            flows.append(statement)

    def new_fork():
        nonlocal fork_counter

        name = f"forkNode_{fork_counter}"
        fork_counter += 1

        declarations.append(f"        fork {name};")
        return name

    def new_join():
        nonlocal join_counter

        name = f"joinNode_{join_counter}"
        join_counter += 1

        declarations.append(f"        join {name};")
        return name

    def new_decision():
        nonlocal decision_counter

        name = f"decision_{decision_counter}"
        decision_counter += 1

        declarations.append(f"        decide {name};")
        return name

    # ---------------------------------------------------------
    # Determine which action is actually tested by an OR branch.
    #
    # Normally OR children in SOPBench are actions.
    # ---------------------------------------------------------

    # def result_action(node_id):
    #     node = nodes_by_id[node_id]
    #
    #     if node["kind"] == "action":
    #         return node_name[node_id]
    #
    #     children = dependencies.get(node_id, [])
    #
    #     if len(children) == 1:
    #         return result_action(children[0])
    #
    #     raise ValueError(
    #         f"Cannot determine success-producing action for "
    #         f"OR alternative node {node_id}: {node['name']}"
    #     )

    # ---------------------------------------------------------
    # Recursive compiler
    #
    # compile_node(v, continuation)
    #
    # means:
    #
    #     generate everything required to execute v,
    #     then continue to `continuation`.
    #
    # Returns the ENTRY point of this generated fragment.
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Action usages
    #
    # A SOPBench node may be shared by several logical branches.
    # Each compiled occurrence must therefore have its own
    # SysML action usage.
    # ---------------------------------------------------------

    action_usages = []
    action_occurrence_count = {}

    # Reserve the original/base names so clone names cannot
    # accidentally collide with another original action name.
    reserved_action_names = {
        node_name[nid]
        for nid, node in nodes_by_id.items()
        if node["kind"] == "action"
    }

    used_action_usage_names = set()

    def new_action_usage(node_id):

        base_name = node_name[node_id]

        count = action_occurrence_count.get(node_id, 0) + 1
        action_occurrence_count[node_id] = count

        # First occurrence keeps the normal action name.
        if count == 1:
            usage_name = base_name

        else:
            suffix = count
            usage_name = f"{base_name}__occ{suffix}"

            while (
                    usage_name in used_action_usage_names
                    or usage_name in reserved_action_names
            ):
                suffix += 1
                usage_name = f"{base_name}__occ{suffix}"

        used_action_usage_names.add(usage_name)

        action_usages.append({
            "node_id": node_id,
            "name": usage_name,
        })

        return usage_name

    recursion_stack = set()

    def compile_node(node_id, success_target, failure_target="done"):

        if node_id in recursion_stack:
            raise ValueError(
                f"Cycle detected in SOPBench dependency graph at node {node_id}"
            )

        recursion_stack.add(node_id)

        node = nodes_by_id[node_id]
        kind = node["kind"]
        operator = node["name"].lower()

        children = dependencies.get(node_id, [])

        # =========================================================
        # ACTION
        # =========================================================
        if kind == "action":

            action_ref = new_action_usage(node_id)

            # ---------------------------------------------------------
            # Helper: emit what happens AFTER this action
            # ---------------------------------------------------------
            def emit_action_exit():

                # No decision needed when both outcomes are identical.
                if success_target == failure_target:
                    emit(
                        f"        first {action_ref} then {success_target};"
                    )
                    return

                decision = new_decision()

                emit(
                    f"        first {action_ref} then {decision};"
                )

                flows.append(
                    f"        first {decision};"
                )
                flows.append(
                    f"            if {action_ref}.success "
                    f"then {success_target};"
                )
                flows.append(
                    f"            else {failure_target};"
                )
                flows.append("")

            # =========================================================
            # No dependencies
            # =========================================================
            if not children:
                emit_action_exit()

                recursion_stack.remove(node_id)
                return action_ref

            # =========================================================
            # One dependency
            # =========================================================
            if len(children) == 1:
                child_entry = compile_node(
                    children[0],
                    success_target=action_ref,
                    failure_target=failure_target
                )

                # IMPORTANT:
                # action exit is emitted AFTER its prerequisite fragment
                emit_action_exit()

                recursion_stack.remove(node_id)
                return child_entry

            # =========================================================
            # Multiple direct dependencies = implicit AND
            # =========================================================

            fork = new_fork()
            join = new_join()

            for child in children:
                child_entry = compile_node(
                    child,
                    success_target=join,
                    failure_target=failure_target
                )

                emit(
                    f"        first {fork} then {child_entry};"
                )

            # All prerequisites completed -> execute action
            emit(
                f"        first {join} then {action_ref};"
            )

            # Then whatever happens after the action
            emit_action_exit()

            recursion_stack.remove(node_id)
            return fork
        # =========================================================
        # AND
        # =========================================================
        if kind == "operator" and operator == "and":

            if not children:
                recursion_stack.remove(node_id)
                return success_target

            # AND with one child is equivalent to that child
            if len(children) == 1:
                entry = compile_node(
                    children[0],
                    success_target=success_target,
                    failure_target=failure_target
                )

                recursion_stack.remove(node_id)
                return entry

            fork = new_fork()
            join = new_join()

            # All successful branches arrive here.
            emit(
                f"        first {join} then {success_target};"
            )

            for child in children:
                child_entry = compile_node(
                    child,
                    success_target=join,
                    failure_target=failure_target
                )

                emit(
                    f"        first {fork} then {child_entry};"
                )

            recursion_stack.remove(node_id)
            return fork

        # =========================================================
        # OR = ORDERED FALLBACK
        # =========================================================
        if kind == "operator" and operator == "or":

            if not children:
                recursion_stack.remove(node_id)
                return failure_target

            # Build fallback chain backwards.
            #
            # OR(A, B, C)
            #
            # A failure -> B
            # B failure -> C
            # C failure -> failure_target
            #
            # Any success -> success_target

            next_failure = failure_target

            # Compile from last alternative to first
            for child in reversed(children):
                child_entry = compile_node(
                    child,
                    success_target=success_target,
                    failure_target=next_failure
                )

                next_failure = child_entry

            recursion_stack.remove(node_id)

            # Entry of the OR is the first alternative
            return next_failure

        # =========================================================
        # Unsupported
        # =========================================================
        recursion_stack.remove(node_id)

        raise ValueError(
            f"Unsupported SOPBench operator: {node['name']}"
        )

    def order_flows_for_output(flow_lines):
        """
        Reorder generated SysML succession statements into causal order.

        Example:
            start -> A
            A -> decision
            decision -> B / C
            ...
            root_action -> done

        The recursive compiler may generate statements backwards,
        especially for OR fallback chains. This function only changes
        textual output order; it does not change the graph semantics.
        """

        import re
        from collections import defaultdict, deque

        # ---------------------------------------------------------
        # 1. Group lines into logical blocks
        #
        # Normal block:
        #     first A then B;
        #
        # Decision block:
        #     first decision_0;
        #         if A.success then X;
        #         else Y;
        # ---------------------------------------------------------

        blocks = []

        i = 0

        while i < len(flow_lines):

            line = flow_lines[i]

            # Skip standalone empty lines
            if not line.strip():
                i += 1
                continue

            stripped = line.strip()

            # -----------------------------------------------------
            # Decision block
            # -----------------------------------------------------
            decision_match = re.match(
                r"first\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
                stripped
            )

            if decision_match:

                source = decision_match.group(1)

                block_lines = [line]
                targets = []

                i += 1

                while i < len(flow_lines):

                    next_line = flow_lines[i]
                    next_stripped = next_line.strip()

                    if not next_stripped:
                        i += 1
                        break

                    # Stop if another flow statement begins
                    if next_stripped.startswith("first "):
                        break

                    block_lines.append(next_line)

                    # if ... then TARGET;
                    success_match = re.search(
                        r"\bthen\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
                        next_stripped
                    )

                    if success_match:
                        targets.append(success_match.group(1))

                    # else TARGET;
                    else_match = re.match(
                        r"else\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
                        next_stripped
                    )

                    if else_match:
                        targets.append(else_match.group(1))

                    i += 1

                blocks.append({
                    "source": source,
                    "targets": targets,
                    "lines": block_lines,
                })

                continue

            # -----------------------------------------------------
            # Normal succession:
            #
            # first A then B;
            # -----------------------------------------------------
            normal_match = re.match(
                r"first\s+([A-Za-z_][A-Za-z0-9_]*)"
                r"\s+then\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
                stripped
            )

            if normal_match:
                source = normal_match.group(1)
                target = normal_match.group(2)

                blocks.append({
                    "source": source,
                    "targets": [target],
                    "lines": [line],
                })

                i += 1
                continue

            # Unknown line: preserve it
            blocks.append({
                "source": None,
                "targets": [],
                "lines": [line],
            })

            i += 1

        # ---------------------------------------------------------
        # 2. Group blocks by source
        #
        # A fork may have several blocks:
        #
        # first fork then A;
        # first fork then B;
        # ---------------------------------------------------------

        blocks_by_source = defaultdict(list)

        for index, block in enumerate(blocks):
            if block["source"] is not None:
                blocks_by_source[block["source"]].append(index)

        # ---------------------------------------------------------
        # 3. Build control-flow graph for ordering
        # ---------------------------------------------------------

        successors = defaultdict(set)
        predecessors = defaultdict(set)

        for block in blocks:

            source = block["source"]

            if source is None:
                continue

            for target in block["targets"]:

                # "done" has no outgoing block
                if target == "done":
                    continue

                successors[source].add(target)
                predecessors[target].add(source)

        # ---------------------------------------------------------
        # 4. Topological order, starting naturally from "start"
        # ---------------------------------------------------------

        all_sources = set(blocks_by_source.keys())

        indegree = {
            source: len(
                predecessors.get(source, set()) & all_sources
            )
            for source in all_sources
        }

        ready = deque()

        # Always prefer start first
        if "start" in all_sources:
            ready.append("start")

        for source in sorted(all_sources):
            if source == "start":
                continue

            if indegree[source] == 0:
                ready.append(source)

        ordered_sources = []
        visited_sources = set()

        while ready:

            source = ready.popleft()

            if source in visited_sources:
                continue

            visited_sources.add(source)
            ordered_sources.append(source)

            for target in successors.get(source, set()):

                if target not in indegree:
                    continue

                indegree[target] -= 1

                if indegree[target] == 0:
                    ready.append(target)

        # In case something was not reached because of an unusual
        # structure, preserve it rather than dropping it.
        for source in blocks_by_source:
            if source not in visited_sources:
                ordered_sources.append(source)

        # ---------------------------------------------------------
        # 5. Rebuild textual flow section
        # ---------------------------------------------------------

        result = []
        emitted_blocks = set()

        for source in ordered_sources:

            for block_index in blocks_by_source[source]:

                if block_index in emitted_blocks:
                    continue

                emitted_blocks.add(block_index)

                block = blocks[block_index]

                result.extend(block["lines"])

                # Blank line after decision blocks for readability
                if (
                        block["lines"]
                        and re.match(
                    r"\s*first\s+\w+\s*;",
                    block["lines"][0]
                )
                ):
                    result.append("")

        # Preserve any unclassified blocks
        for block_index, block in enumerate(blocks):

            if block_index not in emitted_blocks:
                result.extend(block["lines"])

        return result

    # ---------------------------------------------------------
    # Find main/root SOPBench action
    #
    # Root = node which is never itself a dependency.
    # ---------------------------------------------------------

    root_ids = [
        nid
        for nid in nodes_by_id
        if not parents[nid]
        and nodes_by_id[nid]["kind"] == "action"
    ]

    if not root_ids:
        raise ValueError("Could not determine SOPBench root action.")

    # Normally SOPBench has one main action.
    main_action_id = root_ids[0]

    # Compile entire dependency tree.
    workflow_entry = compile_node(
        main_action_id,
        success_target="done",
        failure_target="done"
    )

    emit(
        f"        first start then {workflow_entry};"
    )



    # ---------------------------------------------------------
    # Produce SysML
    # ---------------------------------------------------------

    def escape(text):
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

    lines = []

    lines.append(f"package {package_name} {{")
    lines.append("")
    lines.append(f"    action def {action_def_name} {{")
    lines.append("")

    lines.append(
        f"        part def {agent_part_name} specializes SolutionAgent {{"
    )
    lines.append("        }")
    lines.append("")

    lines.append(
        f"        {agent_instance_name} : {agent_part_name};"
    )
    lines.append("")

    # Actions
    # Actions
    #
    # Declare compiled action usages rather than original graph nodes.
    # Shared DAG nodes may therefore appear several times with
    # different SysML usage names.
    for usage in action_usages:
        node_id = usage["node_id"]
        action_ref = usage["name"]

        node = nodes_by_id[node_id]

        description = escape(
            node.get("description") or node["name"]
        )

        lines.append(f"        action {action_ref} {{")
        lines.append(
            f'            description = "{description}";'
        )
        lines.append(
            f"            agent = {agent_instance_name};"
        )
        lines.append("            out success;")
        lines.append("        }")
        lines.append("")

    # Control nodes
    lines.extend(declarations)

    if declarations:
        lines.append("")

    # Control flow, rendered in causal order
    ordered_flows = order_flows_for_output(flows)
    lines.extend(ordered_flows)

    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)
    # If disconnected components remain, process them deterministically.
    remaining = [nid for nid in sorted(nodes_by_id, key=_id_sort_key) if nid not in activated]
    for nid in remaining:
        emit_first("start", _entry_ref(nid))

    # End only from the main/root SOPBench action, usually original node 0.
    main_action_id = transformed.get("main_action_id", 0)

    if main_action_id in nodes_by_id:
        emit_first(_node_ref(main_action_id), "done")

    lines.extend(flow_lines)

    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def summarize_counts(values):
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "variance": 0.0,
        }

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "variance": statistics.pvariance(values) if len(values) > 1 else 0.0,
    }


def count_flow_elements(transformed):
    nodes = transformed.get("nodes", [])
    edges = transformed.get("edges", [])

    actions = sum(1 for n in nodes if n.get("kind") == "action")
    decisions = sum(1 for n in nodes if n.get("kind") == "decision")
    forks = sum(
        1
        for n in nodes
        if n.get("kind") == "operator" and str(n.get("name", "")).lower() == "and"
    )

    action_ids = {n["id"] for n in nodes if n.get("kind") == "action"}
    incoming_count = {}
    for src, dst in edges:
        incoming_count[dst] = incoming_count.get(dst, 0) + 1

    joins = sum(1 for node_id in action_ids if incoming_count.get(node_id, 0) > 1)

    return actions, decisions, joins, forks


def metrics_for_sopbench(split="train", limit=None, progress_every=1000):
    data = load_sopbench_hf(split=split)
    total = len(data) if limit is None else min(limit, len(data))

    action_counts = []
    decision_counts = []
    join_counts = []
    fork_counts = []

    parse_fail = 0
    transform_fail = 0

    for i in range(total):
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"processed={i + 1}/{total}")

        sample = data[i]

        try:
            graph = extract_sopbench_raw(sample)
        except Exception:
            parse_fail += 1
            continue

        try:
            transformed = transform_or_nodes_to_fallback(graph)
        except Exception:
            transform_fail += 1
            continue

        actions, decisions, joins, forks = count_flow_elements(transformed)
        action_counts.append(actions)
        decision_counts.append(decisions)
        join_counts.append(joins)
        fork_counts.append(forks)

    valid_models = len(action_counts)

    total_actions = sum(action_counts)
    total_decisions = sum(decision_counts)
    total_joins = sum(join_counts)
    total_forks = sum(fork_counts)
    all_flow_items = total_actions + total_decisions + total_joins + total_forks

    by_model = {
        "actions": (sum(1 for x in action_counts if x > 0) / valid_models) if valid_models else 0.0,
        "decisions": (sum(1 for x in decision_counts if x > 0) / valid_models) if valid_models else 0.0,
        "joins": (sum(1 for x in join_counts if x > 0) / valid_models) if valid_models else 0.0,
        "forks": (sum(1 for x in fork_counts if x > 0) / valid_models) if valid_models else 0.0,
    }

    global_freq = {
        "actions": (total_actions / all_flow_items) if all_flow_items else 0.0,
        "decisions": (total_decisions / all_flow_items) if all_flow_items else 0.0,
        "joins": (total_joins / all_flow_items) if all_flow_items else 0.0,
        "forks": (total_forks / all_flow_items) if all_flow_items else 0.0,
    }

    return {
        "processed": total,
        "valid_models": valid_models,
        "parse_fail": parse_fail,
        "transform_fail": transform_fail,
        "actions": summarize_counts(action_counts),
        "decisions": summarize_counts(decision_counts),
        "joins": summarize_counts(join_counts),
        "forks": summarize_counts(fork_counts),
        "frequencies_by_model": by_model,
        "frequencies_global": global_freq,
    }

def dependency_graph_to_execution_graph(graph):
    """
    SOPBench edge: A -> B means A depends on B.
    Execution edge: B -> A.
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    execution_edges = [[dst, src] for src, dst in edges]

    return {
        "nodes": nodes,
        "edges": execution_edges,
        "main_action_id": graph.get("main_action_id", 0),
        "graph_description": graph.get("graph_description", "")
    }

def print_sopbench_metrics(report):
    for key in ["actions", "decisions", "joins", "forks"]:
        stats = report[key]
        print(
            f"{key}: sum={stats['sum']} mean={stats['mean']:.3f} "
            f"min={stats['min']} max={stats['max']} var={stats['variance']:.3f}"
        )

    print(f"frequencies by model: {report['frequencies_by_model']}")
    print(f"frequencies global: {report['frequencies_global']}")


def run_sopbench_transformation_to_csv(
    output_csv: str = "sopbench_sysmlv2.csv",
    splits=("train", "test"),
    limit_per_split=None,
    progress_every: int = 1000,
):
    ds = load_dataset("Zekunli/SOPBench")
    print("Available splits:", ds.keys())

    output_path = output_csv
    if not os.path.isabs(output_path):
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            output_path,
        )

    headers = ["origin", "user_prompt", "sysml_model"]

    ok = 0
    parse_fail = 0
    transform_fail = 0

    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for split in splits:
            if split not in ds:
                print(f"Skipping missing split: {split}")
                continue

            data = ds[split]

            total = (
                len(data)
                if limit_per_split is None
                else min(limit_per_split, len(data))
            )

            print(f"Processing split '{split}' with {total} samples")

            for i in range(total):

                if progress_every > 0 and (i + 1) % progress_every == 0:
                    print(
                        f"split={split} processed={i + 1}/{total} "
                        f"OK={ok} "
                        f"parse_fail={parse_fail} "
                        f"transform_fail={transform_fail}"
                    )

                sample = data[i]

                try:
                    graph = extract_sopbench_raw(sample)
                except Exception:
                    parse_fail += 1
                    continue

                try:
                    pkg_name = sanitize_name(
                        str(sample.get("domain") or "SOPBenchPackage")
                    )

                    action_def_name = sanitize_name(
                        str(sample.get("user_goal") or "Workflow")
                    )

                    sysml_model = build_sysml_from_dependency_graph(
                        graph,
                        package_name=pkg_name,
                        action_def_name=action_def_name,
                        agent_part_name="WorkflowAgent",
                        agent_instance_name="workflowAgent",
                    )
                    # DEBUG: print first 10 instances
                    if i < 10:
                        print("\n" + "=" * 100)
                        print(f"SAMPLE {split}::{i}")
                        print("=" * 100)

                        print("\n--- ORIGINAL SOPBENCH GRAPH ---")
                        print(sample.get("directed_action_graph"))

                        print("\n--- NODES ---")
                        for node in graph["nodes"]:
                            print(
                                f'{node["id"]}: {node["name"]} '
                                f'({node["kind"]})'
                            )

                        print("\n--- DEPENDENCIES ---")

                        nodes_debug = {
                            n["id"]: n for n in graph["nodes"]
                        }

                        for parent_id, child_id in graph["edges"]:
                            parent = nodes_debug[parent_id]
                            child = nodes_debug[child_id]

                            print(
                                f'{parent_id}: {parent["name"]} '
                                f'--depends on--> '
                                f'{child_id}: {child["name"]}'
                            )

                        print("\n--- GENERATED SYSML ---")
                        print(sysml_model)

                        print("\n" + "=" * 100)



                except Exception as e:

                    transform_fail += 1
                    print(f"\n{'=' * 80}")
                    print(f"FAILED sample {split}::{i}")
                    print(type(e).__name__, ":", e)
                    print("\n--- ORIGINAL directed_action_graph ---")
                    print(sample.get("directed_action_graph"))
                    print("\n--- PARSED NODES ---")
                    for node in graph["nodes"]:
                        print(
                            f'{node["id"]}: '
                            f'{node["name"]} '
                            f'({node["kind"]})'

                        )
                    print("\n--- PARSED EDGES / DEPENDENCIES ---")
                    for src, dst in graph["edges"]:
                        src_name = next(
                            n["name"] for n in graph["nodes"]
                            if n["id"] == src
                        )
                        dst_name = next(
                            n["name"] for n in graph["nodes"]
                            if n["id"] == dst
                        )
                        print(
                            f"{src}: {src_name}"
                            f"  ->  "
                            f"{dst}: {dst_name}"
                        )
                    print("=" * 80)
                    continue

                user_prompt = (
                    sample.get("user_prompt")
                    or extract_graph_description(sample)
                )

                origin = f"sopbench::{split}::{i}"

                writer.writerow(
                    {
                        "origin": origin,
                        "user_prompt": user_prompt,
                        "sysml_model": sysml_model,
                    }
                )

                ok += 1

    print(f"Output CSV: {output_path}")
    print(f"OK: {ok}")
    print(f"parse_fail: {parse_fail}")
    print(f"transform_fail: {transform_fail}")


if __name__ == "__main__":
    run_sopbench_transformation_to_csv(
        output_csv="sopbench_sysmlv2.csv",
        splits=("train", "test"),
        limit_per_split=None,
        progress_every=1000,
    )