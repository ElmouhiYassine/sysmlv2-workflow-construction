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


def transform_or_nodes_to_fallback(graph):

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    outgoing = {}
    incoming = {}
    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    new_nodes = []
    new_edges = []
    decisions = []

    next_id = max(nodes_by_id.keys()) + 1 if nodes_by_id else 0

    or_ids = {
        nid for nid, node in nodes_by_id.items()
        if node["kind"] == "operator" and node["name"].lower() == "or"
    }

    # Keep all non-OR original nodes
    for nid, node in nodes_by_id.items():
        if nid not in or_ids:
            new_nodes.append(node)

    def dedup(seq):
        seen = set()
        out = []
        for item in seq:
            t = tuple(item) if isinstance(item, list) else item
            if t not in seen:
                seen.add(t)
                out.append(item)
        return out

    def resolve_first_non_or(node_id):

        visited = set()
        current = node_id

        while current in or_ids:
            if current in visited:
                raise ValueError(f"Cycle detected while resolving nested OR at node {current}")
            visited.add(current)

            children = outgoing.get(current, [])
            if not children:
                raise ValueError(f"OR node {current} has no children")

            current = children[0]

        return current

    # Copy non-OR edges first, except child->succ edges of OR children
    edges_to_skip = set()
    for or_id in or_ids:
        children = outgoing.get(or_id, [])
        for c in children:
            for succ in outgoing.get(c, []):
                edges_to_skip.add((c, succ))

    for src, dst in edges:
        if src in or_ids or dst in or_ids:
            continue
        if (src, dst) in edges_to_skip:
            continue
        new_edges.append([src, dst])

    # Transform each OR
    for or_id in sorted(or_ids):
        preds = incoming.get(or_id, [])
        children = outgoing.get(or_id, [])

        if not children:
            continue

        # Resolve children if some are nested ORs
        resolved_children = [resolve_first_non_or(c) for c in children]

        # Collect continuation targets from original direct children
        continuation_targets = []
        for c in children:
            for succ in outgoing.get(c, []):
                continuation_targets.append(resolve_first_non_or(succ) if succ in or_ids else succ)

        continuation_targets = dedup([[x] for x in continuation_targets])
        continuation_targets = [x[0] for x in continuation_targets]

        # Predecessors now point to first resolved child
        first_child = resolved_children[0]
        for p in preds:
            if p in or_ids:
                continue
            new_edges.append([p, first_child])

        # Build fallback chain
        for i, child in enumerate(resolved_children):
            is_last = i == len(resolved_children) - 1

            if not is_last:
                next_child = resolved_children[i + 1]

                decision_id = next_id
                next_id += 1
                decision_name = f"DecisionAfter_{child}"

                tested_name = nodes_by_id[child]["name"] if child in nodes_by_id else f"node_{child}"

                new_nodes.append({
                    "id": decision_id,
                    "name": decision_name,
                    "kind": "decision",
                    "arguments": {},
                    "description": (
                        f"Decision after trying fallback branch {tested_name} "
                        f"from OR node {or_id}."
                    )
                })

                new_edges.append([child, decision_id])

                for succ in continuation_targets:
                    new_edges.append([decision_id, succ])

                new_edges.append([decision_id, next_child])

                decisions.append({
                    "decision_id": decision_id,
                    "decision_name": decision_name,
                    "source_or_id": or_id,
                    "tested_action_id": child,
                    "tested_action_name": tested_name,
                    "success_targets": continuation_targets,
                    "else_target": next_child,
                    "guard_success": f"if {tested_name}.success then",
                    "guard_else": f"else {nodes_by_id[next_child]['name']}"
                })

            else:
                for succ in continuation_targets:
                    new_edges.append([child, succ])

    new_edges = dedup(new_edges)

    # Final safety: remove dangling edges
    valid_ids = {n["id"] for n in new_nodes}
    new_edges = [e for e in new_edges if e[0] in valid_ids and e[1] in valid_ids]

    # Also clean decisions whose referenced ids disappeared
    cleaned_decisions = []
    for d in decisions:
        if d["decision_id"] not in valid_ids:
            continue
        if d["tested_action_id"] not in valid_ids:
            continue
        if d["else_target"] not in valid_ids:
            continue
        d["success_targets"] = [x for x in d["success_targets"] if x in valid_ids]
        cleaned_decisions.append(d)

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "decisions": cleaned_decisions
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


def build_sysml_from_transformed(
    transformed: dict,
    package_name: str = "SOPBenchPackage",
    action_def_name: str = "Workflow",
    agent_part_name: str = "WorkflowAgent",
    agent_instance_name: str = "workflowAgent"
) -> str:
    """
    Generate SysML v2 code from transformed SOPBench graph.

    Expected transformed format:
    {
        "nodes": [...],
        "edges": [[src, dst], ...],
        "decisions": [...]
    }
    """

    nodes_by_id = {n["id"]: n for n in transformed["nodes"]}
    edges = [
        [src, dst]
        for src, dst in transformed["edges"]
        if src in nodes_by_id and dst in nodes_by_id
    ]
    decisions = [
        d for d in transformed.get("decisions", [])
        if d.get("decision_id") in nodes_by_id and d.get("else_target") in nodes_by_id
    ]

    outgoing = {}
    incoming = {}
    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    node_var_name = {}
    used_names = set()
    for node_id in sorted(nodes_by_id):
        raw_name = nodes_by_id[node_id]["name"]
        safe_name = sanitize_name(raw_name)
        if safe_name in used_names:
            safe_name = f"{safe_name}_{node_id}"
        used_names.add(safe_name)
        node_var_name[node_id] = safe_name

    fork_nodes = {
        node_id: f"forkNode_{node_id}"
        for node_id, node in nodes_by_id.items()
        if node["kind"] == "operator" and node["name"].lower() == "and"
    }

    # Create joins only for action fan-in to avoid invalid join/fork combinations.
    join_nodes = {}
    for node_id in sorted(incoming):
        preds = incoming[node_id]
        target = nodes_by_id.get(node_id)
        if not target:
            continue
        if len(preds) > 1 and target["kind"] == "action":
            join_nodes[node_id] = f"joinNode_{node_id}"

    def _escape_str(text):
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    def _node_ref(node_id):
        node = nodes_by_id[node_id]
        if node_id in fork_nodes:
            return fork_nodes[node_id]
        return node_var_name[node_id]

    def _entry_ref(node_id):
        # Fan-in actions must be entered via their join node.
        if node_id in join_nodes:
            return join_nodes[node_id]
        return _node_ref(node_id)

    lines = []
    lines.append(f"package {package_name} {{")
    lines.append("")
    lines.append(f"    action def {action_def_name} {{")
    lines.append("")
    lines.append(f"        part def {agent_part_name} specializes SolutionAgent {{")
    lines.append("        }")
    lines.append("")
    lines.append(f"        {agent_instance_name} : {agent_part_name};")
    lines.append("")

    # 1) Action definitions first
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        if node["kind"] != "action":
            continue
        name = node_var_name[node_id]
        desc = _escape_str(node.get("description") or node["name"])
        lines.append(f"        action {name} {{")
        lines.append(f'            description = "{desc}";')
        lines.append(f"            agent = {agent_instance_name};")
        lines.append("            out success;")
        lines.append("        }")
        lines.append("")

    # 2) Forks then joins
    for node_id in sorted(fork_nodes):
        lines.append(f"        fork {fork_nodes[node_id]};")
        lines.append("")

    for node_id in sorted(join_nodes):
        lines.append(f"        join {join_nodes[node_id]};")
        lines.append("")

    # 3) Decisions after action/fork/join declarations
    decision_ids = [
        node_id for node_id in sorted(nodes_by_id)
        if nodes_by_id[node_id]["kind"] == "decision"
    ]
    for node_id in decision_ids:
        lines.append(f"        decide {node_var_name[node_id]};")
        lines.append("")

    # Keep unsupported operators as comments for traceability.
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        if node["kind"] == "operator" and node["name"].lower() != "and":
            lines.append(
                f"        // Unsupported operator node kept for traceability: {node_var_name[node_id]} ({node['name']})"
            )
            lines.append("")

    # 4) Successions (event-driven traversal, causal order)
    flow_lines = []
    emitted_first = set()

    def _id_sort_key(value):
        if isinstance(value, int):
            return (0, value)
        return (1, str(value))

    def emit_first(src_name, dst_name):
        stmt = f"        first {src_name} then {dst_name};"
        if stmt not in emitted_first:
            emitted_first.add(stmt)
            flow_lines.append(stmt)

    decisions_by_id = {d["decision_id"]: d for d in decisions}

    adjacency = {nid: [] for nid in nodes_by_id}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    for nid in adjacency:
        adjacency[nid] = sorted(set(adjacency[nid]), key=_id_sort_key)

    # Stable traversal rank from roots for deterministic processing order.
    root_ids = sorted([nid for nid in nodes_by_id if nid not in incoming], key=_id_sort_key)
    if not root_ids and nodes_by_id:
        root_ids = [sorted(nodes_by_id, key=_id_sort_key)[0]]

    rank = {}
    bfs = list(root_ids)
    for rid in root_ids:
        rank[rid] = 0
    i = 0
    while i < len(bfs):
        cur = bfs[i]
        i += 1
        for nxt in adjacency.get(cur, []):
            if nxt not in rank:
                rank[nxt] = rank[cur] + 1
                bfs.append(nxt)
    for nid in nodes_by_id:
        rank.setdefault(nid, 10**9)

    join_expected = {tid: len(set(incoming.get(tid, []))) for tid in join_nodes}
    join_arrivals = {tid: set() for tid in join_nodes}

    activated = set()
    processed = set()
    queue = []

    def enqueue(node_id, prioritize=False):
        if node_id in activated:
            return
        activated.add(node_id)
        if prioritize:
            queue.insert(0, node_id)
        else:
            queue.append(node_id)

    def route_non_decision(src_id, dst_id):
        src_ref = _node_ref(src_id)
        if dst_id in join_nodes:
            emit_first(src_ref, join_nodes[dst_id])
            join_arrivals[dst_id].add(src_id)
            if len(join_arrivals[dst_id]) >= join_expected.get(dst_id, 0):
                emit_first(join_nodes[dst_id], _node_ref(dst_id))
                enqueue(dst_id, prioritize=(nodes_by_id[dst_id]["kind"] == "decision"))
        else:
            emit_first(src_ref, _node_ref(dst_id))
            enqueue(dst_id, prioritize=(nodes_by_id[dst_id]["kind"] == "decision"))

    def activate_decision_target(src_decision_id, target_id):
        if target_id in join_nodes:
            join_arrivals[target_id].add(src_decision_id)
            if len(join_arrivals[target_id]) >= join_expected.get(target_id, 0):
                emit_first(join_nodes[target_id], _node_ref(target_id))
                enqueue(target_id, prioritize=(nodes_by_id[target_id]["kind"] == "decision"))
        else:
            enqueue(target_id, prioritize=(nodes_by_id[target_id]["kind"] == "decision"))

    for rid in root_ids:
        emit_first("start", _entry_ref(rid))
        enqueue(rid)

    while queue:
        current = queue.pop(0)
        if current in processed:
            continue

        node = nodes_by_id[current]

        if node["kind"] == "decision" and current in decisions_by_id:
            d = decisions_by_id[current]
            decision_ref = _node_ref(current)
            else_target = d["else_target"]
            success_targets = [x for x in d.get("success_targets", []) if x in nodes_by_id]
            tested_action_id = d.get("tested_action_id")

            if not success_targets:
                # Degenerate decision: still emit a conditional block so the SysML keeps an if/else form.
                # The tested action already leads into the decision node; here we only attach the conditional exit.
                if tested_action_id in node_var_name:
                    tested_action_name = node_var_name[tested_action_id]
                else:
                    tested_action_name = sanitize_name(d.get("tested_action_name", "condition"))

                flow_lines.append(f"        first {decision_ref};")
                flow_lines.append(
                    f"            if {tested_action_name}.success then {_entry_ref(else_target)};"
                )
                flow_lines.append("            else done;")
                flow_lines.append("")
                activate_decision_target(current, else_target)
            else:
                if tested_action_id in node_var_name:
                    tested_action_name = node_var_name[tested_action_id]
                else:
                    tested_action_name = sanitize_name(d.get("tested_action_name", "condition"))

                flow_lines.append(f"        first {decision_ref};")
                for succ in success_targets:
                    flow_lines.append(
                        f"            if {tested_action_name}.success then {_entry_ref(succ)};"
                    )
                    activate_decision_target(current, succ)
                flow_lines.append(f"            else {_entry_ref(else_target)};")
                flow_lines.append("")
                activate_decision_target(current, else_target)

            processed.add(current)
            continue

        for nxt in adjacency.get(current, []):
            if current in decisions_by_id:
                continue
            route_non_decision(current, nxt)

        processed.add(current)

    # If disconnected components remain, process them deterministically.
    remaining = [nid for nid in sorted(nodes_by_id, key=_id_sort_key) if nid not in activated]
    for nid in remaining:
        emit_first("start", _entry_ref(nid))

    # End transitions from terminal actions only.
    action_leaf_ids = []
    for nid, node in nodes_by_id.items():
        if node["kind"] != "action":
            continue
        if not outgoing.get(nid):
            action_leaf_ids.append(nid)

    for lid in sorted(action_leaf_ids, key=_id_sort_key):
        emit_first(_node_ref(lid), "done")

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
    output_csv: str = "sopbench_sysml.csv",
    splits=("train", "test"),
    limit_per_split=None,
    progress_every: int = 1000,
):
    # Build one CSV with all transformed SOPBench workflows.
    ds = load_dataset("Zekunli/SOPBench")
    print("Available splits:", ds.keys())

    output_path = output_csv
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)

    headers = ["data", "user_prompt", "sysml_code"]
    ok = 0
    parse_fail = 0
    transform_fail = 0

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for split in splits:
            if split not in ds:
                print(f"Skipping missing split: {split}")
                continue

            data = ds[split]
            total = len(data) if limit_per_split is None else min(limit_per_split, len(data))
            print(f"Processing split '{split}' with {total} samples")

            for i in range(total):
                if progress_every > 0 and (i + 1) % progress_every == 0:
                    print(
                        f"split={split} processed={i + 1}/{total} "
                        f"OK={ok} parse_fail={parse_fail} transform_fail={transform_fail}"
                    )

                sample = data[i]

                try:
                    graph = extract_sopbench_raw(sample)
                except Exception:
                    parse_fail += 1
                    continue

                try:
                    transformed = transform_or_nodes_to_fallback(graph)
                    pkg_name = sanitize_name(str(sample.get("domain") or "SOPBenchPackage"))
                    action_def_name = sanitize_name(str(sample.get("user_goal") or "Workflow"))
                    sysml = build_sysml_from_transformed(
                        transformed,
                        package_name=pkg_name,
                        action_def_name=action_def_name,
                        agent_part_name="WorkflowAgent",
                        agent_instance_name="workflowAgent",
                    )
                except Exception:
                    transform_fail += 1
                    continue

                user_prompt = sample.get("user_prompt") or extract_graph_description(sample)

                writer.writerow(
                    {
                        "data": "sop-bench",
                        "user_prompt": user_prompt,
                        "sysml_code": sysml,
                    }
                )
                ok += 1

    print(f"Output CSV: {output_path}")
    print(f"OK: {ok}")
    print(f"parse_fail: {parse_fail}")
    print(f"transform_fail: {transform_fail}")

if __name__ == "__main__":
    # report = metrics_for_sopbench(split="train", Limit=None, progress_every=1000)
    # print_sopbench_metrics(report)
    run_sopbench_transformation_to_csv(
        output_csv="sopbench_sysml.csv",
        splits=("train", "test"),
        limit_per_split=None,
        progress_every=1000,
    )
