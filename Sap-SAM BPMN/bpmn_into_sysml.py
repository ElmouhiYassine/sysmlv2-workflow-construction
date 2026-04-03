from bpmn_parser import extract_bpmn, build_controlflow_per_pool
from collections import Counter, defaultdict,deque
import pandas as pd
import json
from typing import Dict, List, Set, Tuple, Optional
import re


def indent(level: int, spaces: int = 4) -> str:
    return " " * (level * spaces)

def esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')
def to_action_name(name: str, fallback: str = "UnnamedAction"):
    name = " ".join(str(name).split())
    if not name:
        return fallback

    parts = name.split()
    return "".join(p.capitalize() for p in parts)

def define_actions(bpmn):
    actions = {}

    all_nodes = {}
    all_nodes.update(bpmn.get("tasks", {}))
    all_nodes.update(bpmn.get("nodes", {}))

    action_number = 1 #unnamed ones
    for t in all_nodes.values():
        task_id = t.get("id")
        pool_id = t.get("pool")
        lane_id = t.get("lane")

        task_type = t.get("task_type") or t.get("node_type") or t.get("type")

        if task_type not in {"Task", "Function", "Subprocess", "CollapsedSubprocess", 'StartNoneEvent', 'EndNoneEvent'}:
            continue


        name = " ".join(str(t.get("name", "")).split())
        name = " ".join(str(name).split())

        # remove punctuation like , . ; : ( ) etc
        clean_name = re.sub(r"[^\w\s]", "", name)

        action_name = to_action_name(clean_name,fallback=task_type)
        doc = " ".join(str(t.get("documentation", "")).split())


        # pool_name = to_action_name(t.get("pool_name", ""), fallback="Pool")
        lane_name = to_action_name(t.get("lane_name", ""), fallback="UnspecifiedLane")
        agent_instance_name = f"{lane_name[:1].lower()}{lane_name[1:]}"

        if doc:
            description = doc
        elif name:
            description = name
        else:
            action_number += 1
            description = ""

        d = (
            f'{indent(2)}action {action_name} {{\n'
            f'{indent(3)}description = "{esc(description)}";\n'
            f'{indent(3)}agent = {agent_instance_name};\n'
            f'{indent(2)}}}\n'
        )

        if pool_id not in actions:
            actions[pool_id] = {}

        actions[pool_id][task_id] = {
            "task_id": task_id,
            "pool_id": pool_id,
            "lane_id": lane_id,
            "name": name,
            "action_name": action_name,
            "agent_instance_name": agent_instance_name,
            "description": description,
            "code": d,
        }

    return actions
def create_fork_and_join(pool_cf, bpmn):
    result = {}

    # node types that map to an action or decision — no joinNode needed
    action_types = {"Task", "Function", "Subprocess", "CollapsedSubprocess",
                    "StartNoneEvent", "EndNoneEvent"}
    decision_types = {"Exclusive_Databased_Gateway"}
    no_join_types = action_types | decision_types

    shape_index = bpmn.get("shape_index", {})

    for pool_id, pool_data in pool_cf.items():
        result[pool_id] = {
            "blocks": [],
            "by_split_bpmn_id": {},
            "by_join_bpmn_id": {},
        }

        parallel_blocks = pool_data.get("parallel_blocks", [])

        for i, blk in enumerate(parallel_blocks, start=1):
            fork_id = f"forkNode{i}"
            block_id = f"parallelBlock{i}"

            join_bpmn_id = blk.get("join")
            join_bpmn_type = shape_index.get(join_bpmn_id, {}).get("type", "")

            # Only create a synthetic joinNode if the convergence point is a
            # ParallelGateway — otherwise the existing action/decision represents it
            if join_bpmn_type in no_join_types:
                join_id = None      # will resolve naturally via pool_defs or decision_map
                join_code = None
            else:
                join_id = f"joinNode{i}"
                join_code = f"{indent(2)}join {join_id};\n"

            item = {
                "block_id": block_id,

                "fork_id": fork_id,
                "join_id": join_id,         # may be None

                "split_bpmn_id": blk.get("split"),
                "join_bpmn_id": join_bpmn_id,

                "branch_entries": blk.get("branch_entries", []),
                "branch_exits": blk.get("branch_exits", []),
                "branches": blk.get("branches", []),

                "fork_code": f"{indent(2)}fork {fork_id};\n",
                "join_code": join_code,     # may be None
            }

            result[pool_id]["blocks"].append(item)

            split_bpmn_id = item["split_bpmn_id"]

            if split_bpmn_id:
                result[pool_id]["by_split_bpmn_id"][split_bpmn_id] = item

            # Only register in by_join_bpmn_id if we actually made a joinNode —
            # otherwise resolve_id() will correctly fall through to pool_defs/decision_map
            if join_id and join_bpmn_id:
                result[pool_id]["by_join_bpmn_id"][join_bpmn_id] = item

    return result

def renumber_by_position(pool_id, fork_join_by_pool, decisions_by_pool, pool_cf, bpmn):
    """Renumber forkNode/joinNode/decisionNode by their topological position."""

    # Build position index from sequence edges
    position = {}
    pos_counter = 0
    for eid in pool_cf[pool_id].get("sequence_edges", []):
        e = bpmn["edges"].get(eid)
        if not e:
            continue
        for raw in (e.get("src"), e.get("tgt")):
            if raw and raw not in position:
                src_pool = bpmn["shape_index"].get(raw, {}).get("pool")
                if src_pool == pool_id:  # local nodes only
                    position[raw] = pos_counter
                    pos_counter += 1

    # Renumber fork/join blocks by split bpmn id position
    pool_fj = fork_join_by_pool.get(pool_id, {})
    blocks = pool_fj.get("blocks", [])
    blocks_sorted = sorted(
        blocks,
        key=lambda blk: position.get(blk["split_bpmn_id"], 9999)
    )
    new_by_split = {}
    new_by_join = {}
    for i, blk in enumerate(blocks_sorted, start=1):
        old_fork = blk["fork_id"]
        old_join = blk["join_id"]

        blk["fork_id"] = f"forkNode{i}"
        blk["fork_code"] = f"{indent(2)}fork forkNode{i};\n"

        if old_join:   # only if a joinNode was actually created
            blk["join_id"] = f"joinNode{i}"
            blk["join_code"] = f"{indent(2)}join joinNode{i};\n"

        if blk["split_bpmn_id"]:
            new_by_split[blk["split_bpmn_id"]] = blk
        if blk["join_bpmn_id"] and blk["join_id"]:
            new_by_join[blk["join_bpmn_id"]] = blk

    pool_fj["blocks"] = blocks_sorted
    pool_fj["by_split_bpmn_id"] = new_by_split
    pool_fj["by_join_bpmn_id"] = new_by_join

    # Renumber decision nodes by their bpmn id position
    pool_dec = decisions_by_pool.get(pool_id, {})
    nodes = pool_dec.get("nodes", [])
    nodes_sorted = sorted(
        nodes,
        key=lambda d: position.get(d["bpmn_id"], 9999)
    )
    new_by_bpmn = {}
    for i, d in enumerate(nodes_sorted, start=1):
        d["decision_id"] = f"decisionNode{i}"
        d["code"] = f"{indent(2)}decide decisionNode{i};\n"
        new_by_bpmn[d["bpmn_id"]] = d

    pool_dec["nodes"] = nodes_sorted
    pool_dec["by_bpmn_id"] = new_by_bpmn

def resolve_id(node_id, home_pool_id, pool_defs, split_map, join_map, decision_map, definitions_by_pool, all_pools):
    if node_id in pool_defs:
        return pool_defs[node_id]["action_name"]
    if node_id in split_map:
        return split_map[node_id]["fork_id"]
    if node_id in join_map:
        return join_map[node_id]["join_id"]
    if node_id in decision_map:
        return decision_map[node_id]["decision_id"]

    for other_pool_id, other_defs in definitions_by_pool.items():
        if other_pool_id == home_pool_id:
            continue
        if node_id in other_defs:
            action_name = other_defs[node_id]["action_name"]
            foreign_pool = all_pools.get(other_pool_id, {})
            lane_id = other_defs[node_id]["lane_id"]
            lane = bpmn.get("lanes", {}).get(lane_id, {})
            lane_name = to_action_name(" ".join(str(lane['name']).split()),
                                       fallback="UnspecifiedLane")
            package_name = to_action_name(
                " ".join(str(foreign_pool.get("name", other_pool_id)).split()),
                fallback="Package"
            )
            return f"{package_name}::{package_name}::{action_name}"

    return None


from collections import deque
from graphlib import TopologicalSorter

def write_sequence_edges(
    bpmn,
    pool_cf,
    pool_id,
    definitions_by_pool,
    fork_join_by_pool,
    pool_decisions,
    all_pools,
    debug=False,
):
    pool_defs = definitions_by_pool.get(pool_id, {})
    pool_fj = fork_join_by_pool.get(pool_id, {})

    split_map = pool_fj.get("by_split_bpmn_id", {})
    join_map = pool_fj.get("by_join_bpmn_id", {})
    decision_map = pool_decisions.get("by_bpmn_id", {})


    def resolve_idd(node_id, home_pool_id=pool_id):
        return resolve_id(
            node_id,
            home_pool_id,
            pool_defs,
            split_map,
            join_map,
            decision_map,
            definitions_by_pool,
            all_pools,
        )

    shape_index = bpmn.get("shape_index", {})

    # --------------------------------------------------
    # 1. Build resolved local graph
    # --------------------------------------------------
    successors = {}
    predecessors = {}
    bpmn_id_of = {}
    entry_targets = []

    for raw_bpmn_id, d_item in decision_map.items():
        bpmn_id_of[d_item["decision_id"]] = raw_bpmn_id

    for eid in pool_cf[pool_id].get("sequence_edges", []):
        e = bpmn["edges"].get(eid)
        if not e:
            continue

        raw_src = e.get("src")
        raw_tgt = e.get("tgt")

        src_info = shape_index.get(raw_src, {})
        if src_info.get("type") == "StartNoneEvent":
            resolved_src = resolve_idd(raw_src, pool_id)
            if resolved_src and "." not in resolved_src:
                entry_targets.append(resolved_src)

        src = resolve_idd(raw_src, pool_id)
        tgt = resolve_idd(raw_tgt, pool_id)


        if not src or not tgt:
            continue

        # Only keep LOCAL -> LOCAL for intra-pool writing
        # if "." in src or "." in tgt:
        #     dbg("  SKIP: foreign endpoint")
        #     continue

        bpmn_id_of.setdefault(src, raw_src)
        bpmn_id_of.setdefault(tgt, raw_tgt)

        successors.setdefault(src, []).append(tgt)
        predecessors.setdefault(tgt, set()).add(src)
        predecessors.setdefault(src, set())

    seen = set()
    entry_targets = [x for x in entry_targets if not (x in seen or seen.add(x))]

    # --------------------------------------------------
    # 2. Reachability from real local starts
    # --------------------------------------------------
    reachable_from_entry = set()
    q = deque(entry_targets)

    while q:
        node = q.popleft()
        if node in reachable_from_entry:
            continue
        reachable_from_entry.add(node)

        for tgt in successors.get(node, []):
            q.append(tgt)

    # --------------------------------------------------
    # 3. Position from traversal starting at local starts
    # --------------------------------------------------
    position = {}
    q = deque(entry_targets)
    seen_pos = set()
    pos_counter = 0

    while q:
        node = q.popleft()
        if node in seen_pos:
            continue
        seen_pos.add(node)

        if node not in position:
            position[node] = pos_counter
            pos_counter += 1

        for tgt in successors.get(node, []):
            if tgt not in seen_pos:
                q.append(tgt)

    fallback_base = pos_counter + 1000
    for node in predecessors:
        if node not in position:
            position[node] = fallback_base
            fallback_base += 1

    for node in list(position.keys()):
        if node.startswith("decisionNode"):
            preds = predecessors.get(node, set())
            pred_positions = [position[p] for p in preds if p in position]
            if pred_positions:
                position[node] = max(position[node], max(pred_positions) + 1)
    # --------------------------------------------------
    # 4. Detect local nodes returned from foreign pools
    # --------------------------------------------------
    returned_from_foreign = set()

    for eid, e in bpmn["edges"].items():
        if e.get("type") != "SequenceFlow":
            continue

        raw_src = e.get("src")
        raw_tgt = e.get("tgt")

        src_pool = shape_index.get(raw_src, {}).get("pool")
        tgt_pool = shape_index.get(raw_tgt, {}).get("pool")

        if src_pool != pool_id and tgt_pool == pool_id:
            resolved_tgt = resolve_idd(raw_tgt, pool_id)
            if resolved_tgt and "." not in resolved_tgt:
                returned_from_foreign.add(resolved_tgt)

    true_starts = {
        resolve_idd(sid, pool_id)
        for sid, info in shape_index.items()
        if info.get("type") == "StartNoneEvent" and info.get("pool") == pool_id
    }

    foreign_entry_nodes = {
        n for n in predecessors
        if "." not in n
           and not predecessors[n]
           and n not in true_starts
           and n in returned_from_foreign
    }
    # --------------------------------------------------
    # 5. Local back-edges (loops)
    # --------------------------------------------------
    back_edges = set()
    visited = set()
    stack = set()

    def dfs(node):
        visited.add(node)
        stack.add(node)
        for tgt in successors.get(node, []):
            if tgt not in visited:
                dfs(tgt)
            elif tgt in stack:
                back_edges.add((node, tgt))
        stack.discard(node)

    for node in predecessors:
        if node not in visited:
            dfs(node)

    for src, tgt in back_edges:
        predecessors.get(tgt, set()).discard(src)

    # --------------------------------------------------
    # 6. Emit
    # --------------------------------------------------
    ts = TopologicalSorter(predecessors)
    ts.prepare()

    lines = []
    decision_done = set()
    pending_ready = set()

    def edge_label(e):
        name = " ".join(str(e.get("name", "")).split())
        if name:
            return name

        props = e.get("props", {}) or {}
        cond = " ".join(str(props.get("conditionexpression", "")).split())
        if cond and cond != "None":
            return cond

        return ""

    def node_kind(resolved_id):
        if resolved_id.startswith("forkNode"):
            return "fork"
        if resolved_id.startswith("joinNode"):
            return "join"
        if resolved_id.startswith("decisionNode"):
            return "decision"
        return "action"

    def needs_condition_execution(decision_resolved_id):
        if not decision_resolved_id.startswith("decisionNode"):
            return False
        raw_id = bpmn_id_of.get(decision_resolved_id)
        if not raw_id:
            return False
        info = decision_map.get(raw_id, {})
        return bool(info.get("conditions"))

    step = 0

    while ts.is_active() or pending_ready:
        newly_ready = list(ts.get_ready())
        for n in newly_ready:
            pending_ready.add(n)

        main_ready = [
            n for n in pending_ready
            if n in reachable_from_entry and n not in returned_from_foreign
        ]

        deferred_ready = [
            n for n in pending_ready
            if n not in main_ready
        ]

        if main_ready:
            ready = sorted(main_ready, key=lambda n: (position.get(n, 10**9), n))
        else:
            ready = sorted(deferred_ready, key=lambda n: (position.get(n, 10**9), n))


        step += 1

        if not ready:
            break

        # Process only one node at a time to preserve ordering
        src = ready[0]
        pending_ready.remove(src)

        kind = node_kind(src)
        targets = successors.get(src, [])

        if kind == "decision" and src not in decision_done:
            decision_done.add(src)

            raw_id = bpmn_id_of.get(src)
            decision_blocks = pool_cf[pool_id].get("decision_blocks", {})
            if isinstance(decision_blocks, dict):
                block = decision_blocks.get(raw_id)
            else:
                block = next((x for x in decision_blocks if x.get("decision") == raw_id), None)

            decision_info = decision_map.get(raw_id, {})
            decision_conditions = decision_info.get("conditions", [])
            condition_ids = [c.get("condition_id") for c in decision_conditions if c.get("condition_id")]
            condition_fork_id = decision_info.get("condition_fork_id")

            if condition_ids:
                incoming = sorted(predecessors.get(src, set()), key=lambda n: (position.get(n, 10**9), n))

                if len(condition_ids) == 1:
                    c_id = condition_ids[0]
                    for pred in incoming:
                        lines.append(f"{indent(2)}first {pred} then {c_id};\n")
                    lines.append(f"{indent(2)}first {c_id} then {src};\n")
                else:
                    if condition_fork_id:
                        for pred in incoming:
                            lines.append(f"{indent(2)}first {pred} then {condition_fork_id};\n")
                        for c_id in condition_ids:
                            lines.append(f"{indent(2)}first {condition_fork_id} then {c_id};\n")
                    else:
                        for pred in incoming:
                            for c_id in condition_ids:
                                lines.append(f"{indent(2)}first {pred} then {c_id};\n")

                    for c_id in condition_ids:
                        lines.append(f"{indent(2)}first {c_id} then {src};\n")

            lines.append(f"{indent(2)}first {src};\n")

            emitted_any = False
            branch_conds = decision_info.get("branch_conditions", [])
            binary_if_else = decision_info.get("binary_if_else", False)
            binary_condition_id = decision_info.get("binary_condition_id")

            if block:
                branch_targets = []
                for b in block.get("branches", []):
                    resolved_tgt = resolve_idd(b.get("raw_target"), pool_id)
                    if resolved_tgt:
                        branch_targets.append(resolved_tgt)

                default_block = block.get("default")
                if default_block:
                    resolved_default = resolve_idd(default_block.get("raw_target"), pool_id)
                    if resolved_default:
                        branch_targets.append(resolved_default)

                if binary_if_else and binary_condition_id and len(branch_targets) >= 2:
                    lines.append(f"{indent(3)}if {binary_condition_id}.result then {branch_targets[0]};\n")
                    lines.append(f"{indent(3)}else {branch_targets[1]};\n")
                    emitted_any = True
                else:
                    for i, b in enumerate(block.get("branches", [])):
                        resolved_tgt = resolve_idd(b.get("raw_target"), pool_id)
                        if not resolved_tgt:
                            continue

                        #if i == 0 else "else if"
                        keyword = "if"
                        branch_cond = branch_conds[i] if i < len(branch_conds) else None
                        cond_ref = branch_cond["condition_id"] if isinstance(branch_cond, dict) else None
                        cond_negate = branch_cond.get("negate", False) if isinstance(branch_cond, dict) else False

                        if cond_ref:
                            cond_expr = f"not {cond_ref}.result" if cond_negate else f"{cond_ref}.result"
                            lines.append(f"{indent(3)}{keyword} {cond_expr} then {resolved_tgt};\n")
                        else:
                            if i == len(block.get("branches", [])) - 1 and not block.get("default"):
                                lines.append(f"{indent(3)}else {resolved_tgt};\n")
                            else:
                                cond = b.get("cond", '"unspecifiedCondition"')
                                lines.append(f"{indent(3)}{keyword} {cond} then {resolved_tgt};\n")
                        emitted_any = True

                    if default_block:
                        resolved_tgt = resolve_idd(default_block.get("raw_target"), pool_id)
                        if resolved_tgt:
                            lines.append(f"{indent(3)}else {resolved_tgt};\n")
                            emitted_any = True

            if not emitted_any:
                raw_id = bpmn_id_of.get(src)

                fallback_edges = []
                if raw_id:
                    for eid in pool_cf[pool_id].get("sequence_edges", []):
                        e = bpmn["edges"].get(eid)
                        if not e:
                            continue
                        if e.get("src") != raw_id:
                            continue

                        resolved_tgt = resolve_idd(e.get("tgt"), pool_id)
                        if not resolved_tgt:
                            continue

                        cond = edge_label(e) or "unspecifiedCondition"
                        fallback_edges.append((cond, resolved_tgt))

                for i, (cond, tgt) in enumerate(fallback_edges):
                    if binary_if_else and binary_condition_id and len(fallback_edges) >= 2:
                        lines.append(f"{indent(3)}if {binary_condition_id}.result then {fallback_edges[0][1]};\n")
                        lines.append(f"{indent(3)}else {fallback_edges[1][1]};\n")
                        emitted_any = True
                        break

                    #if i == 0 else "else if"
                    keyword = "if"
                    branch_cond = branch_conds[i] if i < len(branch_conds) else None
                    cond_ref = branch_cond["condition_id"] if isinstance(branch_cond, dict) else None
                    cond_negate = branch_cond.get("negate", False) if isinstance(branch_cond, dict) else False

                    if cond_ref:
                        cond_expr = f"not {cond_ref}.result" if cond_negate else f"{cond_ref}.result"
                        lines.append(f"{indent(3)}{keyword} {cond_expr} then {tgt};\n")
                    else:
                        if i == len(fallback_edges) - 1:
                            lines.append(f"{indent(3)}else {tgt};\n")
                        else:
                            lines.append(f'{indent(3)}{keyword} "{cond}" then {tgt};\n')

                    emitted_any = True

        else:
            for tgt in targets:
                if (src, tgt) not in back_edges:
                    if needs_condition_execution(tgt):
                        continue
                    lines.append(f"{indent(2)}first {src} then {tgt};\n")

        ts.done(src)

    for src, tgt in back_edges:
        if node_kind(src) == "decision" or node_kind(tgt) == "decision":
            continue
        lines.append(f"{indent(2)}first {src} then {tgt};\n")

    return lines

def find_start_end_actions(bpmn, pool_cf, pool_id, definitions_by_pool, all_pools):
    start_actions = set()
    end_actions = set()

    pool_defs = definitions_by_pool.get(pool_id, {})
    shape_index = bpmn.get("shape_index", {})

    # build set of local pool node ids for cross-pool check
    local_node_ids = set(pool_defs.keys())

    for eid in pool_cf[pool_id].get("sequence_edges", []):
        e = bpmn["edges"].get(eid)
        if not e:
            continue

        src, tgt = e.get("src"), e.get("tgt")
        src_info = shape_index.get(src, {})
        tgt_info = shape_index.get(tgt, {})

        src_pool = src_info.get("pool")
        tgt_pool = tgt_info.get("pool")

        # start: src is a local StartNoneEvent
        if src_info.get("type") == "StartNoneEvent" and src in pool_defs:
            start_actions.add(pool_defs[src]["action_name"])

        # end: tgt is a local EndNoneEvent AND src is also local
        # (cross-pool edge landing on a foreign EndNoneEvent doesn't end this pool)
        if (tgt_info.get("type") == "EndNoneEvent"
                and tgt in pool_defs
                and src_pool == pool_id):
            end_actions.add(pool_defs[tgt]["action_name"])

    return start_actions, end_actions

def define_agents(bpmn):
    agents = {}

    all_pools = {}
    all_pools.update(bpmn.get("pools", {}))
    all_pools.update(bpmn.get("collapsed_pools", {}))

    for pool_id, pool in all_pools.items():
        pool_name_raw = " ".join(str(pool.get("name", pool_id)).split())
        pool_name = to_action_name(pool_name_raw, fallback="Pool")

        for lane_id in pool.get("lanes", []):
            lane = bpmn.get("lanes", {}).get(lane_id, {})
            lane_name_raw = " ".join(str(lane.get("name", lane_id)).split())
            lane_name = to_action_name(lane_name_raw, fallback="UnspecifiedLane")

            part_def_name = f"{lane_name}"
            instance_name = f"{lane_name[:1].lower()}{lane_name[1:]}"

            part_def_code = (
                f"{indent(1)}part def {part_def_name} specializes SolutionAgent {{\n"
                f"{indent(1)}}}\n"
            )

            instance_code = (
                f"{indent(1)}{instance_name} : {part_def_name};\n"
            )

            if pool_id not in agents:
                agents[pool_id] = {}

            agents[pool_id][lane_id] = {
                "pool_id": pool_id,
                "lane_id": lane_id,
                "pool_name": pool_name_raw,
                "lane_name": lane_name_raw,
                "part_def_name": part_def_name,
                "instance_name": instance_name,
                "part_def_code": part_def_code,
                "instance_code": instance_code,
            }

    return agents

def define_decisions(pool_cf, bpmn):
    decisions_by_pool = {}

    for pool_id in pool_cf:
        decisions_by_pool[pool_id] = {
            "nodes": [],
            "by_bpmn_id": {}
        }

        i = 1
        condition_counter = 1
        condition_fork_counter = 1

        def normalize_condition_text(raw_cond):
            cond = " ".join(str(raw_cond or "").split())
            if len(cond) >= 2 and cond[0] == '"' and cond[-1] == '"':
                cond = cond[1:-1]
            return cond or "unspecifiedCondition"

        def is_unspecified_condition(cond_text):
            return normalize_condition_text(cond_text).lower() in {
                "unspecifiedcondition",
                "unspecified condition",
                "unspecified",
            }

        def format_condition_description(true_cond, false_cond=None):
            if false_cond:
                return f"return True if {true_cond}, return False if {false_cond}."
            return f"return True if {true_cond}, return False otherwise."

        def edge_label(e):
            name = " ".join(str(e.get("name", "")).split())
            if name:
                return name
            props = e.get("props", {}) or {}
            cond = " ".join(str(props.get("conditionexpression", "")).split())
            if cond and cond != "None":
                return cond
            return ""

        for nid, n in bpmn.get("nodes", {}).items():
            if n.get("pool") != pool_id:
                continue

            if n.get("gateway_type") != "Exclusive_Databased_Gateway":
                continue

            decision_id = f"decisionNode{i}"
            code = f"{indent(2)}decide {decision_id};\n"

            lane_name = to_action_name(n.get("lane_name", ""), fallback="UnspecifiedLane")
            agent_instance_name = f"{lane_name[:1].lower()}{lane_name[1:]}"

            decision_blocks = pool_cf[pool_id].get("decision_blocks", {})
            if isinstance(decision_blocks, dict):
                block = decision_blocks.get(nid)
            else:
                block = next((x for x in decision_blocks if x.get("decision") == nid), None)

            branch_specs = []
            if block:
                for b in block.get("branches", []):
                    branch_specs.append(
                        {
                            "cond": b.get("cond"),
                            "raw_target": b.get("raw_target"),
                        }
                    )
                default_block = block.get("default") if isinstance(block, dict) else None
                if default_block and default_block.get("raw_target"):
                    branch_specs.append(
                        {
                            "cond": "unspecifiedCondition",
                            "raw_target": default_block.get("raw_target"),
                        }
                    )
            else:
                for eid in pool_cf[pool_id].get("sequence_edges", []):
                    e = bpmn.get("edges", {}).get(eid)
                    if not e or e.get("src") != nid:
                        continue
                    branch_specs.append(
                        {
                            "cond": edge_label(e) or "unspecifiedCondition",
                            "raw_target": e.get("tgt"),
                        }
                    )

            conditions = []
            branch_conditions = []
            binary_if_else = len(branch_specs) == 2
            binary_condition_id = None

            normalized_conds = [normalize_condition_text(spec.get("cond")) for spec in branch_specs]
            branch_count = len(branch_specs)

            # Build executable conditions:
            # - 2 branches  => 1 condition (if/else)
            # - N branches  => N-1 conditions (if/else if/.../else)
            if branch_count >= 2:
                condition_count = 1 if branch_count == 2 else branch_count - 1

                for idx in range(condition_count):
                    this_cond = normalized_conds[idx]
                    next_cond = normalized_conds[idx + 1] if (idx + 1) < branch_count else None

                    true_text = this_cond if not is_unspecified_condition(this_cond) else f"branch {idx + 1}"
                    false_text = None
                    if next_cond and not is_unspecified_condition(next_cond):
                        false_text = next_cond

                    cond_desc = format_condition_description(true_text, false_text)

                    condition_id = f"condition{condition_counter}"
                    condition_code = (
                        f"{indent(2)}action {condition_id} : decisionStep {{\n"
                        f'{indent(3)}:>> description = "{esc(cond_desc)}";\n'
                        f"{indent(3)}:>> agent = {agent_instance_name};\n"
                        # f"{indent(3)}out result : Boolean;\n"
                        f"{indent(2)}}}\n"
                    )

                    conditions.append(
                        {
                            "condition_id": condition_id,
                            "description": cond_desc,
                            "raw_target": branch_specs[idx].get("raw_target"),
                            "code": condition_code,
                        }
                    )
                    condition_counter += 1

                if branch_count == 2:
                    first_condition_id = conditions[0]["condition_id"]
                    branch_conditions.append({"condition_id": first_condition_id, "negate": False})
                    branch_conditions.append({"condition_id": first_condition_id, "negate": True})
                    binary_condition_id = first_condition_id
                else:
                    for idx in range(branch_count):
                        if idx < len(conditions):
                            branch_conditions.append(
                                {
                                    "condition_id": conditions[idx]["condition_id"],
                                    "negate": False,
                                }
                            )
                        else:
                            branch_conditions.append(None)
            elif branch_count == 1:
                branch_conditions.append(None)

            condition_fork_id = None
            condition_fork_code = None
            if len(conditions) > 1:
                condition_fork_id = f"forkCondition{condition_fork_counter}"
                condition_fork_code = f"{indent(2)}fork {condition_fork_id};\n"
                condition_fork_counter += 1

            item = {
                "bpmn_id": nid,
                "decision_id": decision_id,
                "code": code,
                "conditions": conditions,
                "branch_conditions": branch_conditions,
                "binary_if_else": binary_if_else,
                "binary_condition_id": binary_condition_id,
                "condition_fork_id": condition_fork_id,
                "condition_fork_code": condition_fork_code,
            }

            decisions_by_pool[pool_id]["nodes"].append(item)
            decisions_by_pool[pool_id]["by_bpmn_id"][nid] = item

            i += 1

    return decisions_by_pool

def sysml_code(bpmn):
    lines = []

    definitions_by_pool = define_actions(bpmn)

    all_pools = {}
    all_pools.update(bpmn.get("pools", {}))
    all_pools.update(bpmn.get("collapsed_pools", {}))
    agents_by_pool = define_agents(bpmn)
    pool_cf = build_controlflow_per_pool(bpmn)
    fork_join_by_pool = create_fork_and_join(pool_cf, bpmn)
    decisions_by_pool = define_decisions(pool_cf, bpmn)

    for pool_id, pool in all_pools.items():
        renumber_by_position(pool_id, fork_join_by_pool, decisions_by_pool, pool_cf, bpmn)
        pool_name = " ".join(str(pool.get("name", pool_id)).split())
        package_name = to_action_name(pool_name, fallback="PackageName")
        process_name = to_action_name(pool_name, fallback="ProcessName")
        lines.append("private import ScalarValues::*;\n")
        lines.append("private import MOSAICO::*;\n\n")

        lines.append(f"package {package_name} {{\n\n")
        lines.append(f"{indent(1)}action def {process_name} {{\n\n")

        pool_agents = agents_by_pool.get(pool_id, {})
        for info in pool_agents.values():
            lines.append(info["part_def_code"])
            lines.append("\n")

        for info in pool_agents.values():
            lines.append(info["instance_code"])
        if pool_agents:
            lines.append("\n")
        pool_defs = definitions_by_pool.get(pool_id, {})

        # action definitions
        for info in pool_defs.values():
            lines.append(info["code"])
            lines.append("\n")

        #fork / join definitions
        pool_fj = fork_join_by_pool.get(pool_id, {})
        for blk in pool_fj.get("blocks", []):
            lines.append(f"{indent(2)}fork {blk['fork_id']};\n")
            if blk["join_code"]:  # only emit if a joinNode was created
                lines.append(blk["join_code"])
        lines.append("\n")

        pool_decisions = decisions_by_pool.get(pool_id, {})

        # synthetic fork nodes to execute multiple condition actions before a decision
        for d in pool_decisions.get("nodes", []):
            if d.get("condition_fork_code"):
                lines.append(d["condition_fork_code"])
        if any(d.get("condition_fork_code") for d in pool_decisions.get("nodes", [])):
            lines.append("\n")

        # condition actions backing decision branches
        for d in pool_decisions.get("nodes", []):
            for c in d.get("conditions", []):
                lines.append(c["code"])
                lines.append("\n")

        for d in pool_decisions.get("nodes", []):
            lines.append(d["code"])
        if pool_decisions.get("nodes"):
            lines.append("\n")

        start_actions, end_actions = find_start_end_actions(
            bpmn,
            pool_cf,
            pool_id,
            definitions_by_pool,
            all_pools
        )

        # Dans sysml_code, pour déterminer si un pool a un StartNoneEvent local :
        has_local_start = any(
            info.get("type") == "StartNoneEvent" and info.get("pool") == pool_id
            for info in bpmn["shape_index"].values()
        )

        for a in start_actions:
            lines.append(f"{indent(2)}first start then {a};\n")
        if not start_actions and has_local_start:
            lines.append(f"{indent(2)}first start;\n")
        # si ni start_actions ni has_local_start → pool invoqué depuis l'extérieur, rien émettre

        seq_lines = write_sequence_edges(
            bpmn=bpmn,
            pool_cf=pool_cf,
            pool_id=pool_id,
            definitions_by_pool=definitions_by_pool,
            fork_join_by_pool=fork_join_by_pool,
            pool_decisions=pool_decisions,
            all_pools=all_pools,
            debug=True,
        )

        lines.extend(seq_lines)


        for a in end_actions:
            lines.append(f"{indent(2)}first {a} then done;\n")
        # if not end_actions:
        #     lines.append(f"{indent(2)}then end;\n")

        lines.append(f"{indent(1)}}}\n")
        lines.append("}\n\n")

    return "".join(lines)

if __name__ == "__main__":
    df = pd.read_csv("D:\\sap_sam_2022\\models\\10000.csv")
    model = json.loads(df.loc[0, "Model JSON"])

    #7278
    #551 200000
    # with open("bpmn_hr_it.json") as f:
    #     model = json.load(f)

    bpmn = extract_bpmn(model)

    print(sysml_code(bpmn))
