from collections import defaultdict, deque
import pandas as pd
import json
from typing import Dict, List, Set, Tuple, Optional


pool_types = {"Pool", "CollapsedPool"}
lane_types = {"Lane"}
edge_types = {
    "SequenceFlow"
}
task_types = {
    "Task",
    "Function",
    "Subprocess",
    "CollapsedSubprocess",
    "StartNoneEvent",
    "EndNoneEvent",

}
gateway_types = {
    "Exclusive_Databased_Gateway",
    "ParallelGateway"
}

ignore_types = {"BPMNDiagram"}  # Root, not a process element

unspecified_pool = "UnspecifiedPool"
unspecified_lane = "UnspecifiedLane"
unnamed_action = "UnnamedAction"
unspecified_gateway = "UnspecifiedGateway"
unspecified_node = "UnspecifiedNode"

# Walk all nested BPMN shapes with their parent
def walk(shape, parent=None):
    yield shape, parent
    for ch in shape.get("childShapes", []) or []:
        yield from walk(ch, shape)

def extract_bpmn(doc, strict=False):
    shapes = {}
    parent_of = {}
    extraction_errors = []

    def log_error(stage, sid=None, exc=None, extra=None):
        msg = {
            "stage": stage,
            "sid": sid,
            "error_type": type(exc).__name__ if exc else None,
            "error": str(exc) if exc else None,
            "extra": extra,
        }
        extraction_errors.append(msg)
        if strict and exc is not None:
            raise exc

    # First pass: index shapes and parent links
    try:
        walked = walk(doc)
    except Exception as e:
        log_error("walk(doc)", exc=e)
        return {
            "pools": {},
            "collapsed_pools": {},
            "lanes": {},
            "tasks": {},
            "nodes": {},
            "edges": {},
            "shape_index": {},
            "errors": extraction_errors,
        }

    for k, item in enumerate(walked):
        sid = None
        try:
            shape, parent = item
        except Exception as e:
            log_error("unpack_walk_item", exc=e, extra={"item_index": k, "item_repr": repr(item)[:300]})
            continue

        try:
            if not isinstance(shape, dict):
                log_error("shape_not_dict", extra={"item_index": k, "shape_type": type(shape).__name__})
                continue

            sid = shape.get("resourceId")
            if not sid:
                continue

            shapes[sid] = shape

            if isinstance(parent, dict) and parent.get("resourceId"):
                parent_of[sid] = parent["resourceId"]
        except Exception as e:
            log_error("pass1_index_shapes", sid=sid, exc=e, extra={"item_index": k})
            continue

    def stencil_id(sid):
        try:
            return (shapes.get(sid, {}).get("stencil") or {}).get("id") or ""
        except Exception as e:
            log_error("stencil_id", sid=sid, exc=e)
            return ""

    def props_of(sid):
        try:
            props = shapes.get(sid, {}).get("properties")
            return props if isinstance(props, dict) else {}
        except Exception as e:
            log_error("props_of", sid=sid, exc=e)
            return {}

    def get_text(props, key):
        try:
            if not isinstance(props, dict):
                return ""
            v = props.get(key)
            if v is None:
                return ""
            return str(v).strip()
        except Exception as e:
            log_error("get_text", exc=e, extra={"key": key, "props_repr": repr(props)[:200]})
            return ""

    def find_ancestor(sid, stencil_set):
        try:
            cur = sid
            seen = set()
            while cur in parent_of and cur not in seen:
                seen.add(cur)
                cur = parent_of[cur]
                if stencil_id(cur) in stencil_set:
                    return cur
            return None
        except Exception as e:
            log_error("find_ancestor", sid=sid, exc=e, extra={"stencil_set": list(stencil_set)})
            return None

    pools, collapsed_pools, lanes = {}, {}, {}
    tasks, nodes, edges = {}, {}, {}

    # Second pass: classify pools, lanes, tasks, nodes and edges

    for sid in list(shapes.keys()):
        try:
            st = stencil_id(sid)
            if st in ignore_types:
                continue

            props = props_of(sid)
            name = get_text(props, "name")
            documentation = get_text(props, "documentation")

            pool_id = find_ancestor(sid, pool_types)
            lane_id = find_ancestor(sid, lane_types)

            if st in pool_types:
                entry = {
                    "id": sid,
                    "type": st,
                    "name": name or unspecified_pool,
                    "lanes": [],
                    "nodes": [],
                    "tasks": [],
                }
                if st == "CollapsedPool":
                    collapsed_pools[sid] = entry
                else:
                    pools[sid] = entry

            elif st in lane_types:
                lanes[sid] = {
                    "id": sid,
                    "type": st,
                    "name": name or unspecified_lane,
                    "pool": pool_id,
                    "nodes": [],
                    "tasks": [],
                }

            elif st in edge_types:
                target_obj = shapes[sid].get("target")
                tgt = target_obj.get("resourceId") if isinstance(target_obj, dict) else None

                edges[sid] = {
                    "id": sid,
                    "type": st,
                    "name": name,
                    "documentation": documentation,
                    "props": props,
                    "src": None,
                    "tgt": tgt,
                }

            elif st in task_types:
                tasks[sid] = {
                    "id": sid,
                    "type": "Task",
                    "task_type": st,
                    "name": name or unnamed_action,
                    "documentation": documentation,
                    "props": props,
                    "pool": pool_id,
                    "lane": lane_id,
                }
            elif st in gateway_types:
                nodes[sid] = {
                    "id": sid,
                    "type": "Gateway",
                    "gateway_type": st,
                    "name": name or unspecified_gateway,
                    "documentation": documentation,
                    "props": props,
                    "pool": pool_id,
                    "lane": lane_id,
                }

            else:
                nodes[sid] = {
                    "id": sid,
                    "type": "Node",
                    "node_type": st,
                    "name": name or unspecified_node,
                    "documentation": documentation,
                    "props": props,
                    "pool": pool_id,
                    "lane": lane_id,
                }

        except Exception as e:
            log_error("pass2_classify_shape", sid=sid, exc=e)
            continue

    # Create a default pool when the diagram has none
    default_pool_id = "__DEFAULT_POOL__"
    try:
        if not pools and not collapsed_pools:
            pools[default_pool_id] = {
                "id": default_pool_id,
                "type": "Pool",
                "name": unspecified_pool,
                "lanes": [],
                "nodes": [],
                "tasks": [],
            }

        all_pools = {}
        all_pools.update(pools)
        all_pools.update(collapsed_pools)
    except Exception as e:
        log_error("pass2_5_default_pool", exc=e)
        all_pools = {}

    # Attach lanes to a valid pool
    for lid, lane in list(lanes.items()):
        try:
            if lane.get("pool") not in all_pools:
                lane["pool"] = default_pool_id
        except Exception as e:
            log_error("attach_lane_to_pool", sid=lid, exc=e)
            continue

    # Attach tasks to a valid pool
    for tid, t in list(tasks.items()):
        try:
            if t.get("pool") in all_pools:
                continue
            lid = t.get("lane")
            if lid in lanes and lanes[lid].get("pool") in all_pools:
                t["pool"] = lanes[lid]["pool"]
            else:
                t["pool"] = default_pool_id
        except Exception as e:
            log_error("attach_task_to_pool", sid=tid, exc=e)
            continue

    # Attach nodes to a valid pool
    for nid, n in list(nodes.items()):
        try:
            if n.get("pool") in all_pools:
                continue
            lid = n.get("lane")
            if lid in lanes and lanes[lid].get("pool") in all_pools:
                n["pool"] = lanes[lid]["pool"]
            else:
                n["pool"] = default_pool_id
        except Exception as e:
            log_error("attach_node_to_pool", sid=nid, exc=e)
            continue

    # Recover edge sources from each shape's outgoing list
    for sid, shape in list(shapes.items()):
        try:
            outgoing = shape.get("outgoing") or []
            if not isinstance(outgoing, list):
                log_error("outgoing_not_list", sid=sid, extra={"outgoing_type": type(outgoing).__name__})
                continue

            for out in outgoing:
                if not isinstance(out, dict):
                    log_error("outgoing_item_not_dict", sid=sid, extra={"out_repr": repr(out)[:200]})
                    continue
                eid = out.get("resourceId")
                if eid in edges:
                    edges[eid]["src"] = sid
        except Exception as e:
            log_error("pass3_recover_edge_sources", sid=sid, exc=e)
            continue

    # Register lanes inside pool
    for lid, lane in list(lanes.items()):
        try:
            pid = lane.get("pool")
            if pid in all_pools:
                all_pools[pid]["lanes"].append(lid)
        except Exception as e:
            log_error("pass4_attach_lanes", sid=lid, exc=e)
            continue

    # Register tasks and nodes under lanes or directly under pools
    def attach_to_lane_or_pool(obj_id, obj, lane_key="lane", pool_key="pool", lane_list="nodes", pool_list="nodes"):
        try:
            lid = obj.get(lane_key)
            pid = obj.get(pool_key)
            if lid in lanes:
                lanes[lid][lane_list].append(obj_id)
            elif pid in all_pools:
                all_pools[pid][pool_list].append(obj_id)
        except Exception as e:
            log_error("attach_to_lane_or_pool", sid=obj_id, exc=e, extra={"lane_list": lane_list, "pool_list": pool_list})

    for tid, t in list(tasks.items()):
        attach_to_lane_or_pool(tid, t, lane_list="tasks", pool_list="tasks")

    for nid, n in list(nodes.items()):
        attach_to_lane_or_pool(nid, n, lane_list="nodes", pool_list="nodes")

    # Build quick maps for readable pool and lane names
    try:
        pool_name = {pid: p["name"] for pid, p in all_pools.items()}
        lane_name = {lid: l["name"] for lid, l in lanes.items()}
    except Exception as e:
        log_error("build_name_resolution_maps", exc=e)
        pool_name, lane_name = {}, {}

    for tid, t in list(tasks.items()):
        try:
            t["pool_name"] = pool_name.get(t.get("pool"), "")
            t["lane_name"] = lane_name.get(t.get("lane"), "")
        except Exception as e:
            log_error("add_task_name_resolution", sid=tid, exc=e)

    for nid, n in list(nodes.items()):
        try:
            n["pool_name"] = pool_name.get(n.get("pool"), "")
            n["lane_name"] = lane_name.get(n.get("lane"), "")
        except Exception as e:
            log_error("add_node_name_resolution", sid=nid, exc=e)

    # Build a unified index to resolve any shape quickly.
    shape_index = {}

    for pid, p in list(all_pools.items()):
        try:
            shape_index[pid] = {
                "id": pid,
                "kind": "pool",
                "type": p["type"],
                "name": p["name"],
                "pool": pid,
                "lane": None,
            }
        except Exception as e:
            log_error("shape_index_pool", sid=pid, exc=e)

    for lid, l in list(lanes.items()):
        try:
            shape_index[lid] = {
                "id": lid,
                "kind": "lane",
                "type": l["type"],
                "name": l["name"],
                "pool": l["pool"],
                "lane": lid,
            }
        except Exception as e:
            log_error("shape_index_lane", sid=lid, exc=e)

    for tid, t in list(tasks.items()):
        try:
            shape_index[tid] = {
                "id": tid,
                "kind": "task",
                "type": t["task_type"],
                "name": t["name"],
                "pool": t["pool"],
                "lane": t["lane"],
            }
        except Exception as e:
            log_error("shape_index_task", sid=tid, exc=e)

    for nid, n in list(nodes.items()):
        try:
            shape_index[nid] = {
                "id": nid,
                "kind": "node",
                 "type": n.get("gateway_type") or n.get("node_type", ""),
                "name": n["name"],
                "pool": n["pool"],
                "lane": n["lane"],
            }
        except Exception as e:
            log_error("shape_index_node", sid=nid, exc=e)

    return {
        "pools": pools,
        "collapsed_pools": collapsed_pools,
        "lanes": lanes,
        "tasks": tasks,
        "nodes": nodes,
        "edges": edges,
        "shape_index": shape_index,
        "errors": extraction_errors,
    }

# Return the pool id for any known shape id.
def pool_of_any(sid, bpmn):
    idx = bpmn.get("shape_index", {})
    if sid in idx:
        return idx[sid]["pool"]
    return None

def analyze_gateways(bpmn):
    edges = bpmn["edges"]

    nodes = bpmn["nodes"]
    tasks = bpmn.get("tasks", {})

    # Merge all flow endpoints for degree computation.
    all_endpoints = {}
    all_endpoints.update(tasks)
    all_endpoints.update(nodes)

    incoming = defaultdict(list)
    outgoing = defaultdict(list)

    # Only SequenceFlow contributes to control flow.
    for e in edges.values():
        if e.get("type") != "SequenceFlow":
            continue

        src = e.get("src")
        tgt = e.get("tgt")

        if not src or not tgt:
            continue

        if src not in all_endpoints or tgt not in all_endpoints:
            continue

        outgoing[src].append(tgt)
        incoming[tgt].append(src)

    gateway_info = {}

    for nid, n in nodes.items():
        if n.get("gateway_type") != "ParallelGateway":
            continue

        in_deg = len(incoming[nid])
        out_deg = len(outgoing[nid])

        if in_deg == 1 and out_deg > 1:
            role = "AND_SPLIT"
        elif in_deg > 1 and out_deg == 1:
            role = "AND_JOIN"
        elif in_deg > 1 and out_deg > 1:
            role = "AND_JOIN_SPLIT"
        else:
            role = "UNCLASSIFIED"

        gateway_info[nid] = {
            "name": n.get("name", nid),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "role": role,
            "incoming": incoming[nid],
            "outgoing": outgoing[nid],
        }

    return gateway_info

def build_controlflow_per_pool(bpmn):
    flow_node_types = {
        "Task", "Function", "Subprocess", "CollapsedSubprocess",
        "StartNoneEvent",
        "EndNoneEvent",
        "ParallelGateway",
        "Exclusive_Databased_Gateway",

    }

    pools = bpmn.get("pools", {})
    collapsed_pools = bpmn.get("collapsed_pools", {})
    tasks = bpmn.get("tasks", {})
    nodes = bpmn.get("nodes", {})
    edges = bpmn.get("edges", {})
    idx = bpmn.get("shape_index", {})

    all_pools = {}
    all_pools.update(pools)
    all_pools.update(collapsed_pools)

    all_nodes = {}
    all_nodes.update(tasks)
    all_nodes.update(nodes)
    all_nodes.update(edges)


    def bpmn_type(sid: str):
        if sid in tasks:
            return tasks[sid].get("task_type", "")
        if sid in nodes and nodes[sid].get("type") == "Node" :
            return nodes[sid].get("node_type", "")
        if sid in nodes:
            return nodes[sid].get("gateway_type", "")
        if sid in edges:
            return edges[sid].get("type", "")


    def pool_of_any(sid: str):
        if sid in idx:
            return idx[sid].get("pool")
        if sid in all_nodes:
            return all_nodes[sid].get("pool")
        return None

    pool_nodes: Dict[str, Set[str]] = defaultdict(set)
    for nid in all_nodes:
        pid = pool_of_any(nid)
        if not pid:
            continue
        if bpmn_type(nid) in flow_node_types:
            pool_nodes[pid].add(nid)



    pool_seq_edges: Dict[str, List[str]] = defaultdict(list)
    succ: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    pred: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    warnings: Dict[str, List[str]] = defaultdict(list)

    for eid, e in edges.items():
        if e.get("type") != "SequenceFlow":
            continue

        src, tgt = e.get("src"), e.get("tgt")
        if not src or not tgt:
            continue

        sp = pool_of_any(src)
        tp = pool_of_any(tgt)

        if sp is None or tp is None:
            if sp is not None:
                warnings[sp].append(f"SequenceFlow {eid} has endpoint with unknown pool.")
            continue

        if sp != tp:
            # still register the edge in the source pool's sequence edges
            # we emit cross-pool reference
            pool_seq_edges[sp].append(eid)
            succ[sp][src].append(tgt)
            # do NOT add to pred[sp] — tgt is foreign, don't count it for local degree
            warnings[sp].append(f"SequenceFlow {eid} crosses pools ({sp} -> {tp}). Registered as cross-pool.")
            continue

        pool_id = sp

        if src not in all_nodes or tgt not in all_nodes:

            warnings[pool_id].append(f"SequenceFlow {eid} connects unknown endpoint(s): {src} -> {tgt}. Skipped.")
            continue

        pool_seq_edges[pool_id].append(eid)

        # if bpmn_type(src) not in flow_node_types :
        #     print(src)
        #
        # if bpmn_type(src) not in flow_node_types or bpmn_type(tgt) not in flow_node_types:
        #     continue

        succ[pool_id][src].append(tgt)
        pred[pool_id][tgt].append(src)



    start_nodes: Dict[str, List[str]] = {}
    end_nodes: Dict[str, List[str]] = {}
    for pool_id, nset in pool_nodes.items():
        start_nodes[pool_id] = [
            n for n in pool_nodes[pool_id]  # only local nodes
            if len(pred[pool_id].get(n, [])) == 0
        ]
        end_nodes[pool_id] = [
            n for n in pool_nodes[pool_id]  # only local nodes
            if len(succ[pool_id].get(n, [])) == 0
        ]

    # Detect gateway roles per pool
    gateways_parallel: Dict[str, Dict[str, dict]] = defaultdict(dict)
    gateways_exclusive: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for pool_id, nset in pool_nodes.items():
        for nid in nset:
            t = bpmn_type(nid)
            in_deg = sum(
                1 for pred_id in pred[pool_id].get(nid, [])
                if pred_id in pool_nodes[pool_id]
            )
            out_deg = sum(
                1 for succ_id in succ[pool_id].get(nid, [])
                if succ_id in pool_nodes[pool_id]
            )

            if t == "ParallelGateway":
                if in_deg == 1 and out_deg > 1:
                    role = "AND_SPLIT"
                elif in_deg > 1 and out_deg == 1:
                    role = "AND_JOIN"
                elif in_deg > 1 and out_deg > 1:
                    role = "AND_JOIN_SPLIT"
                else:
                    role = "UNCLASSIFIED"

                gateways_parallel[pool_id][nid] = {
                    "role": role,
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "incoming": list(pred[pool_id].get(nid, [])),
                    "outgoing": list(succ[pool_id].get(nid, [])),
                }

            elif t == "Exclusive_Databased_Gateway":
                if in_deg == 1 and out_deg > 1:
                    role = "XOR_SPLIT"
                elif in_deg > 1 and out_deg == 1:
                    role = "XOR_JOIN"
                elif in_deg > 1 and out_deg > 1:
                    role = "XOR_JOIN_SPLIT"
                else:
                    role = "UNCLASSIFIED"

                gateways_exclusive[pool_id][nid] = {
                    "role": role,
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "incoming": list(pred[pool_id].get(nid, [])),
                    "outgoing": list(succ[pool_id].get(nid, [])),
                }

    # Helper functions for parallel block extraction.
    def find_reachable(pool_id: str, start: str, stop: str) -> Set[str]:
        seen = set()
        q = deque([start])
        while q:
            u = q.popleft()
            if u == stop:
                continue
            if u in seen:
                continue
            seen.add(u)
            for v in succ[pool_id].get(u, []):
                if v not in seen and v != stop:
                    q.append(v)
        return seen

    # def reachable_without_other_join(pool_id: str, start: str, target: str, forbidden_joins: set):
    #     q = deque([start])
    #     seen = set()
    #
    #     while q:
    #         u = q.popleft()
    #
    #         if u == target:
    #             return True
    #
    #         if u in seen:
    #             continue
    #         seen.add(u)
    #
    #         if u in forbidden_joins and u != start:
    #             continue
    #
    #         for v in succ[pool_id].get(u, []):
    #             if v not in seen:
    #                 q.append(v)
    #
    #     return False

    def shortest_distance(pool_id: str, start: str, target: str):
        q = deque([(start, 0)])
        seen = set()

        while q:
            u, d = q.popleft()
            if u == target:
                return d
            if u in seen:
                continue
            seen.add(u)

            for v in succ[pool_id].get(u, []):
                if v not in seen:
                    q.append((v, d + 1))

        return None

    def is_exclusive_gateway(pool_id: str, nid: str) -> bool:
        return nid in gateways_exclusive[pool_id]

    def find_common_join(pool_id: str, split_id: str):
        branch_entries = succ[pool_id].get(split_id, [])
        if len(branch_entries) < 2:
            return None

        # Try explicit parallel joins first.
        explicit_joins = [
            gid for gid, g in gateways_parallel[pool_id].items()
            if g["role"] in {"AND_JOIN", "AND_JOIN_SPLIT"}
        ]

        valid_explicit = []
        for join_id in explicit_joins:
            distances = []
            valid = True
            for entry in branch_entries:
                d = shortest_distance(pool_id, entry, join_id)
                if d is None:
                    valid = False
                    break
                distances.append(d)
            if valid:
                valid_explicit.append((max(distances), sum(distances), join_id))

        if valid_explicit:
            valid_explicit.sort()
            return valid_explicit[0][2]

        # If needed, infer the nearest common reachable join.
        reachable_sets = []
        for entry in branch_entries:
            r = find_reachable(pool_id, entry, stop="__NONE__")
            reachable_sets.append(r | {entry})  # include entry itself

        common = set.intersection(*reachable_sets) if reachable_sets else set()
        common.discard(split_id)

        if not common:
            return None

        candidates = []
        for nid in common:
            # Skip decision gateways when inferring a parallel join.
            if is_exclusive_gateway(pool_id, nid):
                continue

            distances = []
            valid = True
            for entry in branch_entries:
                d = shortest_distance(pool_id, entry, nid)
                if d is None:
                    valid = False
                    break
                distances.append(d)

            if valid:
                candidates.append((max(distances), sum(distances), nid))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def extract_branches(pool_id: str, split_id: str, join_id: str) -> Tuple[List[List[str]], List[str], List[str]]:
        branch_entries = list(succ[pool_id].get(split_id, []))
        branches = []
        branch_exits = []

        for entry in branch_entries:
            r = find_reachable(pool_id, entry, stop=join_id)
            exits = [n for n in r if join_id in succ[pool_id].get(n, [])]
            branch_exits.extend(exits)
            branches.append(sorted(r))

        seen = set()
        branch_exits_unique = []
        for x in branch_exits:
            if x not in seen:
                seen.add(x)
                branch_exits_unique.append(x)

        return branches, branch_entries, branch_exits_unique

    # Build parallel split/join blocks.
    parallel_blocks: Dict[str, List[dict]] = defaultdict(list)

    for pool_id, gw_map in gateways_parallel.items():
        for gid, g in gw_map.items():

            if g["role"] not in {"AND_SPLIT", "AND_JOIN_SPLIT"}:
                continue

            join_id = find_common_join(pool_id, gid)
            if not join_id:
                warnings[pool_id].append(f"Could not find matching AND_JOIN for split {gid}.")
                continue

            branches, entries, exits = extract_branches(pool_id, gid, join_id)
            parallel_blocks[pool_id].append({
                "split": gid,
                "join": join_id,
                "branch_entries": entries,
                "branch_exits": exits,
                "branches": branches,
            })


    # Build decision blocks from exclusive gateways.
    decision_blocks = defaultdict(dict)

    def extract_xor_branches(pool_id: str, gateway_id: str):
        gw = gateways_exclusive[pool_id][gateway_id]
        branches = []

        for tgt in gw["outgoing"]:
            label = ""
            condition = ""
            condition_type = ""
            edge_id = None
            is_default = False

            # Match the sequence flow edge for this branch.
            for eid, e in edges.items():
                if e.get("type") != "SequenceFlow":
                    continue
                if e.get("src") == gateway_id and e.get("tgt") == tgt:
                    edge_id = eid
                    label = e.get("name", "").strip()
                    condition = e.get("condition", "").strip()
                    condition_type = e.get("condition_type", "").strip()
                    is_default = e.get("is_default", False)
                    break

            branches.append({
                "edge_id": edge_id,
                "label": label,
                "condition": condition,
                "condition_type": condition_type,
                "is_default": is_default,
                "target": tgt,
            })

        return branches

    # Keep raw BPMN target ids and resolve names later.
    def build_decision_block(pool_id: str, gateway_id: str):
        branches = extract_xor_branches(pool_id, gateway_id)

        condition_branches = []
        default_branch = None

        for b in branches:
            target_bpmn_id = b["target"]

            cond_text = (b.get("condition") or "").strip()
            label_text = (b.get("label") or "").strip()

            if b.get("is_default", False):
                default_branch = {"raw_target": target_bpmn_id, "text": "else"}
            else:
                if cond_text:
                    cond = cond_text
                elif label_text:
                    cond = f'"{label_text}"'
                else:
                    cond = '"unspecified condition"'

                condition_branches.append({"raw_target": target_bpmn_id, "cond": cond})

        result = {
            "decision": gateway_id,
            "branches": condition_branches,
            "default": default_branch,
        }

        return result

    for pool_id, gw_map in gateways_exclusive.items():
        for gid, g in gw_map.items():
            if g["role"] not in {"XOR_SPLIT", "XOR_JOIN_SPLIT"}:
                continue
            block = build_decision_block(pool_id, gid)

            decision_blocks[pool_id][gid] = block

    result = {}
    for pool_id, p in all_pools.items():
        result[pool_id] = {
            "pool_name": p["name"],
            "nodes": pool_nodes.get(pool_id, set()),
            "sequence_edges": pool_seq_edges.get(pool_id, []),
            "succ": dict(succ.get(pool_id, {})),
            "pred": dict(pred.get(pool_id, {})),
            "start_nodes": start_nodes.get(pool_id, []),
            "end_nodes": end_nodes.get(pool_id, []),
            "gateways": {
                "parallel": gateways_parallel.get(pool_id, {}),
                "exclusive": gateways_exclusive.get(pool_id, {}),
            },
            "parallel_blocks": parallel_blocks.get(pool_id, []),
            "decision_blocks": decision_blocks.get(pool_id, {}),
            "warnings": warnings.get(pool_id, []),
        }

    return result

# Render a readable name for debug output.
def pretty_name(bpmn, sid):
    idx = bpmn.get("shape_index", {})
    info = idx.get(sid)

    if not info:
        return sid

    name = " ".join(info.get("name", sid).split())
    bpmn_type = info.get("type", "")

    if name.startswith("sid-") or name == sid:
        return f"{bpmn_type}({sid})"

    return f"{name} [{bpmn_type}]"


def print_pool_controlflow(bpmn: dict, pool_cf: dict, pool_id: str):
    print("\n POOL:", pool_cf[pool_id]["pool_name"], " - ", pool_id)
    print("Start nodes:")
    for n in pool_cf[pool_id]["start_nodes"]:
        print(" -", pretty_name(bpmn, n))
    print("End nodes:")
    for n in pool_cf[pool_id]["end_nodes"]:
        print(" -", pretty_name(bpmn, n))

    print("\nParallel blocks:")
    for blk in pool_cf[pool_id]["parallel_blocks"]:
        print(" * Split:", pretty_name(bpmn, blk["split"]))
        print("   Join :", pretty_name(bpmn, blk["join"]))
        print("   Entries:")
        for e in blk["branch_entries"]:
            print("    -", pretty_name(bpmn, e))
        print("   Exits:")
        for x in blk["branch_exits"]:
            print("    -", pretty_name(bpmn, x))
        print("   Branch node sets:")
        for i, br in enumerate(blk["branches"], 1):
            print(f"    Branch {i}:")
            for n in br:
                print("      -", pretty_name(bpmn, n))

    if pool_cf[pool_id]["warnings"]:
        print("\nWarnings:")
        for w in pool_cf[pool_id]["warnings"]:
            print(" -", w)

flow_node_types = {
    "Task", "CollapsedSubprocess",
    "StartNoneEvent", "EndNoneEvent",
    "ParallelGateway", "ExclusiveGateway",
    "InclusiveGateway", "EventBasedGateway",
}

data_node_types = {"DataObject", "DataStore"}

def extract_data_associations(bpmn: dict,
                             data_node_types: set,
                             flow_node_types: set) -> dict:

    pools = bpmn.get("pools", {})
    collapsed_pools = bpmn.get("collapsed_pools", {})
    tasks = bpmn.get("tasks", {})
    nodes = bpmn.get("nodes", {})
    edges = bpmn.get("edges", {})
    idx = bpmn.get("shape_index", {})

    # union pools
    all_pools = {}
    all_pools.update(pools)
    all_pools.update(collapsed_pools)

    def pool_name(pid: Optional[str]) -> str:
        if pid and pid in all_pools:
            return all_pools[pid].get("name", "")
        return ""

    def bpmn_type_of(sid: str) -> str:
        info = idx.get(sid)
        if info:
            return info.get("type", "")
        if sid in tasks:
            return tasks[sid].get("task_type", "")
        if sid in nodes:
            return nodes[sid].get("node_type", "")
        return ""

    def kind_of(sid: str) -> str:
        t = bpmn_type_of(sid)
        if t in data_node_types:
            return "DATA"
        if t in flow_node_types:
            return "FLOW"
        return "OTHER"

    # Collect data nodes from both task and node containers.

    all_endpoints = {}
    all_endpoints.update(tasks)
    all_endpoints.update(nodes)

    data_nodes = {
        sid: all_endpoints[sid]
        for sid in all_endpoints
        if bpmn_type_of(sid) in data_node_types
    }

    links = []
    by_data = defaultdict(list)
    by_flow = defaultdict(list)

    for eid, e in edges.items():
        if e.get("type") not in {"Association_Unidirectional", "Association_Undirected"}:
            continue
        src, tgt = e.get("src"), e.get("tgt")
        if not src or not tgt:
            continue

        src_info = idx.get(src, {"type": bpmn_type_of(src), "name": src, "pool": None})
        tgt_info = idx.get(tgt, {"type": bpmn_type_of(tgt), "name": tgt, "pool": None})

        pool_id = src_info.get("pool") or tgt_info.get("pool")

        link = {
            "edge_id": eid,
            "edge_type": e.get("type", ""),
            "src_id": src,
            "tgt_id": tgt,
            "src_kind": kind_of(src),
            "tgt_kind": kind_of(tgt),
            "src_name": src_info.get("name", src),
            "tgt_name": tgt_info.get("name", tgt),
            "src_type": src_info.get("type", "UNKNOWN"),
            "tgt_type": tgt_info.get("type", "UNKNOWN"),
            "pool": pool_id,
            "pool_name": pool_name(pool_id),
            "direction": "undirected" if e.get("type") == "Association_Undirected" else "src_to_tgt",
        }
        links.append(link)

        if link["src_kind"] == "DATA":
            by_data[src].append(link)
        if link["tgt_kind"] == "DATA":
            by_data[tgt].append(link)
        if link["src_kind"] == "FLOW":
            by_flow[src].append(link)
        if link["tgt_kind"] == "FLOW":
            by_flow[tgt].append(link)

    return {
        "data_nodes": data_nodes,
        "links": links,
        "by_data": dict(by_data),
        "by_flow": dict(by_flow),
    }



def print_data_summary(data_model: dict, limit: int = 50):
    print("\n DATA ASSOCIATIONS :")
    shown = 0
    for link in data_model["links"]:
        if not ((link["src_kind"] == "DATA" and link["tgt_kind"] == "FLOW") or
                (link["src_kind"] == "FLOW" and link["tgt_kind"] == "DATA") or
                (link["src_kind"] == "DATA" and link["tgt_kind"] == "DATA")):
            continue

        arrow = "<->" if link["direction"] == "undirected" else "->"
        print(
            f"- {link['edge_id']} [{link['edge_type']}]: "
            f"{link['src_name']} [{link['src_type']}] {arrow} {link['tgt_name']} [{link['tgt_type']}] "
            f"| pool={link['pool_name'] or link['pool']}"
        )
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("No Data<->Flow associations found.")

if __name__ == "__main__":
    # run for one sample model
    df = pd.read_csv("D:\\sap_sam_2022\\models\\40000.csv")
    model = json.loads(df.loc[0, "Model JSON"])

    bpmn = extract_bpmn(model)

    bad = [e for e in bpmn["edges"].values() if not e["src"] or not e["tgt"]]
    print("Edges missing src or tgt:", len(bad))

    print("\n POOLS:")
    for p in bpmn["pools"].values():
        print("-", p["name"])


    print("\n\n LANES:")
    for l in bpmn["lanes"].values():
        print("-", l["name"], "| pool:", l["pool"])

    print("\n\nSEQUENCE EDGES:")
    for e in bpmn.get("edges", {}).values():
        if e.get("type") != "SequenceFlow":
            continue
        src = e.get("src")
        tgt = e.get("tgt")
        if not src or not tgt:
            continue
        print(f"{pretty_name(bpmn, src)} -> {pretty_name(bpmn, tgt)}")

    print("\n\nTASKS:")
    for t in bpmn.get("tasks", {}).values():
        name = " ".join(t.get("name", "").split())
        doc = t.get("documentation", "")
        task_type = t.get("task_type", "")
        print(f"- {name} [{task_type}]")
        if doc:
            clean_doc = " ".join(str(doc).split())
            print("   doc:", clean_doc)


    # gw = analyze_gateways(bpmn)
    #
    # print("\n\nPARALLEL GATEWAYS:")
    # for gid, g in gw.items():
    #     print("-", g["name"])
    #     print("   role:", g["role"])
    #     print("   in :", g["in_degree"])
    #     print("   out:", g["out_degree"])
    #
    #
    # print("\n\nInspecting pools:")
    # pool_cf = build_controlflow_per_pool(bpmn)
    # for pool_id in bpmn["pools"]:
    #     print_pool_controlflow(bpmn, pool_cf, pool_id)
    #
    # def reachable_from(pool_id, start):
    #     seen = set()
    #     q = deque([start])
    #     while q:
    #         u = q.popleft()
    #         if u in seen:
    #             continue
    #         seen.add(u)
    #         for v in pool_cf[pool_id]["succ"].get(u, []):
    #             if v not in seen:
    #                 q.append(v)
    #     return seen
    #
    # demo_pool_id = next(iter(bpmn["pools"]), None)
    # if demo_pool_id:
    #     start_nodes = pool_cf[demo_pool_id]["start_nodes"]
    #     start = start_nodes[0] if start_nodes else None
    #     reach = reachable_from(demo_pool_id, start)
    #
    #     print("\n\nREACHABLE FROM START:")
    #     for nid in reach:
    #         print("-", pretty_name(bpmn, nid))
    #
    #
    # data_model = extract_data_associations(bpmn,data_node_types,flow_node_types)
    #
    # print_data_summary(data_model)