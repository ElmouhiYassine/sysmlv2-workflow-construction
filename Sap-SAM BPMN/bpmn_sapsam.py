from collections import Counter, defaultdict,deque
import pandas as pd
import json
from typing import Dict, List, Set, Tuple, Optional


pool_types = {"Pool", "CollapsedPool"}
lane_types = {"Lane"}
edge_types = {
    "SequenceFlow",
    "MessageFlow",
    "Association_Unidirectional",
    "Association_Undirected",
}

ignore_types = {"BPMNDiagram"}  # root, not a real node

#  Yield (shape, parent) for all nested shapes.
def walk(shape, parent=None):
    yield shape, parent
    for ch in shape.get("childShapes", []) or []:
        yield from walk(ch, shape)

def extract_bpmn(doc):
    shapes = {}
    parent_of = {}

    pool_types = {"Pool", "CollapsedPool"}
    lane_types = {"Lane"}
    edge_types = {"SequenceFlow", "ControlFlow"}

    task_types = {
        "Task",
        "Function",
        "Subprocess",
        "CollapsedSubprocess",
        "StartNoneEvent",
        "StartMessageEvent",
        "StartTimerEvent",
        "EndNoneEvent",
        "EndMessageEvent",
        "EndEscalationEvent",
        "EndTerminateEvent",
        "IntermediateEvent",
        "IntermediateTimerEvent",
        "IntermediateMessageEventCatching",
        "IntermediateMessageEventThrowing",
        "IntermediateEscalationEvent",
        "IntermediateErrorEvent",
    }
    ignore_types = []

    # Pass 1: index shapes + containment
    for shape, parent in walk(doc):
        sid = shape.get("resourceId")
        if not sid:
            continue
        shapes[sid] = shape
        if parent and parent.get("resourceId"):
            parent_of[sid] = parent["resourceId"]

    def stencil_id(sid):
        return (shapes.get(sid, {}).get("stencil") or {}).get("id") or ""

    def props_of(sid):
        return shapes.get(sid, {}).get("properties") or {}

    def get_text(props, key):
        v = props.get(key)
        if v is None:
            return ""
        return str(v).strip()

    def find_ancestor(sid, stencil_set):
        cur = sid
        while cur in parent_of:
            cur = parent_of[cur]
            if stencil_id(cur) in stencil_set:
                return cur
        return None

    pools, collapsed_pools, lanes = {}, {}, {}
    tasks, nodes, edges = {}, {}, {}

    # Pass 2: classify shapes
    for sid in shapes:
        st = stencil_id(sid)
        if st in ignore_types:
            continue

        props = props_of(sid)
        name = get_text(props, "name")
        documentation = get_text(props, "documentation")

        pool_id = find_ancestor(sid, pool_types)
        lane_id = find_ancestor(sid, lane_types)

        # Pools
        if st in pool_types:
            entry = {
                "id": sid,
                "type": st,                # Pool or CollapsedPool
                "name": name or sid,
                "lanes": [],
                "nodes": [],
                "tasks": [],
            }
            if st == "CollapsedPool":
                collapsed_pools[sid] = entry
            else:
                pools[sid] = entry

        # Lanes
        elif st in lane_types:
            lanes[sid] = {
                "id": sid,
                "type": st,
                "name": name or sid,
                "pool": pool_id,  # parent pool (may be None)
                "nodes": [],
                "tasks": [],
            }

        # Edges
        elif st in edge_types:
            tgt = (shapes[sid].get("target") or {}).get("resourceId")
            edges[sid] = {
                "id": sid,
                "type": st,
                "name": name,
                "documentation": documentation,
                "props": props,
                "src": None,
                "tgt": tgt,
            }

        # Task-like nodes
        elif st in task_types:
            tasks[sid] = {
                "id": sid,
                "type": "Task",
                "task_type": st,
                "name": name or sid,
                "documentation": documentation,
                "props": props,
                "pool": pool_id,
                "lane": lane_id,
            }

        # Generic nodes (everything else that isn't pool/lane/edge)
        else:
            nodes[sid] = {
                "id": sid,
                "type": "Node",
                "node_type": st,
                "name": name or sid,
                "documentation": documentation,
                "props": props,
                "pool": pool_id,
                "lane": lane_id,
            }

    # Pass 2.5: default pool if none exist
    default_pool_id = "__DEFAULT_POOL__"
    if not pools and not collapsed_pools:
        pools[default_pool_id] = {
            "id": default_pool_id,
            "type": "Pool",
            "name": "Default Pool",
            "lanes": [],
            "nodes": [],
            "tasks": [],
        }

    # helper: union view for "all pools"
    all_pools = {}
    all_pools.update(pools)
    all_pools.update(collapsed_pools)

    # If lanes exist but their pool is None, we attach them to default pool
    for lid, lane in lanes.items():
        if lane.get("pool") not in all_pools:
            lane["pool"] = default_pool_id

    # Attach tasks to pools
    for tid, t in tasks.items():
        if t.get("pool") in all_pools:
            continue
        lid = t.get("lane")
        if lid in lanes and lanes[lid].get("pool") in all_pools:
            t["pool"] = lanes[lid]["pool"]
        else:
            t["pool"] = default_pool_id

    # Attach nodes to pools
    for nid, n in nodes.items():
        if n.get("pool") in all_pools:
            continue
        lid = n.get("lane")
        if lid in lanes and lanes[lid].get("pool") in all_pools:
            n["pool"] = lanes[lid]["pool"]
        else:
            n["pool"] = default_pool_id

    # Pass 3: recover edge sources from outgoing refs
    for sid, shape in shapes.items():
        for out in (shape.get("outgoing") or []):
            eid = out.get("resourceId")
            if eid in edges:
                edges[eid]["src"] = sid

    # Pass 4: attach lanes to pools
    for lid, lane in lanes.items():
        pid = lane.get("pool")
        if pid in all_pools:
            all_pools[pid]["lanes"].append(lid)

    # Pass 5: attach tasks/nodes to lanes or pools
    def attach_to_lane_or_pool(obj_id, obj, lane_key="lane", pool_key="pool", lane_list="nodes", pool_list="nodes"):
        lid = obj.get(lane_key)
        pid = obj.get(pool_key)
        if lid in lanes:
            lanes[lid][lane_list].append(obj_id)
        elif pid in all_pools:
            all_pools[pid][pool_list].append(obj_id)

    for tid, t in tasks.items():
        attach_to_lane_or_pool(tid, t, lane_list="tasks", pool_list="tasks")

    for nid, n in nodes.items():
        attach_to_lane_or_pool(nid, n, lane_list="nodes", pool_list="nodes")

    # Helpful name resolution
    pool_name = {pid: p["name"] for pid, p in all_pools.items()}
    lane_name = {lid: l["name"] for lid, l in lanes.items()}

    for t in tasks.values():
        t["pool_name"] = pool_name.get(t["pool"], "")
        t["lane_name"] = lane_name.get(t["lane"], "")

    for n in nodes.values():
        n["pool_name"] = pool_name.get(n["pool"], "")
        n["lane_name"] = lane_name.get(n["lane"], "")

    # Global index for any shape endpoint lookup (task/node/lane/pool)
    shape_index = {}

    for pid, p in all_pools.items():
        shape_index[pid] = {"id": pid, "kind": "pool", "type": p["type"], "name": p["name"], "pool": pid, "lane": None}

    for lid, l in lanes.items():
        shape_index[lid] = {"id": lid, "kind": "lane", "type": l["type"], "name": l["name"], "pool": l["pool"], "lane": lid}

    for tid, t in tasks.items():
        shape_index[tid] = {"id": tid, "kind": "task", "type": t["task_type"], "name": t["name"], "pool": t["pool"], "lane": t["lane"]}

    for nid, n in nodes.items():
        shape_index[nid] = {"id": nid, "kind": "node", "type": n.get("node_type", ""), "name": n["name"], "pool": n["pool"], "lane": n["lane"]}

    return {
        "pools": pools,
        "collapsed_pools": collapsed_pools,
        "lanes": lanes,
        "tasks": tasks,
        "nodes": nodes,
        "edges": edges,
        "shape_index": shape_index,
    }

# Return pool id for any shape id (node/lane/pool), else None.
def pool_of_any(sid, bpmn):
    idx = bpmn.get("shape_index", {})
    if sid in idx:
        return idx[sid]["pool"]
    return None

def analyze_gateways(bpmn):
    edges = bpmn["edges"]

    # gateways are stored in bpmn["nodes"]
    nodes = bpmn["nodes"]
    tasks = bpmn.get("tasks", {})

    # combine all possible endpoints for degree computations
    all_endpoints = {}
    all_endpoints.update(tasks)
    all_endpoints.update(nodes)

    incoming = defaultdict(list)
    outgoing = defaultdict(list)

    # Only SequenceFlow defines control-flow
    for e in edges.values():
        if e.get("type") != "SequenceFlow":
            continue

        src = e.get("src")
        tgt = e.get("tgt")

        # skip broken edges
        if not src or not tgt:
            continue

        # skip if endpoints are not known nodes/tasks
        if src not in all_endpoints or tgt not in all_endpoints:
            continue

        outgoing[src].append(tgt)
        incoming[tgt].append(src)

    gateway_info = {}

    for nid, n in nodes.items():
        # IMPORTANT: gateways are stored as generic nodes with node_type
        if n.get("node_type") != "ParallelGateway":
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
    # Original BPMN types you want to treat as "control-flow nodes"
    flow_node_types = {
        # tasks / subprocess
        "Task", "Function", "Subprocess", "CollapsedSubprocess",
        # start/end
        "StartNoneEvent", "StartMessageEvent", "StartTimerEvent",
        "EndNoneEvent", "EndMessageEvent", "EndEscalationEvent", "EndTerminateEvent",
        # intermediate events (optional for your graph)
        "IntermediateEvent", "IntermediateTimerEvent",
        "IntermediateMessageEventCatching", "IntermediateMessageEventThrowing",
        "IntermediateEscalationEvent", "IntermediateErrorEvent",
        # gateways
        "ParallelGateway", "Exclusive_Databased_Gateway", "InclusiveGateway", "EventbasedGateway", "Decision",
    }

    pools = bpmn.get("pools", {})
    collapsed_pools = bpmn.get("collapsed_pools", {})
    lanes = bpmn.get("lanes", {})
    tasks = bpmn.get("tasks", {})
    nodes = bpmn.get("nodes", {})      # generic nodes (gateways are here)
    edges = bpmn.get("edges", {})
    idx = bpmn.get("shape_index", {})

    # union of pools
    all_pools = {}
    all_pools.update(pools)
    all_pools.update(collapsed_pools)

    # union of endpoints (both tasks + generic nodes)
    all_nodes = {}
    all_nodes.update(tasks)
    all_nodes.update(nodes)

    # helper: original BPMN type of any endpoint
    def bpmn_type(sid: str) -> str:
        if sid in tasks:
            return tasks[sid].get("task_type", "")
        if sid in nodes:
            return nodes[sid].get("node_type", "")
        return ""

    # helper: pool of any shape via shape_index
    def pool_of_any(sid: str):
        if sid in idx:
            return idx[sid].get("pool")
        # fallback: try endpoint payload
        if sid in all_nodes:
            return all_nodes[sid].get("pool")
        return None

    # Build per-pool node sets (flow nodes = endpoints whose BPMN type is in flow_node_types)
    pool_nodes: Dict[str, Set[str]] = defaultdict(set)
    for nid in all_nodes:
        pid = pool_of_any(nid)
        if not pid:
            continue
        if bpmn_type(nid) in flow_node_types:
            pool_nodes[pid].add(nid)

    # Per-pool sequence edges and adjacency
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
            warnings[sp].append(f"SequenceFlow {eid} crosses pools ({sp} -> {tp}). Ignored for control-flow.")
            continue

        pool_id = sp

        # endpoints must be known nodes/tasks
        if src not in all_nodes or tgt not in all_nodes:
            warnings[pool_id].append(f"SequenceFlow {eid} connects unknown endpoint(s): {src} -> {tgt}. Skipped.")
            continue

        pool_seq_edges[pool_id].append(eid)

        # keep adjacency only for flow nodes
        if bpmn_type(src) not in flow_node_types or bpmn_type(tgt) not in flow_node_types:
            continue

        succ[pool_id][src].append(tgt)
        pred[pool_id][tgt].append(src)

    # Compute start/end nodes per pool (within indexed node set)
    start_nodes: Dict[str, List[str]] = {}
    end_nodes: Dict[str, List[str]] = {}
    for pool_id, nset in pool_nodes.items():
        sn = [n for n in nset if len(pred[pool_id].get(n, [])) == 0]
        en = [n for n in nset if len(succ[pool_id].get(n, [])) == 0]
        start_nodes[pool_id] = sn
        end_nodes[pool_id] = en

    # Parallel gateway analysis per pool
    gateways_parallel: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for pool_id, nset in pool_nodes.items():
        for nid in nset:
            if bpmn_type(nid) != "ParallelGateway":
                continue

            in_deg = len(pred[pool_id].get(nid, []))
            out_deg = len(succ[pool_id].get(nid, []))

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

    # --- BFS reachability ---
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

    def find_common_join(pool_id: str, split_id: str) -> Optional[str]:
        outs = succ[pool_id].get(split_id, [])
        if len(outs) < 2:
            return None

        reachable_sets = [find_reachable(pool_id, o, stop="__NONE__") for o in outs]
        common = set.intersection(*reachable_sets) if reachable_sets else set()

        candidates = [
            gid for gid, g in gateways_parallel[pool_id].items()
            if g["role"] == "AND_JOIN" and gid in common
        ]
        if not candidates:
            return None

        dist = {split_id: 0}
        q = deque([split_id])
        while q:
            u = q.popleft()
            for v in succ[pool_id].get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        candidates.sort(key=lambda x: dist.get(x, 10**9))
        return candidates[0]

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

    parallel_blocks: Dict[str, List[dict]] = defaultdict(list)
    for pool_id, gw_map in gateways_parallel.items():
        for gid, g in gw_map.items():
            if g["role"] != "AND_SPLIT":
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

    # Pack output per pool (include collapsed pools too)
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
            "gateways": {"parallel": gateways_parallel.get(pool_id, {})},
            "parallel_blocks": parallel_blocks.get(pool_id, []),
            "warnings": warnings.get(pool_id, []),
        }
    return result

#print
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


def extract_message_interactions(bpmn: dict) -> dict:
    pools = bpmn.get("pools", {})
    collapsed_pools = bpmn.get("collapsed_pools", {})
    edges = bpmn.get("edges", {})
    idx = bpmn.get("shape_index", {})

    # union pools
    all_pools = {}
    all_pools.update(pools)
    all_pools.update(collapsed_pools)

    def pool_of_any(sid: str) -> Optional[str]:
        info = idx.get(sid)
        if info and info.get("pool"):
            return info["pool"]
        # if endpoint is literally a pool id
        if sid in all_pools:
            return sid
        return None

    def pool_name(pid: Optional[str]) -> str:
        if pid and pid in all_pools:
            return all_pools[pid].get("name", "")
        return ""

    messages = []
    by_pool = defaultdict(lambda: {"out": [], "in": []})

    for eid, e in edges.items():
        if e.get("type") != "MessageFlow":
            continue
        src, tgt = e.get("src"), e.get("tgt")
        if not src or not tgt:
            continue

        sp = pool_of_any(src)
        tp = pool_of_any(tgt)

        # shape_index entries now include: kind, type, name, pool, lane
        src_info = idx.get(src, {"type": "UNKNOWN", "name": src})
        tgt_info = idx.get(tgt, {"type": "UNKNOWN", "name": tgt})

        msg = {
            "edge_id": eid,
            "src_id": src,
            "tgt_id": tgt,
            "src_pool": sp,
            "tgt_pool": tp,
            "src_pool_name": pool_name(sp),
            "tgt_pool_name": pool_name(tp),
            "src_kind": src_info.get("kind", "UNKNOWN"),
            "tgt_kind": tgt_info.get("kind", "UNKNOWN"),
            "src_type": src_info.get("type", "UNKNOWN"),
            "tgt_type": tgt_info.get("type", "UNKNOWN"),
            "src_name": src_info.get("name", src),
            "tgt_name": tgt_info.get("name", tgt),
            "cross_pool": (sp is not None and tp is not None and sp != tp),
        }
        messages.append(msg)

        if sp:
            by_pool[sp]["out"].append(msg)
        if tp:
            by_pool[tp]["in"].append(msg)

    return {"messages": messages, "by_pool": dict(by_pool)}

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

    # collect data nodes (from BOTH tasks and nodes)

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

def print_message_summary(msg_model: dict):
    print("\n MESSAGE FLOWS :")
    for m in msg_model["messages"]:
        print(
            f"- {m['edge_id']}: "
            f"{m['src_pool_name'] or m['src_pool']} ({m['src_type']}) "
            f"-> {m['tgt_pool_name'] or m['tgt_pool']} ({m['tgt_type']}) "
            f"| src={m['src_name']} -> tgt={m['tgt_name']}"
        )
    if not msg_model['messages']:
        print('No messages flow.')



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
    #D:\\sap_sam_2022\\models\\30000.csv
    # D:\\sap_sam_2022\\models\\40000.csv
    # D:\\sap_sam_2022\\models\\30000.csv
    df = pd.read_csv("D:\\sap_sam_2022\\models\\10000.csv")
    model = json.loads(df.loc[0, "Model JSON"])

    bpmn = extract_bpmn(model)
    # print(bpmn['edges'])
    c = Counter(n["type"] for n in bpmn["nodes"].values())
    # print(c.most_common(30))

    # ec = Counter(e["type"] for e in bpmn["edges"].values())
    # print(ec)

    bad = [e for e in bpmn["edges"].values() if not e["src"] or not e["tgt"]]
    # print("Edges missing src or tgt:", len(bad))

    msgs = [e for e in bpmn["edges"].values() if e["type"] == "MessageFlow" and e["src"] and e["tgt"]]
    # for e in msgs:
    #     sp = pool_of_any(e["src"], bpmn)
    #     tp = pool_of_any(e["tgt"], bpmn)
    #     print(e["id"], sp, "->", tp)

    print("\n POOLS:")
    for p in bpmn["pools"].values():
        print("-", p["name"])


    print("\n\n LANES:")
    for l in bpmn["lanes"].values():
        print("-", l["name"], "| pool:", l["pool"])

    print("\n\nTASKS:")
    for t in bpmn.get("tasks", {}).values():
        name = " ".join(t.get("name", "").split())
        doc = t.get("documentation", "")
        task_type = t.get("task_type", "")
        print(f"- {name} [{task_type}]")
        if doc:
            clean_doc = " ".join(str(doc).split())
            print("   doc:", clean_doc)


    gw = analyze_gateways(bpmn)

    print("\n\nPARALLEL GATEWAYS:")
    for gid, g in gw.items():
        print("-", g["name"])
        print("   role:", g["role"])
        print("   in :", g["in_degree"])
        print("   out:", g["out_degree"])


    print('\n\n Inspecting Pools:')
    pool_cf = build_controlflow_per_pool(bpmn)
    for pool_id in bpmn["pools"]:
        print_pool_controlflow(bpmn, pool_cf, pool_id)

    def reachable_from(pool_id, start):
        seen = set()
        q = deque([start])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            for v in pool_cf[pool_id]["succ"].get(u, []):
                if v not in seen:
                    q.append(v)
        return seen

    start_nodes = pool_cf[pool_id]["start_nodes"]
    start = start_nodes[0] if start_nodes else None
    reach = reachable_from(pool_id, start)

    print("\n\nREACHABLE FROM START:")
    for nid in reach:
        print("-", pretty_name(bpmn, nid))


    def print_pool_controlflow(bpmn: dict, pool_cf: dict, pool_id: str):
        print("\n POOL:", pool_cf[pool_id]["pool_name"], " - ", pool_id)

        print("Start nodes:")
        for n in pool_cf[pool_id]["start_nodes"]:
            print(" -", pretty_name(bpmn, n))

        print("End nodes:")
        for n in pool_cf[pool_id]["end_nodes"]:
            print(" -", pretty_name(bpmn, n))

        print("\nSEQUENCE EDGES:")
        for eid in pool_cf[pool_id]["sequence_edges"]:
            e = bpmn["edges"][eid]
            print(" -", pretty_name(bpmn, e["src"]), "->", pretty_name(bpmn, e["tgt"]))

        print("\nParallel blocks:")
        for blk in pool_cf[pool_id]["parallel_blocks"]:
            print("   Split:", pretty_name(bpmn, blk["split"]))
            print("   Join :", pretty_name(bpmn, blk["join"]))
            print("   Entries:")
            for x in blk["branch_entries"]:
                print("    -", pretty_name(bpmn, x))
            print("   Exits:")
            for x in blk["branch_exits"]:
                print("    -", pretty_name(bpmn, x))


    pool_cf = build_controlflow_per_pool(bpmn)

    for pool_id, pool_data in bpmn["pools"].items():
        print_pool_controlflow(bpmn, pool_cf, pool_id)

    msg_model = extract_message_interactions(bpmn)
    data_model = extract_data_associations(bpmn,data_node_types,flow_node_types)

    print_message_summary(msg_model)
    print_data_summary(data_model)