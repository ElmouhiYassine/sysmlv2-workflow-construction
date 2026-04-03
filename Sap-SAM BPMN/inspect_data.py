import os
# Set PYTHONHASHSEED for reproducible behavior across runs
os.environ['PYTHONHASHSEED'] = '0'

import glob
import json
import statistics
from collections import Counter
import pandas as pd
from bpmn_parser import extract_bpmn, build_controlflow_per_pool
from bpmn_into_sysml import sysml_code
from langdetect import LangDetectException, detect_langs


def iter_shapes(shape):

    yield shape
    for child in shape.get("childShapes", []) or []:
        yield from iter_shapes(child)


def collect_stencil_ids(model_dict):
    stencil_ids = []

    root_stencil = (model_dict.get("stencil") or {}).get("id")
    if root_stencil:
        stencil_ids.append(root_stencil)

    for top in model_dict.get("childShapes", []) or []:
        for shp in iter_shapes(top):
            sid = ((shp.get("stencil") or {}).get("id"))
            if sid:
                stencil_ids.append(sid)

    return stencil_ids


def scan_csv_bpmn_types(
        csv_path="10000.csv",
        json_col="Model JSON",
        limit=None
):

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    counts = Counter()
    per_model_unique = Counter()
    errors = []

    for i in range(len(df)):

        raw = df.loc[i, json_col]

        # If it's NaN or empty
        if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            errors.append((i, "Empty/NaN JSON cell"))
            continue

        try:
            model = json.loads(raw)
        except Exception as e:
            errors.append((i, f"json.loads failed: {e}"))
            continue

        try:
            stencil_ids = collect_stencil_ids(model)
            if "BPMNDiagram" not in stencil_ids:
                continue
            counts.update(stencil_ids)
            per_model_unique.update(set(stencil_ids))
        except Exception as e:
            errors.append((i, f"collect_stencil_ids failed: {e}"))
            continue

    return counts, per_model_unique, errors


def pretty_print_top(counter, top_n=70):
    for k, v in counter.most_common(top_n):
        print(f"{k:35s} {v}")


errors = []


def run_optional_quick_scan(csv_path="10000.csv", json_col="Model JSON", limit=None):
    # Run this helper only when the local demo CSV exists.
    if not os.path.exists(csv_path):
        return Counter(), Counter(), []
    return scan_csv_bpmn_types(csv_path=csv_path, json_col=json_col, limit=limit)

# print("\nMost frequent stencil types (by occurrences):")
# print('------'*10)
# pretty_print_top(counts)

# print("\nMost frequent stencil types per model (BPMN only):")
# print('------'*10)
# pretty_print_top(per_model_unique)
# print("\n\n\n")

# print(f"\nErrors: {len(errors)}")
# if errors:
#     print("First 10 errors:")
#     for e in errors[:10]:
#         print(e)


sysml_v2_els = [
    "BPMNDiagram",
    "Pool",
    "Lane",
    "CollapsedPool",

    "Task",
    "Function",
    "Subprocess",
    "CollapsedSubprocess",

    "StartNoneEvent",

    "EndNoneEvent",

    "Exclusive_Databased_Gateway", # this is decision aka xor
    "ParallelGateway",
    # "InclusiveGateway",
    # "EventbasedGateway",
    # "Decision", # we dont have in BPMNs

    "SequenceFlow",
    "ControlFlow",

    "InputData",
    "DataObject"
]
#776 diagrams total


def _is_english_with_confidence(text: str, min_prob: float = 0.8) -> bool:
    text = " ".join(str(text or "").split())
    if not text:
        return False
    try:
        langs = detect_langs(text)
        return bool(langs) and langs[0].lang == "en" and langs[0].prob > min_prob
    except LangDetectException:
        return False


def _passes_english_filter(model: dict, bpmn: dict) -> bool:
    # Ignore the model language metadata and detect language directly from BPMN text.
    text_parts = []
    for t in bpmn.get("tasks", {}).values():
        name = " ".join(str(t.get("name", "")).split())
        doc = " ".join(str(t.get("documentation", "")).split())
        text_parts.append(f"{name} {doc}")
    full_text = " ".join(text_parts).strip()
    return _is_english_with_confidence(full_text, min_prob=0.8)


def _summarize_counts(values):
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


def _count_model_flow_elements(bpmn: dict):
    # Actions map to task-like BPMN elements extracted in bpmn["tasks"].
    action_count = len(bpmn.get("tasks", {}))

    pool_cf = build_controlflow_per_pool(bpmn)

    decision_count = 0
    fork_count = 0
    join_count = 0

    for pool_data in pool_cf.values():
        decision_blocks = pool_data.get("decision_blocks", {})
        if isinstance(decision_blocks, dict):
            decision_count += len(decision_blocks)
        else:
            decision_count += len(decision_blocks or [])

        parallel_blocks = pool_data.get("parallel_blocks", [])
        fork_count += sum(1 for blk in parallel_blocks if blk.get("split"))
        join_count += sum(1 for blk in parallel_blocks if blk.get("join"))

    return action_count, decision_count, join_count, fork_count


def metrics_for_valid_bpmn(
    csv_dir="D:\\sap_sam_2022\\models",
    json_col="Model JSON",
    limit_per_file=None,
    progress_every=2000,
):
    """
    Compute metrics on valid BPMN only.
    Valid BPMN = BPMNDiagram + only sysml_v2_els + English metadata/detected.
    """
    action_counts = []
    decision_counts = []
    join_counts = []
    fork_counts = []

    valid_bpmn = 0
    ok_sysml_bpmn = 0
    total_rows = 0
    sysml_generation_fail = 0

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if limit_per_file is not None:
            df = df.head(limit_per_file)

        for i in range(len(df)):
            total_rows += 1
            if progress_every > 0 and total_rows % progress_every == 0:
                print(f"processed={total_rows}, valid_bpmn={valid_bpmn}")

            raw = df.loc[i, json_col]
            if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
                continue

            try:
                model = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue

            try:
                stencil_ids = collect_stencil_ids(model)
            except Exception:
                continue

            if "BPMNDiagram" not in stencil_ids:
                continue

            try:
                bpmn = extract_bpmn(model)
            except Exception:
                continue

            # Step 2: detect english from BPMN text only.
            if not _passes_english_filter(model, bpmn):
                continue

            # Step 3: keep only diagrams with allowed SysML-v2-related BPMN stencils.
            if any(sid not in sysml_v2_els for sid in stencil_ids):
                continue

            # Step 4: validate transformation by generating SysML.
            try:
                _ = sysml_code(bpmn)
            except Exception:
                sysml_generation_fail += 1
                continue

            ok_sysml_bpmn += 1

            actions, decisions, joins, forks = _count_model_flow_elements(bpmn)

            action_counts.append(actions)
            decision_counts.append(decisions)
            join_counts.append(joins)
            fork_counts.append(forks)
            valid_bpmn += 1

    total_actions = sum(action_counts)
    total_decisions = sum(decision_counts)
    total_joins = sum(join_counts)
    total_forks = sum(fork_counts)

    all_flow_items = total_actions + total_decisions + total_joins + total_forks

    model_frequencies = {
        "actions": (sum(1 for x in action_counts if x > 0) / ok_sysml_bpmn) if ok_sysml_bpmn else 0.0,
        "decisions": (sum(1 for x in decision_counts if x > 0) / ok_sysml_bpmn) if ok_sysml_bpmn else 0.0,
        "joins": (sum(1 for x in join_counts if x > 0) / ok_sysml_bpmn) if ok_sysml_bpmn else 0.0,
        "forks": (sum(1 for x in fork_counts if x > 0) / ok_sysml_bpmn) if ok_sysml_bpmn else 0.0,
    }

    total_frequencies = {
        "actions": (total_actions / all_flow_items) if all_flow_items else 0.0,
        "decisions": (total_decisions / all_flow_items) if all_flow_items else 0.0,
        "joins": (total_joins / all_flow_items) if all_flow_items else 0.0,
        "forks": (total_forks / all_flow_items) if all_flow_items else 0.0,
    }

    return {
        "total_rows": total_rows,
        "valid_bpmn": valid_bpmn,
        "ok_sysml_bpmn": ok_sysml_bpmn,
        "sysml_generation_fail": sysml_generation_fail,
        "actions": {
            "per_model": action_counts,
            "stats": _summarize_counts(action_counts),
        },
        "decisions": {
            "per_model": decision_counts,
            "stats": _summarize_counts(decision_counts),
        },
        "joins": {
            "per_model": join_counts,
            "stats": _summarize_counts(join_counts),
        },
        "forks": {
            "per_model": fork_counts,
            "stats": _summarize_counts(fork_counts),
        },
        "totals": {
            "actions": total_actions,
            "decisions": total_decisions,
            "joins": total_joins,
            "forks": total_forks,
            "all_flow_items": all_flow_items,
        },
        "frequencies": {
            "by_model": model_frequencies,
            "global": total_frequencies,
        },
    }


def _load_bpmn_index(bpmns_csv_path: str):
    if not os.path.exists(bpmns_csv_path):
        raise FileNotFoundError(f"Index file not found: {bpmns_csv_path}")

    index_df = pd.read_csv(bpmns_csv_path)
    required = {"csv_name", "row_index"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in bpmns.csv: {sorted(missing)}")

    return index_df


def metrics_for_indexed_bpmn(
    csv_dir="D:\\sap_sam_2022\\models",
    bpmns_csv_path="bpmns.csv",
    json_col="Model JSON",
    progress_every=2000,
):
    """
    Compute metrics only on BPMNs listed in bpmns.csv.
    The index is expected to be produced by run_transformation.py after filters:
    BPMN -> English (langdetect) -> SysML stencil set -> successful transform.
    """
    action_counts = []
    decision_counts = []
    join_counts = []
    fork_counts = []

    processed_index_rows = 0
    total_index_rows = 0
    index_csv_missing = 0
    index_row_oob = 0
    read_csv_fail = 0
    empty_json = 0
    json_parse_fail = 0
    extract_fail = 0

    index_df = _load_bpmn_index(bpmns_csv_path)
    total_index_rows = len(index_df)

    # Process by source CSV to avoid loading all model files in memory at once.
    grouped_rows = {}
    for k in range(total_index_rows):
        csv_name = str(index_df.loc[k, "csv_name"])
        grouped_rows.setdefault(csv_name, []).append((k, index_df.loc[k, "row_index"]))

    for csv_name, refs in grouped_rows.items():
        csv_path = os.path.join(csv_dir, csv_name)
        if not os.path.exists(csv_path):
            index_csv_missing += len(refs)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            read_csv_fail += len(refs)
            continue

        for k, row_index_raw in refs:
            if progress_every > 0 and (k + 1) % progress_every == 0:
                print(f"indexed_processed={k + 1}/{total_index_rows}")

            try:
                row_index = int(row_index_raw)
            except Exception:
                index_row_oob += 1
                continue

            if row_index < 0 or row_index >= len(df):
                index_row_oob += 1
                continue

            raw = df.loc[row_index, json_col]
            if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
                empty_json += 1
                continue

            try:
                model = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                json_parse_fail += 1
                continue

            try:
                bpmn = extract_bpmn(model)
            except Exception:
                extract_fail += 1
                continue

            actions, decisions, joins, forks = _count_model_flow_elements(bpmn)
            action_counts.append(actions)
            decision_counts.append(decisions)
            join_counts.append(joins)
            fork_counts.append(forks)
            processed_index_rows += 1

    total_actions = sum(action_counts)
    total_decisions = sum(decision_counts)
    total_joins = sum(join_counts)
    total_forks = sum(fork_counts)
    all_flow_items = total_actions + total_decisions + total_joins + total_forks

    model_frequencies = {
        "actions": (sum(1 for x in action_counts if x > 0) / processed_index_rows) if processed_index_rows else 0.0,
        "decisions": (sum(1 for x in decision_counts if x > 0) / processed_index_rows) if processed_index_rows else 0.0,
        "joins": (sum(1 for x in join_counts if x > 0) / processed_index_rows) if processed_index_rows else 0.0,
        "forks": (sum(1 for x in fork_counts if x > 0) / processed_index_rows) if processed_index_rows else 0.0,
    }

    total_frequencies = {
        "actions": (total_actions / all_flow_items) if all_flow_items else 0.0,
        "decisions": (total_decisions / all_flow_items) if all_flow_items else 0.0,
        "joins": (total_joins / all_flow_items) if all_flow_items else 0.0,
        "forks": (total_forks / all_flow_items) if all_flow_items else 0.0,
    }

    return {
        "total_rows": total_index_rows,
        "valid_bpmn": processed_index_rows,
        "ok_sysml_bpmn": processed_index_rows,
        "sysml_generation_fail": 0,
        "actions": {
            "per_model": action_counts,
            "stats": _summarize_counts(action_counts),
        },
        "decisions": {
            "per_model": decision_counts,
            "stats": _summarize_counts(decision_counts),
        },
        "joins": {
            "per_model": join_counts,
            "stats": _summarize_counts(join_counts),
        },
        "forks": {
            "per_model": fork_counts,
            "stats": _summarize_counts(fork_counts),
        },
        "totals": {
            "actions": total_actions,
            "decisions": total_decisions,
            "joins": total_joins,
            "forks": total_forks,
            "all_flow_items": all_flow_items,
        },
        "frequencies": {
            "by_model": model_frequencies,
            "global": total_frequencies,
        },
        "index_issues": {
            "csv_missing": index_csv_missing,
            "row_out_of_bounds_or_invalid": index_row_oob,
            "read_csv_fail": read_csv_fail,
            "empty_json": empty_json,
            "json_parse_fail": json_parse_fail,
            "extract_fail": extract_fail,
        },
    }


def print_valid_bpmn_metrics(report: dict):
    print("\n=== Valid BPMN metrics ===")
    print("Total rows:", report["total_rows"])
    print("Valid BPMN:", report["valid_bpmn"])
    print("OK BPMN (passed SysML generation):", report["ok_sysml_bpmn"])
    print("SysML generation fail (filtered out):", report["sysml_generation_fail"])

    for key in ["actions", "decisions", "joins", "forks"]:
        stats = report[key]["stats"]
        print(
            f"{key}: sum={stats['sum']} mean={stats['mean']:.3f} "
            f"min={stats['min']} max={stats['max']} var={stats['variance']:.3f}"
        )

    print("frequencies by model:", report["frequencies"]["by_model"])
    print("frequencies global:", report["frequencies"]["global"])
    if "index_issues" in report:
        print("index_issues:", report["index_issues"])

def check_model(bpmn):
    text_parts = []

    for t in bpmn.get("tasks", {}).values():
        name = " ".join(str(t.get("name", "")).split())
        doc = " ".join(str(t.get("documentation", "")).split())
        text_parts.append(f"{name} {doc}")

    full_text = " ".join(text_parts).strip()

    try:
        langs = detect_langs(full_text)
        if langs and langs[0].lang == "en" and langs[0].prob >= 0.8:
            return True
    except LangDetectException:
        pass

    return False

def count_diagrams(csv_path="D:\\sap_sam_2022\\models\\30000.csv",
                   json_col="Model JSON"):
    df = pd.read_csv(csv_path)

    count_ = 0
    errors = []

    for i in range(len(df)):
        raw = df.loc[i, json_col]

        if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            errors.append((i, "Empty/NaN JSON cell"))
            continue

        try:
            model = json.loads(raw)
        except Exception as e:
            errors.append((i, f"json.loads failed: {e}"))
            continue

        try:
            stencil_ids = collect_stencil_ids(model)


            if "BPMNDiagram" not in stencil_ids:
                continue

            try:
                bpmn = extract_bpmn(model)
            except Exception as e:
                errors.append((i, f"extract_bpmn failed: {e}"))
                continue

            # Always use language detection from BPMN textual content.
            if not check_model(bpmn):
                continue

            ok = True
            for j in stencil_ids:
                if j not in sysml_v2_els:
                    ok = False
                    break

            if ok:
                count_ += 1

        except Exception as e:
            errors.append((i, f"processing failed: {e}"))
            continue

    return count_, errors



def count_diagrams_all():
    count_all = 0

    for csv_path in glob.glob("D:\\sap_sam_2022\\models\\*.csv") :
        print(count_all)
        p, e = count_diagrams(csv_path)
        count_all += p

    return count_all

# print(count_diagrams_all())

#14093

def scan_csv_languages_from_json(
        csv_path="10000.csv",
        json_col="Model JSON",
        lang_key="language",
        limit=None
):

    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)

    counts = Counter()
    per_model_unique = Counter()
    errors = []

    for i in range(len(df)):

        raw = df.loc[i, json_col]

        # empty cell
        if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            errors.append((i, "Empty/NaN JSON cell"))
            continue

        try:
            model = json.loads(raw)
        except Exception as e:
            errors.append((i, f"json.loads failed: {e}"))
            continue

        try:
            stencil_ids = collect_stencil_ids(model)
            if "BPMNDiagram" not in stencil_ids:
                continue

            try:
                bpmn = extract_bpmn(model)
            except Exception as e:
                errors.append((i, f"extract_bpmn failed: {e}"))
                continue

            text_parts = []
            for t in bpmn.get("tasks", {}).values():
                name = " ".join(str(t.get("name", "")).split())
                doc = " ".join(str(t.get("documentation", "")).split())
                text_parts.append(f"{name} {doc}")

            full_text = " ".join(text_parts).strip()
            if not full_text:
                detected_lang = "unknown"
            else:
                try:
                    langs = detect_langs(full_text)
                    detected_lang = langs[0].lang if langs else "unknown"
                except LangDetectException:
                    detected_lang = "unknown"

            counts.update([detected_lang])
            per_model_unique.update({detected_lang})

        except Exception as e:
            errors.append((i, f"language extraction failed: {e}"))
            continue
    for k, v in counts.most_common(10):
        print(f"{k:35s} {v}")

    return counts, per_model_unique, errors

# print(count_diagrams())
#776 total


def count_bpmn_ids(
        csv_path="D:\\sap_sam_2022\\models\\30000.csv",
        json_col="Model JSON",
):

    df = pd.read_csv(csv_path)

    counts = Counter()
    errors = []

    for i in range(len(df)):

        raw = df.loc[i, json_col]

        if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            errors.append((i, "Empty/NaN JSON cell"))
            continue

        try:
            model = json.loads(raw)
        except Exception as e:
            errors.append((i, f"json.loads failed: {e}"))
            continue

        try:
            stencil_ids = collect_stencil_ids(model)

            # we keep only BPMN diagrams
            if "BPMNDiagram" not in stencil_ids:
                continue

            counts.update(stencil_ids)

        except Exception as e:
            errors.append((i, f"collect_stencil_ids failed: {e}"))
            continue

    return counts, errors

# print("Languages (for 1 csv file):")
# print('------'*10)

# scan_csv_languages_from_json()


def check_english(
        csv_path="D:\\sap_sam_2022\\models\\30000.csv",
        json_col="Model JSON",
):

    df = pd.read_csv(csv_path)

    errors = []
    count_it = 0
    for i in range(len(df)):

        raw = df.loc[i, json_col]

        if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            errors.append((i, "Empty/NaN JSON cell"))
            continue

        try:
            model = json.loads(raw)
        except Exception as e:
            errors.append((i, f"json.loads failed: {e}"))
            continue

        try:
            stencil_ids = collect_stencil_ids(model)
            if "BPMNDiagram" not in stencil_ids:
                continue

            bpmn = extract_bpmn(model)
            if not check_model(bpmn):
                continue

            ok = True
            for j in stencil_ids:
                if j not in sysml_v2_els:
                    ok = False
                    break

            if ok:
                count_it += 1


        except Exception as e:
            errors.append((i, f"collect_stencil_ids failed: {e}"))
            continue
    print('BPMN models (with respect to the filtered set) detected as english :', count_it)
    return count_it

# check_english()


def check_cross_pool_connections(bpmn: dict) -> dict:
    """
    Detects sequence flows and tasks that cross pool boundaries.
    Returns a report dict with all violations found.
    """
    shape_index = bpmn.get("shape_index", {})
    edges = bpmn.get("edges", {})
    tasks = bpmn.get("tasks", {})

    all_pools = {}
    all_pools.update(bpmn.get("pools", {}))
    all_pools.update(bpmn.get("collapsed_pools", {}))

    issues = {
        "cross_pool_sequence_flows": [],   # SequenceFlow crossing pools
        "cross_pool_message_flows": [],    # MessageFlow crossing pools (expected but worth logging)
        "unknown_pool_endpoints": [],      # edges where src or tgt pool can't be resolved
        "tasks_with_no_pool": [],          # tasks not attached to any pool
    }

    for eid, e in edges.items():
        src, tgt = e.get("src"), e.get("tgt")
        edge_type = e.get("type", "")

        if not src or not tgt:
            continue

        src_pool = shape_index.get(src, {}).get("pool")
        tgt_pool = shape_index.get(tgt, {}).get("pool")

        if src_pool is None or tgt_pool is None:
            issues["unknown_pool_endpoints"].append({
                "edge_id": eid,
                "edge_type": edge_type,
                "src": src,
                "src_name": shape_index.get(src, {}).get("name", "?"),
                "src_pool": src_pool,
                "tgt": tgt,
                "tgt_name": shape_index.get(tgt, {}).get("name", "?"),
                "tgt_pool": tgt_pool,
            })
            continue

        if src_pool == tgt_pool:
            continue  # same pool, fine

        entry = {
            "edge_id": eid,
            "edge_type": edge_type,
            "src": src,
            "src_name": shape_index.get(src, {}).get("name", "?"),
            "src_pool": src_pool,
            "src_pool_name": all_pools.get(src_pool, {}).get("name", src_pool),
            "tgt": tgt,
            "tgt_name": shape_index.get(tgt, {}).get("name", "?"),
            "tgt_pool": tgt_pool,
            "tgt_pool_name": all_pools.get(tgt_pool, {}).get("name", tgt_pool),
        }

        if edge_type == "SequenceFlow":
            issues["cross_pool_sequence_flows"].append(entry)
        elif edge_type == "MessageFlow":
            issues["cross_pool_message_flows"].append(entry)

    for tid, t in tasks.items():
        if not t.get("pool") or t["pool"] not in all_pools:
            issues["tasks_with_no_pool"].append({
                "task_id": tid,
                "task_name": t.get("name", "?"),
                "task_type": t.get("task_type", "?"),
                "pool": t.get("pool"),
            })

    return issues


def print_cross_pool_report(report: dict):
    seq = report["cross_pool_sequence_flows"]
    msg = report["cross_pool_message_flows"]
    unk = report["unknown_pool_endpoints"]
    nop = report["tasks_with_no_pool"]

    # print(f"\n=== Cross-pool connection report ===")
    if seq:
        print(f"\n[ERRORS] SequenceFlow crossing pools ({len(seq)}):")
        for e in seq:
            print(f"  {e['edge_id']}: {e['src_name']} [{e['src_pool_name']}]"
                  f" → {e['tgt_name']} [{e['tgt_pool_name']}]")
    # if not seq:
    #     print("  none")

    if msg:
        print(f"\n[INFO] MessageFlow crossing pools ({len(msg)}):")
        for e in msg:
            print(f"  {e['edge_id']}: {e['src_name']} [{e['src_pool_name']}]"
                  f" → {e['tgt_name']} [{e['tgt_pool_name']}]")
    # if not msg:
    #     print("  none")
    if unk:
        print(f"\n[WARN] Edges with unresolved pool ({len(unk)}):")
        for e in unk:
            print(f"  {e['edge_id']} [{e['edge_type']}]: "
                  f"src_pool={e['src_pool']} tgt_pool={e['tgt_pool']}")
    # if not unk:
    #     print("  none")
    if nop:
        print(f"\n[WARN] Tasks with no pool ({len(nop)}):")
        for t in nop:
            print(f"  {t['task_id']}: {t['task_name']} [{t['task_type']}]")
    # if not nop:
    #     print("  none")

def main():
    # Optional local scan: only runs if 10000.csv exists in the current directory.
    # run_optional_quick_scan(csv_path="10000.csv", json_col="Model JSON", limit=None)
    #
    # for csv_path in glob.glob("D:\\sap_sam_2022\\models\\*.csv"):
    #     df = pd.read_csv(csv_path)
    #
    #     print(f"--------------{csv_path}-----------------------")
    #
    #     for i in range(len(df)):
    #         if (i + 1) % 1000 == 0:
    #             print(f"processed {i + 1}/{len(df)} in {os.path.basename(csv_path)}")
    #
    #         raw = df.loc[i, "Model JSON"]
    #
    #         if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
    #             errors.append((i, "Empty/NaN JSON cell"))
    #             continue
    #
    #         try:
    #             model = json.loads(raw)
    #         except Exception as e:
    #             errors.append((i, f"json.loads failed: {e}"))
    #             continue
    #
    #         try:
    #             stencil_ids = collect_stencil_ids(model)
    #
    #             # we keep only BPMN diagrams
    #             if "BPMNDiagram" not in stencil_ids:
    #                 continue
    #
    #             bpmn = extract_bpmn(model)
    #             report = check_cross_pool_connections(bpmn)
    #             # print_cross_pool_report(report)
    #
    #         except Exception as e:
    #             errors.append((i, f"error: {e}"))
    #             continue

    report = metrics_for_indexed_bpmn(
        csv_dir="D:\\sap_sam_2022\\models",
        bpmns_csv_path="bpmns.csv",
        json_col="Model JSON",
        progress_every=2000,
    )
    print_valid_bpmn_metrics(report)


if __name__ == "__main__":
    main()

