import os
# Set PYTHONHASHSEED for reproducible behavior across runs
os.environ['PYTHONHASHSEED'] = '0'

import glob
import json
import csv
from collections import Counter
from typing import Optional

import pandas as pd
from langdetect import LangDetectException, detect_langs

from bpmn_parser import extract_bpmn
from bpmn_into_sysml import sysml_code


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

    "Exclusive_Databased_Gateway",

    "ParallelGateway",
    "SequenceFlow",
    "ControlFlow",

    "InputData",
    "DataObject",
]


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

def is_english_with_confidence(text: str, min_prob: float = 0.8) -> bool:
    text = " ".join(str(text or "").split())
    if not text:
        return False
    try:
        langs = detect_langs(text)
        return bool(langs) and langs[0].lang == "en" and langs[0].prob > min_prob
    except LangDetectException:
        return False

def check_english(model: dict, bpmn: Optional[dict] = None) -> bool:
    """
    Detect English directly from BPMN task/documentation text.
    The `language` field from the raw model is intentionally ignored.
    """
    if bpmn is None:
        bpmn = extract_bpmn(model)

    text_parts = []
    for t in bpmn.get("tasks", {}).values():
        name = " ".join(str(t.get("name", "")).split())
        doc = " ".join(str(t.get("documentation", "")).split())
        text_parts.append(f"{name} {doc}")

    full_text = " ".join(text_parts).strip()
    if not full_text:
        return False

    try:
        return is_english_with_confidence(full_text, min_prob=0.8)
    except LangDetectException:
        return False


def run_bpmn_batch_transform_folder(
    csv_dir: str,
    json_col: str = "Model JSON",
    limit_per_file: Optional[int] = None,
    progress_every: int = 1000,
    bpmns_index_csv: str = "bpmns.csv",
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, "sapsam_sysmlv2.csv")

    headers = ["origin", "user_prompt", "sysml_model"]

    out_file = open(output_csv, mode="w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=headers)
    writer.writeheader()

    counts = Counter(
        {
            "total": 0,
            "bpmn": 0,
            "bpmn_english_detected": 0,
            "bpmn_english_total": 0,
            "bpmn_sysml_v2_only": 0,
            "ok": 0,
            "empty_json": 0,
            "json_parse_fail": 0,
            "non_bpmn": 0,
            "non_sysml_v2_stencil": 0,
            "non_english_detected": 0,
            "extract_fail": 0,
            "transform_fail": 0,
            "read_csv_fail": 0,
            "indexed_bpmns": 0,
            "index_row_oob": 0,
            "index_json_col_missing": 0,
        }
    )

    index_df = pd.read_csv(bpmns_index_csv)
    if "csv_name" not in index_df.columns or "row_index" not in index_df.columns:
        raise ValueError("bpmns.csv must contain columns: csv_name,row_index")

    selected_rows_by_csv = {}
    for _, r in index_df.iterrows():
        name = str(r["csv_name"])
        try:
            idx = int(r["row_index"])
        except Exception:
            continue
        if name not in selected_rows_by_csv:
            selected_rows_by_csv[name] = set()
        selected_rows_by_csv[name].add(idx)

    print(f"Loaded {len(index_df)} indexed BPMN rows from: {bpmns_index_csv}")

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    print(f"Found {len(csv_files)} csv file(s) in: {csv_dir}")

    total_rows = 0

    try:
        for csv_path in csv_files:
            csv_name = os.path.splitext(os.path.basename(csv_path))[0]
            csv_file_name = os.path.basename(csv_path)

            selected_rows = selected_rows_by_csv.get(csv_file_name)
            if selected_rows is None:
                selected_rows = selected_rows_by_csv.get(csv_name)
            if not selected_rows:
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                counts["read_csv_fail"] += 1
                continue

            if limit_per_file is not None:
                df = df.head(limit_per_file)

            print(f"Processing: {csv_path} ({len(df)} rows)")
            total_rows += len(df)

            if json_col not in df.columns:
                counts["index_json_col_missing"] += len(selected_rows)
                continue

            for i in range(len(df)):
                if i not in selected_rows:
                    continue

                counts["total"] += 1

                if progress_every > 0 and (i + 1) % progress_every == 0:
                    print(
                        "  processed", i + 1,
                        "OK", counts["ok"],
                        "non_english_detected", counts["non_english_detected"],
                        "extract_fail", counts["extract_fail"],
                        "transform_fail", counts["transform_fail"],
                    )

                raw = df.loc[i, json_col]

                if pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
                    counts["empty_json"] += 1
                    continue

                try:
                    model = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    counts["json_parse_fail"] += 1
                    continue

                try:
                    bpmn = extract_bpmn(model)
                except Exception:
                    counts["extract_fail"] += 1
                    continue

                # Legacy filters intentionally kept as comments.
                # The active behavior now trusts bpmns.csv indexing and does not re-filter.
                #
                # stencil_ids = collect_stencil_ids(model)
                # if "BPMNDiagram" not in stencil_ids:
                #     counts["non_bpmn"] += 1
                #     continue
                # counts["bpmn"] += 1
                #
                # if not check_english(model, bpmn):
                #     counts["non_english_detected"] += 1
                #     continue
                # counts["bpmn_english_detected"] += 1
                # counts["bpmn_english_total"] += 1
                #
                # ids_set = set(stencil_ids)
                # if not ids_set.issubset(set(sysml_v2_els)):
                #     counts["non_sysml_v2_stencil"] += 1
                #     continue
                # counts["bpmn_sysml_v2_only"] += 1

                try:
                    generated_sysml_code = sysml_code(bpmn)
                except Exception:
                    counts["transform_fail"] += 1
                    continue

                counts["ok"] += 1
                counts["indexed_bpmns"] += 1
                origin = f"sapsam::{csv_name}::{i}"

                writer.writerow(
                    {
                        "origin": origin,
                        "user_prompt": "",
                        "sysml_model": generated_sysml_code,
                    }
                )

            # Track indexed rows that do not exist in this CSV range.
            max_row = len(df) - 1
            counts["index_row_oob"] += sum(1 for idx in selected_rows if idx < 0 or idx > max_row)
    finally:
        out_file.close()

    print("Total rows:", counts["total"])
    print("BPMN:", counts["bpmn"])
    print(
        "BPMN en anglais (detect_langs):",
        counts["bpmn_english_total"],
        "(detecte:",
        counts["bpmn_english_detected"],
        ")",
    )
    print("BPMN avec uniquement sysml_v2_stencil:", counts["bpmn_sysml_v2_only"])
    print("OK:", counts["ok"])
    print("empty_json:", counts["empty_json"])
    print("json_parse_fail:", counts["json_parse_fail"])
    print("non_bpmn:", counts["non_bpmn"])
    print("non_sysml_v2_stencil:", counts["non_sysml_v2_stencil"])
    print("non_english_detected:", counts["non_english_detected"])
    print("extract_fail:", counts["extract_fail"])
    print("transform_fail:", counts["transform_fail"])
    print("read_csv_fail:", counts["read_csv_fail"])
    print("indexed_bpmns:", counts["indexed_bpmns"])
    print("index_row_oob:", counts["index_row_oob"])
    print("index_json_col_missing:", counts["index_json_col_missing"])
    print(
        "Sum:",
        counts["ok"]
        + counts["empty_json"]
        + counts["json_parse_fail"]
        + counts["non_bpmn"]
        + counts["non_sysml_v2_stencil"]
        + counts["non_english_detected"]
        + counts["extract_fail"]
        + counts["transform_fail"]
        + counts["read_csv_fail"],
        + counts["index_row_oob"]
        + counts["index_json_col_missing"]
    )
    print("Output CSV:", output_csv)
    print("BPMN index source CSV:", bpmns_index_csv)


if __name__ == "__main__":
    run_bpmn_batch_transform_folder(
        csv_dir=r"D:\sap_sam_2022\models",
        json_col="Model JSON",
        limit_per_file=None,
        progress_every=1000,
    )