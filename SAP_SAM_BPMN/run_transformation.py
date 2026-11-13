import os
import argparse
from pathlib import Path
import glob
import json
import csv
from collections import Counter
from typing import Optional

import pandas as pd

if __package__:
    from .bpmn_parser import extract_bpmn
    from .bpmn_into_sysml import sysml_code
else:
    from bpmn_parser import extract_bpmn
    from bpmn_into_sysml import sysml_code

PROJECT_DIR = Path(__file__).resolve().parent


def run_bpmn_batch_transform_folder(
    csv_dir: str,
    json_col: str = "Model JSON",
    limit_per_file: Optional[int] = None,
    progress_every: int = 1000,
    bpmns_index_csv: str = "bpmns.csv",
    output_dir: str | Path = PROJECT_DIR / "sap_sam_sysmlv2",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "sapsam_sysmlv2.csv"

    headers = ["origin", "user_prompt", "sysml_model"]

    out_file = open(str(output_csv), mode="w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=headers)
    writer.writeheader()

    counts = Counter(
        {
            "total": 0,
            "ok": 0,
            "empty_json": 0,
            "json_parse_fail": 0,
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
    print("Successful transformations:", counts["ok"])
    print("Empty JSON:", counts["empty_json"])
    print("JSON parse failures:", counts["json_parse_fail"])
    print("BPMN extraction failures:", counts["extract_fail"])
    print("SysML transformation failures:", counts["transform_fail"])
    print("CSV read failures:", counts["read_csv_fail"])
    print("Rows missing from source CSV:", counts["index_row_oob"])
    print("Rows missing the JSON column:", counts["index_json_col_missing"])
    print("Output CSV:", output_csv)
    print("BPMN index source CSV:", bpmns_index_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform selected SAP-SAM BPMN models into SysML v2."
    )

    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=PROJECT_DIR / "sap_sam_2022" / "models",
        help="Path to the SAP-SAM models folder (default: sap_sam_2022/models).",
    )

    parser.add_argument(
        "--bpmns_index_csv",
        type=Path,
        default=PROJECT_DIR / "sap_sam_2022" / "bpmns.csv",
        help="CSV containing selected BPMN rows (default: sap_sam_2022/bpmns.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "sap_sam_sysmlv2",
        help="Directory for the generated SysML CSV.",
    )

    args = parser.parse_args()

    run_bpmn_batch_transform_folder(
        csv_dir=args.csv_dir,
        bpmns_index_csv=args.bpmns_index_csv,
        output_dir=args.output_dir,
    )