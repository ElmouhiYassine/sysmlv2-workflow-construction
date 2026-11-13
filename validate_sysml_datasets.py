"""Validate dataset SysML models with the Specialized_sysml_generator kernel logic."""

from __future__ import annotations

import argparse
import csv
import os
from queue import Empty
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parent
ENRICHED_DIR = ROOT / "enriched dataset" / "fixed"
DATASETS = {
    # "SOPBench": ENRICHED_DIR / "sopbench_nemotron_together.csv",
    # "WorFBench": ENRICHED_DIR / "worfbench_nemotron_together.csv",
    "SAP-SAM": ENRICHED_DIR / "sapsam_nemotron_together.csv",
}
OUTPUT_DIR = Path(
    os.environ.get("SYSML_VALIDATION_OUTPUT_DIR", ROOT / "validation_results_enriched")
)
KERNEL_DIR = (
    Path(os.environ["SYSML_KERNEL_DIR"])
    if os.environ.get("SYSML_KERNEL_DIR")
    else None
)
MOSAICO_PATH = Path(
    os.environ.get(
        "SYSML_MOSAICO_PATH",
        ROOT.parent / "Specialized_sysml_generator" / "library" / "MOSAICO.sysml",
    )
)
REQUIRED_COLUMN = "sysml_model"
CHECKPOINT_EVERY = 100
IOPUB_TIMEOUT_SECONDS = 300
csv.field_size_limit(sys.maxsize)


def find_sysml_kernel_dir(kernel_dir: Path | None = None) -> Path:
    if kernel_dir is None:
        ksm = KernelSpecManager()
    else:
        ksm = KernelSpecManager(kernel_dirs=[str(kernel_dir)])
    kernels = ksm.find_kernel_specs()
    if "sysml" not in kernels:
        searched = str(kernel_dir) if kernel_dir else "Jupyter's default paths"
        raise FileNotFoundError(
            f"No 'sysml' Jupyter kernel found in {searched}; "
            "set SYSML_KERNEL_DIR if it is installed elsewhere."
        )
    return Path(kernels["sysml"]).resolve()


def start_sysml_kernel(kernel_dir: Path | None = None):
    spec_dir = find_sysml_kernel_dir(kernel_dir)
    ksm = KernelSpecManager(kernel_dirs=[str(spec_dir.parent)])
    km = KernelManager(kernel_name="sysml", kernel_spec_manager=ksm)
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)
    return km, kc


def execute_and_collect(kc, code: str):
    msg_id = kc.execute(code)
    outputs = []
    jupyter_errors = []

    while True:
        try:
            msg = kc.get_iopub_msg(timeout=IOPUB_TIMEOUT_SECONDS)
        except Empty as exc:
            try:
                kc.interrupt_kernel()
            except Exception:
                pass
            raise TimeoutError(
                f"SysML kernel produced no output for "
                f"{IOPUB_TIMEOUT_SECONDS} seconds"
            ) from exc
        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "stream":
            text = content.get("text", "")
            if text:
                outputs.append(text)
        elif msg_type in ("display_data", "execute_result"):
            data = content.get("data", {})
            if "text/plain" in data:
                outputs.append(data["text/plain"])
        elif msg_type == "error":
            text = f"{content.get('ename', '')}: {content.get('evalue', '')}"
            traceback = content.get("traceback", [])
            if traceback:
                text += "\n" + "\n".join(traceback)
            jupyter_errors.append(text)
        elif (
            msg_type == "status"
            and content.get("execution_state") == "idle"
        ):
            break

    return outputs, jupyter_errors


def check_sysml(kc, sysml_model: str):
    outputs, jupyter_errors = execute_and_collect(kc, sysml_model)
    parser_output = "\n".join(outputs + jupyter_errors).strip()
    error_lines = [
        line.strip()
        for line in parser_output.splitlines()
        if line.strip().startswith("ERROR:")
    ]
    warning_lines = [
        line.strip()
        for line in parser_output.splitlines()
        if line.strip().startswith("WARNING:")
    ]
    valid = len(error_lines) == 0 and len(jupyter_errors) == 0
    return {
        "valid": valid,
        "errors": error_lines,
        "warnings": warning_lines,
        "output": parser_output,
        "jupyter_errors": jupyter_errors,
    }


def load_mosaico_library(kc, mosaico_path: Path):
    with mosaico_path.open("r", encoding="utf-8") as f:
        mosaico_code = f.read()
    outputs, jupyter_errors = execute_and_collect(kc, mosaico_code)
    parser_output = "\n".join(outputs + jupyter_errors).strip()
    error_lines = [
        line.strip()
        for line in parser_output.splitlines()
        if line.strip().startswith("ERROR:")
    ]
    if error_lines or jupyter_errors:
        raise RuntimeError(
            "MOSAICO.sysml could not be loaded.\n" + parser_output
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [
                name.strip("\ufeff\"' ") for name in reader.fieldnames
            ]
        if not reader.fieldnames or REQUIRED_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain the {REQUIRED_COLUMN!r} column; "
                f"found {reader.fieldnames!r}"
            )
        return list(reader)


def result_path(dataset: str) -> Path:
    return OUTPUT_DIR / f"{dataset.lower().replace('-', '_')}_validation.csv"


def load_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            row["workflow_id"]: row
            for row in csv.DictReader(f)
            if row.get("workflow_id")
        }


def write_results(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workflow_id",
        "origin",
        "sysml_model",
        "is_valid",
        "parser_error",
    ]
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_invalid_results(
    path: Path, rows: Iterable[dict[str, str]]
) -> None:
    write_results(path.with_name(path.stem.replace("_validation", "_invalid") + ".csv"),
                  (row for row in rows if row["is_valid"] == "False"))


def validate_dataset(
    name: str,
    path: Path,
    kc,
    limit: int | None = None,
) -> dict[str, float | int]:
    source_rows = read_rows(path)
    output = result_path(name)
    completed = load_completed(output)
    rows = source_rows if limit is None else source_rows[:limit]
    source_by_id = {source["origin"]: source for source in rows}
    results = {
        row["workflow_id"]: {
            **row,
            "origin": source_by_id[row["workflow_id"]]["origin"],
            "sysml_model": source_by_id[row["workflow_id"]][REQUIRED_COLUMN],
        }
        for row in completed.values()
        if row["workflow_id"] in {source["origin"] for source in rows}
    }
    valid = sum(row["is_valid"] == "True" for row in results.values())
    invalid = len(results) - valid

    for index, source in enumerate(rows, start=1):
        workflow_id = source["origin"]
        if workflow_id in results:
            continue
        try:
            result = check_sysml(kc, source[REQUIRED_COLUMN])
            parser_error = "\n".join(
                result["errors"] + result["jupyter_errors"]
            )
            is_valid = result["valid"]
        except TimeoutError as exc:
            print(
                f"[{name}] timeout at {index} / {len(rows)}: {workflow_id}",
                flush=True,
            )
            parser_error = str(exc)
            is_valid = False
        results[workflow_id] = {
            "workflow_id": workflow_id,
            "origin": workflow_id,
            "sysml_model": source[REQUIRED_COLUMN],
            "is_valid": str(is_valid),
            "parser_error": parser_error,
        }
        if is_valid:
            valid += 1
        else:
            invalid += 1

        if index % CHECKPOINT_EVERY == 0 or index == len(rows):
            ordered = [
                results[source_row["origin"]] for source_row in rows[:index]
            ]
            write_results(output, ordered)
            write_invalid_results(output, ordered)
            rate = valid / index * 100 if index else 0
            print(
                f"[{name}] {index} / {len(rows)} checked\n"
                f"valid={valid}\ninvalid={invalid}\nrate={rate:.2f}%",
                flush=True,
            )

    ordered = [results[source_row["origin"]] for source_row in rows]
    write_results(output, ordered)
    write_invalid_results(output, ordered)
    total = len(rows)
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "validity_rate": valid / total * 100 if total else 0.0,
    }


def write_summary(summary: dict[str, dict[str, float | int]]) -> None:
    path = OUTPUT_DIR / "validation_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "total", "valid", "invalid", "validity_rate"],
        )
        writer.writeheader()
        for dataset, values in summary.items():
            writer.writerow({"dataset": dataset, **values})


def main() -> int:
    global KERNEL_DIR, MOSAICO_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=KERNEL_DIR,
        help="Jupyter kernels directory containing the 'sysml' kernel.",
    )
    parser.add_argument(
        "--mosaico-path",
        type=Path,
        default=MOSAICO_PATH,
        help="Path to the MOSAICO.sysml standard library file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for resumable validation results.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate only the first five workflows in each dataset.",
    )
    args = parser.parse_args()

    KERNEL_DIR = args.kernel_dir
    MOSAICO_PATH = args.mosaico_path
    OUTPUT_DIR = args.output_dir

    for name, path in DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not MOSAICO_PATH.is_file():
        raise FileNotFoundError(MOSAICO_PATH)

    km, kc = start_sysml_kernel(KERNEL_DIR)
    try:
        load_mosaico_library(kc, MOSAICO_PATH)
        summary = {
            name: validate_dataset(
                name, path, kc, limit=5 if args.smoke_test else None
            )
            for name, path in DATASETS.items()
        }
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    total = sum(int(values["total"]) for values in summary.values())
    valid = sum(int(values["valid"]) for values in summary.values())
    overall = {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "validity_rate": valid / total * 100 if total else 0.0,
    }
    if not args.smoke_test:
        write_summary(summary)
    for name, values in (*summary.items(), ("Overall", overall)):
        print(
            f"{name}:\n"
            f"    total: {values['total']}\n"
            f"    valid: {values['valid']}\n"
            f"    invalid: {values['invalid']}\n"
            f"    validity rate: {values['validity_rate']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
