"""Validate the sysml_model column of one CSV with the project's kernel validator (no resume, small files)."""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import validate_sysml_datasets as v


def validate_file(path: Path, out: Path) -> tuple[int, int]:
    rows = v.read_rows(path)
    km, kc = v.start_sysml_kernel()
    results = []
    try:
        v.load_mosaico_library(kc, v.MOSAICO_PATH)
        for i, r in enumerate(rows, 1):
            try:
                res = v.check_sysml(kc, r["sysml_model"])
                ok, err = res["valid"], "\n".join(res["errors"] + res["jupyter_errors"])
            except TimeoutError as exc:
                ok, err = False, str(exc)
            results.append({"origin": r["origin"], "is_valid": str(ok), "parser_error": err})
            if i % 100 == 0:
                print(f"{i}/{len(rows)} invalid={sum(x['is_valid']=='False' for x in results)}", flush=True)
    finally:
        kc.stop_channels(); km.shutdown_kernel(now=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["origin", "is_valid", "parser_error"]); w.writeheader(); w.writerows(results)
    bad = sum(x["is_valid"] == "False" for x in results)
    return len(results) - bad, bad


if __name__ == "__main__":
    ok, bad = validate_file(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"valid={ok} invalid={bad}")
