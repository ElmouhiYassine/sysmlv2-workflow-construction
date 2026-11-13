"""Prepare the WorFBench software-engineering subset for manual annotation."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "worfbench_software_engineering.csv"
    destination = ROOT / "worfbench_se_test.csv"
    data = pd.read_csv(source)
    data["detailed_description"] = ""
    data = data[
        ["origin", "user_prompt", "detailed_description", "sysml_model"]
    ]
    data.to_csv(destination, index=False)
    print(f"Saved {len(data)} workflows to {destination}")


if __name__ == "__main__":
    main()
