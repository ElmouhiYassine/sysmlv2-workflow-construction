import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(PROJECT_DIR / "sap_sam_2022" / "models" / "40000.csv")

    input_path = PROJECT_DIR / "model.json"
    output_path = PROJECT_DIR / "bpmn_pretty.json"
    with input_path.open("w", encoding="utf-8") as f:
        f.write(df.loc[0, "Model JSON"])

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Wrote:", output_path)


if __name__ == "__main__":
    main()