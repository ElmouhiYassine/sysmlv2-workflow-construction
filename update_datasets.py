import argparse
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_COLUMNS = ["origin", "user_prompt", "sysml_model"]


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_dataset(path):
    """
    Load a dataset that already contains:
    origin, user_prompt, sysml_model
    """
    df = pd.read_csv(path)

    missing_columns = [
        col for col in OUTPUT_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {missing_columns}"
        )

    df = df[OUTPUT_COLUMNS].copy()

    for col in OUTPUT_COLUMNS:
        df[col] = df[col].apply(clean_text)

    return df


def print_dataset_summary(name, df):
    print(f"\n{name}")
    print("-" * 80)
    print("Rows:", len(df))
    print("Empty prompts:", (df["user_prompt"].str.len() == 0).sum())
    print("Empty SysML:", (df["sysml_model"].str.len() == 0).sum())
    print(
        "Unique prompt+code:",
        df.drop_duplicates(
            subset=["user_prompt", "sysml_model"]
        ).shape[0]
    )

    print("\nOrigins:")
    print(df["origin"].head())


def unify_datasets(
    sopbench_path,
    worfbench_path,
    sapsam_path,
    output_path,
):
    # Load datasets
    sopbench = load_dataset(sopbench_path)
    worfbench = load_dataset(worfbench_path)
    sapsam = load_dataset(sapsam_path)

    # Debug individual datasets
    print_dataset_summary("SOPBench", sopbench)
    print_dataset_summary("WorFBench", worfbench)
    print_dataset_summary("SAP-SAM", sapsam)

    # Concatenate all datasets
    unified = pd.concat(
        [
            sopbench,
            worfbench,
            sapsam,
        ],
        ignore_index=True,
    )

    print("\n" + "=" * 80)
    print("BEFORE CLEANING")
    print("=" * 80)
    print("Total rows:", len(unified))

    # Remove rows without prompt or SysML
    unified = unified[
        (unified["user_prompt"].str.len() > 0)
        &
        (unified["sysml_model"].str.len() > 0)
    ].copy()

    print("After removing empty rows:", len(unified))

    # Remove exact duplicate prompt + SysML pairs
    unified = unified.drop_duplicates(
        subset=["user_prompt", "sysml_model"]
    ).reset_index(drop=True)

    print("After removing duplicates:", len(unified))

    # Dataset statistics
    print("\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print("Total rows:", len(unified))

    print("\nRows by dataset:")
    print(
        unified["origin"]
        .str.split("::")
        .str[0]
        .value_counts()
    )

    print("\nExample origins:")
    for dataset in ["sopbench", "worfbench", "sapsam"]:
        examples = unified[
            unified["origin"].str.startswith(dataset + "::")
        ]["origin"].head(3)

        print(f"\n{dataset}:")
        for origin in examples:
            print(" ", origin)

    # Save final unified dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unified.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(f"Saved unified dataset to:")
    print(output_path)

    return unified


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine source-specific SysML v2 workflow CSVs."
    )
    parser.add_argument(
        "--sopbench",
        type=Path,
        default=ROOT / "SOPBench" / "sopbench_sysmlv2.csv",
    )
    parser.add_argument(
        "--worfbench",
        type=Path,
        default=ROOT / "WorfBench" / "worfbench_dataset" / "worfbench_sysmlv2.csv",
    )
    parser.add_argument(
        "--sapsam",
        type=Path,
        default=ROOT / "SAP_SAM_BPMN" / "sap_sam_sysmlv2" / "sapsam_sysmlv2.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "unified" / "unified_sysmlv2.csv",
    )
    args = parser.parse_args(argv)
    unified = unify_datasets(
        sopbench_path=args.sopbench,
        worfbench_path=args.worfbench,
        sapsam_path=args.sapsam,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())