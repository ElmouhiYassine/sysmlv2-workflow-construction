import os
import zipfile
import requests
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7012043/files/sap_sam_2022.zip?download=1"

BASE_DIR = Path(__file__).resolve().parent / "sap_sam_2022"
ZIP_PATH = BASE_DIR / "sap_sam.zip"
EXTRACT_DIR = BASE_DIR / "temp_extract"
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SAP-SAM dataset...")

    response = requests.get(ZENODO_URL, stream=True)
    response.raise_for_status()

    with ZIP_PATH.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to: {ZIP_PATH}")

    print("Extracting ZIP...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"Extracted to: {EXTRACT_DIR}")

    print("Searching for CSV files...")

    csv_count = 0

    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".csv"):
                src = Path(root) / file
                dst = MODELS_DIR / file

                src.rename(dst)
                csv_count += 1

    print(f"Moved {csv_count} CSV files to {MODELS_DIR}")

    ZIP_PATH.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()