"""
extract_dataset.py
==================
Safely extracts downloaded flood dataset archive into dataset/ directory
without modifying or deleting the original downloaded files.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from src.data_inspection import run_inspection


def find_downloaded_zip(search_dirs=None) -> Path:
    """Find the downloaded flooding dataset zip archive."""
    if search_dirs is None:
        search_dirs = [
            Path(r"C:\Users\Acer\Downloads"),
            Path("dataset"),
            Path(".")
        ]

    candidates = ["flooding-image-dataset.zip", "archive.zip", "Flood Images.zip"]
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        # Direct name check
        for name in candidates:
            cand = s_dir / name
            if cand.exists() and cand.stat().st_size > 100_000_000:  # > 100MB
                return cand
        # Pattern check
        for f in s_dir.glob("*.zip"):
            if "flood" in f.name.lower() or "archive" in f.name.lower():
                if f.stat().st_size > 100_000_000:
                    return f

    return None


def extract_archive(zip_path: Path, target_dir: Path = Path("dataset")) -> bool:
    """Extract zip archive preserving original file."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} ({zip_path.stat().st_size / (1024**3):.2f} GB) to {target_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print("Extraction completed successfully.")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def main():
    zip_path = find_downloaded_zip()
    if zip_path is None:
        print("Dataset zip archive not found or still actively downloading.")
        return

    dest = Path("dataset")
    success = extract_archive(zip_path, dest)
    if success:
        print("Running automatic dataset inspection on extracted files...")
        run_inspection(dataset_dir="dataset", output_dir="reports")


if __name__ == "__main__":
    main()
