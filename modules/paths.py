"""
============================================================
PhotoGuard AI
Project Path Definitions & Directory Setup

Author : Vinay Kirithic

Description:
Robust path resolution supporting Colab Drive mounts and local environments.
Robustly searches drive and local filesystem without relying on strict casing.
============================================================
"""

import os
from pathlib import Path

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_dataset_dir() -> Path:
    """Finds the dataset directory in Colab Drive or Local workspace."""
    # 1. Direct Colab Drive check
    colab_default = Path("/content/drive/MyDrive/PhotoGuard/datasets")
    if colab_default.exists():
        return colab_default

    # 2. Case-insensitive Search under /content/drive/MyDrive
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        for item in drive_root.rglob("*"):
            if item.is_dir() and item.name.lower() == "raw" and any(p.name.lower() == "photoguard" for p in item.parents):
                return item.parent

    # 3. Fallback to Local repository dataset folder
    return BASE_DIR / "datasets"

DATASET_DIR = resolve_dataset_dir()

RAW_DIR = DATASET_DIR / "raw"
if not RAW_DIR.exists():
    # If folder is capitalized as 'Raw' or 'RAW'
    for candidate in DATASET_DIR.iterdir() if DATASET_DIR.exists() else []:
        if candidate.is_dir() and candidate.name.lower() == "raw":
            RAW_DIR = candidate
            break

PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
BALANCED_DIR = DATASET_DIR / "balanced"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

METADATA_DIR = BASE_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"

def ensure_directories_exist():
    """Creates output folders safely."""
    for directory in [
        PREPROCESSED_DIR, BALANCED_DIR, TRAIN_DIR, VALIDATION_DIR,
        TEST_DIR, METADATA_DIR, MODELS_DIR, OUTPUT_DIR,
        PREDICTION_DIR, GRADCAM_DIR, LOG_DIR, METRICS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories_exist()

if __name__ == "__main__":
    print("=" * 60)
    print("PhotoGuard AI - Path Configuration")
    print("=" * 60)
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Dataset Directory    : {DATASET_DIR}")
    print(f"Raw Images Directory : {RAW_DIR}")
    print(f"RAW_DIR Exists?      : {RAW_DIR.exists()}")
    if RAW_DIR.exists():
        subfolders = [f.name for f in RAW_DIR.iterdir() if f.is_dir()]
        print(f"RAW Subfolders       : {subfolders}")
    print("=" * 60)