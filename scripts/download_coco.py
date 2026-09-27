import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# ===========================
# Configuration
# ===========================

COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "datasets" / "downloads"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
COCO_DIR = RAW_DIR / "COCO"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
COCO_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DOWNLOAD_DIR / "train2017.zip"

# ===========================
# Download Function
# ===========================

def download_file(url, destination):

    if destination.exists():
        print("✅ COCO ZIP already exists.")
        return

    print("Downloading COCO Dataset...\n")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print("\n✅ Download Completed!")

# ===========================
# Extract Function
# ===========================

def extract_zip(zip_path, output_folder):

    image_folder = output_folder / "train2017"

    if image_folder.exists():
        print("✅ Dataset already extracted.")
        return

    print("\nExtracting ZIP...\n")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    print("✅ Extraction Completed!")

# ===========================
# Count Images
# ===========================

def count_images(folder):

    total = 0

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        total += len(list(folder.rglob(ext)))

    return total

# ===========================
# Main
# ===========================

def main():

    download_file(COCO_URL, ZIP_PATH)

    extract_zip(ZIP_PATH, COCO_DIR)

    image_folder = COCO_DIR / "train2017"

    total = count_images(image_folder)

    print("\n==========================")
    print("COCO Dataset Ready!")
    print("==========================")
    print(f"Images : {total}")
    print(f"Location: {image_folder}")

    # Uncomment this if you want to delete the ZIP
    # ZIP_PATH.unlink()

if __name__ == "__main__":
    main()