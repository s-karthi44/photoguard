"""
============================================================
PhotoGuard AI
Image Preprocessing Module

Author : Vinay Kirithic

Description:
Scans raw dataset directories, validates images, removes corrupted files,
resizes images to 224x224, converts all to standard RGB JPEG format,
logs metadata, and handles balanced dataset extraction.
============================================================
"""

import time
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from modules.paths import (
    RAW_DIR,
    PREPROCESSED_DIR,
    METADATA_DIR,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
IMAGE_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

METADATA_FILE = METADATA_DIR / "metadata.csv"
LOG_FILE = METADATA_DIR / "preprocessing_log.txt"

# ==========================================================
# PREPROCESSING PIPELINE CLASS
# ==========================================================
class ImagePreprocessor:
    def __init__(self, raw_dir: Path = RAW_DIR, output_dir: Path = PREPROCESSED_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.metadata = []
        self.processed_count = 0
        self.corrupted_count = 0
        self.dataset_stats = {}

    def get_all_images(self):
        """Scans raw folder for supported image formats."""
        if not self.raw_dir.exists():
            return []
        return [
            file for file in self.raw_dir.rglob("*")
            if file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def process_single_image(self, image_path: Path):
        """Validates, converts to RGB, resizes, and saves single image."""
        try:
            with Image.open(image_path) as img:
                orig_width, orig_height = img.size
                orig_format = img.format

                # Convert to standard RGB format
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

                # Maintain directory relative structure
                relative_path = image_path.relative_to(self.raw_dir)
                save_path = self.output_dir / relative_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path = save_path.with_suffix(".jpg")

                # Save as optimized JPEG
                img.save(save_path, "JPEG", quality=95, optimize=True)

                parts = relative_path.parts
                label = parts[0] if len(parts) > 0 else "Unknown"
                dataset_name = parts[1] if len(parts) > 1 else "General"

                self.metadata.append({
                    "filename": image_path.name,
                    "label": label,
                    "dataset": dataset_name,
                    "original_width": orig_width,
                    "original_height": orig_height,
                    "original_format": orig_format,
                    "output_format": "JPEG",
                    "output_width": IMAGE_SIZE[0],
                    "output_height": IMAGE_SIZE[1],
                    "path": str(save_path)
                })

                self.dataset_stats[dataset_name] = self.dataset_stats.get(dataset_name, 0) + 1
                self.processed_count += 1

        except (UnidentifiedImageError, OSError, Exception) as err:
            self.corrupted_count += 1
            METADATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"Corrupted File: {image_path}\nError: {err}\n\n")

    def run(self):
        """Executes full preprocessing pipeline."""
        start_time = time.time()
        images = self.get_all_images()

        print("\n" + "=" * 60)
        print(f"PhotoGuard AI - Preprocessing Pipeline")
        print("=" * 60)
        print(f"Found {len(images)} Raw Images")

        if not images:
            print(f"No raw images found in: {self.raw_dir}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc="Preprocessing Images", colour="green"):
            self.process_single_image(img_path)

        # Save metadata CSV
        if self.metadata:
            df = pd.DataFrame(self.metadata)
            METADATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(METADATA_FILE, index=False)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("-" * 60)
        print(f"Processed Images  : {self.processed_count}")
        print(f"Corrupted Images  : {self.corrupted_count}")
        print(f"Output Directory  : {self.output_dir}")
        print(f"Metadata Saved To : {METADATA_FILE}")
        print(f"Total Time Taken  : {elapsed:.2f} seconds")
        print("=" * 60 + "\n")

def main():
    processor = ImagePreprocessor()
    processor.run()

if __name__ == "__main__":
    main()