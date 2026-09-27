"""
============================================================
PhotoGuard AI
Dataset Splitting & Class Balancing Module

Author : Vinay Kirithic

Description:
Splits preprocessed images into Train (70%), Validation (15%), and Test (15%) sets.
Organizes output cleanly by top-level class folders ('real' vs 'ai').
Supports class balancing (e.g., capping majority Real class at 12,000 to match AI class).
============================================================
"""

import random
import shutil
from pathlib import Path
from tqdm import tqdm

from modules.paths import (
    PREPROCESSED_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
)

# ==========================================================
# CONFIGURATION & RATIOS
# ==========================================================
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Maximum samples per top-level class ('real' vs 'ai')
MAX_SAMPLES_PER_CLASS = 20000  

RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class DatasetSplitter:
    def __init__(
        self,
        input_dir: Path = PREPROCESSED_DIR,
        max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
        seed: int = RANDOM_SEED
    ):
        self.input_dir = input_dir
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed
        random.seed(self.seed)

    def clean_output_folders(self):
        """Cleans existing train/val/test directories for a fresh split."""
        for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
        print("Previous dataset splits cleared. Fresh directories initialized.")

    def get_top_level_classes(self):
        """Retrieves top-level class directories ('real', 'ai')."""
        if not self.input_dir.exists():
            return []
        return [d for d in self.input_dir.iterdir() if d.is_dir()]

    def copy_files(self, files, destination_dir):
        """Copies list of image files to target directory."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, destination_dir / file.name)

    def split(self):
        """Executes stratified train/validation/test split with top-level class balancing."""
        if not self.input_dir.exists():
            print(f"Preprocessed dataset directory not found: {self.input_dir}")
            return

        self.clean_output_folders()
        top_classes = self.get_top_level_classes()

        print("\n" + "=" * 60)
        print("PhotoGuard AI - Dataset Splitting & Class Balancing")
        print("=" * 60)

        total_train, total_val, total_test = 0, 0, 0

        for class_dir in top_classes:
            class_name = class_dir.name  # 'real' or 'ai'
            
            # Recursively collect all preprocessed images within this class (including subfolders like flicker/coco)
            images = [
                f for f in class_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            random.shuffle(images)

            # Apply top-level class balancing cap
            if self.max_samples_per_class and len(images) > self.max_samples_per_class:
                print(f"Balancing Top Class [{class_name}]: Capped from {len(images)} -> {self.max_samples_per_class} images")
                images = images[:self.max_samples_per_class]

            total = len(images)
            train_cnt = int(total * TRAIN_RATIO)
            val_cnt = int(total * VALIDATION_RATIO)

            train_imgs = images[:train_cnt]
            val_imgs = images[train_cnt : train_cnt + val_cnt]
            test_imgs = images[train_cnt + val_cnt :]

            # Copy to split directory flattened by top class name for PyTorch ImageFolder readiness
            self.copy_files(train_imgs, TRAIN_DIR / class_name)
            self.copy_files(val_imgs, VALIDATION_DIR / class_name)
            self.copy_files(test_imgs, TEST_DIR / class_name)

            total_train += len(train_imgs)
            total_val += len(val_imgs)
            total_test += len(test_imgs)

            print(f"  Class [{class_name}] -> Total: {total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

        print("\n" + "=" * 60)
        print("Dataset Split Completed Successfully!")
        print("-" * 60)
        print(f"Total Train Images : {total_train}")
        print(f"Total Val Images   : {total_val}")
        print(f"Total Test Images  : {total_test}")
        print("=" * 60 + "\n")

def main():
    splitter = DatasetSplitter()
    splitter.split()

if __name__ == "__main__":
    main()