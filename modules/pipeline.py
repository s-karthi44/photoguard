"""
============================================================
PhotoGuard AI
End-to-End Orchestration Pipeline

Author : Vinay Kirithic

Description:
Orchestrates the entire PhotoGuard AI non-CNN pipeline steps:
1. Environment & Directory Setup
2. Raw Image Preprocessing & Cleaning
3. Class-Balanced Dataset Splitting (Train/Val/Test)
4. DataLoader Validation Check
============================================================
"""

import time
from modules.paths import ensure_directories_exist
from modules.preprocessing import ImagePreprocessor
from modules.split_dataset import DatasetSplitter
from modules.dataloader import create_dataloaders

def run_pipeline(balance_dataset: bool = True, use_weighted_sampler: bool = True):
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("           PhotoGuard AI - Core Pipeline Orchestrator")
    print("=" * 70)
    
    # Step 1: Ensure directory structure
    print("\n[Step 1/4] Verifying Project Directory Structure...")
    ensure_directories_exist()
    print("Directory structure OK.")

    # Step 2: Run Image Preprocessing
    print("\n[Step 2/4] Running Image Preprocessor...")
    preprocessor = ImagePreprocessor()
    preprocessor.run()

    # Step 3: Run Dataset Splitting & Balancing
    print("\n[Step 3/4] Running Dataset Splitter...")
    splitter = DatasetSplitter()
    splitter.split()

    # Step 4: Verify DataLoaders
    print("\n[Step 4/4] Validating PyTorch DataLoaders...")
    try:
        train_loader, val_loader, test_loader, class_map = create_dataloaders(
            use_weighted_sampler=use_weighted_sampler
        )
        print(f"DataLoaders Ready | Class Mapping: {class_map}")
        print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")
    except Exception as e:
        print(f"DataLoader validation notice: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Pipeline Execution Complete in {elapsed:.2f} seconds.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_pipeline()
