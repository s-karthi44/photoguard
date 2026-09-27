"""
============================================================
PhotoGuard AI
DataLoader & Sampling Module

Author : Vinay Kirithic

Description:
Builds PyTorch Datasets and DataLoaders for Training,
Validation, and Testing. Features WeightedRandomSampler to
handle class imbalance automatically.
============================================================
"""

import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from modules.paths import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
)
from modules.cnn.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# SAMPLER FOR CLASS IMBALANCE
# ============================================

def get_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler to oversample the minority class 
    (e.g., AI images) in real-time during training batch generation.
    """
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts

# ============================================
# BUILD DATALOADERS
# ============================================

def create_dataloaders(use_weighted_sampler=False):
    """
    Instantiates Datasets and DataLoaders for train/val/test splits.
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
        
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )

    if use_weighted_sampler:
        sampler, counts = get_weighted_sampler(train_dataset)
        print(f"Weighted Sampler initialized. Train counts per class: {counts}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

if __name__ == "__main__":
    print("=" * 60)
    print("PhotoGuard AI - DataLoader Module")
    print("=" * 60)

    try:
        train_l, val_l, test_l, class_map = create_dataloaders(use_weighted_sampler=True)
        print("\nClass Mapping:", class_map)
        print("Training Batches  :", len(train_l))
        print("Validation Batches:", len(val_l))
        print("Test Batches      :", len(test_l))
    except Exception as e:
        print(f"DataLoader setup note: {e}")