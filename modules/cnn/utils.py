"""
=========================================================
PhotoGuard AI
CNN Utility Functions

Author : Vinay

Description:
Reusable utility functions for CNN training.

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import random
import numpy as np
import torch

from pathlib import Path

from modules.cnn.config import (
    SEED,
    BEST_MODEL,
    LAST_MODEL,
    DEVICE
)

# =========================================================
# SET RANDOM SEED
# =========================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(f"Random Seed Set : {SEED}")


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model, optimizer, epoch, accuracy, filepath):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "accuracy": accuracy

    }

    torch.save(checkpoint, filepath)

    print(f"\nModel Saved : {filepath}")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model, optimizer=None, filepath=BEST_MODEL):

    if not Path(filepath).exists():

        print("Checkpoint not found.")

        return model, optimizer, 0, 0

    checkpoint = torch.load(

        filepath,

        map_location=DEVICE

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    if optimizer is not None:

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

    epoch = checkpoint["epoch"]

    accuracy = checkpoint["accuracy"]

    print(f"\nLoaded Model : {filepath}")

    return model, optimizer, epoch, accuracy


# =========================================================
# COUNT PARAMETERS
# =========================================================

def count_parameters(model):

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    return total, trainable


# =========================================================
# FREEZE BACKBONE
# =========================================================

def freeze_backbone(model):

    for param in model.model.features.parameters():

        param.requires_grad = False

    print("Backbone Frozen")


# =========================================================
# UNFREEZE BACKBONE
# =========================================================

def unfreeze_backbone(model):

    for param in model.model.features.parameters():

        param.requires_grad = True

    print("Backbone Unfrozen")


# =========================================================
# PRINT MODEL INFO
# =========================================================

def print_model_info(model):

    total, trainable = count_parameters(model)

    print("\n" + "=" * 60)

    print("Model Information")

    print("=" * 60)

    print(f"Total Parameters      : {total:,}")

    print(f"Trainable Parameters  : {trainable:,}")

    print("=" * 60)


# =========================================================
# CALCULATE ACCURACY
# =========================================================

def calculate_accuracy(outputs, labels):

    predicted = outputs.argmax(dim = 1)

    correct = (predicted == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


# =========================================================
# GET LEARNING RATE
# =========================================================

def get_learning_rate(optimizer):

    return optimizer.param_groups[0]["lr"]


# =========================================================
# AVERAGE METER
# =========================================================

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0

        self.count = 0

        self.avg = 0

    def update(self, value, n=1):

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count


# =========================================================
# TEST
# =========================================================

def main():

    print("=" * 60)

    print("PhotoGuard AI")

    print("Utility Module")

    print("=" * 60)

    set_seed()

    print("\nUtilities Loaded Successfully")


if __name__ == "__main__":

    main()