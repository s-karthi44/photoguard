"""
=========================================================
PhotoGuard AI
CNN Training Engine

Author : Vinay Kirithic

Description:
Main training script.

Responsibilities

1. Load Dataset
2. Build CNN
3. Configure Optimizer
4. Configure Scheduler
5. Resume Training
6. Save Best Model
7. Save Checkpoints

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

from datetime import datetime
import json
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from modules.cnn.config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

from modules.cnn.utils import (
    set_seed,
    get_learning_rate
)


from modules.cnn.dataset import (
    get_dataloaders
)

from modules.cnn.model import (
    build_model
)

from modules.cnn.trainer import (
    train_one_epoch,
    validate_one_epoch
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Check if running on Google Colab
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/PhotoGuard")
if COLAB_DRIVE_ROOT.exists():
    WEIGHTS_DIR = COLAB_DRIVE_ROOT / "weights"
    CHECKPOINT_DIR = COLAB_DRIVE_ROOT / "checkpoints"
    HISTORY_DIR = COLAB_DRIVE_ROOT / "history"
else:
    WEIGHTS_DIR = BASE_DIR / "weights"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    HISTORY_DIR = BASE_DIR / "history"

BEST_MODEL = WEIGHTS_DIR / "best_model.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"
HISTORY_FILE = HISTORY_DIR / "history.json"
# =========================================================
# CREATE PROJECT FOLDERS
# =========================================================

def create_directories():

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# CREATE MODEL
# =========================================================

def create_model():

    model = build_model()

    model = model.to(DEVICE)

    return model


# =========================================================
# LOSS FUNCTION
# =========================================================

def build_loss():

    return nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

def build_optimizer(model):

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    return optimizer


# =========================================================
# LR SCHEDULER
# =========================================================

def build_scheduler(optimizer):

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3

    )

    return scheduler


# =========================================================
# FIND LATEST CHECKPOINT
# =========================================================

def get_latest_checkpoint_path():
    """Finds the most recent checkpoint file across all run directories."""
    latest_checkpoint = None
    latest_time = 0.0
    
    # Search in all run subdirectories checkpoints/run_YYYYMMDD_HHMMSS/
    if CHECKPOINT_DIR.exists():
        for run_dir in CHECKPOINT_DIR.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                for cp_file in run_dir.glob("checkpoint_epoch_*.pth"):
                    mtime = cp_file.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_checkpoint = cp_file
                        
    # Fallback to default if no run folders exist
    if latest_checkpoint is None and LAST_CHECKPOINT.exists():
        latest_checkpoint = LAST_CHECKPOINT
        
    return latest_checkpoint


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(model, optimizer, default_run_id):

    start_epoch = 1

    best_accuracy = 0.0

    run_id = default_run_id

    latest_checkpoint = get_latest_checkpoint_path()

    if latest_checkpoint is not None:

        print(f"\nLoading Previous Checkpoint: {latest_checkpoint.name}...\n")

        checkpoint = torch.load(

            latest_checkpoint,

            map_location=DEVICE

        )

        try:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            start_epoch = checkpoint["epoch"] + 1
            best_accuracy = checkpoint["best_accuracy"]
            run_id = checkpoint.get("run_id", default_run_id)
            print(
                f"Resuming From Epoch {start_epoch} (Run: {run_id})"
            )
        except Exception as e:
            print("\n[NOTE] Previous checkpoint architecture differs from Dual-Stream model. Starting fresh training.\n")
            start_epoch = 1
            best_accuracy = 0.0
            run_id = default_run_id

    return (

        model,

        optimizer,

        start_epoch,

        best_accuracy,

        run_id

    )
# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    create_directories()

    print("\n" + "=" * 70)
    print("             PhotoGuard AI - CNN Training")
    print("=" * 70)

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader, validation_loader, _ = get_dataloaders()

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_model()

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = build_loss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = build_optimizer(model)

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = build_scheduler(optimizer)

    # -----------------------------------------------------
    # Resume Checkpoint
    # -----------------------------------------------------

    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, optimizer, start_epoch, best_accuracy, run_id = load_checkpoint(

        model,
        optimizer,
        default_run_id

    )

    run_checkpoint_dir = CHECKPOINT_DIR / f"run_{run_id}"
    run_weights_dir = WEIGHTS_DIR / f"run_{run_id}"

    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_weights_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Training History
    # -----------------------------------------------------

    history = {

        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []

    }

    patience = 0

    EARLY_STOPPING = EARLY_STOPPING_PATIENCE

    # -----------------------------------------------------
    # Epoch Loop
    # -----------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):

        print("\n" + "=" * 70)

        print(f"Epoch {epoch}/{EPOCHS}")

        print("=" * 70)

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(

            model,
            train_loader,
            criterion,
            optimizer

        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        validation_loss, validation_accuracy = validate_one_epoch(

            model,
            validation_loader,
            criterion

        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(

            validation_accuracy

        )

        # -------------------------------------------------
        # Save History
        # -------------------------------------------------

        history["train_loss"].append(

            train_loss

        )

        history["train_accuracy"].append(

            train_accuracy

        )

        history["validation_loss"].append(

            validation_loss

        )

        history["validation_accuracy"].append(

            validation_accuracy

        )

        # -------------------------------------------------
        # Print Results
        # -------------------------------------------------

        print()

        print(f"Train Loss        : {train_loss:.4f}")

        print(f"Train Accuracy    : {train_accuracy*100:.2f}%")

        print(f"Validation Loss   : {validation_loss:.4f}")

        print(f"Validation Acc    : {validation_accuracy*100:.2f}%")

        print(

            f"Learning Rate     : "

            f"{get_learning_rate(optimizer):.7f}"

        )

        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            epoch_best_model = run_weights_dir / f"best_model_epoch_{epoch}_val_acc_{validation_accuracy:.4f}.pth"

            torch.save(

                model.state_dict(),

                epoch_best_model

            )

            print(f"\n✓ Best Model Saved to {epoch_best_model.name}")

            try:
                shutil.copy2(epoch_best_model, BEST_MODEL)
                print("✓ Updated default best_model.pth link")
            except Exception as e:
                print(f"Warning: Could not copy best_model: {e}")

            patience = 0

        else:

            patience += 1

        # -------------------------------------------------
        # Save Checkpoint
        # -------------------------------------------------

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "best_accuracy": best_accuracy,

            "run_id": run_id

        }

        epoch_checkpoint = run_checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save(

            checkpoint,

            epoch_checkpoint

        )

        try:
            shutil.copy2(epoch_checkpoint, LAST_CHECKPOINT)
        except Exception as e:
            pass

        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------

        if patience >= EARLY_STOPPING:

            print("\nEarly Stopping Triggered.")

            break

    # -----------------------------------------------------
    # Save History
    # -----------------------------------------------------

    with open(

        HISTORY_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            history,

            file,

            indent=4

        )

    print("\n" + "=" * 70)

    print("Training Completed Successfully")

    print(f"Best Validation Accuracy : {best_accuracy*100:.2f}%")

    print("=" * 70)

    # =========================================================
# PRINT TRAINING SUMMARY
# =========================================================

def print_training_summary():

    print("\n")
    print("=" * 70)
    print("              PhotoGuard AI - Training Summary")
    print("=" * 70)

    print(f"\nBest Model")

    print(f"{BEST_MODEL}")

    print(f"\nCheckpoint")

    print(f"{LAST_CHECKPOINT}")

    print(f"\nTraining History")

    print(f"{HISTORY_FILE}")

    print("\nModel Training Finished Successfully.")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 70)
    print("            PhotoGuard AI - CNN Training Engine")
    print("=" * 70)

    print(f"\nDevice : {DEVICE}")

    set_seed()

    train_model()

    print_training_summary()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()