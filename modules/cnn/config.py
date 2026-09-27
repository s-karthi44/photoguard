"""
=========================================================
PhotoGuard AI
CNN Configuration File

Author : Vinay
Model  : EfficientNet-B0
=========================================================
"""

from pathlib import Path
import torch

from modules.paths import (
    BASE_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
)

# =========================================================
# PROJECT ROOT
# =========================================================

# =========================================================
# DATASET PATHS
# =========================================================



MODEL_NAME = "efficientnet_b0"

BEST_MODEL = MODELS_DIR / "best_model.pth"

LAST_MODEL = MODELS_DIR / "last_model.pth"

# =========================================================
# OUTPUT PATHS
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTION_DIR = OUTPUT_DIR / "predictions"

GRADCAM_DIR = OUTPUT_DIR / "gradcam"

LOG_DIR = OUTPUT_DIR / "logs"

METRICS_DIR = OUTPUT_DIR / "metrics"

PREDICTION_DIR.mkdir(exist_ok=True)

GRADCAM_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

METRICS_DIR.mkdir(exist_ok=True)

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMAGE_SIZE = 224

CHANNELS = 3

NUM_CLASSES = 2

CLASS_NAMES = [

    "AI",

    "Real"

]

# =========================================================
# TRAINING SETTINGS
# =========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

PIN_MEMORY = True

SAVE_EVERY = 1

# =========================================================
# EARLY STOPPING
# =========================================================

EARLY_STOPPING_PATIENCE = 5

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "mps"

    if torch.backends.mps.is_available()

    else "cpu"

)

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42


# =========================================================
# SET RANDOM SEED
# =========================================================

def set_seed():

    import random

    random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)

# =========================================================
# EARLY STOPPING
# =========================================================



# =========================================================
# MODEL
# =========================================================

PRETRAINED = True

DROPOUT = 0.30

# =========================================================
# DATA AUGMENTATION
# =========================================================

HORIZONTAL_FLIP = True

RANDOM_ROTATION = 10

COLOR_JITTER = True

# =========================================================
# PRINT CONFIG
# =========================================================

def show_config():

    print("=" * 60)

    print("PhotoGuard AI")

    print("=" * 60)

    print()

    print("Dataset")

    print(TRAIN_DIR)

    print()

    print("Validation")

    print(VALIDATION_DIR)

    print()

    print("Test")

    print(TEST_DIR)

    print()

    print(f"Image Size      : {IMAGE_SIZE}")

    print(f"Batch Size      : {BATCH_SIZE}")

    print(f"Epochs          : {EPOCHS}")

    print(f"Learning Rate   : {LEARNING_RATE}")

    print(f"Device          : {DEVICE}")

    print("=" * 60)


if __name__ == "__main__":

    show_config()