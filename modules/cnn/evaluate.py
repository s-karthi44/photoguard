"""
=========================================================
PhotoGuard AI
CNN Model Evaluation Module

Author : Vinay Kirithic

Description:
Evaluates trained PhotoGuard CNN model on the test dataset.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
and saves metrics summary to output directory.
Supports Google Colab Google Drive output persistence.
=========================================================
"""

import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from modules.cnn.config import (
    DEVICE,
)
from modules.cnn.model import build_model
from modules.paths import MODELS_DIR, TEST_DIR, OUTPUT_DIR

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Check if running on Google Colab to persist metrics & plots directly to Google Drive
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/PhotoGuard")
if COLAB_DRIVE_ROOT.exists():
    BEST_MODEL_PATH = COLAB_DRIVE_ROOT / "weights" / "best_model.pth"
    METRICS_OUTPUT_DIR = COLAB_DRIVE_ROOT / "outputs" / "metrics"
else:
    BEST_MODEL_PATH = BASE_DIR / "weights" / "best_model.pth"
    if not BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
    METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"


def evaluate_model(model_path: Path = BEST_MODEL_PATH):
    print("\n" + "=" * 70)
    print("           PhotoGuard AI - Model Evaluation Engine")
    print("=" * 70)
    print(f"Loading Model Weights from: {model_path}")

    if not model_path.exists():
        print(f"Error: Model weights file not found at {model_path}")
        return

    # Import CNN dataloader
    from modules.cnn.dataset import get_dataloaders
    _, _, test_loader = get_dataloaders()
    
    # Extract ground truth class mapping from Dataset (e.g. {'ai': 0, 'real': 1})
    class_to_idx = test_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    print(f"Test Batches: {len(test_loader)} | Dynamically Inferred Classes: {class_names}")

    # Load Trained Model
    model = build_model()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            # For 2-class classification, pass probabilities of positive class (index 1)
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("-" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc:.4f}")
    print("=" * 70)

    # Print Detailed Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    # Save metrics JSON safely
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "class_mapping": class_to_idx
    }
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file_path = METRICS_OUTPUT_DIR / "test_evaluation_metrics.json"
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[OK] Metrics JSON saved to: {metrics_file_path}")

    # Plot Confusion Matrix safely
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("PhotoGuard AI - Confusion Matrix")
    plt.tight_layout()
    cm_path = METRICS_OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Confusion Matrix plot saved to: {cm_path}\n")

if __name__ == "__main__":
    evaluate_model()
