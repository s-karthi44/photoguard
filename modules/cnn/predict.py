"""
=========================================================
PhotoGuard AI
CNN Inference & Prediction Engine

Author : Vinay Kirithic

Description:
Runs single-image or batch-image inference using trained PhotoGuard CNN.
Outputs prediction class ('AI' vs 'Real'), confidence scores, and probability distribution.
=========================================================
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modules.cnn.config import DEVICE, CLASS_NAMES, IMAGE_SIZE
from modules.cnn.model import build_model
from modules.paths import MODELS_DIR

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
if not BEST_MODEL_PATH.exists():
    BEST_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "best_model.pth"

# Inference Transform Pipeline
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class PhotoGuardPredictor:
    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()
        self.patch_size = IMAGE_SIZE  # 224
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _load_model(self):
        model = build_model()
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _extract_informative_patches(self, image: Image.Image, max_patches: int = 9, min_std: float = 18.0) -> list[torch.Tensor]:
        """
        Extracts native-resolution unscaled crops (224x224) focusing on active subject regions.
        Filters out featureless background regions (studio backdrops, solid walls, paper margins)
        that have low textural standard deviation and would trigger false-positive CNN predictions.
        """
        w, h = image.size
        crop_size = self.patch_size
        if w < crop_size or h < crop_size:
            return []

        # Candidate sampling points across subject area (center, 1/4, 1/2, 3/4 coordinates)
        # Avoids extreme canvas edges which are typically paper borders, studio backdrops, or solid margins
        candidates = [(w // 2 - crop_size // 2, h // 2 - crop_size // 2)]
        xs = [int(w * 0.25), int(w * 0.50), int(w * 0.75)]
        ys = [int(h * 0.25), int(h * 0.50), int(h * 0.75)]
        for x in xs:
            for y in ys:
                candidates.append((x - crop_size // 2, y - crop_size // 2))

        inset_x, inset_y = int(w * 0.12), int(h * 0.12)
        candidates.extend([
            (inset_x, inset_y),
            (w - crop_size - inset_x, inset_y),
            (inset_x, h - crop_size - inset_y),
            (w - crop_size - inset_x, h - crop_size - inset_y)
        ])

        valid_patches = []
        seen = set()
        for cx, cy in candidates:
            x = max(0, min(w - crop_size, cx))
            y = max(0, min(h - crop_size, cy))
            if (x, y) in seen:
                continue
            seen.add((x, y))

            patch = image.crop((x, y, x + crop_size, y + crop_size))
            patch_gray = np.array(patch.convert("L"), dtype=np.float32)
            std_val = float(np.std(patch_gray))

            # Filter out featureless background regions (studio backdrops, solid walls, paper borders)
            # which lack semantic structure and cause false positive CNN collapse
            if std_val >= min_std:
                valid_patches.append((patch, std_val))

        # Prioritize patches with richest structural texture
        valid_patches.sort(key=lambda item: item[1], reverse=True)
        selected_patches = [self.normalize(transforms.ToTensor()(vp[0])) for vp in valid_patches[:max_patches]]
        return selected_patches

    def predict_image(self, image_path: Path):
        """
        Runs multi-scale inference on a single image file:
        1. Evaluates the global resized image (224x224) for holistic scene context.
        2. Evaluates informative native resolution crops (224x224) with sufficient texture.
        3. Dynamically fuses global and patch-level probabilities.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")

        # 1. Global View Inference (Resized)
        tensor_global = inference_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs_global = self.model(tensor_global)
            probabilities_global = F.softmax(outputs_global, dim=1)[0]
        global_ai_prob = float(probabilities_global[0].item())

        # 2. Content-Aware Native Resolution Patch Inference (Unscaled)
        patch_tensors = self._extract_informative_patches(image, max_patches=9, min_std=18.0)

        if patch_tensors:
            batch = torch.stack(patch_tensors).to(DEVICE)
            with torch.no_grad():
                out_batch = self.model(batch)
                patch_probs = F.softmax(out_batch, dim=1)[:, 0]  # AI probabilities

            patch_ai_scores = patch_probs.tolist()
            mean_patch_ai = float(sum(patch_ai_scores) / len(patch_ai_scores))
            max_patch_ai = float(max(patch_ai_scores))
            ai_ratio = float(sum(1 for a in patch_ai_scores if a >= 0.50) / len(patch_ai_scores))
        else:
            mean_patch_ai = global_ai_prob
            max_patch_ai = global_ai_prob
            ai_ratio = 1.0 if global_ai_prob >= 0.50 else 0.0

        # 3. Dynamic Multi-Scale Fusion Gate:
        # If the global view is strongly clean/authentic (global_ai_prob < 0.20):
        # Keep the clean global prediction so authentic camera photographs are not corrupted by patch false alarms.
        # Patch-level analytics are passed to the Fusion Engine for physical sensor cross-validation.
        if global_ai_prob < 0.20:
            if ai_ratio >= 0.70 and mean_patch_ai >= 0.65:
                combined_ai_prob = 0.35 * global_ai_prob + 0.65 * mean_patch_ai
            else:
                combined_ai_prob = global_ai_prob
        elif ai_ratio >= 0.50 and mean_patch_ai >= 0.50:
            combined_ai_prob = 0.25 * global_ai_prob + 0.75 * mean_patch_ai
        elif ai_ratio >= 0.25 and max_patch_ai >= 0.85:
            combined_ai_prob = 0.40 * global_ai_prob + 0.60 * mean_patch_ai
        else:
            combined_ai_prob = 0.70 * global_ai_prob + 0.30 * mean_patch_ai

        pred_idx = 0 if combined_ai_prob >= 0.50 else 1
        conf = combined_ai_prob if pred_idx == 0 else (1.0 - combined_ai_prob)
        predicted_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

        return {
            "image_name": image_path.name,
            "prediction": predicted_class,
            "confidence": round(conf * 100, 2),
            "probabilities": {
                "AI": round(combined_ai_prob * 100, 2),
                "Real": round((1.0 - combined_ai_prob) * 100, 2)
            },
            "patch_analytics": {
                "patch_count": len(patch_tensors),
                "patch_mean_ai": round(mean_patch_ai * 100, 2),
                "patch_max_ai": round(max_patch_ai * 100, 2),
                "patch_ai_ratio": round(ai_ratio * 100, 2),
                "global_ai": round(global_ai_prob * 100, 2)
            }
        }


def main():
    predictor = PhotoGuardPredictor()
    print("Predictor Engine ready. Call predict_image(image_path) for inference.")

if __name__ == "__main__":
    main()
