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

    def predict_image(self, image_path: Path):
        """
        Runs inference on a single image file.
        Returns dictionary with predicted label, confidence, and class probabilities.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[pred_idx.item()] if pred_idx.item() < len(CLASS_NAMES) else str(pred_idx.item())

        return {
            "image_name": image_path.name,
            "prediction": predicted_class,
            "confidence": round(conf.item() * 100, 2),
            "probabilities": {
                CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}": round(prob.item() * 100, 2)
                for i, prob in enumerate(probabilities)
            }
        }


def main():
    predictor = PhotoGuardPredictor()
    print("Predictor Engine ready. Call predict_image(image_path) for inference.")

if __name__ == "__main__":
    main()
