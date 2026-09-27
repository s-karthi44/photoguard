"""
============================================================
PhotoGuard AI - Localized Splicing & Inpainting Mask Engine
Author : PhotoGuard AI Team

Description:
Fuses Grad-CAM neural attention heatmaps and Error Level Analysis (ELA)
compression residuals into a calibrated localized segmentation mask.
Identifies and bounds tampered, spliced, or in-painted regions inside
images with bounding boxes and contour overlays.
============================================================
"""

from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

import torch
from torchvision import transforms
from modules.cnn.config import DEVICE, IMAGE_SIZE
from modules.cnn.model import build_model
from modules.cnn.gradcam import GradCAM, BEST_MODEL_PATH
from modules.paths import OUTPUT_DIR

class SplicingDetector:
    def __init__(self, ela_quality: int = 95, activation_threshold: float = 0.65):
        """
        Args:
            ela_quality: JPEG re-compression quality for ELA disparity.
            activation_threshold: Anomaly score cutoff for binary segmentation mask.
        """
        self.ela_quality = ela_quality
        self.activation_threshold = activation_threshold
        self.splicing_dir = OUTPUT_DIR / "splicing"
        self.splicing_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load model for Grad-CAM
        self._model = None
        self._gradcam = None

    def _init_gradcam(self):
        if self._gradcam is None:
            self._model = build_model()
            if BEST_MODEL_PATH.exists():
                checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                else:
                    self._model.load_state_dict(checkpoint)
            self._model.eval()
            target_layer = self._model.spatial_features[-1]
            self._gradcam = GradCAM(self._model, target_layer)

    def _compute_ela_map(self, img_pil: Image.Image) -> np.ndarray:
        """Computes normalized 2D Error Level Analysis map [0.0, 1.0]."""
        # Save temporary compressed jpeg in memory or file
        temp_path = self.splicing_dir / f"temp_ela_{os.getpid()}.jpg"
        img_rgb = img_pil.convert("RGB")
        img_rgb.save(temp_path, "JPEG", quality=self.ela_quality)
        
        compressed = Image.open(temp_path)
        diff = ImageChops.difference(img_rgb, compressed)
        
        # Convert to grayscale array
        diff_gray = np.array(diff.convert("L"), dtype=np.float32)
        
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)

        # Normalize difference map
        p98 = np.percentile(diff_gray, 98)
        if p98 > 0:
            diff_norm = np.clip(diff_gray / p98, 0.0, 1.0)
        else:
            diff_norm = np.zeros_like(diff_gray)
            
        return diff_norm

    def _compute_gradcam_map(self, img_pil: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
        """Computes normalized Grad-CAM attention map scaled to target_size."""
        self._init_gradcam()
        resized = img_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(resized).unsqueeze(0).to(DEVICE)
        
        # Class 0: AI, Class 1: Real
        cam = self._gradcam.generate_heatmap(tensor, class_idx=0)
        cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
        return cam_resized

    def analyze(self, image_path: str | Path, alpha: float = 0.50) -> dict:
        """
        Fuses ELA compression residuals and Grad-CAM attention into a single
        binary segmentation mask and visual bounding box overlay.

        Args:
            image_path: Path to the image file.
            alpha: Weight for Grad-CAM in fusion (1 - alpha for ELA).
        
        Returns:
            dict containing:
                - is_spliced: bool
                - spliced_regions_count: int
                - spliced_area_percentage: float
                - regions: list of detected bounding boxes with confidence
                - mask_path: path to binary segmentation mask PNG
                - overlay_path: path to bounding box overlay JPG
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        orig_pil = Image.open(image_path).convert("RGB")
        w, h = orig_pil.size
        orig_np = np.array(orig_pil)

        # 1. Compute individual spatial anomaly maps
        gradcam_map = self._compute_gradcam_map(orig_pil, (w, h))
        ela_map = self._compute_ela_map(orig_pil)

        # If PNG or lossless, ELA is uniform; rely primarily on Grad-CAM
        is_png = image_path.suffix.lower() == ".png"
        if is_png:
            fused_map = gradcam_map
        else:
            fused_map = alpha * gradcam_map + (1.0 - alpha) * ela_map

        # 2. Smooth fused map with Gaussian blur to consolidate clusters
        smoothed = cv2.GaussianBlur(fused_map, (15, 15), 0)

        # 3. Create binary segmentation mask via adaptive thresholding
        binary_mask = (smoothed >= self.activation_threshold).astype(np.uint8) * 255

        # Morphological operations to clean isolated pixel noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Extract contours and compute bounding boxes
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_pixels = w * h
        min_region_pixels = int(total_pixels * 0.015)  # Ignore regions < 1.5% of total image

        detected_regions = []
        overlay_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        red_tint_mask = np.zeros_like(overlay_bgr)

        spliced_pixels = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_region_pixels:
                continue

            spliced_pixels += area
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Compute regional confidence from smoothed fused map
            patch_scores = smoothed[y:y+bh, x:x+bw]
            region_conf = float(np.mean(patch_scores)) if patch_scores.size > 0 else 0.0
            area_pct = round((area / total_pixels) * 100.0, 2)

            detected_regions.append({
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh),
                "confidence": round(region_conf * 100.0, 2),
                "area_percentage": area_pct
            })

            # Draw semi-transparent red mask fill
            cv2.drawContours(red_tint_mask, [cnt], -1, (0, 0, 220), -1)

            # Draw prominent glowing border
            cv2.rectangle(overlay_bgr, (x, y), (x + bw, y + bh), (0, 70, 255), 3)

            # Draw tag box with label
            label = f"SPLICED: {area_pct}% (conf: {round(region_conf * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(overlay_bgr, (x, max(0, y - th - 10)), (x + tw + 10, max(0, y)), (0, 50, 200), -1)
            cv2.putText(overlay_bgr, label, (x + 5, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Apply translucent tint over tampered area
        overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, red_tint_mask, 0.35, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        total_spliced_pct = round((spliced_pixels / total_pixels) * 100.0, 2)
        has_high_confidence_region = any(r.get("confidence", 0) >= 65.0 for r in detected_regions)
        is_spliced = len(detected_regions) > 0 and total_spliced_pct >= 2.5 and has_high_confidence_region

        # 5. Save visual artifacts
        mask_filename = f"splicing_mask_{image_path.stem}.png"
        overlay_filename = f"splicing_overlay_{image_path.stem}.jpg"
        
        mask_path = self.splicing_dir / mask_filename
        overlay_path = self.splicing_dir / overlay_filename

        Image.fromarray(binary_mask).save(mask_path)
        Image.fromarray(overlay_rgb).save(overlay_path, quality=92)

        return {
            "is_spliced": is_spliced,
            "spliced_regions_count": len(detected_regions),
            "spliced_area_percentage": total_spliced_pct,
            "regions": detected_regions,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "mask_filename": mask_filename,
            "overlay_filename": overlay_filename
        }

if __name__ == "__main__":
    detector = SplicingDetector()
    print("SplicingDetector initialized successfully.")
