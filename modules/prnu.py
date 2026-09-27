"""
============================================================
PhotoGuard AI
PRNU (Photo Response Non-Uniformity) Noise Residual Analyzer

Author : Vinay Kirithic

Description:
Extracts high-frequency camera sensor noise residuals and analyzes
statistical properties (spatial correlation, variance, and entropy).
Real photos contain stochastic camera sensor noise (PRNU), while
AI-generated images exhibit low-noise entropy or structured interpolation artifacts.
============================================================
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class PRNUAnalyzer:
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def extract_noise_residual(self, img_np: np.ndarray) -> np.ndarray:
        """
        Extracts high-frequency camera sensor noise residual W = I - F(I)
        using edge-preserving median filtering.
        """
        img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.kernel_size)
        return img_np - denoised.astype(np.float32)

    def analyze(self, image_path: str | Path) -> dict:
        """
        Extracts and analyzes high-frequency camera sensor noise features.
        
        Natural camera sensors introduce stochastic, spatially uncorrelated
        photo-response non-uniformity (PRNU) across the sensor surface.
        In contrast, AI generative models either produce unnaturally smooth
        regions (missing noise floor) or spatially correlated upsampling artifacts.
        
        Returns:
            dict containing:
                - prnu_anomaly_score: float (0.0 to 1.0, high indicates anomalous/synthetic noise profile)
                - noise_variance: float (variance of the flat-region noise residual)
                - noise_spatial_correlation: float (correlation between adjacent noise pixels)
                - noise_entropy: float (entropy of flat-region noise residual)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Load image in grayscale
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, dtype=np.float32)

        # 2. Extract edge-preserving noise residual
        residual = self.extract_noise_residual(img_np)

        # 3. Mask out edges using Sobel gradients to isolate flat/smooth regions
        # Sensor noise is only cleanly measurable in flat/low-gradient regions;
        # edge regions naturally have strong spatial correlation in both real & AI images.
        gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Bottom 40th percentile gradients represent flat/smooth patches
        p40 = np.percentile(grad_mag, 40)
        flat_mask = grad_mag <= max(float(p40), 1.0)
        flat_residual = residual[flat_mask]

        if len(flat_residual) < 100:
            flat_residual = residual.flatten()
            flat_mask = np.ones_like(residual, dtype=bool)

        # 4. Compute flat-region noise variance (physical sensor noise floor)
        variance = float(np.var(flat_residual))

        # 5. Compute flat-region spatial correlation (lag-1 autocorrelation)
        shifted = np.roll(residual, shift=1, axis=1)
        flat_shifted = shifted[flat_mask]

        std_res = np.std(flat_residual)
        std_shift = np.std(flat_shifted)
        if std_res > 1e-4 and std_shift > 1e-4:
            correlation = float(np.corrcoef(flat_residual, flat_shifted)[0, 1])
        else:
            correlation = 0.0

        # 6. Compute Shannon Entropy on flat residual
        hist, _ = np.histogram(flat_residual, bins=64, range=(-20, 20), density=False)
        hist = hist[hist > 0]
        probs = hist / np.sum(hist)
        noise_entropy = float(-np.sum(probs * np.log2(probs)))

        # 7. Compute Calibrated PRNU Anomaly Score:
        # Real camera photos:
        # - Physical sensor shot noise gives variance >= 0.90 in flat regions
        # - Spatial correlation is low (<= 0.35)
        # AI / Synthetic images:
        # - Excessively smooth flat regions without sensor noise floor (variance < 0.70)
        # - Interpolation / upsampling correlations (correlation > 0.50)
        corr_score = max(0.0, min((correlation - 0.35) / 0.35, 1.0))
        noise_floor_score = max(0.0, min((0.85 - variance) / 0.85, 1.0))

        anomaly_score = 0.5 * corr_score + 0.5 * noise_floor_score

        return {
            "prnu_anomaly_score": round(float(anomaly_score), 4),
            "noise_entropy": round(float(noise_entropy), 4),
            "noise_variance": round(float(variance), 4),
            "noise_spatial_correlation": round(float(correlation), 4)
        }

if __name__ == "__main__":
    analyzer = PRNUAnalyzer()
    print("PRNU Noise Analyzer Initialized.")
