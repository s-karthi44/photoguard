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

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

class PRNUAnalyzer:
    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def extract_noise_residual(self, img_np: np.ndarray) -> np.ndarray:
        """Extracts the noise residual W = I - F(I) using a Gaussian denoising filter."""
        denoised = gaussian_filter(img_np, sigma=self.sigma)
        return img_np - denoised

    def analyze(self, image_path: str | Path) -> dict:
        """
        Extracts and analyzes high-frequency noise features from the image.
        
        Returns:
            dict containing:
                - prnu_anomaly_score: float (0.0 to 1.0, high indicates anomalous/synthetic noise profile)
                - noise_entropy: float (entropy of the noise residual)
                - noise_variance: float (variance of the noise residual)
                - noise_spatial_correlation: float (correlation between adjacent noise pixels)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Load image in grayscale
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, dtype=np.float32)

        # 2. Extract noise residual
        residual = self.extract_noise_residual(img_np)

        # 3. Calculate Noise Variance
        variance = np.var(residual)

        # 4. Calculate Noise Entropy (Shannon Entropy of residual histogram)
        hist, _ = np.histogram(residual.flatten(), bins=256, range=(-50, 50), density=True)
        hist = hist[hist > 0]
        noise_entropy = -np.sum(hist * np.log2(hist))

        # 5. Calculate Spatial Correlation of Noise (AI images often show correlation patterns due to upsampling)
        # Shift residual to the right and compute correlation coefficient
        shifted_residual = np.roll(residual, shift=1, axis=1)
        correlation = np.corrcoef(residual.flatten(), shifted_residual.flatten())[0, 1]

        # 6. Compute Anomaly Score
        # Natural sensor noise typically:
        # - Has high entropy (usually > 4.5)
        # - Has low spatial correlation (close to 0, e.g., < 0.1)
        # AI-generated images typically:
        # - Have lower noise entropy due to blurring/filtering
        # - Have higher spatial correlation due to CNN grid generation
        entropy_factor = max(0.0, min((5.0 - noise_entropy) / 1.5, 1.0))
        correlation_factor = max(0.0, min((abs(correlation) - 0.05) / 0.25, 1.0))

        anomaly_score = 0.4 * entropy_factor + 0.6 * correlation_factor

        return {
            "prnu_anomaly_score": round(float(anomaly_score), 4),
            "noise_entropy": round(float(noise_entropy), 4),
            "noise_variance": round(float(variance), 4),
            "noise_spatial_correlation": round(float(correlation), 4)
        }

if __name__ == "__main__":
    analyzer = PRNUAnalyzer()
    print("PRNU Noise Analyzer Initialized.")
