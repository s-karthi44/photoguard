"""
============================================================
PhotoGuard AI
FFT Frequency Domain Grid Artifact Detector

Author : Vinay Kirithic

Description:
Computes the 2D Fast Fourier Transform (FFT) of an image and analyzes
the high-frequency spectrum. AI generative models (GANs, Diffusion) leave
periodic grid artifacts (upsampling noise) that show up as discrete
spectral peaks, whereas natural camera images display a smooth decay.
============================================================
"""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from modules.paths import OUTPUT_DIR
import cv2

class FFTAnalyzer:
    def __init__(self):
        self.metrics_dir = OUTPUT_DIR / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.fft_dir = OUTPUT_DIR / "fft"
        self.fft_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, image_path: str | Path) -> dict:
        """
        Calculates spectral peaks in the high-frequency band of the 2D FFT.
        
        Returns:
            dict containing:
                - fft_grid_score: float (0.0 to 1.0, high indicates strong periodic grid artifacts)
                - peak_count: int (number of detected high-frequency peak anomalies)
                - spectral_peakiness: float (ratio of peak amplitude to high-frequency mean)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Load image in grayscale
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, dtype=np.float32)

        # 2. Perform 2D FFT & shift low frequencies to center
        f_transform = np.fft.fft2(img_np)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Handle log scale for statistical robustness
        log_magnitude = np.log(magnitude + 1)
        h, w = log_magnitude.shape

        # 3. Create a circular mask to exclude the center low-frequencies (DC component)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 8  # Exclude central 1/8th of frequencies

        y, x = np.ogrid[:h, :w]
        mask_center = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        # High frequency area (medium and high bands)
        high_freq_mag = np.copy(log_magnitude)
        high_freq_mag[mask_center] = 0

        # 4. Detect local peaks using a sliding maximum filter
        # Neighborhood of size 15x15 to locate distinct peaks
        neighborhood_size = 15
        local_max = maximum_filter(high_freq_mag, size=neighborhood_size) == high_freq_mag
        
        # Keep peaks only where the frequency magnitude is high enough
        threshold = np.mean(high_freq_mag[high_freq_mag > 0]) + 3 * np.std(high_freq_mag[high_freq_mag > 0])
        peaks = local_max & (high_freq_mag > threshold)
        
        peak_y, peak_x = np.where(peaks)
        peak_count = len(peak_y)

        # 5. Measure "peakiness" of the high-frequency spectrum
        mean_val = np.mean(high_freq_mag[high_freq_mag > 0]) if np.any(high_freq_mag > 0) else 1.0
        max_val = np.max(high_freq_mag) if np.any(high_freq_mag > 0) else 1.0
        peakiness = max_val / mean_val

        # Map peakiness and peak_count to [0.0, 1.0] anomaly score
        # Camera images rarely have high frequency spectral peaks (peak_count <= 2, peakiness < 2.0)
        # AI images with grids often have multiple peaks or extremely high peakiness.
        score_count = min(peak_count / 10.0, 1.0)
        score_peakiness = min(max((peakiness - 1.8) / 2.0, 0.0), 1.0)
        
        grid_score = 0.4 * score_count + 0.6 * score_peakiness

        # 6. Generate normalized 2D FFT magnitude spectrum visualizer
        norm_mag = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_spectrum = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        
        output_filename = f"fft_{image_path.stem}.png"
        fft_output_path = self.fft_dir / output_filename
        cv2.imwrite(str(fft_output_path), color_spectrum)

        return {
            "fft_grid_score": round(float(grid_score), 4),
            "peak_count": int(peak_count),
            "spectral_peakiness": round(float(peakiness), 4),
            "fft_output_path": str(fft_output_path)
        }

if __name__ == "__main__":
    analyzer = FFTAnalyzer()
    print("FFT Module Initialized.")
