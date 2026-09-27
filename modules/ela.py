"""
============================================================
PhotoGuard AI
Error Level Analysis (ELA) Module

Author : Vinay Kirithic

Description:
Detects JPEG compression anomalies and inconsistencies in images.
AI-generated and digitally manipulated images display uneven ELA patterns,
while original camera photographs exhibit a uniform error profile.
============================================================
"""

import os
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance, ImageStat
from modules.paths import OUTPUT_DIR

class ELAAnalyzer:
    def __init__(self, quality: int = 95):
        self.quality = quality
        self.ela_dir = OUTPUT_DIR / "ela"
        self.ela_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, image_path: str | Path) -> dict:
        """
        Performs Error Level Analysis (ELA) on the input image.
        Saves the ELA residual visual overlay.
        
        Returns:
            dict containing:
                - ela_anomaly_score: float (0.0 to 1.0, where high indicates anomalous compression)
                - ela_output_path: str (path to the saved visual ELA image)
                - avg_difference: float (average pixel error level)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Open original image
        original = Image.open(image_path).convert("RGB")
        
        # 2. Compress and save temporarily
        temp_path = self.ela_dir / f"temp_{image_path.name}.jpg"
        original.save(temp_path, "JPEG", quality=self.quality)
        
        # 3. Open compressed version
        compressed = Image.open(temp_path)
        
        # 4. Compute absolute pixel differences
        ela_image = ImageChops.difference(original, compressed)
        
        # 5. Extract statistics
        stat = ImageStat.Stat(ela_image)
        avg_diff = sum(stat.mean) / len(stat.mean)
        
        # 6. Amplify difference contrast for visualization
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
            
        scale = 255.0 / max_diff
        visual_ela = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Save ELA visualization
        output_filename = f"ela_{image_path.stem}.jpg"
        ela_output_path = self.ela_dir / output_filename
        visual_ela.save(ela_output_path, "JPEG", quality=90)
        
        # Clean up temporary compressed file
        if temp_path.exists():
            os.remove(temp_path)
            
        # Calculate anomaly score: ELA average differences above 4.0 start indicating
        # compression mismatch/AI generation grids. Map to [0.0, 1.0] sigmoid-like scale.
        # Normal camera images usually fall under 1.5 to 2.5 average difference.
        # AI images or double saved local manipulations range from 3.5 to 8+.
        anomaly_score = min(max((avg_diff - 1.5) / 5.0, 0.0), 1.0)

        return {
            "ela_anomaly_score": round(anomaly_score, 4),
            "ela_output_path": str(ela_output_path),
            "avg_difference": round(avg_diff, 4)
        }

if __name__ == "__main__":
    analyzer = ELAAnalyzer()
    print("ELA Module Initialized.")
