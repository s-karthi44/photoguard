"""
============================================================
PhotoGuard AI - Forensic Robustness & Perturbation Test Suite
Author : Vinay Kirithic

Evaluates PhotoGuard Multi-Modal Decision Fusion under common
real-world transmission and social media perturbations:
1. JPEG Re-compression (Quality 90, 75, 50)
2. Gaussian Blur (sigma=1.0, sigma=2.0)
3. Downsampling & Re-upscaling (50% scale)
============================================================
"""

import os
import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageFilter
from modules.fusion import PhotoGuardFusionEngine
from modules.paths import OUTPUT_DIR

def apply_perturbations(input_path: Path, temp_dir: Path) -> dict[str, Path]:
    """Generates perturbed variants of the source image."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    
    variants = {}
    variants["Baseline (Original)"] = input_path
    
    # 1. JPEG Compression variants
    for q in [90, 75, 50]:
        out_path = temp_dir / f"perturb_jpeg_q{q}_{input_path.stem}.jpg"
        img.save(out_path, "JPEG", quality=q)
        variants[f"JPEG (Quality {q})"] = out_path

    # 2. Gaussian Blur variants
    for radius in [1.0, 2.0]:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        out_path = temp_dir / f"perturb_blur_r{int(radius)}_{input_path.stem}.png"
        blurred.save(out_path, "PNG")
        variants[f"Gaussian Blur (r={radius})"] = out_path

    # 3. Downscale 50% & Upscale back (Social media re-scaling artifact)
    orig_w, orig_h = img.size
    downscaled = img.resize((orig_w // 2, orig_h // 2), Image.Resampling.BILINEAR)
    upscaled = downscaled.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
    out_path = temp_dir / f"perturb_rescale_{input_path.stem}.png"
    upscaled.save(out_path, "PNG")
    variants["Downscale 50% & Upscale"] = out_path

    return variants

def run_robustness_suite(image_path: Path):
    if not image_path.exists():
        print(f"Error: Target image '{image_path}' not found.")
        return

    print("=" * 88)
    print("              PHOTOGUARD AI - FORENSIC ROBUSTNESS BENCHMARK")
    print("=" * 88)
    print(f"Target Image: {image_path.resolve()}\n")

    temp_dir = OUTPUT_DIR / "temp" / "robustness_test"
    variants = apply_perturbations(image_path, temp_dir)

    engine = PhotoGuardFusionEngine()

    results = []
    for label, path in variants.items():
        res = engine.analyze_image(path)
        bk = res["breakdown"]
        results.append({
            "Perturbation": label,
            "Verdict": res["prediction"],
            "Confidence": f"{res['confidence']:.1f}%",
            "Fused Score": f"{res['fusion_score']:.3f}",
            "CNN": f"{bk['cnn_score'] * 100:.1f}%",
            "PRNU": f"{bk['prnu_score'] * 100:.1f}%",
            "ELA": f"{bk['ela_score'] * 100:.1f}%",
            "FFT": f"{bk['fft_score'] * 100:.1f}%",
        })

    # Print Table
    header = f"{'Perturbation':<26} | {'Verdict':<10} | {'Conf':<6} | {'Fused':<6} | {'CNN':<6} | {'PRNU':<6} | {'ELA':<6} | {'FFT':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Perturbation']:<26} | {r['Verdict']:<10} | {r['Confidence']:<6} | {r['Fused Score']:<6} | {r['CNN']:<6} | {r['PRNU']:<6} | {r['ELA']:<6} | {r['FFT']:<6}")
    print("=" * len(header))

    # Clean up temporary test files
    for path in variants.values():
        if path != image_path and path.exists():
            path.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

if __name__ == "__main__":
    test_img = Path("outputs/gradcam/gradcam_ai_sample.jpg")
    if len(sys.argv) > 1:
        test_img = Path(sys.argv[1])
    run_robustness_suite(test_img)
