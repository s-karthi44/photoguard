import sys
from pathlib import Path
from modules.fusion import PhotoGuardFusionEngine

def main():
    # Allow passing an image path as command-line argument, default to a verified dataset image
    image = "datasets/train/real/000000000086.jpg"
    if len(sys.argv) > 1:
        image = sys.argv[1]

    image_path = Path(image)
    if not image_path.exists():
        print(f"Error: Target image file '{image}' does not exist.")
        return

    print("=" * 60)
    print("                 PHOTOGUARD AI - FUSION ENGINE")
    print("=" * 60)
    print(f"Analyzing Image: {image_path.name}")
    print(f"Path: {image_path.resolve()}\n")

    # Initialize and run Fusion Engine
    engine = PhotoGuardFusionEngine()
    result = engine.analyze_image(image_path)

    print("-" * 60)
    print("ANALYSIS RESULTS")
    print("-" * 60)
    print(f"Final Verdict      : {result['prediction']}")
    print(f"Confidence         : {result['confidence']}%")
    print(f"Fusion Score (AI)  : {result['fusion_score']}")
    print(f"Verdict Reason     : {result['verdict_reason']}")
    print("-" * 60)
    
    print("\nSUB-DETECTORS BREAKDOWN (Probability of being AI):")
    print("=" * 60)
    bk = result["breakdown"]
    print(f"1. Dual-Stream CNN Score   : {bk['cnn_score'] * 100:.2f}%")
    print(f"2. PRNU Camera Noise Score : {bk['prnu_score'] * 100:.2f}%")
    print(f"3. ELA Error Level Score   : {bk['ela_score'] * 100:.2f}%")
    print(f"4. FFT Spectral Grid Score : {bk['fft_score'] * 100:.2f}%")
    print(f"5. EXIF Metadata Score     : {bk['metadata_score'] * 100:.2f}%")
    print("=" * 60)

    details = result["details"]
    splicing = result.get("splicing", {})
    print("\nIMAGE CAPTURE & FORENSIC LOCALIZATION:")
    print("-" * 60)
    print(f"EXIF Software Tag          : {details['software'] or 'N/A'}")
    print(f"Camera Manufacturer        : {details['camera_make'] or 'N/A'}")
    print(f"Camera Model               : {details['camera_model'] or 'N/A'}")
    print(f"Localized Splicing Found   : {'YES' if splicing.get('is_spliced') else 'NO'}")
    if splicing.get('is_spliced'):
        print(f"Spliced Tampered Area      : {splicing.get('spliced_area_percentage')}%")
        print(f"Spliced Regions Count      : {splicing.get('spliced_regions_count')}")
        print(f"Splicing Mask Visualization: {splicing.get('mask_path')}")
        print(f"Splicing Box Overlay Path  : {splicing.get('overlay_path')}")
    print(f"ELA Output Visualization   : {details['ela_visual_path']}")
    print(f"FFT Output Visualization   : {details.get('fft_visual_path') or 'N/A'}")
    print("=" * 60)

if __name__ == "__main__":
    main()