"""
============================================================
PhotoGuard AI
Multi-Modal Decision Fusion Engine

Author : Vinay Kirithic

Description:
Aggregates predictions and statistical anomaly scores from multiple sub-detectors:
1. Spatial & Frequency CNN (EfficientNet + SRM + FFT features)
2. PRNU Camera Noise Residual (stochastic noise properties)
3. JPEG Error Level Analysis (ELA double-compression anomalies)
4. FFT high-frequency grid-artifacts (periodic structure peaks)
5. EXIF Metadata Tag Verifier (AI/editor tags signature check)
============================================================
"""

from pathlib import Path
from modules.cnn.predict import PhotoGuardPredictor
from modules.prnu import PRNUAnalyzer
from modules.ela import ELAAnalyzer
from modules.fft import FFTAnalyzer
from modules.metadata import MetadataVerifier
from modules.splicing import SplicingDetector

class PhotoGuardFusionEngine:
    def __init__(self):
        self.predictor = PhotoGuardPredictor()
        self.prnu_analyzer = PRNUAnalyzer()
        self.ela_analyzer = ELAAnalyzer()
        self.fft_analyzer = FFTAnalyzer()
        self.metadata_verifier = MetadataVerifier()
        self.splicing_detector = SplicingDetector()

        # Decision Fusion Weights
        self.w_cnn = 0.50
        self.w_prnu = 0.20
        self.w_ela = 0.15
        self.w_fft = 0.15

    def analyze_image(self, image_path: str | Path) -> dict:
        """
        Runs all sub-detectors on the image and fuses their outputs.
        
        Returns:
            dict with final fused prediction, confidence, and individual sub-scores.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Run all sub-analyzers
        cnn_res = self.predictor.predict_image(image_path)
        prnu_res = self.prnu_analyzer.analyze(image_path)
        ela_res = self.ela_analyzer.analyze(image_path)
        fft_res = self.fft_analyzer.analyze(image_path)
        meta_res = self.metadata_verifier.analyze(image_path)
        splicing_res = self.splicing_detector.analyze(image_path)

        # 2. Extract probability of being AI (0.0 to 1.0)
        # Class names: 0 -> AI, 1 -> Real
        cnn_ai_prob = cnn_res["probabilities"].get("AI", cnn_res["probabilities"].get("ai", 0.0)) / 100.0
        prnu_score = prnu_res["prnu_anomaly_score"]
        ela_score = ela_res["ela_anomaly_score"]
        fft_score = fft_res["fft_grid_score"]

        # Adapt weights based on file format (lossless PNG doesn't support ELA double compression)
        is_png = image_path.suffix.lower() == ".png"
        w_cnn = self.w_cnn
        w_prnu = self.w_prnu
        w_ela = self.w_ela
        w_fft = self.w_fft
        
        if is_png:
            w_ela = 0.0
            w_cnn = 0.60
            w_fft = 0.20

        # 3. Override if explicit AI EXIF tags are found
        if meta_res["metadata_anomaly_score"] == 1.0:
            return {
                "image_name": image_path.name,
                "prediction": "AI",
                "confidence": 100.0,
                "fusion_score": 1.0,
                "probabilities": {
                    "AI": 100.0,
                    "Real": 0.0
                },
                "breakdown": {
                    "cnn_score": round(cnn_ai_prob, 4),
                    "prnu_score": round(prnu_score, 4),
                    "ela_score": round(ela_score, 4),
                    "fft_score": round(fft_score, 4),
                    "metadata_score": 1.0
                },
                "details": {
                    "software": meta_res["software_tag"],
                    "camera_make": meta_res["camera_make"],
                    "camera_model": meta_res["camera_model"],
                    "ela_visual_path": ela_res["ela_output_path"],
                    "fft_visual_path": fft_res.get("fft_output_path")
                },
                "verdict_reason": f"Explicit AI generation tag detected in image metadata ({meta_res['software_tag']})."
            }

        # 4. Perform Weighted Fusion
        fused_ai_score = (
            w_cnn * cnn_ai_prob +
            w_prnu * prnu_score +
            w_ela * ela_score +
            w_fft * fft_score
        )

        # 5. Output unified 3-state prediction verdict
        # Calibrated anomaly checks for physical camera sensor & spectral grid footprints
        has_physical_noise_anomaly = prnu_score >= 0.70
        has_spectral_grid_anomaly = fft_score >= 0.60
        is_cnn_deceived = cnn_ai_prob < 0.20

        # Gated Splicing: Only trigger local inpainting verdict if the image has non-negligible
        # AI/manipulation features (avoiding false alarms on intricate real statues/textures)
        is_true_splicing = splicing_res.get("is_spliced") and (cnn_ai_prob >= 0.15 or prnu_score >= 0.40)

        if is_true_splicing:
            prediction = "AI (Spliced)"
            confidence = max(splicing_res["regions"][0]["confidence"], 80.0)
            fused_ai_score = max(fused_ai_score, confidence / 100.0)
            verdict_reason = f"Localized Inpainting / Splicing Detected: Tampered region identified covering {splicing_res['spliced_area_percentage']}% of the image."
        elif has_physical_noise_anomaly and has_spectral_grid_anomaly and is_cnn_deceived:
            # Physical camera sensor footprint is completely missing and strong frequency grid lattice is present
            prediction = "AI"
            fused_ai_score = max(fused_ai_score, 0.68)
            confidence = round(fused_ai_score * 100.0, 2)
            verdict_reason = f"AI Generated: Physical camera sensor noise is absent (PRNU: {round(prnu_score*100)}%) and spectral grid artifacts detected (FFT: {round(fft_score*100)}%) despite deceptive visual realism."
        elif (has_physical_noise_anomaly or has_spectral_grid_anomaly) and is_cnn_deceived and fused_ai_score >= 0.40:
            prediction = "Uncertain"
            confidence = round(max(fused_ai_score, 1.0 - fused_ai_score) * 100.0, 2)
            anomalies = []
            if has_physical_noise_anomaly:
                anomalies.append(f"PRNU sensor noise missing ({round(prnu_score*100)}%)")
            if has_spectral_grid_anomaly:
                anomalies.append(f"FFT frequency grid detected ({round(fft_score*100)}%)")
            verdict_reason = f"Inconclusive / Potential AI: Visual textures appear natural to the CNN, but physical camera forensics show anomalies: {', '.join(anomalies)}."
        elif fused_ai_score >= 0.55:
            prediction = "AI"
            confidence = round(fused_ai_score * 100.0, 2)
            verdict_reason = "AI Generated / Manipulated: Strong generative artifacts detected across spatial and frequency sub-detectors."
        elif fused_ai_score < 0.35:
            prediction = "Real"
            confidence = round((1.0 - fused_ai_score) * 100.0, 2)
            verdict_reason = "Verified Authentic: Consistent camera sensor noise, natural frequency decay, and uniform compression profile."
        else:
            prediction = "Uncertain"
            confidence = round(max(fused_ai_score, 1.0 - fused_ai_score) * 100.0, 2)
            # Detect conflicting signals
            if (cnn_ai_prob > 0.60 and (ela_score < 0.30 or prnu_score < 0.30)) or \
               ((ela_score > 0.60 or prnu_score > 0.60) and cnn_ai_prob < 0.30):
                verdict_reason = "Inconclusive: Conflicting evidence between spatial visual features and forensic noise/compression patterns."
            else:
                verdict_reason = "Inconclusive / Mixed Evidence: Sub-detector signals fall in the borderline uncertainty region (35%-55%). Further forensic review recommended."

        return {
            "image_name": image_path.name,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "fusion_score": round(fused_ai_score, 4),
            "probabilities": {
                "AI": round(fused_ai_score * 100, 2),
                "Real": round((1.0 - fused_ai_score) * 100, 2)
            },
            "breakdown": {
                "cnn_score": round(cnn_ai_prob, 4),
                "prnu_score": round(prnu_score, 4),
                "ela_score": round(ela_score, 4),
                "fft_score": round(fft_score, 4),
                "metadata_score": meta_res["metadata_anomaly_score"]
            },
            "details": {
                "software": meta_res["software_tag"],
                "camera_make": meta_res["camera_make"],
                "camera_model": meta_res["camera_model"],
                "ela_visual_path": ela_res["ela_output_path"],
                "fft_visual_path": fft_res.get("fft_output_path"),
                "splicing_overlay_path": splicing_res.get("overlay_path"),
                "splicing_mask_path": splicing_res.get("mask_path")
            },
            "splicing": splicing_res,
            "verdict_reason": verdict_reason
        }

if __name__ == "__main__":
    engine = PhotoGuardFusionEngine()
    print("Fusion Engine ready.")
