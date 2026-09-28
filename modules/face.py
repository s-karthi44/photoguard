"""
============================================================
PhotoGuard AI
Facial Landmark & GAN Alignment Forensic Analyzer

Author : Vinay Kirithic

Description:
Detects facial presence and evaluates structural/anatomical signatures
characteristic of GAN-generated portraits (StyleGAN / FFHQ):
1. Canonical FFHQ normalized alignment (horizontal eye tilt, fixed eye line height,
   and exact bilateral border centering).
2. Corneal specular reflection (catchlight) geometric disparity between eyes.
============================================================
"""

from pathlib import Path
import cv2
import numpy as np

class FaceForensicAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def analyze(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "face_detected": False,
                "is_ffhq_aligned": False,
                "catchlight_disparity": 0.0,
                "gan_face_score": 0.0,
                "details": {}
            }

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return {
                "face_detected": False,
                "is_ffhq_aligned": False,
                "catchlight_disparity": 0.0,
                "gan_face_score": 0.0,
                "details": {}
            }

        fx, fy, fw, fh = faces[0]
        face_roi = gray[fy:fy+fh, fx:fx+fw]

        # Detect eyes in the upper 55% of the detected face
        eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        valid_eyes = [e for e in eyes if e[1] < fh * 0.55]
        valid_eyes.sort(key=lambda e: e[0])  # Sort left to right

        is_ffhq_aligned = False
        catchlight_disparity = 0.0
        gan_face_score = 0.0
        details = {
            "face_box": [int(fx), int(fy), int(fw), int(fh)],
            "eyes_detected": len(valid_eyes)
        }

        if len(valid_eyes) >= 2:
            e1, e2 = valid_eyes[0], valid_eyes[-1]
            e1_center = (fx + e1[0] + e1[2] // 2, fy + e1[1] + e1[3] // 2)
            e2_center = (fx + e2[0] + e2[2] // 2, fy + e2[1] + e2[3] // 2)

            dy = abs(e1_center[1] - e2_center[1])
            eye_y_ratio = ((e1_center[1] + e2_center[1]) / 2.0) / h
            dist_left = e1_center[0]
            dist_right = w - e2_center[0]
            centering_diff = abs(dist_left - dist_right) / float(w)

            # Canonical FFHQ alignment criteria:
            # - Horizontal eye tilt <= 2% height
            # - Eye line between 38% and 52% height
            # - Symmetrically centered (|left - right| < 5% canvas width)
            is_ffhq_aligned = bool(dy <= max(3, h * 0.02) and 0.36 <= eye_y_ratio <= 0.52 and centering_diff < 0.06)

            # Corneal specular catchlight extraction (top 4% brightest pixels in each eye box)
            eye1_crop = face_roi[e1[1]:e1[1]+e1[3], e1[0]:e1[0]+e1[2]]
            eye2_crop = face_roi[e2[1]:e2[1]+e2[3], e2[0]:e2[0]+e2[2]]
            if eye1_crop.size > 0 and eye2_crop.size > 0:
                p96_1 = np.percentile(eye1_crop, 96)
                p96_2 = np.percentile(eye2_crop, 96)
                c1_area = float(np.sum(eye1_crop >= p96_1))
                c2_area = float(np.sum(eye2_crop >= p96_2))
                max_area = max(c1_area, c2_area, 1.0)
                catchlight_disparity = round(abs(c1_area - c2_area) / max_area, 4)

            details.update({
                "eye_y_ratio": round(float(eye_y_ratio), 3),
                "centering_diff": round(float(centering_diff), 3),
                "eye_tilt_pixels": int(dy),
                "catchlight_disparity": catchlight_disparity
            })

            if is_ffhq_aligned:
                gan_face_score = 0.70 + (0.20 * min(catchlight_disparity * 2.0, 1.0))
            else:
                gan_face_score = 0.30 * min(catchlight_disparity * 1.5, 1.0)

        return {
            "face_detected": True,
            "is_ffhq_aligned": is_ffhq_aligned,
            "catchlight_disparity": catchlight_disparity,
            "gan_face_score": round(float(gan_face_score), 4),
            "details": details
        }
