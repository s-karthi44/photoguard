"""
=========================================================
PhotoGuard AI
FastAPI Inference & Detection Server

Author : Vinay Kirithic

Description:
REST API Backend serving PhotoGuard CNN predictions and Grad-CAM
explainability heatmaps for the Flutter mobile application.
=========================================================
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from modules.fusion import PhotoGuardFusionEngine
from modules.cnn.gradcam import generate_gradcam_overlay
from modules.paths import OUTPUT_DIR

app = FastAPI(
    title="PhotoGuard AI API",
    description="Backend service for AI Image Detection & Explainability",
    version="1.0.0"
)

# Enable CORS for Flutter Web & Mobile requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Engine
predictor = PhotoGuardFusionEngine()

# Static directory for serving Grad-CAM images
GRADCAM_SERVE_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_SERVE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/gradcam", StaticFiles(directory=str(GRADCAM_SERVE_DIR)), name="gradcam")

# Static directory for serving FFT images
FFT_SERVE_DIR = OUTPUT_DIR / "fft"
FFT_SERVE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/fft", StaticFiles(directory=str(FFT_SERVE_DIR)), name="fft")

# Static directory for serving ELA images
ELA_SERVE_DIR = OUTPUT_DIR / "ela"
ELA_SERVE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/ela", StaticFiles(directory=str(ELA_SERVE_DIR)), name="ela")

# Static directory for serving Splicing Mask & Overlay images
SPLICING_SERVE_DIR = OUTPUT_DIR / "splicing"
SPLICING_SERVE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/splicing", StaticFiles(directory=str(SPLICING_SERVE_DIR)), name="splicing")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "PhotoGuard AI Inference Server",
        "version": "1.0.0"
    }


@app.post("/api/v1/predict")
async def predict_photo(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns:
    - Prediction label ('AI' vs 'Real')
    - Confidence percentage
    - Class probability scores
    - Grad-CAM heatmap base64 string
    """
    # Check image extension or content type
    ext = Path(file.filename).suffix.lower()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    is_image_type = file.content_type and file.content_type.startswith("image/")
    
    if not (is_image_type or ext in valid_exts):
        raise HTTPException(status_code=400, detail="File must be an image (.jpg, .jpeg, .png, .webp)")

    # Save uploaded file temporarily inside outputs/temp
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Run Multi-Modal Fusion Inference
        result = predictor.analyze_image(temp_path)

        # Generate Grad-CAM Heatmap
        gradcam_filename = f"gradcam_{Path(file.filename).stem}.jpg"
        gradcam_output_path = generate_gradcam_overlay(temp_path, output_filename=gradcam_filename)

        # Encode Grad-CAM overlay image to Base64 for instant Flutter display
        with open(gradcam_output_path, "rb") as image_file:
            encoded_gradcam = base64.b64encode(image_file.read()).decode("utf-8")

        result["gradcam_base64"] = encoded_gradcam
        result["gradcam_url"] = f"/static/gradcam/{gradcam_filename}"

        # Encode ELA visualization to Base64
        ela_visual_path = result["details"].get("ela_visual_path")
        if ela_visual_path and os.path.exists(ela_visual_path):
            with open(ela_visual_path, "rb") as image_file:
                encoded_ela = base64.b64encode(image_file.read()).decode("utf-8")
            result["ela_base64"] = encoded_ela
            result["ela_url"] = f"/static/ela/{Path(ela_visual_path).name}"
        else:
            result["ela_base64"] = None
            result["ela_url"] = None

        # Encode FFT visualization to Base64
        fft_visual_path = result["details"].get("fft_visual_path")
        if fft_visual_path and os.path.exists(fft_visual_path):
            with open(fft_visual_path, "rb") as image_file:
                encoded_fft = base64.b64encode(image_file.read()).decode("utf-8")
            result["fft_base64"] = encoded_fft
            result["fft_url"] = f"/static/fft/{Path(fft_visual_path).name}"
        else:
            result["fft_base64"] = None
            result["fft_url"] = None

        # Encode Localized Splicing Mask & Overlay to Base64
        splicing_overlay_path = result.get("details", {}).get("splicing_overlay_path")
        if splicing_overlay_path and os.path.exists(splicing_overlay_path):
            with open(splicing_overlay_path, "rb") as image_file:
                result["splicing_overlay_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
            result["splicing_overlay_url"] = f"/static/splicing/{Path(splicing_overlay_path).name}"
        else:
            result["splicing_overlay_base64"] = None
            result["splicing_overlay_url"] = None

        splicing_mask_path = result.get("details", {}).get("splicing_mask_path")
        if splicing_mask_path and os.path.exists(splicing_mask_path):
            with open(splicing_mask_path, "rb") as image_file:
                result["splicing_mask_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
            result["splicing_mask_url"] = f"/static/splicing/{Path(splicing_mask_path).name}"
        else:
            result["splicing_mask_base64"] = None
            result["splicing_mask_url"] = None

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

    finally:
        # Clean temp uploaded file
        if temp_path.exists():
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
