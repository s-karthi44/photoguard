"""
============================================================
PhotoGuard AI
EXIF Metadata Verifier Module

Author : Vinay Kirithic

Description:
Parses EXIF metadata tags from images to check for digital manipulation
signatures, explicit AI creator tags, or complete absence of camera capture tags.
============================================================
"""

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

class MetadataVerifier:
    def __init__(self):
        # Software or software comments linked to AI generation or editors
        self.ai_keywords = {
            "midjourney", "stable diffusion", "dall-e", "dalle",
            "firefly", "ideogram", "novelai", "wombo", "craiyon",
            "photoshop", "gimp", "canva", "sdxl", "sd3"
        }

    def analyze(self, image_path: str | Path) -> dict:
        """
        Parses EXIF tags to identify authenticity.
        
        Returns:
            dict containing:
                - metadata_anomaly_score: float (0.0 to 1.0, where 1.0 means explicit AI/Editor tags)
                - software_tag: str (software used if found)
                - camera_make: str (camera manufacturer if found)
                - camera_model: str (camera model if found)
                - has_exif: bool (whether EXIF tags exist)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        has_exif = False
        software_tag = None
        camera_make = None
        camera_model = None
        explicit_ai_found = False

        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                if exif_data:
                    has_exif = True
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Convert to string and inspect
                        val_str = str(value).strip()
                        
                        if tag_name == "Software":
                            software_tag = val_str
                            if any(k in val_str.lower() for k in self.ai_keywords):
                                explicit_ai_found = True
                                
                        elif tag_name == "Make":
                            camera_make = val_str
                        elif tag_name == "Model":
                            camera_model = val_str
                            
                        # Also check description/user comments for AI signatures
                        elif tag_name in ["ImageDescription", "UserComment"]:
                            if any(k in val_str.lower() for k in self.ai_keywords):
                                explicit_ai_found = True
        except Exception:
            pass

        # Calculate anomaly score
        if explicit_ai_found:
            anomaly_score = 1.0
        elif not has_exif:
            # Completely missing EXIF is common for internet/AI images, but not a positive proof
            anomaly_score = 0.5
        elif camera_make is None and camera_model is None:
            # Has EXIF but lacks any standard camera make/model (suspect)
            anomaly_score = 0.4
        else:
            # Standard camera details present and no AI keywords found
            anomaly_score = 0.0

        return {
            "metadata_anomaly_score": float(anomaly_score),
            "software_tag": software_tag,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "has_exif": has_exif
        }

if __name__ == "__main__":
    verifier = MetadataVerifier()
    print("Metadata Verifier Initialized.")
