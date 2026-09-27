from pathlib import Path
import random
import shutil

# ----------------------------
# Configuration
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = PROJECT_ROOT / "datasets" / "raw" / "COCO" / "train2017"
DESTINATION = PROJECT_ROOT / "datasets" / "processed" / "COCO"

NUMBER_OF_IMAGES = 10000

DESTINATION.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Get all images
# ----------------------------

images = list(SOURCE.glob("*.jpg"))

print(f"Total Images Found: {len(images)}")

if len(images) < NUMBER_OF_IMAGES:
    NUMBER_OF_IMAGES = len(images)

selected = random.sample(images, NUMBER_OF_IMAGES)

print(f"Copying {NUMBER_OF_IMAGES} images...\n")

for image in selected:
    shutil.copy2(image, DESTINATION / image.name)

print("Done!")
print(f"Saved to: {DESTINATION}")