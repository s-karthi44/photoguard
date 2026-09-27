from pathlib import Path
import random
import shutil

# Set a fixed seed so the same images are selected every time
random.seed(42)

# Source and destination folders
SOURCE = Path("datasets/raw/Flickr30k")
DESTINATION = Path("datasets/processed/Flickr30k")

DESTINATION.mkdir(parents=True, exist_ok=True)

# Supported image formats
extensions = (".jpg", ".jpeg", ".png")

# Find all images
images = [
    img for img in SOURCE.iterdir()
    if img.suffix.lower() in extensions
]

print(f"Found {len(images)} images.")

# Number of images to sample
sample_size = min(10000, len(images))

selected = random.sample(images, sample_size)

print(f"Copying {sample_size} images...")

for image in selected:
    shutil.copy2(image, DESTINATION / image.name)

print(f"\nDone! {sample_size} images copied to:")
print(DESTINATION)