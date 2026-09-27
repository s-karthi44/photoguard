from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Load dataset
dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")

# Where to save
BASE_DIR = Path("../datasets/raw/Defactify")

label_map = {
    0: "Real",
    1: "SD21",
    2: "SDXL",
    3: "SD3",
    4: "DALLE3",
    5: "Midjourney"
}

# Number of images per class
MAX_IMAGES = 2000

saved = {name: 0 for name in label_map.values()}

for split in ["train", "validation", "test"]:

    print(f"\nProcessing {split}...")

    for sample in tqdm(dataset[split]):

        folder = label_map[sample["Label_B"]]

        if saved[folder] >= MAX_IMAGES:
            continue

        save_dir = BASE_DIR / folder
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{folder}_{saved[folder]:05d}.jpg"

        sample["Image"].save(save_dir / filename)

        saved[folder] += 1

    if all(v >= MAX_IMAGES for v in saved.values()):
        break

print("\nDone!\n")

for k, v in saved.items():
    print(f"{k}: {v}")