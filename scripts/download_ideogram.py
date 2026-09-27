from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "datasets" / "raw" / "ai" / "Ideogram"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading bitmind/ideogram-27k dataset in streaming mode...")
dataset = load_dataset("bitmind/ideogram-27k", streaming=True)

# Check for existing downloads to resume efficiently
existing_files = list(SAVE_DIR.glob("ideogram_*.jpg"))
saved_count = len(existing_files)

dataset_train = dataset["train"]
if saved_count > 0:
    print(f"Found {saved_count} existing images in target directory. Resuming download...")
    dataset_train = dataset_train.skip(saved_count)

print(f"\nDownloading and saving all Ideogram images to:")
print(f"Directory: {SAVE_DIR}\n")

for sample in tqdm(dataset_train, initial=saved_count, total=27000):
    filename = f"ideogram_{saved_count:05d}.jpg"
    save_path = SAVE_DIR / filename

    try:
        # Convert to RGB and save as JPEG
        sample["image"].convert("RGB").save(save_path, "JPEG", quality=95)
        saved_count += 1
    except Exception as e:
        print(f"\nError saving image {saved_count}: {e}")

print(f"\nSuccessfully processed/downloaded {saved_count} Ideogram images!")
