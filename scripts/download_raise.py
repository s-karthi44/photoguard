from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "RAISE_6k.csv"
SAVE_DIR = PROJECT_ROOT / "datasets" / "raw" / "RAISE"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

print(f"Found {len(df)} images")

for _, row in tqdm(df.iterrows(), total=len(df)):
    url = row["TIFF"]
    filename = Path(url).name
    save_path = SAVE_DIR / filename

    # Skip already downloaded files
    if save_path.exists():
        continue

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    except Exception as e:
        print(f"Failed: {filename} -> {e}")

print("Download completed!")