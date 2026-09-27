import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw" / "OpenImages"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

command = [
    "oidv6",
    "downloader",
    "--dataset",
    "train",
    "--type_data",
    "all",
    "--limit",
    "10000",
    "--output_folder",
    str(OUTPUT_DIR)
]

subprocess.run(command)