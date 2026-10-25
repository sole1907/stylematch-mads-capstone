"""Download the DeepFashion Category and Attribute Prediction (CAP) subset.

Six files, all from the official project's shared Google Drive folder:
https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
- img.zip: the clothes images (standard-res)
- list_attr_cloth.txt / list_attr_img.txt: attribute vocabulary / per-image labels
- list_category_cloth.txt / list_category_img.txt: category vocabulary / per-image labels
- list_eval_partition.txt: the official train/val/test split
"""

import zipfile
from pathlib import Path

import gdown
import typer
from loguru import logger

from stylematch.config import RAW_DATA_DIR

app = typer.Typer()

DEEPFASHION_DIR = RAW_DATA_DIR / "deepfashion"

# (source URL, destination filename)
# Small annotation files first, large img.zip last — so a quota/rate-limit
# hit on the large file doesn't block the small ones from downloading too.
FILES = [
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pYnBKQVBOaHR1WWs/view"
        "?usp=drive_link&resourcekey=0-N8iNXTUvwLteEixb7emjwg",
        "list_attr_cloth.txt",
    ),
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pWXE4QWotX2hxQ1U/view"
        "?usp=drive_link&resourcekey=0-3w0aVsDrCfaDNG1eH6Pirw",
        "list_attr_img.txt",
    ),
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pWnFiNlNGTVloLUk/view"
        "?usp=drive_link&resourcekey=0-ClgptLwFKxgFU4HfAakqmw",
        "list_category_cloth.txt",
    ),
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pTGNoWkhZeVpzbFk/view"
        "?usp=drive_link&resourcekey=0-Dm6R_wuniTptUQSrWT7tSg",
        "list_category_img.txt",
    ),
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pdS1FMlNreEwtc1E/view"
        "?usp=drive_link&resourcekey=0-_AL8Vhcl1UUYuw7NbGnZtA",
        "list_eval_partition.txt",
    ),
    (
        "https://drive.google.com/file/d/0B7EVK8r0v71pa2EyNEJ0dE9zbU0/view"
        "?usp=drive_link&resourcekey=0-CPiKS-AiE8IDonk54WJ5_w",
        "img.zip",
    ),
]


@app.command()
def main(
    skip_existing: bool = typer.Option(
        True, help="Skip re-downloading files that already exist."
    ),
):
    """Download DeepFashion CAP images + annotations into data/raw/deepfashion/.

    Keeps going if one file fails (e.g. Google Drive rate-limiting) rather
    than aborting the whole run, and ends with a single short summary line
    (meant for cron/notification delivery, not just the detailed log above).
    """
    DEEPFASHION_DIR.mkdir(parents=True, exist_ok=True)

    already_had = 0
    newly_downloaded = 0
    failed = []

    for url, filename in FILES:
        output_path = DEEPFASHION_DIR / filename
        if skip_existing and output_path.exists():
            already_had += 1
            logger.info(f"Skipping {filename}, already exists.")
            continue
        logger.info(f"Downloading {filename}...")
        try:
            gdown.download(url=url, output=str(output_path), quiet=True)
            newly_downloaded += 1
        except Exception as e:
            failed.append(filename)
            logger.error(f"Failed: {filename}: {e}")

    done_count = already_had + newly_downloaded
    total_count = len(FILES)

    images_dir = DEEPFASHION_DIR / "img"
    zip_path = DEEPFASHION_DIR / "img.zip"
    extracted = images_dir.exists()
    if zip_path.exists() and not extracted:
        try:
            logger.info("Extracting img.zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DEEPFASHION_DIR)
            extracted = True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

    if done_count == total_count and extracted:
        summary = "All files downloaded successfully, yipee!"
        logger.success(summary)
    elif done_count == 0:
        summary = f"Failed, rate limit likely (0/{total_count} files)"
        logger.warning(summary)
    else:
        summary = f"{done_count}/{total_count} files done so far, rate limit likely blocking the rest"
        logger.warning(summary)

    # Deliberately short and on its own line: this is what a cron job
    # should deliver as the notification text, not the full log above.
    print(f"SUMMARY: {summary}")


if __name__ == "__main__":
    app()
