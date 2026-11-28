"""One-off convenience script: copies a small per-category sample of
each dataset's real held-out images (DeepFashion/Polyvore test split,
AFRIFASHION1600 holdout) into data/raw/demo_test_data/<dataset>/<category>/,
for manually picking an upload image when testing the demo app. Not
used by any pipeline/training code -- a copy, not a move, safe to
re-run anytime.

Similarity (DeepFashion) only includes the 21 categories the model
was actually trained/evaluated on, not all 46 real categories.

Run directly:
    python scripts/build_demo_test_samples.py
"""

from pathlib import Path
import re
import shutil

from loguru import logger

from stylematch.compatibility.dataset import PolyvoreItemDataset
from stylematch.config import AFRIFASHION_DIR, DEEPFASHION_DIR, POLYVORE_DIR, RAW_DATA_DIR
from stylematch.similarity.dataset import _iter_annotation_rows, get_qualifying_categories

OUTPUT_DIR = RAW_DATA_DIR / "demo_test_data"
SAMPLES_PER_CATEGORY = 5
CATEGORY_INCLUSION_THRESHOLD = 1000
_AFRIFASHION_FILENAME_RE = re.compile(r"^(.+)_\d+\.png$")


def _load_deepfashion_category_names() -> list[str]:
    """0-indexed, same order as the category labels used everywhere
    else (confirmed: list_category_cloth.txt's real format is a
    count, a header, then one '<name> <type>' line per category)."""
    with open(DEEPFASHION_DIR / "list_category_cloth.txt") as f:
        lines = f.readlines()[2:]
    return [line.split()[0] for line in lines]


def build_deepfashion_samples() -> None:
    category_names = _load_deepfashion_category_names()
    allowed = get_qualifying_categories(CATEGORY_INCLUSION_THRESHOLD, split="train")

    counts: dict[int, int] = {}
    for image_name, category_label, _ in _iter_annotation_rows(DEEPFASHION_DIR, "test"):
        if category_label not in allowed:
            continue
        counts[category_label] = counts.get(category_label, 0) + 1
        if counts[category_label] > SAMPLES_PER_CATEGORY:
            continue

        category_dir = OUTPUT_DIR / "deepfashion" / category_names[category_label]
        category_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEEPFASHION_DIR / image_name, category_dir / Path(image_name).name)

    logger.success(f"DeepFashion: sampled {len(allowed)} categories into {OUTPUT_DIR / 'deepfashion'}")


def build_polyvore_samples() -> None:
    dataset = PolyvoreItemDataset(split="test")
    counts: dict[str, int] = {}

    for item_id in dataset.item_ids:
        # Already the real string name (e.g. "bags"), not an index --
        # confirmed via PolyvoreItemDataset's own __getitem__, which
        # looks this exact field up directly, no SEMANTIC_CATEGORIES
        # indexing needed.
        category_name = dataset.metadata[item_id]["semantic_category"]
        counts[category_name] = counts.get(category_name, 0) + 1
        if counts[category_name] > SAMPLES_PER_CATEGORY:
            continue

        category_dir = OUTPUT_DIR / "polyvore" / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(POLYVORE_DIR / "images" / f"{item_id}.jpg", category_dir / f"{item_id}.jpg")

    logger.success(f"Polyvore: sampled {len(counts)} categories into {OUTPUT_DIR / 'polyvore'}")


def build_afrifashion_samples() -> None:
    holdout_dir = AFRIFASHION_DIR / "holdout"
    counts: dict[str, int] = {}

    for path in sorted(holdout_dir.glob("*.png")):
        match = _AFRIFASHION_FILENAME_RE.match(path.name)
        class_name = match.group(1)
        counts[class_name] = counts.get(class_name, 0) + 1
        if counts[class_name] > SAMPLES_PER_CATEGORY:
            continue

        category_dir = OUTPUT_DIR / "afrifashion" / class_name
        category_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, category_dir / path.name)

    logger.success(f"AFRIFASHION1600: sampled {len(counts)} classes into {OUTPUT_DIR / 'afrifashion'}")


def main() -> None:
    build_deepfashion_samples()
    build_polyvore_samples()
    build_afrifashion_samples()


if __name__ == "__main__":
    main()
