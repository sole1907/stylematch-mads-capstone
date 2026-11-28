"""One-off script: splits AFRIFASHION1600 into a catalog pool (90%)
and a held-out query pool (10%), moved into a holdout/ subfolder.
AfrifashionDataset's catalog glob (root.glob("*.png"), non-recursive)
naturally excludes that subfolder, so no code change was needed there
-- confirmed real pathlib behavior, not assumed.

Deterministic (sorted filenames, every 10th per class), not
randomized -- run independently on the local machine and on Great
Lakes, both produce the identical split from the identical files,
nothing needs to be transferred between them.

Run directly, once per machine:
    python scripts/split_afrifashion_holdout.py
"""

import re
import shutil

from loguru import logger

from stylematch.config import AFRIFASHION_DIR

HOLDOUT_DIR = AFRIFASHION_DIR / "holdout"
HOLDOUT_STRIDE = 10  # every 10th image per class, ~10%, deterministic
_FILENAME_RE = re.compile(r"^(.+)_\d+\.png$")


def main() -> None:
    HOLDOUT_DIR.mkdir(exist_ok=True)

    by_class: dict[str, list] = {}
    for path in sorted(AFRIFASHION_DIR.glob("*.png")):
        match = _FILENAME_RE.match(path.name)
        by_class.setdefault(match.group(1), []).append(path)

    moved = 0
    for class_name, paths in sorted(by_class.items()):
        holdout = paths[::HOLDOUT_STRIDE]
        for path in holdout:
            shutil.move(str(path), str(HOLDOUT_DIR / path.name))
            moved += 1
        logger.info(f"{class_name}: held out {len(holdout)}/{len(paths)}")

    logger.success(f"Moved {moved} images into {HOLDOUT_DIR}, excluded from the catalog build")


if __name__ == "__main__":
    main()
