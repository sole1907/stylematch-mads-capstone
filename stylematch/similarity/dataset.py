"""PyTorch Dataset for the DeepFashion Category and Attribute Prediction
(CAP) subset, used to train the similarity engine.

See `scripts/download_deepfashion.py` to fetch the underlying files, and
`references/deepfashion_schema.md` for what's inside them.
"""

from pathlib import Path

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stylematch.config import DEEPFASHION_DIR

NUM_ATTRIBUTES = 1000

# Fallback preprocessing size when no transform is passed in; real
# callers always pass the backbone's own preprocessing instead.
DEFAULT_IMAGE_SIZE = 224

# Consecutive load failures tolerated before __getitem__ gives up and
# raises, rather than looping through the whole dataset.
MAX_CONSECUTIVE_LOAD_RETRIES = 20


def _iter_annotation_rows(root: Path, split: str):
    """Reads the three per-image annotation files in lockstep,
    yielding (image_name, category_label, attribute_label) for `split`.
    """
    category_path = root / "list_category_img.txt"
    attr_path = root / "list_attr_img.txt"
    partition_path = root / "list_eval_partition.txt"

    with (
        open(category_path) as cat_f,
        open(attr_path) as attr_f,
        open(partition_path) as part_f,
    ):
        for _ in range(2):  # skip the two header lines in each file
            next(cat_f)
            next(attr_f)
            next(part_f)

        for cat_line, attr_line, part_line in zip(cat_f, attr_f, part_f):
            cat_parts = cat_line.split()
            attr_parts = attr_line.split()
            part_parts = part_line.split()

            image_name = cat_parts[0]
            if attr_parts[0] != image_name or part_parts[0] != image_name:
                raise ValueError(f"annotation files out of sync at {image_name!r}")
            if part_parts[1] != split:
                continue

            # Files are 1-indexed; PyTorch labels are expected 0-indexed.
            category_label = int(cat_parts[1]) - 1
            attribute_label = [int(v) for v in attr_parts[1:]]
            yield image_name, category_label, attribute_label


def get_qualifying_categories(
    min_images_per_category: int, split: str = "train", root: Path = DEEPFASHION_DIR
) -> set[int]:
    """Category labels with >= min_images_per_category images in
    `split`. Decide once from train, reuse elsewhere."""
    counts: dict[int, int] = {}
    for _, category_label, _ in _iter_annotation_rows(root, split):
        counts[category_label] = counts.get(category_label, 0) + 1
    return {cat for cat, count in counts.items() if count >= min_images_per_category}


class DeepFashionCAPDataset(Dataset):
    """PyTorch Dataset over the downloaded DeepFashion CAP subset.
    Returns single (image, category_label, attribute_label) tuples;
    triplet construction happens downstream, in the batch sampler."""

    def __init__(
        self,
        root: Path = DEEPFASHION_DIR,
        split: str = "train",
        transform=None,
        sample_size: int | None = 100,
        min_images_per_category: int | None = None,
        num_categories: int | None = None,
        max_images_per_category: int | None = None,
        allowed_categories: set[int] | None = None,
    ) -> None:
        """
        Args:
            root: directory with the downloaded CAP files.
            split: one of "train", "val", "test".
            transform: preprocessing pipeline; a minimal placeholder is
                used if None.
            sample_size: cap on images loaded, in file order (ignored
                if min_images_per_category is set).
            min_images_per_category: if set, collects num_categories
                categories with at least this many images each, so a
                batch sampler always has same/different-category
                images to mine triplets from.
            num_categories: how many qualifying categories to collect;
                None means every qualifying one.
            max_images_per_category: caps images collected per
                category; None means no cap.
            allowed_categories: use exactly these labels, decided
                externally, instead of re-deriving qualification here.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self.root = Path(root)
        self.transform = transform
        if min_images_per_category is not None or allowed_categories is not None:
            self.samples = self._load_stratified_samples(
                split,
                min_images_per_category,
                num_categories,
                max_images_per_category,
                allowed_categories,
            )
        else:
            self.samples = self._load_samples(split, sample_size)

    def _load_samples(
        self, split: str, sample_size: int | None
    ) -> list[tuple[str, int, list[int]]]:
        """Stops early once `sample_size` matches are found, rather than
        reading the full attribute file (888MB, ~289K rows) when a
        caller only wants a handful of images."""
        samples = []
        for image_name, category_label, attribute_label in _iter_annotation_rows(self.root, split):
            samples.append((image_name, category_label, attribute_label))
            if sample_size is not None and len(samples) >= sample_size:
                break
        return samples

    def _load_stratified_samples(
        self,
        split: str,
        min_images_per_category: int | None,
        num_categories: int | None,
        max_images_per_category: int | None = None,
        allowed_categories: set[int] | None = None,
    ) -> list[tuple[str, int, list[int]]]:
        """Collects num_categories categories with >= min_images_per_category
        images each, up to max_images_per_category each. If
        allowed_categories is given, that set is used directly instead."""
        by_category: dict[int, list[tuple[str, int, list[int]]]] = {}
        category_counts: dict[int, int] = {}
        previous_category_label = None

        for image_name, category_label, attribute_label in _iter_annotation_rows(self.root, split):
            if allowed_categories is not None and category_label not in allowed_categories:
                continue

            if (
                allowed_categories is None
                and num_categories is not None
                and previous_category_label is not None
                and category_label != previous_category_label
            ):
                qualifying = sum(1 for c in category_counts.values() if c >= min_images_per_category)
                if qualifying >= num_categories:
                    break
            previous_category_label = category_label

            category_counts[category_label] = category_counts.get(category_label, 0) + 1

            bucket = by_category.setdefault(category_label, [])
            if max_images_per_category is not None and len(bucket) >= max_images_per_category:
                continue  # this category has reached its per-category cap

            bucket.append((image_name, category_label, attribute_label))

        if allowed_categories is not None:
            qualifying_categories = list(by_category.keys())
        else:
            qualifying_categories = [
                cat for cat, count in category_counts.items() if count >= min_images_per_category
            ][:num_categories]
            if num_categories is not None and len(qualifying_categories) < num_categories:
                raise ValueError(
                    f"only found {len(qualifying_categories)} categories with >= "
                    f"{min_images_per_category} images in split {split!r}, "
                    f"requested {num_categories}"
                )

        samples = []
        for category in qualifying_categories:
            samples.extend(by_category[category])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, idx: int) -> Image.Image:
        """Opens the real image file for `idx`. Separated out from
        __getitem__ so the retry loop there can call this repeatedly
        without duplicating the open+convert logic."""
        image_name, _, _ = self.samples[idx]
        return Image.open(self.root / image_name).convert("RGB")

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor]:
        """Returns (image, category_label, attribute_label), 0-indexed
        label into list_category_cloth.txt. Falls back to the next
        index on load failure, raising after too many in a row."""
        image = None
        category_label, attribute_label = None, None
        last_error: Exception | None = None
        for attempt in range(MAX_CONSECUTIVE_LOAD_RETRIES):
            try_idx = (idx + attempt) % len(self.samples)
            try:
                image = self._load_image(try_idx)
                if attempt > 0:
                    logger.warning(
                        f"Skipped {attempt} unreadable sample(s) starting at index {idx}, "
                        f"substituted index {try_idx} instead."
                    )
                _, category_label, attribute_label = self.samples[try_idx]
                break
            except (OSError, ValueError) as error:
                last_error = error
                logger.warning(f"Failed to load sample at index {try_idx}: {error}")
        if image is None:
            raise RuntimeError(
                f"{MAX_CONSECUTIVE_LOAD_RETRIES} consecutive samples failed to load "
                f"starting at index {idx}; something is likely wrong with the dataset "
                f"itself (bad root path, missing files), not an isolated bad image."
            ) from last_error

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        attribute_tensor = torch.tensor(attribute_label, dtype=torch.long)
        return image, category_label, attribute_tensor
