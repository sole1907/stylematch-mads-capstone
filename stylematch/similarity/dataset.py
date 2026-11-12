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

# Default preprocessing size, used only when no transform is passed in
# (every real caller passes the backbone's own preprocessing instead,
# model.py's preprocess_train/preprocess_val; this is a fallback for
# tests and ad-hoc use, not something awaiting a future replacement).
DEFAULT_IMAGE_SIZE = 224

# How many consecutive samples __getitem__ will try before giving up and
# raising, if it hits unreadable images back to back. Bounds the retry
# loop so a genuinely broken dataset still fails loudly and quickly,
# rather than looping through every remaining sample.
MAX_CONSECUTIVE_LOAD_RETRIES = 20


class DeepFashionCAPDataset(Dataset):
    """PyTorch Dataset over the downloaded DeepFashion CAP subset.

    Interface contract:
    `__getitem__` returns a single (image, category_label, attribute_label)
    tuple, one example at a time. Triplet construction for the similarity
    engine's loss does not happen here, it happens downstream, via a batch
    sampler that groups multiple images per category into each batch, and
    an online triplet miner (pytorch-metric-learning) that picks actual
    anchor/positive/negative triples from within each batch, using category
    and attribute labels together. Keeping this class single-item keeps it
    reusable for validation-set retrieval evaluation too, which needs
    individual embeddings, not triplets.
    """

    def __init__(
        self,
        root: Path = DEEPFASHION_DIR,
        split: str = "train",
        transform=None,
        sample_size: int | None = 100,
        min_images_per_category: int | None = None,
        num_categories: int | None = None,
        max_images_per_category: int | None = None,
    ) -> None:
        """
        Args:
            root: directory containing the downloaded CAP files (see
                `scripts/download_deepfashion.py`).
            split: one of "train", "val", "test", matching the partitions
                in `list_eval_partition.txt`.
            transform: image preprocessing/augmentation pipeline, applied
                to each image before it's returned. Expected to match
                whatever the chosen backbone (Marqo-FashionCLIP) requires.
                If None, a minimal placeholder (resize + scale to [0, 1])
                is used instead.
            sample_size: cap on how many matching images to load, in
                file order. Defaults to 100. Pass None to load every
                image in the split. Ignored if min_images_per_category
                is set (see below), since the underlying annotation file
                is clustered by category in blocks, confirmed: the first
                16 matches are all a single category, so a plain
                first-N sample can't give a batch sampler the category
                diversity it needs.
            min_images_per_category: if set, switches loading from
                "first `sample_size` matching images" to "collect
                `num_categories` distinct categories that each have at
                least this many images." Needed so a category-balanced
                batch sampler (sampler.py) always has same-category
                positives and different-category negatives to mine
                triplets from. None (default) preserves the plain
                first-N behavior.
            num_categories: only used when min_images_per_category is
                set; how many distinct qualifying categories to collect.
                None means every qualifying category in the split.
            max_images_per_category: only used when min_images_per_category
                is set; caps how many images get collected per qualifying
                category. None (default) means no cap, collect every real
                image that category has.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self.root = Path(root)
        self.transform = transform
        if min_images_per_category is not None:
            self.samples = self._load_stratified_samples(
                split, min_images_per_category, num_categories, max_images_per_category
            )
        else:
            self.samples = self._load_samples(split, sample_size)

    def _iter_annotation_rows(self, split: str):
        """Reads the three per-image annotation files in lockstep,
        yielding (image_name, category_label, attribute_label) for every
        image in `split`.
        """
        category_path = self.root / "list_category_img.txt"
        attr_path = self.root / "list_attr_img.txt"
        partition_path = self.root / "list_eval_partition.txt"

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

    def _load_samples(
        self, split: str, sample_size: int | None
    ) -> list[tuple[str, int, list[int]]]:
        """Stops early once `sample_size` matches are found, rather than
        reading the full attribute file (888MB, ~289K rows) when a
        caller only wants a handful of images."""
        samples = []
        for image_name, category_label, attribute_label in self._iter_annotation_rows(split):
            samples.append((image_name, category_label, attribute_label))
            if sample_size is not None and len(samples) >= sample_size:
                break
        return samples

    def _load_stratified_samples(
        self,
        split: str,
        min_images_per_category: int,
        num_categories: int | None,
        max_images_per_category: int | None = None,
    ) -> list[tuple[str, int, list[int]]]:
        """Collects `num_categories` distinct categories that each have
        at least `min_images_per_category` images, up to
        `max_images_per_category` each (None = no cap, every real image
        that category has). Scans further into the file than
        `_load_samples` would, since categories arrive in contiguous
        blocks, not interleaved. num_categories=None means collect every
        qualifying category.
        """
        by_category: dict[int, list[tuple[str, int, list[int]]]] = {}
        previous_category_label = None

        for image_name, category_label, attribute_label in self._iter_annotation_rows(split):
            # Block boundary: the previous category (if any) is now
            # guaranteed fully scanned, safe to check whether enough
            # qualifying categories already exist, before touching
            # this new category's block at all.
            if (
                num_categories is not None
                and previous_category_label is not None
                and category_label != previous_category_label
            ):
                qualifying = sum(
                    1 for b in by_category.values() if len(b) >= min_images_per_category
                )
                if qualifying >= num_categories:
                    break
            previous_category_label = category_label

            bucket = by_category.setdefault(category_label, [])
            if max_images_per_category is not None and len(bucket) >= max_images_per_category:
                continue  # this category has reached its per-category cap

            bucket.append((image_name, category_label, attribute_label))

        qualifying_categories = [
            cat for cat, bucket in by_category.items() if len(bucket) >= min_images_per_category
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
        """
        Returns:
            image: float tensor, shape (3, H, W), already preprocessed.
            category_label: int, index into the category vocabulary
                (`list_category_cloth.txt`), 0-indexed.
            attribute_label: int tensor, shape (num_attributes,), values
                in {-1, 1} per `list_attr_img.txt`'s own encoding
                (1 = has the attribute, -1 = does not).

        Falls back to the next index (wrapping around),
        logging a warning each time, bounded by
        MAX_CONSECUTIVE_LOAD_RETRIES consecutive failures before giving
        up and raising clearly, since that many in a row means
        something is fundamentally wrong (bad root path, empty
        dataset), not an isolated bad file worth skipping past.
        """
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
