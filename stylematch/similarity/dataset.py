"""PyTorch Dataset for the DeepFashion Category and Attribute Prediction
(CAP) subset, used to train the similarity engine.

See `scripts/download_deepfashion.py` to fetch the underlying files, and
`references/deepfashion_schema.md` for what's inside them.

This is a mock pipeline: real file parsing and real images, but capped
to a small sample by default, fast enough for local iteration while the
full dataset stays untouched.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stylematch.config import DEEPFASHION_DIR

NUM_ATTRIBUTES = 1000

# Placeholder preprocessing size, used only when no transform is passed in.
# Stands in until the real backbone's own preprocessing (Marqo-FashionCLIP,
# via open_clip) replaces this.
MOCK_IMAGE_SIZE = 224


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
                triplets from. Mock stand-in for the real pipeline;
                None (default) preserves the original first-N behavior.
            num_categories: required when min_images_per_category is
                set; how many distinct qualifying categories to collect.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        if min_images_per_category is not None and num_categories is None:
            raise ValueError("num_categories is required when min_images_per_category is set")
        self.root = Path(root)
        self.transform = transform
        if min_images_per_category is not None:
            self.samples = self._load_stratified_samples(
                split, min_images_per_category, num_categories
            )
        else:
            self.samples = self._load_samples(split, sample_size)

    def _load_samples(
        self, split: str, sample_size: int | None
    ) -> list[tuple[str, int, list[int]]]:
        """Reads the three per-image annotation files in lockstep, they
        share the same image order and line count (confirmed: 289,224
        lines each, 289,222 images plus 2 header lines). Stops early once
        `sample_size` matches are found, rather than loading the full
        attribute file (888MB, ~289K rows) into memory."""
        samples = []
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
                samples.append((image_name, category_label, attribute_label))

                if sample_size is not None and len(samples) >= sample_size:
                    break

        return samples

    def _load_stratified_samples(
        self, split: str, min_images_per_category: int, num_categories: int
    ) -> list[tuple[str, int, list[int]]]:
        """Collects `num_categories` distinct categories that each have
        at least `min_images_per_category` images, scanning further into
        the file than `_load_samples` would, since categories arrive in
        contiguous blocks, not interleaved. Duplicates some parsing logic
        from `_load_samples` rather than sharing it, this is mock/
        temporary code meant to be replaced wholesale by the real
        pipeline, not worth a shared abstraction for.
        """
        by_category: dict[int, list[tuple[str, int, list[int]]]] = {}
        category_path = self.root / "list_category_img.txt"
        attr_path = self.root / "list_attr_img.txt"
        partition_path = self.root / "list_eval_partition.txt"

        with (
            open(category_path) as cat_f,
            open(attr_path) as attr_f,
            open(partition_path) as part_f,
        ):
            for _ in range(2):
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

                category_label = int(cat_parts[1]) - 1
                bucket = by_category.setdefault(category_label, [])
                if len(bucket) >= min_images_per_category:
                    continue  # this category already has enough

                attribute_label = [int(v) for v in attr_parts[1:]]
                bucket.append((image_name, category_label, attribute_label))

                qualifying = sum(
                    1 for b in by_category.values() if len(b) >= min_images_per_category
                )
                if qualifying >= num_categories:
                    break

        qualifying_categories = [
            cat for cat, bucket in by_category.items() if len(bucket) >= min_images_per_category
        ][:num_categories]
        if len(qualifying_categories) < num_categories:
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

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor]:
        """
        Returns:
            image: float tensor, shape (3, H, W), already preprocessed.
            category_label: int, index into the category vocabulary
                (`list_category_cloth.txt`), 0-indexed.
            attribute_label: int tensor, shape (num_attributes,), values
                in {-1, 1} per `list_attr_img.txt`'s own encoding
                (1 = has the attribute, -1 = does not).
        """
        image_name, category_label, attribute_label = self.samples[idx]
        image = Image.open(self.root / image_name).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.resize((MOCK_IMAGE_SIZE, MOCK_IMAGE_SIZE))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        attribute_tensor = torch.tensor(attribute_label, dtype=torch.long)
        return image, category_label, attribute_tensor
