"""PyTorch Datasets for the Polyvore Outfits dataset, used to train and
evaluate the compatibility engine. See references/polyvore_outfits_schema.md
for the raw files and the nondisjoint-split decision."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stylematch.config import POLYVORE_DIR

DEFAULT_IMAGE_SIZE = 224

# Consecutive load failures tolerated before __getitem__ gives up,
# same bounded-retry policy as DeepFashionCAPDataset.
MAX_CONSECUTIVE_LOAD_RETRIES = 20

# The real metadata's semantic_category vocabulary (confirmed: exactly
# these 11 values). Sorted so the label mapping is deterministic.
SEMANTIC_CATEGORIES = sorted(
    [
        "accessories",
        "all-body",
        "bags",
        "bottoms",
        "hats",
        "jewellery",
        "outerwear",
        "scarves",
        "shoes",
        "sunglasses",
        "tops",
    ]
)
CATEGORY_TO_LABEL = {name: i for i, name in enumerate(SEMANTIC_CATEGORIES)}

VALID_SPLITS = ("train", "valid", "test")


def _load_item_metadata(root: Path) -> dict:
    with open(root / "polyvore_item_metadata.json") as f:
        return json.load(f)


def _load_outfits(root: Path, split: str) -> list[dict]:
    with open(root / "nondisjoint" / f"{split}.json") as f:
        return json.load(f)


def _build_outfit_index(outfits: list[dict]) -> dict[str, dict[int, str]]:
    """set_id -> {index: item_id}, for resolving the composite
    "<set_id>_<index>" references used in the compatibility/FITB eval
    files back to real item IDs."""
    index: dict[str, dict[int, str]] = {}
    for outfit in outfits:
        index[outfit["set_id"]] = {item["index"]: item["item_id"] for item in outfit["items"]}
    return index


def _resolve_composite_id(ref: str, outfit_index: dict[str, dict[int, str]]) -> str:
    set_id, index_str = ref.rsplit("_", 1)
    return outfit_index[set_id][int(index_str)]


def _load_and_transform_image(root: Path, transform, item_id: str) -> Tensor:
    image = Image.open(root / "images" / f"{item_id}.jpg").convert("RGB")
    if transform is not None:
        return transform(image)
    image = image.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


class PolyvoreOutfitDataset(Dataset):
    """PyTorch Dataset over real outfits, for training. `__getitem__`
    returns one whole outfit (compatibility is a set property, not a
    single-item one); positive/negative construction happens downstream
    via a batch sampler, not baked in here."""

    def __init__(
        self,
        root: Path = POLYVORE_DIR,
        split: str = "train",
        transform=None,
    ) -> None:
        """
        Args:
            root: directory with the downloaded Polyvore Outfits files
                (nondisjoint split).
            split: one of "train", "valid", "test".
            transform: preprocessing/augmentation applied to each image.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.transform = transform
        self.outfits = _load_outfits(self.root, split)
        self.metadata = _load_item_metadata(self.root)

    def __len__(self) -> int:
        return len(self.outfits)

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[str], list[int], str]:
        """
        Returns:
            item_images: one preprocessed image per item in the outfit,
                shape (3, H, W) each, same order as item_ids.
            item_ids: Polyvore item IDs, same order as item_images.
            category_labels: one per item, from `semantic_category` in
                `polyvore_item_metadata.json`.
            outfit_id: the outfit's `set_id`.

        Falls back to the next outfit on a load failure, same
        bounded-retry policy as DeepFashionCAPDataset.__getitem__.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_CONSECUTIVE_LOAD_RETRIES):
            try_idx = (idx + attempt) % len(self.outfits)
            outfit = self.outfits[try_idx]
            try:
                item_ids = [item["item_id"] for item in outfit["items"]]
                item_images = [
                    _load_and_transform_image(self.root, self.transform, item_id)
                    for item_id in item_ids
                ]
                category_labels = [
                    CATEGORY_TO_LABEL[self.metadata[item_id]["semantic_category"]]
                    for item_id in item_ids
                ]
                if attempt > 0:
                    logger.warning(
                        f"Skipped {attempt} unreadable outfit(s) starting at index {idx}, "
                        f"substituted index {try_idx} instead."
                    )
                return item_images, item_ids, category_labels, outfit["set_id"]
            except (OSError, ValueError, KeyError) as error:
                last_error = error
                logger.warning(f"Failed to load outfit at index {try_idx}: {error}")
        raise RuntimeError(
            f"{MAX_CONSECUTIVE_LOAD_RETRIES} consecutive outfits failed to load "
            f"starting at index {idx}; something is likely wrong with the dataset "
            f"itself (bad root path, missing files), not an isolated bad outfit."
        ) from last_error


class PolyvoreItemDataset(Dataset):
    """PyTorch Dataset over individual items (not outfits) -- every
    unique item appearing in `split`'s outfits, each returned once.
    For catalog-building (one embedding per garment), not training."""

    def __init__(
        self,
        root: Path = POLYVORE_DIR,
        split: str = "train",
        transform=None,
    ) -> None:
        """
        Args:
            root: same as `PolyvoreOutfitDataset`.
            split: one of "train", "valid", "test".
            transform: same as `PolyvoreOutfitDataset`.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.transform = transform
        self.metadata = _load_item_metadata(self.root)

        seen = set()
        self.item_ids = []
        for outfit in _load_outfits(self.root, split):
            for item in outfit["items"]:
                item_id = item["item_id"]
                if item_id not in seen:
                    seen.add(item_id)
                    self.item_ids.append(item_id)

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        """
        Returns:
            image: preprocessed item image, shape (3, H, W).
            category_label: from `semantic_category` in
                `polyvore_item_metadata.json`.
            item_id: the real Polyvore item ID.
        """
        item_id = self.item_ids[idx]
        image = _load_and_transform_image(self.root, self.transform, item_id)
        category_label = CATEGORY_TO_LABEL[self.metadata[item_id]["semantic_category"]]
        return image, category_label, item_id


class FITBDataset(Dataset):
    """PyTorch Dataset over the fill-in-the-blank eval task: one
    question per `__getitem__` call, context items plus 4 answer
    candidates. A fixed, official eval set, not sampler-reshaped."""

    def __init__(
        self,
        root: Path = POLYVORE_DIR,
        split: str = "valid",
        transform=None,
    ) -> None:
        """
        Args:
            root: same as `PolyvoreOutfitDataset`.
            split: "valid" or "test", matching
                `fill_in_blank_{split}.json`.
            transform: same as `PolyvoreOutfitDataset`.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.transform = transform
        self.outfit_index = _build_outfit_index(_load_outfits(self.root, split))
        with open(self.root / "nondisjoint" / f"fill_in_blank_{split}.json") as f:
            self.questions = json.load(f)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[Tensor], int]:
        """
        Returns:
            context_images: the outfit's items minus the blank one.
            candidate_images: the 4 answer candidates, exactly one
                correct.
            correct_index: index into candidate_images (0-3) of the
                correct answer. Always 0, confirmed against real data.
        """
        question = self.questions[idx]
        context_ids = [_resolve_composite_id(ref, self.outfit_index) for ref in question["question"]]
        candidate_ids = [_resolve_composite_id(ref, self.outfit_index) for ref in question["answers"]]
        context_images = [
            _load_and_transform_image(self.root, self.transform, item_id) for item_id in context_ids
        ]
        candidate_images = [
            _load_and_transform_image(self.root, self.transform, item_id) for item_id in candidate_ids
        ]
        correct_index = 0
        return context_images, candidate_images, correct_index


class CompatibilityDataset(Dataset):
    """PyTorch Dataset over the compatibility (AUC) eval task: one
    candidate outfit per `__getitem__` call with a real/fake label,
    both fixed by the official eval file (not sampled fresh)."""

    def __init__(
        self,
        root: Path = POLYVORE_DIR,
        split: str = "valid",
        transform=None,
    ) -> None:
        """
        Args:
            root: same as `PolyvoreOutfitDataset`.
            split: "valid" or "test", matching
                `compatibility_{split}.txt`.
            transform: same as `PolyvoreOutfitDataset`.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.transform = transform
        self.outfit_index = _build_outfit_index(_load_outfits(self.root, split))
        self.questions: list[tuple[int, list[str]]] = []
        with open(self.root / "nondisjoint" / f"compatibility_{split}.txt") as f:
            for line in f:
                parts = line.split()
                self.questions.append((int(parts[0]), parts[1:]))

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> tuple[list[Tensor], int]:
        """
        Returns:
            outfit_images: this candidate outfit's items.
            label: 1 = compatible (real outfit), 0 = incompatible
                (generated negative).
        """
        label, refs = self.questions[idx]
        item_ids = [_resolve_composite_id(ref, self.outfit_index) for ref in refs]
        outfit_images = [
            _load_and_transform_image(self.root, self.transform, item_id) for item_id in item_ids
        ]
        return outfit_images, label
