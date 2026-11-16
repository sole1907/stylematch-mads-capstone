"""PyTorch Datasets for the Polyvore Outfits dataset, used to train and
evaluate the compatibility engine.

See `references/polyvore_outfits_schema.md` for what's inside the raw
files, and the nondisjoint-split decision recorded there (matches the
split Vasileva et al.'s cited numbers are reported on).
"""

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

# How many consecutive outfits __getitem__ will try before giving up and
# raising, same bounded-retry policy as DeepFashionCAPDataset.
MAX_CONSECUTIVE_LOAD_RETRIES = 20

# The real metadata's semantic_category vocabulary (confirmed via the
# downloaded data: exactly these 11 values, see
# references/polyvore_outfits_schema.md). Sorted so the label mapping is
# deterministic across machines/runs.
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
    """PyTorch Dataset over real outfits, for training.

    `__getitem__` returns one whole outfit (a variable-length list of
    items) — this is the unavoidable shape here: compatibility is a
    property of a set of items, not of any single one. Positive/negative
    pair or triplet construction (same-outfit items as positive signal,
    cross-outfit items as negative) happens downstream, via a batch
    sampler, mirroring how `CategoryBalancedBatchSampler` does this for
    the similarity engine rather than baking it into the dataset itself.
    """

    def __init__(
        self,
        root: Path = POLYVORE_DIR,
        split: str = "train",
        transform=None,
    ) -> None:
        """
        Args:
            root: directory containing the downloaded Polyvore Outfits
                files (nondisjoint split), see
                `references/polyvore_outfits_schema.md`.
            split: one of "train", "valid", "test", matching
                `{split}.json`.
            transform: image preprocessing/augmentation pipeline,
                applied to each item image before it's returned.
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

        Falls back to the next outfit (wrapping around) on a load
        failure, same bounded-retry policy as
        DeepFashionCAPDataset.__getitem__.
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


class FITBDataset(Dataset):
    """PyTorch Dataset over the fill-in-the-blank eval task.

    Interface contract: one FITB question per `__getitem__` call —
    the outfit's context items plus 4 answer candidates, exactly one
    of which is correct. Separate from `PolyvoreOutfitDataset` for the
    same reason `evaluate.py`'s retrieval evaluation is separate from
    the similarity engine's training dataset: this is a fixed,
    official eval set, not something a training sampler reshapes.
    """

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
                correct answer. Always 0 — confirmed against real data
                (see references/polyvore_outfits_schema.md), not
                assumed.
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
    """PyTorch Dataset over the compatibility (AUC) eval task.

    Interface contract: one candidate outfit per `__getitem__` call,
    with a real/fake label — real outfits (label=1) and generated
    negatives (label=0), both already fixed by the official eval file,
    not sampled fresh each time like training negatives are.
    """

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
