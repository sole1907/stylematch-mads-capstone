"""PyTorch Datasets for the Polyvore Outfits dataset, used to train and
evaluate the compatibility engine.

See `references/polyvore_outfits_schema.md` for what's inside the raw
files, and the nondisjoint-split decision recorded there (matches the
split Vasileva et al.'s cited numbers are reported on).
"""

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from stylematch.config import POLYVORE_DIR


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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[str], list[int], str]:
        """
        Returns:
            item_images: one preprocessed image per item in the outfit,
                shape (3, H, W) each, same order as item_ids.
            item_ids: Polyvore item IDs, same order as item_images.
            category_labels: one per item, from `semantic_category` in
                `polyvore_item_metadata.json`.
            outfit_id: the outfit's `set_id`.
        """
        raise NotImplementedError


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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[Tensor], int]:
        """
        Returns:
            context_images: the outfit's items minus the blank one.
            candidate_images: the 4 answer candidates, exactly one
                correct.
            correct_index: index into candidate_images (0-3) of the
                correct answer.
        """
        raise NotImplementedError


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
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[list[Tensor], int]:
        """
        Returns:
            outfit_images: this candidate outfit's items.
            label: 1 = compatible (real outfit), 0 = incompatible
                (generated negative).
        """
        raise NotImplementedError
