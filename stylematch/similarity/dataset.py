"""PyTorch Dataset for the DeepFashion Category and Attribute Prediction
(CAP) subset, used to train the similarity engine.

See `scripts/download_deepfashion.py` to fetch the underlying files, and
`references/deepfashion_schema.md` for what's inside them.
"""

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from stylematch.config import DEEPFASHION_DIR


class DeepFashionCAPDataset(Dataset):
    """PyTorch Dataset over the downloaded DeepFashion CAP subset.

    Interface contract, to be agreed with Yuzuru before implementation:
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
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor]:
        """
        Returns:
            image: float tensor, shape (3, H, W), already preprocessed.
            category_label: int, index into the category vocabulary
                (`list_category_cloth.txt`).
            attribute_label: int tensor, shape (num_attributes,), values
                in {-1, 1} per `list_attr_img.txt`'s own encoding
                (1 = has the attribute, -1 = does not).
        """
        raise NotImplementedError
