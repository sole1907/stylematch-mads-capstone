"""Shared demo-catalog builder: embeds every item in a dataset via a
model, saves (embeddings, identifiers, category_labels) to disk for
a live nearest-neighbor search. Reused across the similarity
(DeepFashion), compatibility (Polyvore), and stretch-goal (African
garments) catalogs -- the mechanics never change, only which
model/dataset/identifiers get passed in.
"""

from pathlib import Path
import re

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from stylematch.config import AFRIFASHION_DIR

# The real class names, confirmed from the actual filenames (e.g.
# "Gown_166.png", "African_Blouse_123.png" -- some class names
# contain underscores themselves, so parsing strips a trailing
# "_<number>.png", not a first-underscore split).
AFRIFASHION_CLASSES = sorted(
    [
        "African_Blouse",
        "African_Shirts",
        "Agbada",
        "Buba and Trouser",
        "Gele",
        "Gown",
        "Skirt and Blouse",
        "Wrapper and Blouse",
    ]
)
AFRIFASHION_CATEGORY_TO_LABEL = {name: i for i, name in enumerate(AFRIFASHION_CLASSES)}
_AFRIFASHION_FILENAME_RE = re.compile(r"^(.+)_\d+\.png$")


class AfrifashionDataset(Dataset):
    """Flat dataset over the AFRIFASHION1600 images. No subfolders or
    annotation file -- class name is encoded in the filename itself
    (e.g. "Gown_166.png"), confirmed against the real downloaded files."""

    def __init__(self, root: Path = AFRIFASHION_DIR, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_names = sorted(p.name for p in self.root.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        """
        Returns:
            image: preprocessed item image, shape (3, H, W).
            category_label: parsed from the filename.
            image_name: the real filename, e.g. "Gown_166.png".
        """
        image_name = self.image_names[idx]
        match = _AFRIFASHION_FILENAME_RE.match(image_name)
        if match is None:
            raise ValueError(f"filename {image_name!r} doesn't match the expected '<class>_<n>.png' pattern")
        category_label = AFRIFASHION_CATEGORY_TO_LABEL[match.group(1)]

        image = Image.open(self.root / image_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.resize((224, 224))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image, category_label, image_name


@torch.no_grad()
def build_catalog(
    model,
    dataset,
    identifiers: list,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    """Embeds every item in `dataset` via `model`, saves the result
    to `output_path`.

    Args:
        dataset: yields (image, category_label, *ignored) tuples --
            the shape DeepFashionCAPDataset and PolyvoreItemDataset
            already return.
        identifiers: in the same order as `dataset`'s own iteration
            (index 0..len(dataset)-1) -- image paths or item IDs,
            whatever the caller uses to look an image back up later.
    """
    if len(identifiers) != len(dataset):
        raise ValueError(f"identifiers has {len(identifiers)} entries, dataset has {len(dataset)}")

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_category_labels = []
    for images, category_labels, *_ in loader:
        all_embeddings.append(model(images).cpu())
        all_category_labels.append(category_labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.cat(all_embeddings),
            "identifiers": identifiers,
            "category_labels": torch.cat(all_category_labels),
        },
        output_path,
    )
    logger.success(f"Saved catalog ({len(identifiers)} items) to {output_path}")
