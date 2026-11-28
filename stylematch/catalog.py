"""Shared demo-catalog builder: embeds every item in a dataset via a
model, saves (embeddings, identifiers, category_labels) to disk for
a live nearest-neighbor search. Reused across the similarity
(DeepFashion), compatibility (Polyvore), and stretch-goal (African
garments) catalogs -- the mechanics never change, only which
model/dataset/identifiers get passed in.
"""

from pathlib import Path

from loguru import logger
import torch
from torch.utils.data import DataLoader


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
