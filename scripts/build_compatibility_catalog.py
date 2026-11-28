"""Precomputes a compatibility-engine demo catalog: embeds every
unique real val item (not outfit -- individual garments, see
PolyvoreItemDataset) via the real trained checkpoint. Val, not test
-- same reasoning as the similarity catalog.

Run directly:
    python scripts/build_compatibility_catalog.py
"""

import torch

from stylematch.catalog import build_catalog
from stylematch.compatibility.dataset import PolyvoreItemDataset
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import MODELS_DIR

CHECKPOINT_PATH = MODELS_DIR / "compatibility_final.pt"
CATALOG_PATH = MODELS_DIR / "compatibility_catalog.pt"
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = CompatibilityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))

    dataset = PolyvoreItemDataset(split="valid", transform=model.preprocess_val)
    item_ids = list(dataset.item_ids)

    build_catalog(model, dataset, item_ids, CATALOG_PATH)


if __name__ == "__main__":
    main()
