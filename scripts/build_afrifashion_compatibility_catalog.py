"""Precomputes an AFRIFASHION1600 catalog via the compatibility
engine's checkpoint. Stretch goal: lets the demo search African
garments against OTHER African garments for compatible suggestions.

Only 1,600 images -- small enough to run locally (~8 min at the
measured local CPU rate), no Great Lakes job needed.

Note: category_labels here are AFRIFASHION_CLASSES indices (8
classes), a different label space from the Polyvore catalog's 11
semantic categories -- never mix the two.

Run directly:
    python scripts/build_afrifashion_compatibility_catalog.py
"""

import torch

from stylematch.catalog import AfrifashionDataset, build_catalog
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import MODELS_DIR

CHECKPOINT_PATH = MODELS_DIR / "compatibility_final.pt"
CATALOG_PATH = MODELS_DIR / "afrifashion_compatibility_catalog.pt"
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = CompatibilityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))

    dataset = AfrifashionDataset(transform=model.preprocess_val)
    image_names = list(dataset.image_names)

    build_catalog(model, dataset, image_names, CATALOG_PATH)


if __name__ == "__main__":
    main()
