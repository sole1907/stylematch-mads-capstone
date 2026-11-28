"""Precomputes an AFRIFASHION1600 catalog via the similarity engine's
checkpoint. Stretch goal: lets the demo search African garments
against OTHER African garments (not the DeepFashion catalog -- that's
a different, bias-generalization question, not this one).

Note: category_labels here are AFRIFASHION_CLASSES indices (8
classes), a different label space from the DeepFashion catalog's 21
categories -- never mix the two.

Run directly:
    python scripts/build_afrifashion_similarity_catalog.py
"""

import torch

from stylematch.catalog import AfrifashionDataset, build_catalog
from stylematch.config import MODELS_DIR
from stylematch.similarity.model import SimilarityModel

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"
CATALOG_PATH = MODELS_DIR / "afrifashion_similarity_catalog.pt"
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))

    dataset = AfrifashionDataset(transform=model.preprocess_val)
    image_names = list(dataset.image_names)

    build_catalog(model, dataset, image_names, CATALOG_PATH)


if __name__ == "__main__":
    main()
