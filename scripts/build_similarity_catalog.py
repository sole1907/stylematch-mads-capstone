"""Precomputes a similarity-engine demo catalog: embeds every real val
image across the 21 qualifying categories (same set the model was
trained/evaluated on) via the real trained checkpoint. Val, not test
-- test stays reserved for the one official number already reported;
val is already the split used throughout this project for
exploratory/iterative work.

Run directly:
    python scripts/build_similarity_catalog.py
"""

import torch

from stylematch.catalog import build_catalog
from stylematch.config import MODELS_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset, get_qualifying_categories
from stylematch.similarity.model import SimilarityModel

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"
CATALOG_PATH = MODELS_DIR / "similarity_catalog.pt"
CATEGORY_INCLUSION_THRESHOLD = 1000
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))

    allowed_categories = get_qualifying_categories(CATEGORY_INCLUSION_THRESHOLD, split="train")
    dataset = DeepFashionCAPDataset(
        split="val",
        transform=model.preprocess_val,
        allowed_categories=allowed_categories,
        max_images_per_category=None,
    )
    image_paths = [image_name for image_name, _, _ in dataset.samples]

    build_catalog(model, dataset, image_paths, CATALOG_PATH)


if __name__ == "__main__":
    main()
