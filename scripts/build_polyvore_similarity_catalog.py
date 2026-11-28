"""Precomputes a Polyvore catalog via the SIMILARITY engine's
checkpoint (not compatibility). Lets the Infer tab's category
inference search real Polyvore items directly, in Polyvore's own
native 11-category vocabulary -- no DeepFashion-to-Polyvore mapping
needed, and fixes a real blind spot: DeepFashion has no shoes/bags/
jewelry/etc. categories at all, so inference against it could never
produce those categories regardless of what was actually uploaded.

SimilarityModel was only ever officially evaluated on DeepFashion's
own test split -- embedding Polyvore images here is a reasonable,
deliberate generalization step, not a verified-quality operation like
the rest of this project's real numbers.

Run directly:
    python scripts/build_polyvore_similarity_catalog.py
"""

import torch

from stylematch.catalog import build_catalog
from stylematch.compatibility.dataset import PolyvoreItemDataset
from stylematch.config import MODELS_DIR
from stylematch.similarity.model import SimilarityModel

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"
CATALOG_PATH = MODELS_DIR / "polyvore_similarity_catalog.pt"
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))

    dataset = PolyvoreItemDataset(split="valid", transform=model.preprocess_val)
    item_ids = list(dataset.item_ids)

    build_catalog(model, dataset, item_ids, CATALOG_PATH)


if __name__ == "__main__":
    main()
