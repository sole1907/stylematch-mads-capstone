"""Official, one-time test-set evaluation for the trained similarity
engine. Uses the real held-out test split (never touched by the sweep
or training) and the full image pool per category, uncapped -- not
the sweep/sanity-check's thin per-category cap.

Run directly:
    python scripts/evaluate_similarity_final.py
"""

from loguru import logger
import torch

from stylematch.config import MODELS_DIR
from stylematch.similarity.dataset import get_qualifying_categories
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy
from stylematch.similarity.model import SimilarityModel

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"
CATEGORY_INCLUSION_THRESHOLD = 1000
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))
    logger.info(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    allowed_categories = get_qualifying_categories(CATEGORY_INCLUSION_THRESHOLD, split="train")

    results = evaluate_retrieval_accuracy(
        model,
        split="test",
        allowed_categories=allowed_categories,
        max_images_per_category=None,
    )
    logger.success(f"Official test-set evaluation: {results}")


if __name__ == "__main__":
    main()
