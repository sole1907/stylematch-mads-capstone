"""Zero-shot baseline: the raw, pretrained Marqo-FashionCLIP backbone
(no projection head, no fine-tuning) evaluated on the exact same real
test-set protocol as the official evaluation (scripts/evaluate_similarity_final.py)
-- same 21 categories, same uncapped test split -- so the numbers are
directly comparable to our trained model's 0.6635 accuracy.

Run directly:
    python scripts/evaluate_similarity_baseline.py
"""

from loguru import logger

from stylematch.baseline import RawBackboneModel
from stylematch.similarity.dataset import get_qualifying_categories
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy

CATEGORY_INCLUSION_THRESHOLD = 1000


def main() -> None:
    model = RawBackboneModel()
    allowed_categories = get_qualifying_categories(CATEGORY_INCLUSION_THRESHOLD, split="train")

    results = evaluate_retrieval_accuracy(
        model,
        split="test",
        allowed_categories=allowed_categories,
        max_images_per_category=None,
    )
    logger.success(f"Zero-shot baseline (raw backbone, no fine-tuning) test-set evaluation: {results}")


if __name__ == "__main__":
    main()
