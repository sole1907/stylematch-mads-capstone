"""Zero-shot baseline: the raw, pretrained Marqo-FashionCLIP backbone
(no projection head, no fine-tuning) evaluated on the exact same real
test-set protocol as the official evaluation (scripts/evaluate_compatibility_final.py)
-- same uncapped FITB/AUC test sets -- so the numbers are directly
comparable to our trained model's FITB=0.686/AUC=0.9377.

Run directly:
    python scripts/evaluate_compatibility_baseline.py
"""

from loguru import logger

from stylematch.baseline import RawBackboneModel
from stylematch.compatibility.evaluate import evaluate_compatibility


def main() -> None:
    model = RawBackboneModel()
    results = evaluate_compatibility(model, split="test")
    logger.success(f"Zero-shot baseline (raw backbone, no fine-tuning) test-set evaluation: {results}")


if __name__ == "__main__":
    main()
