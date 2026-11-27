"""Official, one-time test-set evaluation for the trained compatibility
engine. Uses the real held-out test split (never touched by the sweep
or training) and the full FITB/compatibility question sets, uncapped
(max_fitb_questions/max_compat_outfits already default to None here,
unlike the similarity engine's evaluate_retrieval_accuracy).

Run directly:
    python scripts/evaluate_compatibility_final.py
"""

from loguru import logger
import torch

from stylematch.compatibility.evaluate import evaluate_compatibility
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import MODELS_DIR

CHECKPOINT_PATH = MODELS_DIR / "compatibility_final.pt"
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10


def main() -> None:
    model = CompatibilityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=model.device))
    logger.info(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    results = evaluate_compatibility(model, split="test")
    logger.success(f"Official test-set evaluation: {results}")


if __name__ == "__main__":
    main()
