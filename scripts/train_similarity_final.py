"""Final production training for the similarity engine, using the real
winning config confirmed via the full-scale hyperparameter search (see
reports/staged_search_summary.csv for the original 3 stages,
reports/margin_iteration2.csv for the margin/num_steps follow-up that
found the actual final winner once margin's original value turned out
to sit at the edge of its tested range).

Winning config: learning_rate=1e-5, freeze_until_block=10, margin=0.7,
num_steps=100 (improvement +0.167 over an untrained baseline).
num_categories=None (uncapped, every qualifying category) -- matching
how the real sweep itself trained, not TrainingConfig's own default of
20, which exists as a speed shortcut for comparing many configs
quickly, not meant for final production training.

Saves a checkpoint (models/similarity_final.pt) for later use by the
demo app and the official one-time test-set evaluation. Runs a
post-training sanity check on the validation split only -- the test
split stays untouched here, reserved for the separate, deliberate
official evaluation step.

Run directly:
    python scripts/train_similarity_final.py
"""

from loguru import logger

from stylematch.config import MODELS_DIR
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy
from stylematch.similarity.train import run_training_loop

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"


def main() -> None:
    model, losses = run_training_loop(
        num_steps=100,
        learning_rate=1e-5,
        freeze_until_block=10,
        margin=0.7,
        num_categories=None,
        checkpoint_path=CHECKPOINT_PATH,
    )
    logger.success(f"Final training losses: {losses}")
    logger.success(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    # Sanity check only, validation split -- not the real official
    # number, just confirming the saved checkpoint is actually behaving
    # reasonably before handing it off for the one-time test-set run.
    sanity = evaluate_retrieval_accuracy(model, split="val")
    logger.success(f"Post-training validation sanity check: {sanity}")


if __name__ == "__main__":
    main()
