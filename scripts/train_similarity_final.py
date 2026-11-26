"""Final production training for the similarity engine: the real
winning hyperparameter config from the full-scale sweep.

num_steps=39105 (not the sweep's 500) is recalculated for one
full-coverage cycle through the biggest qualifying category.
Resumable via RESUME_PATH, since one run may exceed the 8h Slurm cap.
"""

from loguru import logger

from stylematch.config import MODELS_DIR
from stylematch.similarity.dataset import get_qualifying_categories
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy
from stylematch.similarity.train import run_training_loop

CHECKPOINT_PATH = MODELS_DIR / "similarity_final.pt"
RESUME_PATH = MODELS_DIR / "similarity_final_resume.pt"
CHECKPOINT_EVERY = 2000

CATEGORY_INCLUSION_THRESHOLD = 1000
IMAGES_PER_CATEGORY = 4
CATEGORIES_PER_BATCH = 6
NUM_STEPS = 39105


def main() -> None:
    allowed_categories = get_qualifying_categories(CATEGORY_INCLUSION_THRESHOLD, split="train")
    logger.info(f"Training on {len(allowed_categories)} qualifying categories (threshold={CATEGORY_INCLUSION_THRESHOLD})")

    model, losses = run_training_loop(
        num_steps=NUM_STEPS,
        categories_per_batch=CATEGORIES_PER_BATCH,
        images_per_category=IMAGES_PER_CATEGORY,
        learning_rate=1e-5,
        freeze_until_block=10,
        margin=0.7,
        weight_by_category_size=False,
        allowed_categories=allowed_categories,
        checkpoint_path=CHECKPOINT_PATH,
        resume_path=RESUME_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    logger.success(f"Final training losses: first={losses[0]:.4f} last={losses[-1]:.4f}")
    logger.success(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    # Sanity check only, not the official number -- that's a separate,
    # deliberate test-set evaluation step.
    sanity = evaluate_retrieval_accuracy(
        model,
        split="val",
        min_images_per_category=IMAGES_PER_CATEGORY,
        allowed_categories=allowed_categories,
    )
    logger.success(f"Post-training validation sanity check: {sanity}")


if __name__ == "__main__":
    main()
