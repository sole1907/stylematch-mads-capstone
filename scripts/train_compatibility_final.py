"""Final production training for the compatibility engine: the real
winning hyperparameter config from the full-scale sweep.

num_steps=33320 (not the sweep's 4000) is 5 full epochs over the real
53,306 train outfits (6,664 steps/epoch at outfits_per_batch=8) --
compatibility samples outfits uniformly, no per-category skew to
chase, so "full coverage" here is just multiple real epochs, not a
single biggest-category calculation.
Resumable via RESUME_PATH, since one run may exceed the 8h Slurm cap.
"""

from loguru import logger

from stylematch.compatibility.evaluate import evaluate_compatibility
from stylematch.compatibility.train import run_training_loop
from stylematch.config import MODELS_DIR

CHECKPOINT_PATH = MODELS_DIR / "compatibility_final.pt"
RESUME_PATH = MODELS_DIR / "compatibility_final_resume.pt"
CHECKPOINT_EVERY = 500

OUTFITS_PER_BATCH = 8
NUM_STEPS = 33320


def main() -> None:
    model, losses = run_training_loop(
        num_steps=NUM_STEPS,
        outfits_per_batch=OUTFITS_PER_BATCH,
        learning_rate=1e-5,
        freeze_until_block=10,
        margin=0.1,
        checkpoint_path=CHECKPOINT_PATH,
        resume_path=RESUME_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    logger.success(f"Final training losses: first={losses[0]:.4f} last={losses[-1]:.4f}")
    logger.success(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    # Sanity check only, not the official number -- that's a separate,
    # deliberate test-set evaluation step.
    sanity = evaluate_compatibility(model, split="valid", max_fitb_questions=50, max_compat_outfits=50)
    logger.success(f"Post-training validation sanity check: {sanity}")


if __name__ == "__main__":
    main()
