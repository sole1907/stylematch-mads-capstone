"""Smoke test for the similarity engine's real training loop (real
model, real category-balanced batches, real triplet loss). Requires the
DeepFashion CAP subset to already be downloaded; skipped otherwise.
"""

import pytest

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.train import run_training_loop

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_runs_end_to_end_and_loss_changes():
    losses = run_training_loop(
        num_steps=3, categories_per_batch=4, images_per_category=4, num_categories=20
    )

    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    # Not claiming the loss meaningfully decreases over just 3 steps on a
    # small subset, that's a Week 3 question. Just confirming gradients
    # are actually flowing: successive losses shouldn't be frozen at an
    # identical value, which would indicate a dead/degenerate loss (as
    # the old placeholder silently was, see train.py's git history).
    assert len(set(losses)) > 1
