"""Smoke test for the similarity engine's real training loop (real
model, real category-balanced batches, real triplet loss). Requires the
DeepFashion CAP subset to already be downloaded; skipped otherwise.
"""

import pytest

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.train import run_training_loop

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_runs_end_to_end_and_loss_changes():
    model, losses = run_training_loop(
        num_steps=3, categories_per_batch=4, images_per_category=4, num_categories=20
    )

    assert isinstance(model, SimilarityModel)
    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    # Not claiming the loss meaningfully decreases over just 3 steps on a
    # small subset, that's a Week 3 question. Just confirming gradients
    # are actually flowing: successive losses shouldn't be frozen at an
    # identical value, which would indicate a dead/degenerate loss (as
    # the old placeholder silently was, see train.py's git history).
    assert len(set(losses)) > 1


def test_resume_continues_from_saved_step_instead_of_restarting(tmp_path):
    resume_path = tmp_path / "resume.pt"

    _, first_losses = run_training_loop(
        num_steps=2,
        categories_per_batch=4,
        images_per_category=4,
        num_categories=20,
        resume_path=resume_path,
        checkpoint_every=1,
    )
    assert resume_path.exists()

    _, resumed_losses = run_training_loop(
        num_steps=5,
        categories_per_batch=4,
        images_per_category=4,
        num_categories=20,
        resume_path=resume_path,
        checkpoint_every=1,
    )

    # First 2 losses carried over unchanged from the earlier run, not
    # retrained from step 0 -- only the remaining 3 steps are new.
    assert resumed_losses[:2] == first_losses
    assert len(resumed_losses) == 5


def test_resume_past_num_steps_is_a_noop(tmp_path):
    resume_path = tmp_path / "resume.pt"
    kwargs = dict(
        num_steps=3,
        categories_per_batch=4,
        images_per_category=4,
        num_categories=20,
        resume_path=resume_path,
        checkpoint_every=1,
    )

    _, first_losses = run_training_loop(**kwargs)
    # Same num_steps=3 again: start_step (4) > num_steps (3), so no new
    # training should happen -- the resumed losses come back unchanged.
    _, second_losses = run_training_loop(**kwargs)

    assert second_losses == first_losses
