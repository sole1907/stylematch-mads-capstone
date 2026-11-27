"""Smoke test for the compatibility engine's real training loop (real
model, real outfits, real margin ranking loss). Requires the Polyvore
Outfits dataset to already be downloaded; skipped otherwise.
"""

import pytest

from stylematch.compatibility.model import CompatibilityModel
from stylematch.compatibility.train import run_training_loop
from stylematch.config import POLYVORE_DIR

pytestmark = pytest.mark.skipif(
    not (POLYVORE_DIR / "nondisjoint" / "train.json").exists(),
    reason="Polyvore Outfits dataset not downloaded locally",
)


def test_runs_end_to_end_and_loss_changes():
    model, losses = run_training_loop(num_steps=3, outfits_per_batch=4)

    assert isinstance(model, CompatibilityModel)
    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    # Not claiming the loss meaningfully decreases over 3 steps, just
    # that gradients are actually flowing (successive losses shouldn't
    # be frozen at an identical value).
    assert len(set(losses)) > 1


def test_outfits_per_batch_below_two_raises():
    with pytest.raises(ValueError):
        run_training_loop(num_steps=1, outfits_per_batch=1)


def test_resume_continues_from_saved_step_instead_of_restarting(tmp_path):
    resume_path = tmp_path / "resume.pt"

    _, first_losses = run_training_loop(
        num_steps=2, outfits_per_batch=4, resume_path=resume_path, checkpoint_every=1
    )
    assert resume_path.exists()

    _, resumed_losses = run_training_loop(
        num_steps=5, outfits_per_batch=4, resume_path=resume_path, checkpoint_every=1
    )

    # First 2 losses carried over unchanged from the earlier run, not
    # retrained from step 0 -- only the remaining 3 steps are new.
    assert resumed_losses[:2] == first_losses
    assert len(resumed_losses) == 5


def test_resume_past_num_steps_is_a_noop(tmp_path):
    resume_path = tmp_path / "resume.pt"
    kwargs = dict(num_steps=3, outfits_per_batch=4, resume_path=resume_path, checkpoint_every=1)

    _, first_losses = run_training_loop(**kwargs)
    # Same num_steps=3 again: start_step (4) > num_steps (3), so no new
    # training should happen -- the resumed losses come back unchanged.
    _, second_losses = run_training_loop(**kwargs)

    assert second_losses == first_losses
