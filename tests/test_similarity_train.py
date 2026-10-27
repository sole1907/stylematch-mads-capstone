"""Smoke test for the similarity engine's stub training loop. Requires
the DeepFashion CAP subset to already be downloaded; skipped otherwise.
"""

import pytest

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.train import run_stub_training_loop

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_runs_end_to_end_and_loss_changes():
    losses = run_stub_training_loop(num_steps=3, batch_size=4, sample_size=16)

    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    # Not claiming the loss meaningfully decreases, the placeholder model
    # and loss aren't real. Just confirming gradients are actually
    # flowing: successive losses shouldn't be frozen at an identical value.
    assert len(set(losses)) > 1
