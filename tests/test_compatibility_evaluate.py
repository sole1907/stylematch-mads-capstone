"""Tests for the compatibility engine's FITB accuracy and AUC
evaluation. Requires the Polyvore Outfits dataset to already be
downloaded; skipped otherwise.
"""

import pytest

from stylematch.compatibility.evaluate import evaluate_compatibility
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import POLYVORE_DIR

pytestmark = pytest.mark.skipif(
    not (POLYVORE_DIR / "nondisjoint" / "valid.json").exists(),
    reason="Polyvore Outfits dataset not downloaded locally",
)


def test_evaluate_compatibility_runs_end_to_end():
    model = CompatibilityModel()
    result = evaluate_compatibility(
        model, split="valid", max_fitb_questions=8, max_compat_outfits=8
    )

    assert 0.0 <= result["fitb_accuracy"] <= 1.0
    assert result["fitb_num_questions"] == 8
    assert 0.0 <= result["compatibility_auc"] <= 1.0
    assert result["compatibility_num_outfits"] == 8


def test_max_compat_outfits_is_balanced_across_labels():
    """Real file is grouped by label (all label=1 first, then
    label=0) -- confirms the balancing fix actually avoids grabbing
    only one label, which would otherwise break AUC."""
    model = CompatibilityModel()
    result = evaluate_compatibility(model, split="valid", max_fitb_questions=1, max_compat_outfits=4)
    # Just confirming it didn't raise (roc_auc_score raises if only one
    # class is present) and produced a valid AUC.
    assert 0.0 <= result["compatibility_auc"] <= 1.0
