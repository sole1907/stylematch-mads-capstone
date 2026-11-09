"""Tests for the naive color-histogram retrieval baseline. The
hand-verified cases don't need real data (pure color math); the
end-to-end tests do, and require the DeepFashion CAP subset to already
be downloaded, skipped otherwise.
"""

import pytest
import torch

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.naive_baseline import (
    compare_similarity_vs_naive_baseline,
    compute_color_histogram,
    compute_color_histogram_batch,
    evaluate_naive_color_baseline,
)
from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel


def test_identical_images_have_similarity_one():
    image = torch.rand(3, 16, 16)
    feature_a = compute_color_histogram(image)
    feature_b = compute_color_histogram(image.clone())
    assert (feature_a @ feature_b).item() == pytest.approx(1.0, abs=1e-5)


def test_complementary_colors_have_zero_similarity():
    """Solid red (R=1,G=0,B=0) vs. solid cyan (R=0,G=1,B=1): every
    channel's histogram lands in a completely different bin for the
    two images (no channel's value ever matches), so the dot product
    contribution from each of the 3 channels is exactly 0, confirmed
    by hand and empirically before writing this test.
    """
    red = torch.zeros(3, 8, 8)
    red[0] = 1.0
    cyan = torch.zeros(3, 8, 8)
    cyan[1] = 1.0
    cyan[2] = 1.0

    feature_red = compute_color_histogram(red)
    feature_cyan = compute_color_histogram(cyan)
    assert (feature_red @ feature_cyan).item() == pytest.approx(0.0, abs=1e-6)


def test_feature_shape_matches_bins_times_channels():
    image = torch.rand(3, 16, 16)
    feature = compute_color_histogram(image, bins=10)
    assert feature.shape == (30,)  # 3 channels x 10 bins


def test_feature_is_unit_norm():
    image = torch.rand(3, 16, 16)
    feature = compute_color_histogram(image)
    assert feature.norm().item() == pytest.approx(1.0, abs=1e-5)


requires_deepfashion = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


@requires_deepfashion
def test_compute_color_histogram_batch_runs_end_to_end():
    dataset = DeepFashionCAPDataset(
        split="val", transform=None, min_images_per_category=4, num_categories=10
    )
    features, category_labels = compute_color_histogram_batch(dataset)

    assert features.shape == (len(dataset), 3 * 32)  # default 32 bins/channel
    assert category_labels.shape == (len(dataset),)
    norms = features.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(len(dataset)), atol=1e-4)


@requires_deepfashion
def test_evaluate_naive_color_baseline_runs_end_to_end():
    result = evaluate_naive_color_baseline(split="val", num_categories=10)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["random_baseline"] <= 1.0
    assert result["num_categories"] == 10
    assert result["num_images"] > 0


@requires_deepfashion
def test_compare_similarity_vs_naive_baseline_runs_on_matching_subset():
    """Doesn't assert the trained model beats the naive baseline (a
    freshly-constructed, untrained-projection-head model has no reason
    to), only that both sides ran on the exact same subset and every
    number is in a sane range. The real before/after comparison is a
    separate, documented run against an actually-trained checkpoint.
    """
    model = SimilarityModel()
    result = compare_similarity_vs_naive_baseline(model, split="val", num_categories=10)

    assert 0.0 <= result["similarity_model_accuracy"] <= 1.0
    assert 0.0 <= result["similarity_model_attribute_aware_accuracy"] <= 1.0
    assert 0.0 <= result["naive_baseline_accuracy"] <= 1.0
    assert 0.0 <= result["random_baseline"] <= 1.0
    assert result["num_categories"] == 10
    assert result["num_images"] > 0

    # same subset for both sides: cross-check against each function's
    # own standalone result on identical arguments.
    naive_only = evaluate_naive_color_baseline(split="val", num_categories=10)
    assert result["naive_baseline_accuracy"] == naive_only["accuracy"]
    assert result["num_images"] == naive_only["num_images"]
