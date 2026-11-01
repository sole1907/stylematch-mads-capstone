"""Tests for the similarity engine's retrieval-accuracy evaluation.
Requires the DeepFashion CAP subset to already be downloaded; skipped
otherwise.
"""

import pytest
import torch

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.evaluate import (
    evaluate_retrieval_accuracy,
    random_baseline_accuracy,
    top1_retrieval_accuracy,
)
from stylematch.similarity.model import SimilarityModel

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_top1_retrieval_accuracy_perfect_case():
    """4 embeddings, 2 categories, each embedding's true nearest
    neighbor (by construction) is its category-mate -> accuracy 1.0."""
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],  # near embedding 0, same category
                [0.0, 1.0],
                [0.1, 0.9],  # near embedding 2, same category
            ]
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    assert top1_retrieval_accuracy(embeddings, labels) == 1.0


def test_top1_retrieval_accuracy_never_matches_self():
    """A single embedding per category: every query's nearest *other*
    embedding is necessarily a different category, so accuracy must be
    0, not 1 from matching itself."""
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    labels = torch.tensor([0, 1, 2, 3])  # all distinct, no same-category pair exists
    assert top1_retrieval_accuracy(embeddings, labels) == 0.0


def test_random_baseline_matches_manual_calculation():
    # 2 categories of 2 and 2: chance of a random *other* pick matching
    # category = 1/3 for each of the 4 images (1 same-category partner
    # out of 3 possible others) -> expected accuracy = 1/3.
    labels = torch.tensor([0, 0, 1, 1])
    assert random_baseline_accuracy(labels) == pytest.approx(1 / 3)


def test_evaluate_retrieval_accuracy_runs_end_to_end():
    model = SimilarityModel()
    result = evaluate_retrieval_accuracy(model, split="val", num_categories=10)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["random_baseline"] <= 1.0
    assert result["num_categories"] == 10
    assert result["num_images"] > 0
