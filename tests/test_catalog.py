"""Tests for the shared demo-catalog builder. Uses a small real
DeepFashion sample (fast) with a freshly-initialized SimilarityModel
-- testing the save/shape mechanics, not retrieval quality, so an
untrained model (no checkpoint needed) is fine.
"""

import pytest
import torch

from stylematch.catalog import (
    AFRIFASHION_CLASSES,
    AfrifashionDataset,
    build_catalog,
    search_by_compatibility,
    search_by_similarity,
)
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import AFRIFASHION_DIR, DEEPFASHION_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)

requires_afrifashion = pytest.mark.skipif(
    not AFRIFASHION_DIR.exists(), reason="AFRIFASHION1600 not downloaded locally"
)


def test_build_catalog_saves_matching_shapes(tmp_path):
    model = SimilarityModel()
    dataset = DeepFashionCAPDataset(split="val", transform=model.preprocess_val, sample_size=10)
    identifiers = [image_name for image_name, _, _ in dataset.samples]
    catalog_path = tmp_path / "catalog.pt"

    build_catalog(model, dataset, identifiers, catalog_path)

    catalog = torch.load(catalog_path)
    assert catalog["embeddings"].shape == (len(dataset), 256)
    assert catalog["identifiers"] == identifiers
    assert len(catalog["category_labels"]) == len(dataset)


def test_build_catalog_identifier_count_mismatch_raises(tmp_path):
    model = SimilarityModel()
    dataset = DeepFashionCAPDataset(split="val", transform=model.preprocess_val, sample_size=10)
    wrong_length_identifiers = ["a", "b", "c"]  # deliberately not len(dataset)

    with pytest.raises(ValueError):
        build_catalog(model, dataset, wrong_length_identifiers, tmp_path / "catalog.pt")


def test_search_by_similarity_ranks_correctly():
    catalog = {
        "embeddings": torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]), dim=-1
        ),
        "identifiers": ["near", "near2", "far", "opposite"],
        "category_labels": torch.tensor([0, 0, 1, 1]),
    }
    query = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), dim=0)

    results = search_by_similarity(query, catalog, top_k=4)

    assert [identifier for identifier, _ in results] == ["near", "near2", "far", "opposite"]
    # "near" is the query's exact direction -> cosine similarity 1.0.
    assert results[0][1] == pytest.approx(1.0)


def test_search_by_compatibility_matches_score_outfit_directly():
    """Cross-checks the vectorized shortcut against calling the real
    score_outfit() per candidate -- not re-deriving the same formula
    independently, so a shared mistake in both wouldn't go unnoticed."""
    model = CompatibilityModel()
    torch.manual_seed(0)
    context = torch.nn.functional.normalize(torch.randn(3, 256), dim=-1)
    catalog_embeddings = torch.nn.functional.normalize(torch.randn(10, 256), dim=-1)
    catalog = {
        "embeddings": catalog_embeddings,
        "identifiers": [f"item{i}" for i in range(10)],
        "category_labels": torch.zeros(10, dtype=torch.long),
    }

    expected_scores = [
        model.score_outfit(torch.cat([context, catalog_embeddings[i : i + 1]])).item() for i in range(10)
    ]

    results = search_by_compatibility(context, catalog, top_k=10)
    actual_by_identifier = dict(results)

    for i, expected in enumerate(expected_scores):
        assert actual_by_identifier[f"item{i}"] == pytest.approx(expected, abs=1e-5)


def test_search_by_compatibility_single_context_item_matches_score_outfit():
    """k=1 is the degenerate case (no context-context pairs) -- same
    cross-check, confirming the formula still holds there."""
    model = CompatibilityModel()
    torch.manual_seed(1)
    context = torch.nn.functional.normalize(torch.randn(1, 256), dim=-1)
    catalog_embeddings = torch.nn.functional.normalize(torch.randn(5, 256), dim=-1)
    catalog = {
        "embeddings": catalog_embeddings,
        "identifiers": [f"item{i}" for i in range(5)],
        "category_labels": torch.zeros(5, dtype=torch.long),
    }

    expected_scores = [
        model.score_outfit(torch.cat([context, catalog_embeddings[i : i + 1]])).item() for i in range(5)
    ]
    results = search_by_compatibility(context, catalog, top_k=5)
    actual_by_identifier = dict(results)

    for i, expected in enumerate(expected_scores):
        assert actual_by_identifier[f"item{i}"] == pytest.approx(expected, abs=1e-5)


def test_search_by_compatibility_excludes_categories():
    context = torch.nn.functional.normalize(torch.randn(2, 256), dim=-1)
    catalog_embeddings = torch.nn.functional.normalize(torch.randn(4, 256), dim=-1)
    catalog = {
        "embeddings": catalog_embeddings,
        "identifiers": ["a", "b", "c", "d"],
        "category_labels": torch.tensor([0, 0, 1, 1]),
    }

    results = search_by_compatibility(context, catalog, top_k=4, exclude_category_labels={0})

    assert {identifier for identifier, _ in results} == {"c", "d"}


@requires_afrifashion
def test_afrifashion_dataset_shapes_and_labels():
    """1600 total minus whatever split_afrifashion_holdout.py already
    moved into holdout/ -- 1440 once that one-time split has run,
    confirming the non-recursive glob really does exclude it."""
    dataset = AfrifashionDataset()
    assert len(dataset) in (1600, 1440)

    image, category_label, image_name = dataset[0]
    assert image.shape == (3, 224, 224)
    assert 0 <= category_label < len(AFRIFASHION_CLASSES)
    assert image_name.endswith(".png")


@requires_afrifashion
def test_afrifashion_dataset_parses_underscore_containing_class_names():
    """"African_Blouse_123.png" must parse as class "African_Blouse",
    not split on the first underscore -- a real filename pattern in
    this dataset, not a hypothetical edge case."""
    dataset = AfrifashionDataset()
    matches = [name for name in dataset.image_names if name.startswith("African_Blouse_")]
    assert len(matches) > 0

    idx = dataset.image_names.index(matches[0])
    _, category_label, _ = dataset[idx]
    assert AFRIFASHION_CLASSES[category_label] == "African_Blouse"


@requires_afrifashion
def test_afrifashion_dataset_exactly_8_classes_used():
    """Checks filenames directly, not via __getitem__ for all 1,600
    images -- the label is derivable from the name alone, no need to
    actually load and transform every real image for this check."""
    import re

    dataset = AfrifashionDataset()
    pattern = re.compile(r"^(.+)_\d+\.png$")
    classes_seen = {pattern.match(name).group(1) for name in dataset.image_names}
    assert classes_seen == set(AFRIFASHION_CLASSES)
