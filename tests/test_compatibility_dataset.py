"""Tests for the compatibility-engine Polyvore Outfits datasets.

Two tiers: pure unit tests for the composite-ID resolver (no real data
needed), and smoke tests against the real downloaded files (see
scripts/download_polyvore.py), skipped automatically if they aren't
present locally.
"""

import pytest

from stylematch.compatibility.dataset import (
    CompatibilityDataset,
    FITBDataset,
    PolyvoreOutfitDataset,
    _build_outfit_index,
    _resolve_composite_id,
)
from stylematch.config import POLYVORE_DIR

pytestmark = pytest.mark.skipif(
    not (POLYVORE_DIR / "nondisjoint" / "valid.json").exists(),
    reason="Polyvore Outfits dataset not downloaded locally",
)


def test_resolve_composite_id_matches_outfit_index():
    outfits = [
        {"set_id": "100", "items": [{"item_id": "aaa", "index": 1}, {"item_id": "bbb", "index": 2}]},
        {"set_id": "200", "items": [{"item_id": "ccc", "index": 1}]},
    ]
    index = _build_outfit_index(outfits)
    assert _resolve_composite_id("100_2", index) == "bbb"
    assert _resolve_composite_id("200_1", index) == "ccc"


def test_outfit_dataset_len_and_getitem_shapes():
    dataset = PolyvoreOutfitDataset(split="valid")
    assert len(dataset) > 0

    item_images, item_ids, category_labels, outfit_id = dataset[0]
    assert len(item_images) == len(item_ids) == len(category_labels)
    assert item_images[0].shape == (3, 224, 224)
    assert all(isinstance(c, int) for c in category_labels)
    assert isinstance(outfit_id, str)


def test_outfit_dataset_invalid_split_raises():
    with pytest.raises(ValueError):
        PolyvoreOutfitDataset(split="not_a_real_split")


def test_fitb_dataset_shapes_and_correct_index():
    dataset = FITBDataset(split="valid")
    assert len(dataset) > 0

    context_images, candidate_images, correct_index = dataset[0]
    assert len(candidate_images) == 4
    assert correct_index == 0
    assert context_images[0].shape == (3, 224, 224)
    assert candidate_images[0].shape == (3, 224, 224)


def test_compatibility_dataset_labels_are_binary():
    """Labels arrive in contiguous blocks in the real file (all label=1
    rows first, then label=0), same "grouped, not interleaved" pattern
    as DeepFashion's categories -- so this checks both known blocks
    directly via dataset.questions, rather than assuming the first N
    __getitem__ calls would see a mix."""
    dataset = CompatibilityDataset(split="valid")
    assert len(dataset) > 0

    labels = [label for label, _ in dataset.questions]
    assert set(labels) == {0, 1}

    first_positive_idx = labels.index(1)
    first_negative_idx = labels.index(0)

    for idx in (first_positive_idx, first_negative_idx):
        outfit_images, label = dataset[idx]
        assert label in (0, 1)
        assert len(outfit_images) > 0
