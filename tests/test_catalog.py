"""Tests for the shared demo-catalog builder. Uses a small real
DeepFashion sample (fast) with a freshly-initialized SimilarityModel
-- testing the save/shape mechanics, not retrieval quality, so an
untrained model (no checkpoint needed) is fine.
"""

import pytest
import torch

from stylematch.catalog import build_catalog
from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
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
