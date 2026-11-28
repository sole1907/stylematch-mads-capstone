"""Tests for the shared demo-catalog builder. Uses a small real
DeepFashion sample (fast) with a freshly-initialized SimilarityModel
-- testing the save/shape mechanics, not retrieval quality, so an
untrained model (no checkpoint needed) is fine.
"""

import pytest
import torch

from stylematch.catalog import AFRIFASHION_CLASSES, AfrifashionDataset, build_catalog
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


@requires_afrifashion
def test_afrifashion_dataset_shapes_and_labels():
    dataset = AfrifashionDataset()
    assert len(dataset) == 1600

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
