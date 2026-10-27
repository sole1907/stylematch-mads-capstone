"""Smoke tests for DeepFashionCAPDataset, run against the small sample
pulled from the real downloaded files. Requires the DeepFashion CAP
subset to already be downloaded (see scripts/download_deepfashion.py);
skipped automatically if it isn't.
"""

import pytest
from torch.utils.data import DataLoader

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.dataset import NUM_ATTRIBUTES, DeepFashionCAPDataset

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_len_matches_sample_size():
    dataset = DeepFashionCAPDataset(split="train", sample_size=10)
    assert len(dataset) == 10


def test_getitem_shapes_and_types():
    dataset = DeepFashionCAPDataset(split="train", sample_size=5)
    image, category_label, attribute_label = dataset[0]

    assert image.shape == (3, 224, 224)
    assert isinstance(category_label, int)
    assert category_label >= 0  # 0-indexed, never the raw file's 1-indexed value
    assert attribute_label.shape == (NUM_ATTRIBUTES,)
    assert set(attribute_label.unique().tolist()) <= {-1, 1}


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        DeepFashionCAPDataset(split="not_a_real_split", sample_size=1)


def test_dataloader_batches():
    dataset = DeepFashionCAPDataset(split="train", sample_size=16)
    loader = DataLoader(dataset, batch_size=4)
    images, categories, attributes = next(iter(loader))

    assert images.shape == (4, 3, 224, 224)
    assert categories.shape == (4,)
    assert attributes.shape == (4, NUM_ATTRIBUTES)
