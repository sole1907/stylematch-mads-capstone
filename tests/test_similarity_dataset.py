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


def test_stratified_loading_explicit_cap_is_respected():
    """max_images_per_category, when explicitly set, still caps each
    category exactly as before the fix (preserving the old behavior for
    callers that want a bounded subset, e.g. evaluate.py/
    naive_baseline.py, which pin this deliberately rather than inherit
    an uncapped default).
    """
    dataset = DeepFashionCAPDataset(
        split="train", min_images_per_category=2, num_categories=5, max_images_per_category=2
    )
    counts: dict[int, int] = {}
    for _, category_label, _ in dataset.samples:
        counts[category_label] = counts.get(category_label, 0) + 1

    assert len(counts) == 5
    assert all(count == 2 for count in counts.values())


def test_max_images_per_category_none_loads_more_than_the_minimum():
    """The actual fix: with max_images_per_category left at its default
    (None, uncapped), at least one real category should have strictly
    more than min_images_per_category images collected, proving the cap
    is genuinely decoupled, not silently still capping at the minimum
    (the bug this replaces: num_categories=None with
    min_images_per_category=4 previously returned exactly 46 x 4 = 184
    samples, confirmed empirically before this fix, never more than 4
    per category no matter how many a category actually had).
    """
    capped = DeepFashionCAPDataset(
        split="train", min_images_per_category=2, num_categories=5, max_images_per_category=2
    )
    uncapped = DeepFashionCAPDataset(
        split="train", min_images_per_category=2, num_categories=5, max_images_per_category=None
    )

    assert len(capped) == 10  # 5 categories x 2, the old-style bound
    assert len(uncapped) > len(capped)  # same 5 categories, genuinely more images each


def test_num_categories_early_stop_still_exact_when_uncapped():
    """The block-boundary-only early stop (rewritten as part of this
    fix, see dataset.py) should still return exactly num_categories
    distinct categories, not more, not fewer, even though scanning now
    continues to the end of whichever category's block was in progress
    when the stop condition was first met.
    """
    dataset = DeepFashionCAPDataset(
        split="train", min_images_per_category=2, num_categories=5, max_images_per_category=None
    )
    distinct_categories = {category_label for _, category_label, _ in dataset.samples}
    assert len(distinct_categories) == 5
