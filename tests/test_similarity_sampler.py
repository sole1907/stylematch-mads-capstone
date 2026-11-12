"""Tests for CategoryBalancedBatchSampler. Requires the DeepFashion CAP
subset to already be downloaded; skipped otherwise.
"""

import pytest
import torch
from pytorch_metric_learning import miners
from torch.utils.data import DataLoader

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.sampler import CategoryBalancedBatchSampler

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)

CATEGORIES_PER_BATCH = 4
IMAGES_PER_CATEGORY = 4
NUM_CATEGORIES = 20


def _make_dataset(min_images_per_category=IMAGES_PER_CATEGORY, num_categories=NUM_CATEGORIES):
    return DeepFashionCAPDataset(
        split="train",
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
    )


def test_batch_composition_is_correct():
    """Every batch should contain exactly categories_per_batch distinct
    categories, each contributing exactly images_per_category samples,
    not just the right total count, the right *composition*."""
    dataset = _make_dataset()
    sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
    )

    for batch_indices in sampler:
        assert len(batch_indices) == CATEGORIES_PER_BATCH * IMAGES_PER_CATEGORY
        labels_in_batch = [dataset.samples[i][1] for i in batch_indices]
        distinct_categories = set(labels_in_batch)
        assert len(distinct_categories) == CATEGORIES_PER_BATCH
        for category in distinct_categories:
            assert labels_in_batch.count(category) == IMAGES_PER_CATEGORY


def test_len_matches_actual_batches_produced():
    dataset = _make_dataset()
    sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
    )
    assert len(sampler) == sum(1 for _ in sampler)


def test_no_duplicate_categories_in_self_categories():
    """self.categories is built from dict keys, which can't repeat by
    construction, this just confirms that invariant holds against real
    data rather than assuming it."""
    dataset = _make_dataset()
    sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
    )
    assert len(sampler.categories) == len(set(sampler.categories))


def test_every_category_covered_once_when_not_evenly_divisible():
    """21 categories with categories_per_batch=4 doesn't divide evenly
    (5 full batches of 4, 1 leftover category). Every one of the 21
    should still appear exactly once across the epoch, not 20 with 1
    silently dropped (an earlier version's bug), and not 24 with 3
    duplicated to pad out a sixth batch.

    Critically, the leftover category must be *merged into* the last
    batch (making it bigger, 5 categories instead of 4), not given its
    own undersized trailing batch: a batch built from too few
    categories has no cross-category negatives to mine triplets from
    at all (a 1-category batch has zero, regardless of how many images
    are in it), which is exactly the failure mode this sampler exists
    to prevent. An earlier version of this fix got that backwards,
    caught by hand-checking the actual batch *composition*, not just
    the aggregate image count.
    """
    dataset = _make_dataset(num_categories=21)
    sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
    )

    categories_seen = []
    categories_per_batch_seen = []
    for batch_indices in sampler:
        labels_in_batch = {dataset.samples[i][1] for i in batch_indices}
        categories_per_batch_seen.append(len(labels_in_batch))
        categories_seen.extend(labels_in_batch)

    assert len(categories_seen) == 21  # every category exactly once, no drops, no duplicates
    assert len(set(categories_seen)) == 21
    assert categories_per_batch_seen == [4, 4, 4, 4, 5]  # 4 normal batches, last one absorbs the leftover
    assert all(count >= 2 for count in categories_per_batch_seen), (
        "every batch must have at least 2 distinct categories to have any "
        "cross-category negative to mine a triplet against"
    )
    assert len(sampler) == 5  # floor(21 / 4): no extra batch, the last one just grew


def test_raises_when_too_few_qualifying_categories():
    dataset = _make_dataset(num_categories=3)  # fewer categories than categories_per_batch needs
    with pytest.raises(ValueError):
        CategoryBalancedBatchSampler(
            dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
        )


def test_triplets_mined_bounded_and_nonzero_at_init():
    """Confirms the thing we actually investigated: batches always have
    the same composition (categories_per_batch x images_per_category),
    so the combinatorial ceiling on triplets is fixed and known, but how
    many of those are actually mined depends on the model's current
    embeddings (margin violations), not on data availability. At a
    freshly initialized model, embeddings are essentially random, so in
    practice every batch should still have some real violations, confirm
    that's actually true rather than assume it, and confirm no batch
    ever exceeds the known combinatorial maximum.
    """
    model = SimilarityModel()
    model.eval()
    dataset = DeepFashionCAPDataset(
        split="train",
        min_images_per_category=IMAGES_PER_CATEGORY,
        num_categories=NUM_CATEGORIES,
        transform=model.preprocess_train,
    )
    sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=CATEGORIES_PER_BATCH, images_per_category=IMAGES_PER_CATEGORY
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    miner_fn = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")

    batch_size = CATEGORIES_PER_BATCH * IMAGES_PER_CATEGORY
    positives_per_anchor = IMAGES_PER_CATEGORY - 1
    negatives_per_anchor = batch_size - IMAGES_PER_CATEGORY
    max_possible_triplets = batch_size * positives_per_anchor * negatives_per_anchor

    with torch.no_grad():
        for images, category_labels, _attribute_labels in loader:
            embeddings = model(images)
            indices_tuple = miner_fn(embeddings, category_labels)
            num_triplets = indices_tuple[0].numel()

            assert 0 <= num_triplets <= max_possible_triplets
            assert num_triplets > 0, (
                "found zero triplets at model init, expected some margin "
                "violations given essentially-random starting embeddings"
            )
