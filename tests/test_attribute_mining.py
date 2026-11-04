"""Tests for attribute-aware triplet mining. The synthetic cases are
hand-verified exact expectations, not just sanity checks, see each
test's docstring for the hand-worked-out reasoning. Real-data tests
require the DeepFashion CAP subset to already be downloaded; skipped
otherwise.
"""

import pytest
import torch

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.attribute_mining import mine_attribute_aware_triplets
from stylematch.similarity.dataset import DeepFashionCAPDataset


def _as_triplet_set(indices_tuple):
    anchors, positives, negatives = indices_tuple
    return set(zip(anchors.tolist(), positives.tolist(), negatives.tolist()))


def test_exact_triplets_hand_verified():
    """4 items, categories [0, 0, 1, 1], 3-attribute toy vectors:
    item0 active={0,1}, item1 active={0}, item2 active={2}, item3 active={}.

    Hand-worked expectations (min_shared_attributes=1):
    - anchor0: shares attr0 with item1 (same category) -> positive {1};
      negatives are the different-category items {2, 3}.
    - anchor1: symmetric with anchor0 -> positive {0}; negatives {2, 3}.
    - anchor2: same-category partner is item3, but item3 has zero
      active attributes and item2 (the anchor) does NOT, so the
      fallback does not apply to anchor2 (it only waives the
      requirement when the ANCHOR itself has no attributes) -> overlap
      with item3 is 0, no valid positive -> anchor2 contributes no
      triplets at all.
    - anchor3: item3 itself has zero active attributes, so the fallback
      applies -> item2 (same category) counts as a positive regardless
      of overlap -> positive {2}; negatives {0, 1}.

    This also demonstrates the fallback is intentionally asymmetric:
    (anchor=2, candidate=3) is excluded, but (anchor=3, candidate=2) is
    included, since the fallback depends on which item is the anchor.
    """
    category_labels = torch.tensor([0, 0, 1, 1])
    attribute_labels = torch.tensor(
        [
            [1, 1, -1],  # item0: active {0, 1}
            [1, -1, -1],  # item1: active {0}
            [-1, -1, 1],  # item2: active {2}
            [-1, -1, -1],  # item3: active {} (no active attributes)
        ]
    )

    result = mine_attribute_aware_triplets(category_labels, attribute_labels)

    expected = {
        (0, 1, 2), (0, 1, 3),
        (1, 0, 2), (1, 0, 3),
        (3, 2, 0), (3, 2, 1),
        # anchor 2 contributes nothing, see docstring
    }
    assert _as_triplet_set(result) == expected


def test_never_matches_self():
    category_labels = torch.tensor([0, 0, 0])
    attribute_labels = torch.tensor([[1, 1], [1, 1], [1, 1]])  # all identical, max overlap
    anchors, positives, _negatives = mine_attribute_aware_triplets(category_labels, attribute_labels)
    assert all(a != p for a, p in zip(anchors.tolist(), positives.tolist()))


def test_no_valid_negatives_returns_empty():
    """Single category present -> no different-category item exists for
    any anchor -> no valid triplets at all, not an error."""
    category_labels = torch.tensor([0, 0, 0])
    attribute_labels = torch.tensor([[1, 1], [1, -1], [-1, 1]])
    anchors, positives, negatives = mine_attribute_aware_triplets(category_labels, attribute_labels)
    assert anchors.numel() == 0
    assert positives.numel() == 0
    assert negatives.numel() == 0


def test_min_shared_attributes_threshold_is_respected():
    """Two same-category items sharing exactly 1 attribute: a positive
    under the default threshold (1), but not under a threshold of 2."""
    category_labels = torch.tensor([0, 0, 1])
    attribute_labels = torch.tensor(
        [
            [1, -1, -1],  # item0: active {0}
            [1, -1, -1],  # item1: active {0} -- shares exactly 1 attr with item0
            [-1, 1, 1],  # item2: different category
        ]
    )
    result_threshold_1 = mine_attribute_aware_triplets(
        category_labels, attribute_labels, min_shared_attributes=1
    )
    assert (0, 1, 2) in _as_triplet_set(result_threshold_1)

    result_threshold_2 = mine_attribute_aware_triplets(
        category_labels, attribute_labels, min_shared_attributes=2
    )
    # Neither item0 nor item1 has zero attributes, so the fallback
    # doesn't rescue this pair at the stricter threshold.
    assert _as_triplet_set(result_threshold_2) == set()


requires_deepfashion = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


@requires_deepfashion
def test_real_data_every_returned_triplet_satisfies_its_own_definition():
    """Confirms the actual intent on real data, not just internal
    consistency: every (anchor, positive, negative) triplet mined must
    genuinely satisfy category-match (+ attribute overlap or fallback)
    for its positive, and category-mismatch for its negative.
    """
    dataset = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=10)
    samples = [dataset.samples[i] for i in range(32)]
    category_labels = torch.tensor([s[1] for s in samples])
    attribute_labels = torch.tensor([s[2] for s in samples])

    anchors, positives, negatives = mine_attribute_aware_triplets(category_labels, attribute_labels)
    active = attribute_labels == 1

    for a, p, n in zip(anchors.tolist(), positives.tolist(), negatives.tolist()):
        assert category_labels[a] == category_labels[p], "positive must share the anchor's category"
        assert category_labels[a] != category_labels[n], "negative must differ in category"
        overlap = (active[a] & active[p]).sum().item()
        anchor_has_no_attrs = active[a].sum().item() == 0
        assert overlap >= 1 or anchor_has_no_attrs, (
            f"positive pair ({a}, {p}) has {overlap} shared attributes and the anchor "
            f"has attributes of its own, should not have been mined as a positive"
        )


@requires_deepfashion
def test_real_data_is_subset_of_category_only_mining():
    """Attribute-aware mining only tightens the positive definition, it
    never loosens it, so the resulting triplet count should never
    exceed what plain category-only mining (TripletMarginMiner) would
    find on the same batch.
    """
    from pytorch_metric_learning import miners

    dataset = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=10)
    samples = [dataset.samples[i] for i in range(32)]
    category_labels = torch.tensor([s[1] for s in samples])
    attribute_labels = torch.tensor([s[2] for s in samples])
    dummy_embeddings = torch.nn.functional.normalize(torch.randn(32, 16), dim=-1)

    attribute_aware = mine_attribute_aware_triplets(category_labels, attribute_labels)
    category_only_miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
    category_only = category_only_miner(dummy_embeddings, category_labels)

    assert attribute_aware[0].numel() <= category_only[0].numel()
