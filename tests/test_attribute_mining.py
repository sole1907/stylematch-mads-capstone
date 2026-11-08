"""Tests for attribute-aware triplet mining. The synthetic cases are
hand-verified exact expectations, not just sanity checks, see each
test's docstring for the hand-worked-out reasoning. Real-data tests
require the DeepFashion CAP subset to already be downloaded; skipped
otherwise.
"""

import pytest
import torch

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.attribute_mining import mine_attribute_aware_triplets, mine_graded_attribute_triplets
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


def test_graded_four_tiers_hand_verified():
    """5 items, categories [0, 0, 0, 0, 1]:
    item0 (anchor of interest) active={0,1,2}, item1 active={0,1}
    (overlap 2 -> tier A), item2 active={0} (overlap 1 -> tier B),
    item3 active={9} (overlap 0, disjoint attribute -> tier C), item4
    different category (tier D).

    Every tier pair is compared (not just adjacent), so anchor0 alone
    produces 6 triplets: A-B, A-C, A-D, B-C, B-D, C-D. Hand-verified
    totals across all 5 anchors (including anchor2's empty tier A,
    anchor3's empty tiers A and B, and anchor4's zero-attribute fallback
    with no same-category partner at all) sum to 20 triplets.
    """
    category_labels = torch.tensor([0, 0, 0, 0, 1])
    attribute_labels = torch.tensor(
        [
            [1, 1, 1, -1],  # item0: active {0,1,2}
            [1, 1, -1, -1],  # item1: active {0,1} -> overlap w/ item0 = 2
            [1, -1, -1, -1],  # item2: active {0} -> overlap w/ item0 = 1
            [-1, -1, -1, 1],  # item3: active {3} -> overlap w/ item0 = 0
            [-1, -1, -1, -1],  # item4: different category, no attributes
        ]
    )

    result = mine_graded_attribute_triplets(category_labels, attribute_labels)

    expected = {
        # anchor0: tier_a={1}, tier_b={2}, tier_c={3}, tier_d={4}
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4),
        # anchor1: tier_a={0}, tier_b={2}, tier_c={3}, tier_d={4}
        (1, 0, 2), (1, 0, 3), (1, 0, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4),
        # anchor2: tier_a={} (empty), tier_b={0,1}, tier_c={3}, tier_d={4}
        (2, 0, 3), (2, 1, 3), (2, 0, 4), (2, 1, 4), (2, 3, 4),
        # anchor3: tier_a={}, tier_b={} (empty, nothing shares attr 3), tier_c={0,1,2}, tier_d={4}
        (3, 0, 4), (3, 1, 4), (3, 2, 4),
        # anchor4: zero attributes, no other same-category item exists -> 0 triplets
    }
    assert _as_triplet_set(result) == expected


def test_graded_full_pairwise_beats_adjacent_only_when_middle_tier_empty():
    """3 items, ALL the same category, no different-category item exists
    at all: item0 (anchor) active={0,1,2}, item1 active={0,1} (tier A,
    overlap 2), item2 active={} (tier C, overlap 0). Tier B is empty for
    every anchor here, and there's no tier D (no other category), so
    adjacent-only comparisons (A-vs-B, B-vs-C) would find literally
    nothing. Full pairwise still finds the direct A-vs-C relationship,
    confirmed empirically to be a real gap before choosing this design,
    see attribute_mining.py's docstring.
    """
    category_labels = torch.tensor([0, 0, 0])
    attribute_labels = torch.tensor([[1, 1, 1], [1, 1, -1], [-1, -1, -1]])

    result = mine_graded_attribute_triplets(category_labels, attribute_labels)

    expected = {
        (0, 1, 2),  # anchor0: tier_a={1}, tier_c={2}, directly compared
        (1, 0, 2),  # anchor1: tier_a={0}, tier_c={2}
        # anchor2: zero attributes, fallback -> same-category candidates
        # {0,1}, but no different-category item exists, so 0 triplets.
    }
    assert _as_triplet_set(result) == expected


def test_graded_zero_attribute_anchor_falls_back_to_binary():
    """4 items, categories [0,0,0,1]: item0 (anchor) has zero active
    attributes, items1-2 same category (and DO have attributes
    themselves), item3 different category. item0, as anchor, can't
    distinguish tiers A/B/C for itself, so every same-category item
    counts as a valid positive against every different-category
    negative for IT, the same fallback as the binary miner.

    But items 1 and 2 have real attributes of their own, so they go
    through the normal tiered branch when THEY'RE the anchor, this
    test's first draft only hand-verified anchor0 and wrongly assumed
    anchors 1 and 2 would contribute nothing, they don't get a free
    pass just because one OTHER item in the batch has no attributes.
    Full hand-verification, all 4 anchors:

    - anchor0 (no attributes): fallback -> positives {1,2}, negative {3}
      -> (0,1,3), (0,2,3).
    - anchor1 (active {0,1,2}): same_cat_other={0,2}; item0 (active={})
      -> overlap 0 -> tier C; item2 (active={0}) -> overlap 1 -> tier B;
      tier A empty -> only B-C, B-D, C-D fire -> (1,2,0), (1,2,3), (1,0,3).
    - anchor2 (active {0}): same_cat_other={0,1}; item0 -> overlap 0 ->
      tier C; item1 -> overlap 1 -> tier B; tier A empty -> (2,1,0),
      (2,1,3), (2,0,3).
    - anchor3 (different category, no same-category partner exists at
      all): 0 triplets regardless of its own attributes.
    """
    category_labels = torch.tensor([0, 0, 0, 1])
    attribute_labels = torch.tensor(
        [
            [-1, -1, -1],  # item0 (anchor): no active attributes
            [1, 1, 1],  # item1: same category, has attributes
            [1, -1, -1],  # item2: same category, different attributes
            [-1, -1, 1],  # item3: different category
        ]
    )

    result = mine_graded_attribute_triplets(category_labels, attribute_labels)
    expected = {
        (0, 1, 3), (0, 2, 3),
        (1, 2, 0), (1, 2, 3), (1, 0, 3),
        (2, 1, 0), (2, 1, 3), (2, 0, 3),
    }
    assert _as_triplet_set(result) == expected


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
    exceed the full combinatorial count of category-only triplets
    (every same-category-other item paired with every different-
    category item, for every anchor).

    Deliberately NOT compared against pytorch-metric-learning's
    TripletMarginMiner(type_of_triplets="all"): that doesn't mean "every
    combinatorially possible triplet", it means "every triplet currently
    violating the margin", which depends on whatever embeddings are
    passed in. The original version of this test fed it unseeded random
    embeddings and compared the (embedding-dependent, run-to-run
    variable) result against this function's (deterministic, label-only)
    count, a real, pre-existing flakiness that surfaced as an actual
    failure rather than just a theoretical risk. The combinatorial count
    computed here depends only on the labels, never on any embeddings,
    so this comparison is deterministic.
    """
    dataset = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=10)
    samples = [dataset.samples[i] for i in range(32)]
    category_labels = torch.tensor([s[1] for s in samples])
    attribute_labels = torch.tensor([s[2] for s in samples])

    attribute_aware = mine_attribute_aware_triplets(category_labels, attribute_labels)

    max_possible_category_only_triplets = 0
    for anchor in range(len(samples)):
        same_cat_other = (
            (category_labels == category_labels[anchor]) & (torch.arange(len(samples)) != anchor)
        ).sum().item()
        diff_cat = (category_labels != category_labels[anchor]).sum().item()
        max_possible_category_only_triplets += same_cat_other * diff_cat

    assert attribute_aware[0].numel() <= max_possible_category_only_triplets


def _tier(anchor_idx, candidate_idx, category_labels, active, high_overlap_threshold=2):
    """Reimplements the tier definition independently (not by calling
    into attribute_mining.py) for the real-data invariant test below,
    so the test actually checks the intended semantics rather than just
    re-running the same code and trivially agreeing with itself."""
    if category_labels[anchor_idx] != category_labels[candidate_idx]:
        return "D"
    if active[anchor_idx].sum() == 0:
        return "same-category (anchor has no attributes)"
    overlap = (active[anchor_idx] & active[candidate_idx]).sum().item()
    if overlap >= high_overlap_threshold:
        return "A"
    if overlap == 1:
        return "B"
    return "C"


_TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


@requires_deepfashion
def test_graded_real_data_every_triplet_has_positive_outranking_negative():
    """Confirms the actual intent on real data: for every mined triplet,
    the positive's tier must be strictly closer than the negative's
    tier, independently recomputed, not just internally self-consistent.
    """
    dataset = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=10)
    samples = [dataset.samples[i] for i in range(32)]
    category_labels = torch.tensor([s[1] for s in samples])
    attribute_labels = torch.tensor([s[2] for s in samples])
    active = attribute_labels == 1

    anchors, positives, negatives = mine_graded_attribute_triplets(category_labels, attribute_labels)
    assert anchors.numel() > 0, "expected at least some triplets on a real 32-image batch"

    for a, p, n in zip(anchors.tolist(), positives.tolist(), negatives.tolist()):
        pos_tier = _tier(a, p, category_labels, active)
        neg_tier = _tier(a, n, category_labels, active)
        if pos_tier == "same-category (anchor has no attributes)":
            assert neg_tier == "D", f"anchor {a} has no attributes, fallback expects negative to be tier D"
        else:
            assert _TIER_RANK[pos_tier] < _TIER_RANK[neg_tier], (
                f"triplet ({a}, {p}, {n}): positive tier {pos_tier} should outrank negative tier {neg_tier}"
            )


@requires_deepfashion
def test_graded_real_data_produces_at_least_as_many_triplets_as_binary():
    """Comparing every tier pair (not just positive-vs-negative) should
    never produce FEWER triplets than the binary miner on the same
    batch, since every binary positive/negative pair is still a valid
    tier comparison (same-category-tier vs. tier D) under the graded
    scheme, plus the graded version adds same-category-vs-same-category
    comparisons (A-vs-B, A-vs-C, B-vs-C) the binary miner can't produce
    at all.
    """
    dataset = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=10)
    samples = [dataset.samples[i] for i in range(32)]
    category_labels = torch.tensor([s[1] for s in samples])
    attribute_labels = torch.tensor([s[2] for s in samples])

    binary = mine_attribute_aware_triplets(category_labels, attribute_labels)
    graded = mine_graded_attribute_triplets(category_labels, attribute_labels)

    assert graded[0].numel() >= binary[0].numel()
