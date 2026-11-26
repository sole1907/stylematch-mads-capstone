"""Attribute-aware triplet mining: a positive pair requires BOTH
category match and attribute overlap, not category alone. Builds
(anchor, positive, negative) index triples directly, in the format
pytorch-metric-learning losses accept via `indices_tuple`.

Two miners: the binary version (mine_attribute_aware_triplets) and the
graded, 4-tier version (mine_graded_attribute_triplets) wired into
train.py. Both kept, for direct comparison."""

import torch

# DeepFashion CAP attribute vectors are sparse (~3.5 active of 1000 per
# item). 1 is a realistic binary threshold; 2 is the graded "high
# overlap" cutoff (same-category pairs: 45.4% share 0, 34.1% share 1,
# 16.2% share 2, ~4% share 3+ -- as fine-grained as the data supports).
DEFAULT_MIN_SHARED_ATTRIBUTES = 1
DEFAULT_HIGH_OVERLAP_THRESHOLD = 2


def _category_and_overlap_masks(
    category_labels: torch.Tensor, attribute_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared setup for both miners below.

    Returns:
        (same_category_other, different_category, shared_attribute_counts,
        anchor_has_attributes): first two are (batch, batch) boolean
        matrices (diagonal excluded); shared_attribute_counts is
        (batch, batch) float; anchor_has_attributes is (batch,) boolean.
    """
    batch_size = category_labels.size(0)
    device = category_labels.device

    active = attribute_labels == 1  # (batch, num_attributes)
    # Dot product of two binary vectors = count of positions where both
    # are active, i.e. the number of shared active attributes.
    shared_attribute_counts = active.float() @ active.float().T  # (batch, batch)
    anchor_has_attributes = active.sum(dim=1) > 0  # (batch,)

    same_category = category_labels.unsqueeze(1) == category_labels.unsqueeze(0)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)

    return same_category & not_self, ~same_category, shared_attribute_counts, anchor_has_attributes


def mine_attribute_aware_triplets(
    category_labels: torch.Tensor,
    attribute_labels: torch.Tensor,
    min_shared_attributes: int = DEFAULT_MIN_SHARED_ATTRIBUTES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines every valid (anchor, positive, negative) triplet in a batch,
    matching TripletMarginMiner(type_of_triplets="all") semantics.

    Positive: same category AND (overlap >= min_shared_attributes, OR
    the anchor has zero active attributes -- a real fallback, since a
    non-trivial fraction of items have none). Negative: different
    category.

    Args:
        category_labels: (batch,) int tensor.
        attribute_labels: (batch, num_attributes) tensor, {-1, 1}
            (DeepFashion CAP's encoding).
        min_shared_attributes: overlap required to count as a positive.

    Returns:
        (anchor_idx, positive_idx, negative_idx): parallel 1D long
        tensors, usable directly as a pytorch-metric-learning loss's
        `indices_tuple` (labels=None). Empty if no valid triplet exists.
    """
    device = category_labels.device
    same_category_other, different_category, shared_attribute_counts, anchor_has_attributes = (
        _category_and_overlap_masks(category_labels, attribute_labels)
    )
    enough_overlap = shared_attribute_counts >= min_shared_attributes

    # anchor_has_attributes applied per-row: the fallback depends on the
    # ANCHOR's own attribute richness, not the candidate's.
    positive_mask = same_category_other & (enough_overlap | ~anchor_has_attributes.unsqueeze(1))

    anchors, positives, negatives = [], [], []
    for anchor_idx in range(category_labels.size(0)):
        pos_candidates = positive_mask[anchor_idx].nonzero(as_tuple=True)[0].tolist()
        neg_candidates = different_category[anchor_idx].nonzero(as_tuple=True)[0].tolist()
        for pos in pos_candidates:
            for neg in neg_candidates:
                anchors.append(anchor_idx)
                positives.append(pos)
                negatives.append(neg)

    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


def mine_graded_attribute_triplets(
    category_labels: torch.Tensor,
    attribute_labels: torch.Tensor,
    high_overlap_threshold: int = DEFAULT_HIGH_OVERLAP_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines triplets encoding a 4-tier closeness ordering, not a flat
    positive/negative split. Per anchor, closest to farthest: A (same
    category, >= high_overlap_threshold shared attrs), B (same
    category, 1 shared), C (same category, 0 shared), D (different
    category). If the anchor has zero attributes, A/B/C collapse into
    one "same category" group (same fallback as the binary miner).
    Triplets are mined between every pair of tiers, not just adjacent.

    Args:
        category_labels: (batch,) int tensor.
        attribute_labels: (batch, num_attributes) tensor, {-1, 1}.
        high_overlap_threshold: minimum shared-attribute count for tier A.

    Returns:
        (anchor_idx, positive_idx, negative_idx): parallel 1D long
        tensors, higher tier = positive, lower tier = negative. Usable
        directly as a pytorch-metric-learning loss's `indices_tuple`.
    """
    device = category_labels.device
    same_category_other, different_category, shared_attribute_counts, anchor_has_attributes = (
        _category_and_overlap_masks(category_labels, attribute_labels)
    )

    anchors, positives, negatives = [], [], []
    for anchor_idx in range(category_labels.size(0)):
        diff_cat_candidates = different_category[anchor_idx].nonzero(as_tuple=True)[0].tolist()

        if not anchor_has_attributes[anchor_idx]:
            # Can't distinguish A/B/C here; collapse to same-vs-different,
            # matching the binary miner's fallback.
            same_cat_candidates = same_category_other[anchor_idx].nonzero(as_tuple=True)[0].tolist()
            for pos in same_cat_candidates:
                for neg in diff_cat_candidates:
                    anchors.append(anchor_idx)
                    positives.append(pos)
                    negatives.append(neg)
            continue

        same_cat_idx = same_category_other[anchor_idx].nonzero(as_tuple=True)[0]
        overlaps = shared_attribute_counts[anchor_idx, same_cat_idx]
        tier_a = same_cat_idx[overlaps >= high_overlap_threshold].tolist()
        tier_b = same_cat_idx[overlaps == 1].tolist()
        tier_c = same_cat_idx[overlaps == 0].tolist()
        tiers_closest_to_farthest = [tier_a, tier_b, tier_c, diff_cat_candidates]

        for higher in range(len(tiers_closest_to_farthest)):
            for lower in range(higher + 1, len(tiers_closest_to_farthest)):
                for pos in tiers_closest_to_farthest[higher]:
                    for neg in tiers_closest_to_farthest[lower]:
                        anchors.append(anchor_idx)
                        positives.append(pos)
                        negatives.append(neg)

    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )
