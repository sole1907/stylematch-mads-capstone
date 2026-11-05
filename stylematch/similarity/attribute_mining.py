"""Attribute-aware triplet mining: a positive pair requires BOTH
category match and attribute overlap, not category alone. See
dev-notes/architecture-decision.md's pair/triplet construction note
(committed to mentor feedback, Nov 2026): CAP has no same-item identity
pairing, so category + attribute overlap is the best available proxy
for "these two items are genuinely similar."

Why this can't be done with pytorch-metric-learning's built-in
label-based miner (TripletMarginMiner): that miner treats "same label"
as an equivalence relation (same category <=> positive, transitively).
"Same category AND shares >= N attributes" is NOT transitive, item A
and item B can share one attribute, item B and item C can share a
*different* attribute, without A and C sharing anything, so it can't be
collapsed into a single scalar label the way category alone can. This
module builds the (anchor, positive, negative) index triples directly
instead, in the exact format pytorch-metric-learning's loss functions
accept via their own `indices_tuple` argument (confirmed: TripletMarginLoss
works correctly with labels=None when indices_tuple is supplied directly,
including the empty-triplets case, which returns a loss of 0.0, matching
the built-in miner's existing behavior for batches with no valid triplets).

Two miners are provided: the original binary version
(mine_attribute_aware_triplets, same category + any overlap vs.
different category), and a graded, 4-tier version
(mine_graded_attribute_triplets, see dev-notes/architecture-decision.md's
"Graded similarity tiers" note) that's now the one wired into train.py.
Both are kept, not just the newer one, useful to compare directly.
"""

import torch

# DeepFashion CAP's attribute vectors are extremely sparse: ~3.5 active
# attributes per item on average, out of 1000 (confirmed empirically).
# Requiring more than a small handful of shared attributes would almost
# never be satisfied; 1 is the realistic starting threshold for the
# binary miner, and 2 is the "high overlap" cutoff for the graded one
# (same-category pairs: 45.4% share 0, 34.1% share 1, 16.2% share 2,
# only ~4% share 3+, confirmed empirically, so "2+" is as fine-grained
# as the data meaningfully supports).
DEFAULT_MIN_SHARED_ATTRIBUTES = 1
DEFAULT_HIGH_OVERLAP_THRESHOLD = 2


def _category_and_overlap_masks(
    category_labels: torch.Tensor, attribute_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared setup for both miners below.

    Returns:
        (same_category_other, different_category, shared_attribute_counts,
        anchor_has_attributes): same_category_other and
        different_category are (batch, batch) boolean matrices
        (same_category_other excludes the diagonal, an item is never its
        own candidate); shared_attribute_counts is (batch, batch) float,
        the count of co-active attributes per pair; anchor_has_attributes
        is (batch,) boolean.
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
    matching TripletMarginMiner(type_of_triplets="all")'s semantics (all
    valid combinations, not just one per anchor).

    Positive: same category AND (shares >= min_shared_attributes active
    attributes with the anchor, OR the anchor itself has zero active
    attributes). The fallback exists because confirmed in real data, a
    non-trivial fraction of items have no active attributes at all,
    against which any overlap threshold is meaningless, not a signal
    that the item is dissimilar to everything.
    Negative: different category (unchanged from category-only mining;
    only the positive definition is being tightened here).

    Args:
        category_labels: (batch,) int tensor.
        attribute_labels: (batch, num_attributes) tensor, values in
            {-1, 1} (DeepFashion CAP's own encoding: 1 = has the
            attribute, -1 = does not).
        min_shared_attributes: minimum co-active attribute count
            required for a same-category pair to count as a positive.

    Returns:
        (anchor_idx, positive_idx, negative_idx): parallel 1D long
        tensors, one entry per mined triplet. Directly usable as a
        pytorch-metric-learning loss function's `indices_tuple`
        argument (with labels=None). All three are empty tensors if no
        valid triplet exists in the batch.
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
    """Mines triplets encoding a 4-tier ordering of closeness, not a flat
    positive/negative split (see dev-notes/architecture-decision.md's
    "Graded similarity tiers" note). Per anchor, ordered closest to
    farthest:

        A: same category, >= high_overlap_threshold shared attributes
        B: same category, exactly 1 shared attribute
        C: same category, 0 shared attributes
        D: different category

    If the anchor itself has zero active attributes, A/B/C can't be
    meaningfully distinguished for it (every same-category candidate
    would trivially compute as 0 overlap, not because they're genuinely
    tier C, but because the anchor has nothing to compare against). For
    such anchors, A/B/C collapse into one "same category" group compared
    directly against D, the same fallback principle as
    mine_attribute_aware_triplets, just extended to the tiered case.

    Triplets are mined between EVERY pair of tiers (A-B, A-C, A-D, B-C,
    B-D, C-D), not just adjacent ones. Adjacent-only was the original
    plan, but a hand-verified case showed it silently loses the
    ordering between two tiers whenever the tier between them is empty
    for a given anchor, e.g. tier A and tier C both populated but tier
    B empty, which is common given how sparse real attribute overlap is
    (confirmed: ~45% of same-category pairs share 0 attributes, ~34%
    share exactly 1, so tier B is often thin or empty at typical batch
    sizes). Full pairwise comparison is still cheap at this batch scale
    (candidate pools per anchor are small) and avoids that gap entirely.

    Args:
        category_labels: (batch,) int tensor.
        attribute_labels: (batch, num_attributes) tensor, values in
            {-1, 1} (DeepFashion CAP's own encoding).
        high_overlap_threshold: minimum shared-attribute count for tier A.

    Returns:
        (anchor_idx, positive_idx, negative_idx): parallel 1D long
        tensors, one entry per mined triplet, in the "higher tier is the
        positive, lower tier is the negative" sense for that pair.
        Directly usable as a pytorch-metric-learning loss function's
        `indices_tuple` argument (with labels=None).
    """
    device = category_labels.device
    same_category_other, different_category, shared_attribute_counts, anchor_has_attributes = (
        _category_and_overlap_masks(category_labels, attribute_labels)
    )

    anchors, positives, negatives = [], [], []
    for anchor_idx in range(category_labels.size(0)):
        diff_cat_candidates = different_category[anchor_idx].nonzero(as_tuple=True)[0].tolist()

        if not anchor_has_attributes[anchor_idx]:
            # Can't distinguish A/B/C for this anchor; collapse to a
            # plain same-vs-different split, matching the binary
            # miner's existing fallback behavior.
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
