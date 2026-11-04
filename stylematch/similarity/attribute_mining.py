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
"""

import torch

# DeepFashion CAP's attribute vectors are extremely sparse: ~3.5 active
# attributes per item on average, out of 1000 (confirmed empirically).
# Requiring more than a small handful of shared attributes would almost
# never be satisfied; 1 is the realistic starting threshold.
DEFAULT_MIN_SHARED_ATTRIBUTES = 1


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
    batch_size = category_labels.size(0)
    device = category_labels.device

    active = attribute_labels == 1  # (batch, num_attributes)
    # Dot product of two binary vectors = count of positions where both
    # are active, i.e. the number of shared active attributes.
    shared_attribute_counts = active.float() @ active.float().T  # (batch, batch)
    anchor_has_no_attributes = active.sum(dim=1) == 0  # (batch,)

    same_category = category_labels.unsqueeze(1) == category_labels.unsqueeze(0)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    enough_overlap = shared_attribute_counts >= min_shared_attributes

    # anchor_has_no_attributes applied per-row: the fallback depends on
    # the ANCHOR's own attribute richness, not the candidate's.
    positive_mask = same_category & not_self & (enough_overlap | anchor_has_no_attributes.unsqueeze(1))
    negative_mask = ~same_category

    anchors, positives, negatives = [], [], []
    for anchor_idx in range(batch_size):
        pos_candidates = positive_mask[anchor_idx].nonzero(as_tuple=True)[0].tolist()
        neg_candidates = negative_mask[anchor_idx].nonzero(as_tuple=True)[0].tolist()
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
