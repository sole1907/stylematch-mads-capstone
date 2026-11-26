"""Category-balanced batch sampler for the similarity engine. Plain
random batching could load a single category per batch (annotation
rows are clustered by category); this guarantees every batch has
same-category positives and different-category negatives to mine."""

import random

from torch.utils.data import Sampler

from stylematch.similarity.dataset import DeepFashionCAPDataset

# Used only when weight_by_category_size is enabled: a category's
# weight is how many batch-slots it gets per epoch instead of one.
CATEGORY_WEIGHT_BINS = [
    (500, 1),
    (2000, 2),
    (10000, 3),
    (float("inf"), 4),
]


def _category_weight(num_images: int) -> int:
    """How many independent batch-slot opportunities a category of this
    size gets per epoch, per CATEGORY_WEIGHT_BINS."""
    for threshold, weight in CATEGORY_WEIGHT_BINS:
        if num_images < threshold:
            return weight
    return CATEGORY_WEIGHT_BINS[-1][1]


class CategoryBalancedBatchSampler(Sampler[list[int]]):
    """Builds each batch from categories_per_batch categories,
    images_per_category images each. weight_by_category_size gives
    larger categories multiple batch-slots per epoch via extra
    rounds, instead of one repeated-entries shuffle (which couldn't
    guarantee a category never lands twice in the same batch)."""

    def __init__(
        self,
        dataset: DeepFashionCAPDataset,
        categories_per_batch: int = 4,
        images_per_category: int = 4,
        weight_by_category_size: bool = False,
    ) -> None:
        self.categories_per_batch = categories_per_batch
        self.images_per_category = images_per_category
        self.weight_by_category_size = weight_by_category_size

        by_category: dict[int, list[int]] = {}
        for idx, (_, category_label, _) in enumerate(dataset.samples):
            by_category.setdefault(category_label, []).append(idx)

        self.by_category = {
            cat: idxs for cat, idxs in by_category.items() if len(idxs) >= images_per_category
        }
        self.categories = list(self.by_category)
        if len(self.categories) < categories_per_batch:
            raise ValueError(
                f"only {len(self.categories)} categories have >= "
                f"{images_per_category} images each; need at least "
                f"{categories_per_batch} to build one batch"
            )
        # Per-category unused-image queue, persists across epochs: drawn
        # without replacement until exhausted, then reshuffled.
        self._remaining: dict[int, list[int]] = {}

    def _draw(self, category: int) -> list[int]:
        """images_per_category indices from this category's unused
        pool, reshuffling a fresh full cycle once exhausted."""
        remaining = self._remaining.get(category, [])
        if len(remaining) < self.images_per_category:
            remaining = self.by_category[category].copy()
            random.shuffle(remaining)
        drawn, remaining = remaining[: self.images_per_category], remaining[self.images_per_category :]
        self._remaining[category] = remaining
        return drawn

    def _rounds(self) -> list[list[int]]:
        """One category list per round: round r contains every category
        whose weight is > r, i.e. still has a batch-slot opportunity
        left to place this round. Unweighted case is always exactly one
        round (every category, weight 1 > 0)."""
        if not self.weight_by_category_size:
            return [self.categories]
        max_weight = max(_category_weight(len(self.by_category[cat])) for cat in self.categories)
        return [
            [cat for cat in self.categories if _category_weight(len(self.by_category[cat])) > r]
            for r in range(max_weight)
        ]

    def _batches_for_round(self, categories: list[int]) -> list[list[int]]:
        """Same merge-leftover-into-last-batch logic as the unweighted
        sampler always had, applied to whichever categories are active
        this round."""
        step = self.categories_per_batch
        num_batches = len(categories) // step
        batches = []
        for batch_num in range(num_batches):
            start = batch_num * step
            is_last_batch = batch_num == num_batches - 1
            end = len(categories) if is_last_batch else start + step
            batches.append(categories[start:end])
        return batches

    def __iter__(self):
        """Every category appears in exactly one batch per round. Leftover
        categories merge into the last batch rather than forming an
        undersized one with no cross-category negatives to mine."""
        for round_categories in self._rounds():
            categories = round_categories.copy()
            random.shuffle(categories)
            for category_group in self._batches_for_round(categories):
                batch: list[int] = []
                for category in category_group:
                    batch.extend(self._draw(category))
                yield batch

    def __len__(self) -> int:
        return sum(len(self._batches_for_round(round_categories)) for round_categories in self._rounds())
