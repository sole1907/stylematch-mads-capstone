"""Category-balanced batch sampler for the similarity engine's training
loop. Plain random batching doesn't guarantee a batch contains multiple
images from the same category, and the underlying annotation file is
clustered by category in contiguous blocks, so a naive sample (e.g.
DeepFashionCAPDataset's default first-N loading) can easily load only a
single category, guaranteeing triplet mining finds nothing (confirmed
empirically: a 16-image, single-category sample produced 0 mineable
triplets, every step, every time). This sampler guarantees every batch
has both same-category positives and different-category negatives to
mine from.
"""

import random

from torch.utils.data import Sampler

from stylematch.similarity.dataset import DeepFashionCAPDataset

# Category-size weighting bins, used only when weight_by_category_size
# is enabled: a category's weight is how many independent batch-slot
# opportunities it gets per epoch, instead of exactly one like every
# other category. Boundaries chosen from DeepFashion CAP's real
# training-split distribution (46 qualifying categories, 11-52,138
# images each, confirmed via a real count, not assumed) -- most
# categories are small, a handful are enormous (Dress alone is 52,138
# of 209,222 total images), so treating every category equally wastes
# most of the dataset's real volume on a handful of oversized
# categories that barely get sampled relative to their size.
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
    """Builds each batch from `categories_per_batch` categories,
    `images_per_category` images each (effective batch_size =
    categories_per_batch * images_per_category), except possibly the
    last batch of a round, see __iter__.

    Pair with a dataset loaded via
    DeepFashionCAPDataset(min_images_per_category=images_per_category,
    num_categories=..., ...), so every category present is guaranteed
    to have enough images to draw from.

    weight_by_category_size: if True, larger categories get multiple
    independent batch-slot opportunities per epoch instead of one,
    per CATEGORY_WEIGHT_BINS. Implemented as multiple *rounds*, not one
    shuffle with repeated entries: each round only includes categories
    whose weight is high enough to still be active that round (round 0
    = every category, round 1 = only weight >= 2, etc.), and every
    round independently shuffles and batches its own active categories
    with the same distinct-category-per-batch guarantee as the
    unweighted case. This is what prevents a high-weight category from
    landing twice in the same batch, which a single repeated-entries
    shuffle couldn't guarantee.
    """

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
        """Every category appears in exactly one batch each round, none
        silently dropped. When a round's category count isn't evenly
        divisible by categories_per_batch, the leftover categories are
        merged into the *last* batch of that round instead of forming
        their own undersized trailing batch: a batch built from too few
        categories (as few as 1) has no cross-category negatives to
        mine triplets from at all, exactly the failure mode this
        sampler exists to prevent elsewhere, so the last batch grows
        instead of shrinking. If a later round's active category count
        is itself below categories_per_batch (too few high-weight
        categories to form even one batch that round), that round is
        skipped entirely rather than erroring -- a soft loss of that
        round's extra exposure for just those categories, not a
        correctness problem.
        """
        for round_categories in self._rounds():
            categories = round_categories.copy()
            random.shuffle(categories)
            for category_group in self._batches_for_round(categories):
                batch: list[int] = []
                for category in category_group:
                    batch.extend(random.sample(self.by_category[category], self.images_per_category))
                yield batch

    def __len__(self) -> int:
        return sum(len(self._batches_for_round(round_categories)) for round_categories in self._rounds())
