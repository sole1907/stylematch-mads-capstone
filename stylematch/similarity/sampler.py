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


class CategoryBalancedBatchSampler(Sampler[list[int]]):
    """Builds each batch from `categories_per_batch` categories,
    `images_per_category` images each (effective batch_size =
    categories_per_batch * images_per_category), except possibly the
    last batch of an epoch, see __iter__.

    Pair with a dataset loaded via
    DeepFashionCAPDataset(min_images_per_category=images_per_category,
    num_categories=..., ...), so every category present is guaranteed
    to have enough images to draw from.
    """

    def __init__(
        self,
        dataset: DeepFashionCAPDataset,
        categories_per_batch: int = 4,
        images_per_category: int = 4,
    ) -> None:
        self.categories_per_batch = categories_per_batch
        self.images_per_category = images_per_category

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

    def __iter__(self):
        """Every category appears in exactly one batch each epoch, none
        silently dropped. When len(self.categories) isn't evenly
        divisible by categories_per_batch, the leftover categories are
        merged into the *last* batch instead of forming their own
        undersized trailing batch: a batch built from too few
        categories (as few as 1) has no cross-category negatives to
        mine triplets from at all, exactly the failure mode this
        sampler exists to prevent elsewhere, so the last batch grows
        instead of shrinking.
        """
        categories = self.categories.copy()
        random.shuffle(categories)
        step = self.categories_per_batch
        num_batches = len(categories) // step
        for batch_num in range(num_batches):
            start = batch_num * step
            is_last_batch = batch_num == num_batches - 1
            end = len(categories) if is_last_batch else start + step
            batch: list[int] = []
            for category in categories[start:end]:
                batch.extend(random.sample(self.by_category[category], self.images_per_category))
            yield batch

    def __len__(self) -> int:
        return len(self.categories) // self.categories_per_batch
