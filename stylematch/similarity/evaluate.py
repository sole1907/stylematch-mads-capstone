"""Retrieval-accuracy evaluation: for each image, checks whether its
nearest other image by embedding similarity is actually a match, under
two definitions of "correct" -- category alone, or category + attribute
overlap (see attribute_mining.py)."""

import torch
from torch.utils.data import DataLoader

from stylematch.similarity.attribute_mining import DEFAULT_MIN_SHARED_ATTRIBUTES
from stylematch.similarity.dataset import DeepFashionCAPDataset, get_qualifying_categories
from stylematch.similarity.model import SimilarityModel

_UNSET = object()


@torch.no_grad()
def compute_embeddings(
    model: SimilarityModel, dataset: DeepFashionCAPDataset, batch_size: int = 32, num_workers: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs every image in `dataset` through `model` once. Returns
    (embeddings, category_labels, attribute_labels), stacked across
    the whole dataset."""
    model.eval()  # turns off patch_dropout; we want deterministic embeddings here
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_category_labels = []
    all_attribute_labels = []
    for images, category_labels, attribute_labels in loader:
        all_embeddings.append(model(images).cpu())
        all_category_labels.append(category_labels)
        all_attribute_labels.append(attribute_labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_category_labels),
        torch.cat(all_attribute_labels),
    )


def top1_retrieval_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Finds each embedding's nearest other embedding (dot product, since
    already L2-normalized) and checks category match. Returns the
    fraction of queries where that holds."""
    similarity = embeddings @ embeddings.T  # (N, N)
    similarity.fill_diagonal_(-float("inf"))  # never match an embedding to itself
    nearest_neighbor = similarity.argmax(dim=1)
    correct = labels[nearest_neighbor] == labels
    return correct.float().mean().item()


def attribute_aware_top1_retrieval_accuracy(
    embeddings: torch.Tensor,
    category_labels: torch.Tensor,
    attribute_labels: torch.Tensor,
    min_shared_attributes: int = DEFAULT_MIN_SHARED_ATTRIBUTES,
) -> float:
    """Same as top1_retrieval_accuracy, but "correct" also requires
    attribute overlap (or the query having zero active attributes,
    same fallback rule as training, see attribute_mining.py)."""
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-float("inf"))
    nearest_neighbor = similarity.argmax(dim=1)

    active = attribute_labels == 1
    shared_attribute_counts = active.float() @ active.float().T  # (N, N)
    query_has_no_attributes = active.sum(dim=1) == 0  # (N,)

    matched_category = category_labels[nearest_neighbor] == category_labels
    n = embeddings.size(0)
    overlap_with_match = shared_attribute_counts[torch.arange(n), nearest_neighbor]
    enough_overlap = overlap_with_match >= min_shared_attributes

    correct = matched_category & (enough_overlap | query_has_no_attributes)
    return correct.float().mean().item()


def random_baseline_accuracy(labels: torch.Tensor) -> float:
    """What top1_retrieval_accuracy would be from guessing a random
    other image instead of using the embeddings."""
    counts = torch.bincount(labels)
    # For each category, chance of randomly picking another member of
    # the same category out of all the *other* images available.
    n = labels.numel()
    expected_correct = sum((c * (c - 1)) / (n - 1) for c in counts.tolist() if c > 0)
    return expected_correct / n


def evaluate_retrieval_accuracy(
    model: SimilarityModel,
    split: str = "val",
    min_images_per_category: int = 4,
    num_categories: int | None = 20,
    batch_size: int = 32,
    min_shared_attributes: int = DEFAULT_MIN_SHARED_ATTRIBUTES,
    num_workers: int = 4,
    category_inclusion_threshold: int | None = None,
    allowed_categories: set[int] | None = None,
    max_images_per_category=_UNSET,
) -> dict[str, float]:
    """Loads a stratified subset of `split`, embeds it, and returns
    retrieval accuracy under both correctness definitions plus the
    random baseline.

    Args:
        category_inclusion_threshold: qualification minimum, decided
            from train even when split is val, so categories match
            training. Ignored if allowed_categories is given.
        allowed_categories: use exactly these labels, decided externally,
            instead of deriving qualification here.
        max_images_per_category: caps images embedded per category.
            Defaults to min_images_per_category (sweep/sanity-check
            behavior); pass None explicitly for the real, uncapped,
            official evaluation.
    """
    if allowed_categories is None:
        inclusion_threshold = (
            min_images_per_category if category_inclusion_threshold is None else category_inclusion_threshold
        )
        allowed_categories = get_qualifying_categories(inclusion_threshold, split="train")
        if num_categories is not None:
            allowed_categories = set(sorted(allowed_categories)[:num_categories])

    if max_images_per_category is _UNSET:
        max_images_per_category = min_images_per_category

    dataset = DeepFashionCAPDataset(
        split=split,
        transform=model.preprocess_val,
        allowed_categories=allowed_categories,
        # Pinned deliberately: this embeds the whole dataset directly,
        # with no batch sampler bounding per-step cost.
        max_images_per_category=max_images_per_category,
    )
    embeddings, category_labels, attribute_labels = compute_embeddings(
        model, dataset, batch_size=batch_size, num_workers=num_workers
    )
    return {
        "accuracy": top1_retrieval_accuracy(embeddings, category_labels),
        "attribute_aware_accuracy": attribute_aware_top1_retrieval_accuracy(
            embeddings, category_labels, attribute_labels, min_shared_attributes=min_shared_attributes
        ),
        "random_baseline": random_baseline_accuracy(category_labels),
        "num_images": len(dataset),
        "num_categories": len(allowed_categories),
    }
