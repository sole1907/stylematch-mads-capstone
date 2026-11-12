"""Retrieval-accuracy evaluation for the similarity engine: for each
image, checks whether its nearest *other* image by embedding similarity
is actually a match, under two different definitions of "correct":
category label alone, or category + attribute overlap together (see
attribute_mining.py and dev-notes/architecture-decision.md). Both are
kept, the category-only version remains a useful, simpler comparison
point, not dead code.
"""

import torch
from torch.utils.data import DataLoader

from stylematch.similarity.attribute_mining import DEFAULT_MIN_SHARED_ATTRIBUTES
from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel


@torch.no_grad()
def compute_embeddings(
    model: SimilarityModel, dataset: DeepFashionCAPDataset, batch_size: int = 32, num_workers: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs every image in `dataset` through `model` once. Returns
    (embeddings, category_labels, attribute_labels), each stacked
    across the whole dataset, not just one batch.

    Args:
        num_workers: DataLoader subprocesses for image loading/decoding.
    """
    model.eval()  # turns off patch_dropout; we want deterministic embeddings here
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_category_labels = []
    all_attribute_labels = []
    for images, category_labels, attribute_labels in loader:
        all_embeddings.append(model(images))
        all_category_labels.append(category_labels)
        all_attribute_labels.append(attribute_labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_category_labels),
        torch.cat(all_attribute_labels),
    )


def top1_retrieval_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """For each embedding, finds its nearest *other* embedding (cosine
    similarity, just a dot product since embeddings are already
    L2-normalized) and checks whether it shares the query's category
    label. Returns the fraction of queries where that holds.
    """
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
    """Same mechanism as top1_retrieval_accuracy, but "correct" now
    requires category match AND attribute overlap (or the query itself
    having zero active attributes, the same fallback rule used in
    training, see attribute_mining.py), not category alone.
    """
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
    *other* image rather than actually using the embeddings.
    """
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
    num_categories: int = 20,
    batch_size: int = 32,
    min_shared_attributes: int = DEFAULT_MIN_SHARED_ATTRIBUTES,
    num_workers: int = 4,
) -> dict[str, float]:
    """Loads a stratified subset of `split` (same approach as training,
    see dataset.py), embeds it, and returns retrieval accuracy under
    both correctness definitions (category-only and attribute-aware),
    plus the random baseline, so the numbers mean something on their
    own and can be compared to each other directly.
    """
    dataset = DeepFashionCAPDataset(
        split=split,
        transform=model.preprocess_val,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
        # Pinned to match min_images_per_category deliberately, not left
        # uncapped like train.py's default: this function iterates the
        # whole dataset directly (no batch sampler bounding per-step
        # cost), so an uncapped pool here would mean genuinely embedding
        # every real image in num_categories categories, not a bounded
        # evaluation subset. See dataset.py's max_images_per_category
        # docstring for the full reasoning.
        max_images_per_category=min_images_per_category,
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
        "num_categories": num_categories,
    }
