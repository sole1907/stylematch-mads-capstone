"""Retrieval-accuracy evaluation for the similarity engine: for each
image, checks whether its nearest *other* image by embedding similarity
shares its category label.
"""

import torch
from torch.utils.data import DataLoader

from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel


@torch.no_grad()
def compute_embeddings(
    model: SimilarityModel, dataset: DeepFashionCAPDataset, batch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs every image in `dataset` through `model` once. Returns
    (embeddings, category_labels), each stacked across the whole dataset,
    not just one batch.
    """
    model.eval()  # turns off patch_dropout; we want deterministic embeddings here
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []
    for images, category_labels, _attribute_labels in loader:
        all_embeddings.append(model(images))
        all_labels.append(category_labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


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
) -> dict[str, float]:
    """Loads a stratified subset of `split` (same approach as training,
    see dataset.py), embeds it, and returns both the model's retrieval
    accuracy and the random baseline for the same data, so the number
    means something on its own.
    """
    dataset = DeepFashionCAPDataset(
        split=split,
        transform=model.preprocess_val,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
    )
    embeddings, labels = compute_embeddings(model, dataset, batch_size=batch_size)
    return {
        "accuracy": top1_retrieval_accuracy(embeddings, labels),
        "random_baseline": random_baseline_accuracy(labels),
        "num_images": len(dataset),
        "num_categories": num_categories,
    }
