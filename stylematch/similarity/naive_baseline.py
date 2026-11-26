"""Naive color-histogram retrieval baseline: no learning, just a
hand-crafted feature (per-channel RGB distribution), to show the
trained SimilarityModel adds value beyond this trivial heuristic
(per the Evaluation Plan's naive-baseline commitment). Reuses
evaluate.py's accuracy functions with this feature as the
"embedding" source instead of the model's."""

import torch

from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.evaluate import (
    evaluate_retrieval_accuracy,
    random_baseline_accuracy,
    top1_retrieval_accuracy,
)
from stylematch.similarity.model import SimilarityModel

DEFAULT_BINS_PER_CHANNEL = 32


def compute_color_histogram(image: torch.Tensor, bins: int = DEFAULT_BINS_PER_CHANNEL) -> torch.Tensor:
    """Computes a per-channel color histogram feature for one image.

    Args:
        image: float tensor, shape (3, H, W), values in [0, 1].
        bins: number of histogram bins per channel.

    Returns:
        L2-normalized feature vector, shape (3 * bins,) -- each channel
        normalized to sum to 1 first, then concatenated and
        L2-normalized as a whole for cosine-similarity retrieval.
    """
    channel_histograms = []
    for channel in range(3):
        hist = torch.histc(image[channel], bins=bins, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1e-8)  # normalize to a probability distribution
        channel_histograms.append(hist)
    feature = torch.cat(channel_histograms)
    return torch.nn.functional.normalize(feature, dim=0)


@torch.no_grad()
def compute_color_histogram_batch(
    dataset: DeepFashionCAPDataset, bins: int = DEFAULT_BINS_PER_CHANNEL
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes color-histogram features for every image in `dataset`.
    Mirrors evaluate.py's compute_embeddings, but needs no model or
    batching -- each histogram is independent, no parallelism to gain.

    Returns:
        (features, category_labels), stacked across the whole dataset.
    """
    features = []
    labels = []
    for idx in range(len(dataset)):
        image, category_label, _attribute_label = dataset[idx]
        features.append(compute_color_histogram(image, bins=bins))
        labels.append(category_label)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def evaluate_naive_color_baseline(
    split: str = "val",
    min_images_per_category: int = 4,
    num_categories: int = 20,
    bins: int = DEFAULT_BINS_PER_CHANNEL,
) -> dict[str, float]:
    """Loads a stratified subset of `split`, computes plain
    color-histogram features (no model), and returns retrieval
    accuracy under that heuristic plus the random-chance baseline.
    """
    # transform=None deliberately: dataset.py's placeholder transform
    # gives [0,1]-scaled RGB, which a histogram needs (not CLIP's own
    # normalization).
    dataset = DeepFashionCAPDataset(
        split=split,
        transform=None,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
        # Pinned to match min_images_per_category, same reasoning as
        # evaluate.py's evaluate_retrieval_accuracy (iterates the
        # whole dataset directly, so it needs a bounded subset).
        max_images_per_category=min_images_per_category,
    )
    features, category_labels = compute_color_histogram_batch(dataset, bins=bins)
    return {
        "accuracy": top1_retrieval_accuracy(features, category_labels),
        "random_baseline": random_baseline_accuracy(category_labels),
        "num_images": len(dataset),
        "num_categories": num_categories,
    }


def compare_similarity_vs_naive_baseline(
    model: SimilarityModel,
    split: str = "val",
    min_images_per_category: int = 4,
    num_categories: int = 20,
    batch_size: int = 32,
    bins: int = DEFAULT_BINS_PER_CHANNEL,
) -> dict[str, float]:
    """Runs the trained SimilarityModel and the naive color-histogram
    baseline on the same validation subset, so the two accuracy
    numbers are directly comparable. Same-subset guarantee: identical
    args always select the identical images (dataset.py's stratified
    loading is a fixed-order scan, no shuffling).

    Returns:
        A dict with both models' accuracy, the shared random-chance
        baseline, and the subset size.
    """
    similarity_results = evaluate_retrieval_accuracy(
        model,
        split=split,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
        batch_size=batch_size,
    )
    naive_results = evaluate_naive_color_baseline(
        split=split,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
        bins=bins,
    )
    return {
        "similarity_model_accuracy": similarity_results["accuracy"],
        "similarity_model_attribute_aware_accuracy": similarity_results["attribute_aware_accuracy"],
        "naive_baseline_accuracy": naive_results["accuracy"],
        "random_baseline": similarity_results["random_baseline"],
        "num_images": similarity_results["num_images"],
        "num_categories": num_categories,
    }
