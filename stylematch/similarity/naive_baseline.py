"""Naive color-histogram retrieval baseline: no learning at all, just a
hand-crafted visual feature (per-channel RGB color distribution), to
show the trained SimilarityModel's embeddings add value beyond this
trivial heuristic (per the Evaluation Plan's naive-baseline commitment).

Reuses the same evaluation machinery as the trained model
(top1_retrieval_accuracy, random_baseline_accuracy from evaluate.py),
just feeding it a different "embedding" source, a plain color-histogram
feature vector instead of the model's learned representation. Those
functions don't care where the vectors came from, only that they're a
per-image feature tensor usable for nearest-neighbor comparison.
"""

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
        L2-normalized feature vector, shape (3 * bins,): each channel's
        histogram is normalized to sum to 1 first (so comparisons aren't
        skewed by image size), then all 3 channels are concatenated and
        L2-normalized as a whole, so the result is directly usable with
        cosine-similarity-based retrieval, the same convention the
        trained model's embeddings already use.
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
    Mirrors evaluate.py's compute_embeddings, but needs no model, and no
    batching/DataLoader, each image's histogram is computed independently
    regardless, there's no parallelism to gain from grouping them.

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
    """Loads a stratified subset of `split` (same approach as the real
    model's evaluation, see evaluate.py's evaluate_retrieval_accuracy),
    computes plain color-histogram features (no learning, no model), and
    returns retrieval accuracy under that heuristic, alongside the
    random-chance baseline for the same data.
    """
    # transform=None deliberately: dataset.py's own placeholder transform
    # gives real [0,1]-scaled RGB values; a color histogram needs that,
    # not Marqo-FashionCLIP's own normalization.
    dataset = DeepFashionCAPDataset(
        split=split,
        transform=None,
        min_images_per_category=min_images_per_category,
        num_categories=num_categories,
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
    baseline on the *same* validation subset, so the two accuracy
    numbers are directly comparable, not just two separate runs that
    happen to use similar-sounding settings.

    Same-subset guarantee: DeepFashionCAPDataset's stratified loading
    (split/min_images_per_category/num_categories) scans the annotation
    file in a fixed order and takes the first qualifying categories, no
    shuffling involved (see dataset.py's _load_stratified_samples), so
    calling it twice with identical args always selects the identical
    set of images, regardless of which transform each call uses.

    Returns:
        A dict with both models' accuracy, the shared random-chance
        baseline for this subset, and the subset size, e.g.:
        {"similarity_model_accuracy": ..., "naive_baseline_accuracy":
        ..., "random_baseline": ..., "num_images": ..., "num_categories": ...}
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
