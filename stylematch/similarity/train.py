"""Training loop for the similarity engine.

Uses the real SimilarityModel (model.py), real preprocessing, and the
real pytorch-metric-learning triplet loss now. Batch construction uses
CategoryBalancedBatchSampler (sampler.py), a mock stand-in for the real
DeepFashion pipeline, needed so every batch actually has same-category
positives and different-category negatives for triplet mining to find.
"""

from loguru import logger
from pytorch_metric_learning import losses, miners
import torch
from torch.utils.data import DataLoader

from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.sampler import CategoryBalancedBatchSampler


def run_training_loop(
    num_steps: int = 3,
    categories_per_batch: int = 4,
    images_per_category: int = 4,
    num_categories: int = 20,
    learning_rate: float = 1e-3,
    freeze_until_block: int = 8,
    embedding_dim: int = 256,
    margin: float = 0.2,
) -> list[float]:
    """Runs a handful of steps end-to-end against a real, category-
    balanced DeepFashion subset, using the real SimilarityModel and the
    real triplet loss.

    Returns the loss recorded at each step.
    """
    model = SimilarityModel(embedding_dim=embedding_dim, freeze_until_block=freeze_until_block)

    dataset = DeepFashionCAPDataset(
        split="train",
        transform=model.preprocess_train,
        min_images_per_category=images_per_category,
        num_categories=num_categories,
    )
    logger.info(
        f"Loaded {len(dataset)} images across {num_categories} categories "
        f"(split='train', >= {images_per_category} images each)"
    )

    batch_sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=categories_per_batch, images_per_category=images_per_category
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    steps_per_epoch = len(loader)
    effective_batch_size = categories_per_batch * images_per_category
    logger.info(
        f"effective batch_size={effective_batch_size} "
        f"({categories_per_batch} categories x {images_per_category} images) -> "
        f"{steps_per_epoch} steps per epoch (requested num_steps={num_steps}, "
        f"{num_steps / steps_per_epoch:.2f} epochs)"
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"SimilarityModel: freeze_until_block={freeze_until_block}, "
        f"{trainable:,} / {total:,} trainable parameters ({trainable / total:.1%})"
    )

    # Only trainable parameters, not all of model.parameters(): most of
    # the backbone is frozen now, and handing Adam frozen params would
    # waste memory on momentum buffers that never get used.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Real triplet loss now, not a placeholder. Mining on category_label
    # alone, not category + attributes together, as a deliberate known
    # simplification (see dev-notes/architecture-decision.md's
    # pair/triplet construction note): visually different items sharing
    # a broad category could get pulled together. Attribute-aware
    # pairing is a planned refinement, not blocking this first real pass.
    loss_fn = losses.TripletMarginLoss(margin=margin)
    miner_fn = miners.TripletMarginMiner(margin=margin, type_of_triplets="all")

    losses_recorded = []
    data_iter = iter(loader)
    epoch = 1
    for step in range(1, num_steps + 1):
        try:
            images, category_labels, _attribute_labels = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"-- epoch {epoch} starts (batches reshuffled) --")
            data_iter = iter(loader)
            images, category_labels, _attribute_labels = next(data_iter)

        embeddings = model(images)

        indices_tuple = miner_fn(embeddings, category_labels)
        num_triplets = indices_tuple[0].numel()
        loss = loss_fn(embeddings, category_labels, indices_tuple)

        optimizer.zero_grad()
        loss.backward()
        # Total gradient norm across every trainable parameter, not one
        # hardcoded layer, since there's no single ".weight" that stands
        # in for "the model."
        grad_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in trainable_params if p.grad is not None)
        ).item()
        optimizer.step()

        losses_recorded.append(loss.item())
        logger.info(
            f"step {step}/{num_steps}: images {tuple(images.shape)} -> "
            f"embeddings {tuple(embeddings.shape)} | triplets mined={num_triplets} "
            f"| loss={loss.item():.4f} | grad_norm={grad_norm:.4f} | weights updated"
        )

    logger.success(
        f"Done. Loss went {losses_recorded[0]:.4f} -> {losses_recorded[-1]:.4f} "
        f"over {num_steps} steps."
    )
    return losses_recorded


if __name__ == "__main__":
    recorded_losses = run_training_loop()
    print(f"Training loop losses: {recorded_losses}")
