"""Training loop for the similarity engine.

Uses the real SimilarityModel (model.py), real preprocessing, and the
real pytorch-metric-learning triplet loss, with graded attribute-aware
mining (attribute_mining.py). Batch construction uses
CategoryBalancedBatchSampler (sampler.py), needed so every batch
actually has same-category positives and different-category negatives
for triplet mining to find.
"""

from pathlib import Path

from loguru import logger
from pytorch_metric_learning import losses
import torch
from torch.utils.data import DataLoader

from stylematch.config import MODELS_DIR
from stylematch.similarity.attribute_mining import DEFAULT_HIGH_OVERLAP_THRESHOLD, mine_graded_attribute_triplets
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
    high_overlap_threshold: int = DEFAULT_HIGH_OVERLAP_THRESHOLD,
    checkpoint_path: Path | None = None,
    num_workers: int = 4,
    max_images_per_category: int | None = None,
) -> tuple[SimilarityModel, list[float]]:
    """Runs a handful of steps end-to-end against a real, category-
    balanced DeepFashion subset, using the real SimilarityModel and the
    real triplet loss.

    Args:
        checkpoint_path: if given, saves the trained model's state_dict
            here after training, so it can be reloaded and evaluated in
            a later, separate session. If None (default), nothing is
            saved to disk, only returned in memory.
        num_workers: DataLoader subprocesses for image loading/decoding.
            0 (the DataLoader default) loads every image synchronously
            on the main process, invisible at small local-subset scale
            but a real bottleneck at full-dataset scale, where the GPU
            would otherwise sit idle waiting on file I/O. 4 is a safe
            default for a typical multi-core machine; override higher
            to match whatever CPU allocation a given run (e.g. a Great
            Lakes SLURM job's --cpus-per-task) actually has available.
        max_images_per_category: forwarded to DeepFashionCAPDataset, see
            its docstring. Defaults to None (uncapped, every real image
            each category has) deliberately: CategoryBalancedBatchSampler
            already draws a random images_per_category-sized subset from
            whatever pool exists each batch, so a bigger pool only adds
            real epoch-to-epoch variety, it doesn't increase how many
            images actually get loaded per step. Safe to leave uncapped
            here even for small local runs.

    Returns:
        (model, losses): the trained model (already in memory, usable
        immediately without reloading from disk) and the loss recorded
        at each step.
    """
    model = SimilarityModel(embedding_dim=embedding_dim, freeze_until_block=freeze_until_block)

    dataset = DeepFashionCAPDataset(
        split="train",
        transform=model.preprocess_train,
        min_images_per_category=images_per_category,
        num_categories=num_categories,
        max_images_per_category=max_images_per_category,
    )
    logger.info(
        f"Loaded {len(dataset)} images across {num_categories} categories "
        f"(split='train', >= {images_per_category} images each, "
        f"max_images_per_category={max_images_per_category})"
    )

    batch_sampler = CategoryBalancedBatchSampler(
        dataset, categories_per_batch=categories_per_batch, images_per_category=images_per_category
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
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

    # Real triplet loss. Mining uses graded, 4-tier attribute-aware
    # mining now, not a flat positive/negative split: same-category
    # pairs with more shared attributes are pushed closer than
    # same-category pairs with fewer, which are in turn pushed closer
    # than different-category pairs. A custom miner is used instead of
    # pytorch-metric-learning's built-in label-based TripletMarginMiner,
    # since this ordering isn't expressible as a single scalar label.
    loss_fn = losses.TripletMarginLoss(margin=margin)

    losses_recorded = []
    data_iter = iter(loader)
    epoch = 1
    for step in range(1, num_steps + 1):
        try:
            images, category_labels, attribute_labels = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"-- epoch {epoch} starts (batches reshuffled) --")
            data_iter = iter(loader)
            images, category_labels, attribute_labels = next(data_iter)

        embeddings = model(images)

        indices_tuple = mine_graded_attribute_triplets(
            category_labels, attribute_labels, high_overlap_threshold=high_overlap_threshold
        )
        num_triplets = indices_tuple[0].numel()
        loss = loss_fn(embeddings, labels=None, indices_tuple=indices_tuple)

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

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    return model, losses_recorded


if __name__ == "__main__":
    trained_model, recorded_losses = run_training_loop(
        checkpoint_path=MODELS_DIR / "similarity_first_run.pt"
    )
    print(f"Training loop losses: {recorded_losses}")
