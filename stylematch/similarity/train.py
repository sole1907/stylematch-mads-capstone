"""Training loop for the similarity engine: SimilarityModel (model.py)
with pytorch-metric-learning's triplet loss and graded attribute-aware
mining (attribute_mining.py), batched via CategoryBalancedBatchSampler
(sampler.py) so every batch has triplet-mineable positives/negatives."""

from pathlib import Path

from loguru import logger
from pytorch_metric_learning import losses
import torch
from torch.utils.data import DataLoader

from stylematch.config import MODELS_DIR
from stylematch.similarity.attribute_mining import DEFAULT_HIGH_OVERLAP_THRESHOLD, mine_graded_attribute_triplets
from stylematch.similarity.dataset import DeepFashionCAPDataset, get_qualifying_categories
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.sampler import CategoryBalancedBatchSampler


def run_training_loop(
    num_steps: int = 3,
    categories_per_batch: int = 4,
    images_per_category: int = 4,
    num_categories: int | None = 20,
    learning_rate: float = 1e-3,
    freeze_until_block: int = 8,
    embedding_dim: int = 256,
    margin: float = 0.2,
    high_overlap_threshold: int = DEFAULT_HIGH_OVERLAP_THRESHOLD,
    checkpoint_path: Path | None = None,
    num_workers: int = 4,
    max_images_per_category: int | None = None,
    weight_by_category_size: bool = False,
    category_inclusion_threshold: int | None = None,
    allowed_categories: set[int] | None = None,
    resume_path: Path | None = None,
    checkpoint_every: int | None = None,
) -> tuple[SimilarityModel, list[float]]:
    """Runs num_steps training steps against a category-balanced
    DeepFashion subset.

    Args:
        checkpoint_path: if given, saves a plain state_dict here after
            training (the format other code loads the model from).
        max_images_per_category: forwarded to DeepFashionCAPDataset.
        weight_by_category_size: forwarded to CategoryBalancedBatchSampler.
        category_inclusion_threshold: minimum images to qualify; None
            reuses images_per_category. Ignored if allowed_categories
            is given.
        allowed_categories: use exactly these labels, decided externally,
            instead of deriving qualification here.
        resume_path: if this file exists, resumes model/optimizer/step
            state from it instead of starting at step 0.
        checkpoint_every: if given (with resume_path), saves resume
            state to resume_path every N steps, not just at the end.

    Returns:
        (model, losses): the trained model and per-step loss.
    """
    model = SimilarityModel(embedding_dim=embedding_dim, freeze_until_block=freeze_until_block)

    if allowed_categories is None:
        inclusion_threshold = (
            images_per_category if category_inclusion_threshold is None else category_inclusion_threshold
        )
        allowed_categories = get_qualifying_categories(inclusion_threshold, split="train")
        if num_categories is not None:
            allowed_categories = set(sorted(allowed_categories)[:num_categories])

    dataset = DeepFashionCAPDataset(
        split="train",
        transform=model.preprocess_train,
        allowed_categories=allowed_categories,
        max_images_per_category=max_images_per_category,
    )
    logger.info(
        f"Loaded {len(dataset)} images across {len(allowed_categories)} categories "
        f"(split='train', {images_per_category} sampled per batch, "
        f"max_images_per_category={max_images_per_category})"
    )

    batch_sampler = CategoryBalancedBatchSampler(
        dataset,
        categories_per_batch=categories_per_batch,
        images_per_category=images_per_category,
        weight_by_category_size=weight_by_category_size,
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

    # Only trainable params, not all of model.parameters(): most of the
    # backbone is frozen, and Adam would waste memory on unused buffers.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Custom miner instead of the built-in label-based TripletMarginMiner:
    # graded 4-tier closeness isn't expressible as a single scalar label.
    loss_fn = losses.TripletMarginLoss(margin=margin)

    losses_recorded = []
    start_step = 1
    if resume_path is not None and Path(resume_path).exists():
        resume_state = torch.load(resume_path, map_location=model.device)
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        losses_recorded = resume_state["losses_recorded"]
        start_step = resume_state["step"] + 1
        logger.info(f"Resumed from {resume_path} at step {start_step - 1}/{num_steps}")

    if start_step > num_steps:
        logger.info(f"Already completed {start_step - 1}/{num_steps} steps, nothing to resume.")
    data_iter = iter(loader)
    epoch = 1
    for step in range(start_step, num_steps + 1):
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
        indices_tuple = tuple(idx.to(embeddings.device) for idx in indices_tuple)
        loss = loss_fn(embeddings, labels=None, indices_tuple=indices_tuple)

        optimizer.zero_grad()
        loss.backward()
        # Total gradient norm across every trainable param, not one
        # hardcoded layer -- no single ".weight" stands in for the model.
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

        if checkpoint_every is not None and resume_path is not None and step % checkpoint_every == 0:
            resume_path = Path(resume_path)
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "losses_recorded": losses_recorded,
                },
                resume_path,
            )
            logger.info(f"Resume checkpoint saved at step {step}/{num_steps} -> {resume_path}")

    if losses_recorded:
        logger.success(
            f"Done. Loss went {losses_recorded[0]:.4f} -> {losses_recorded[-1]:.4f} "
            f"over {len(losses_recorded)} steps."
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
