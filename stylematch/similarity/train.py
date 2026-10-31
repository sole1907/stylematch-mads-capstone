"""Training loop for the similarity engine.

Uses the real SimilarityModel (model.py) and real preprocessing now.
"""

from loguru import logger
import torch
from torch.utils.data import DataLoader

from stylematch.similarity.dataset import DeepFashionCAPDataset
from stylematch.similarity.model import SimilarityModel


def run_stub_training_loop(
    num_steps: int = 3,
    batch_size: int = 4,
    sample_size: int = 16,
    learning_rate: float = 1e-3,
    freeze_until_block: int = 8,
    embedding_dim: int = 256,
) -> list[float]:
    """Runs a handful of steps end-to-end against a real DeepFashion
    subset, using the real SimilarityModel.

    Returns the loss recorded at each step.
    """
    model = SimilarityModel(embedding_dim=embedding_dim, freeze_until_block=freeze_until_block)

    dataset = DeepFashionCAPDataset(
        split="train", sample_size=sample_size, transform=model.preprocess_train
    )
    logger.info(f"Loaded {len(dataset)} images (split='train', sample_size={sample_size})")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(loader)
    logger.info(
        f"batch_size={batch_size} -> {steps_per_epoch} steps per epoch "
        f"(requested num_steps={num_steps}, "
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

    losses = []
    data_iter = iter(loader)
    epoch = 1
    for step in range(1, num_steps + 1):
        try:
            images, _category_labels, _attribute_labels = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"-- epoch {epoch} starts (data reshuffled) --")
            data_iter = iter(loader)
            images, _category_labels, _attribute_labels = next(data_iter)

        embeddings = model(images)

        # Placeholder loss: each embedding's alignment with the batch's
        # mean direction, pushed toward 0. NOT embeddings.pow(2).mean()
        # (the old stub's placeholder) — once the model L2-normalizes its
        # output, every embedding's squared values always sum to 1 by
        # construction, so mean(embeddings**2) is a *constant*
        # (1/embedding_dim, confirmed: 0.0039 = 1/256) with zero gradient
        # everywhere. This version depends on the embeddings' directions
        # relative to each other instead, which normalization doesn't fix,
        # so it actually has a gradient. Still arbitrary, still stands in
        # for the real triplet loss.
        batch_direction = embeddings.mean(dim=0)
        loss = (embeddings @ batch_direction).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        # Total gradient norm across every trainable parameter, not one
        # hardcoded layer, since there's no single ".weight" that stands
        # in for "the model" the way the old placeholder's did.
        grad_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in trainable_params if p.grad is not None)
        ).item()
        optimizer.step()

        losses.append(loss.item())
        logger.info(
            f"step {step}/{num_steps}: images {tuple(images.shape)} -> "
            f"embeddings {tuple(embeddings.shape)} | loss={loss.item():.4f} "
            f"| grad_norm={grad_norm:.4f} (confirms backward() computed "
            f"real gradients) | weights updated"
        )

    logger.success(
        f"Done. Loss went {losses[0]:.4f} -> {losses[-1]:.4f} over "
        f"{num_steps} steps (not meaningful yet, just confirms it moved)"
    )
    return losses


if __name__ == "__main__":
    recorded_losses = run_stub_training_loop()
    print(f"Stub training loop losses: {recorded_losses}")
