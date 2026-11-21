"""Training loop for the compatibility engine.

Uses the real CompatibilityModel (model.py) and a margin ranking loss
on whole-outfit scores, not single-embedding triplets: for each real
outfit in a batch, a negative is built by swapping one of its items
for an item from a different outfit in the same batch, then the loss
pushes score_outfit(real) above score_outfit(negative) by at least
`margin` -- the same margin-based ranking spirit as the similarity
engine's TripletMarginLoss, applied to outfit-level scores instead of
embedding distances.
"""

import random
from pathlib import Path

from loguru import logger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stylematch.compatibility.dataset import PolyvoreOutfitDataset
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import MODELS_DIR


def _identity_collate(batch: list) -> list:
    """Leaves the batch as a plain list of per-outfit tuples, instead
    of DataLoader's default collate (which would try to stack
    variable-length outfits into one tensor and fail). A module-level
    function, not a lambda: DataLoader's worker processes need to
    pickle this, and lambdas aren't picklable."""
    return batch


def _build_negative_outfit(item_images: list, batch: list, own_index: int) -> list:
    """Swaps one random item in item_images for a random item drawn
    from a different outfit in the same batch."""
    donor_index = random.choice([j for j in range(len(batch)) if j != own_index])
    donor_images = batch[donor_index][0]
    fake_images = item_images.copy()
    fake_images[random.randrange(len(fake_images))] = random.choice(donor_images)
    return fake_images


def run_training_loop(
    num_steps: int = 3,
    outfits_per_batch: int = 8,
    learning_rate: float = 1e-3,
    freeze_until_block: int = 8,
    embedding_dim: int = 256,
    margin: float = 0.2,
    checkpoint_path: Path | None = None,
    num_workers: int = 4,
) -> tuple[CompatibilityModel, list[float]]:
    """Runs a handful of steps end-to-end against real Polyvore
    outfits, using the real CompatibilityModel and a margin ranking
    loss between real and swapped-item negative outfits.

    Args:
        outfits_per_batch: how many outfits per step. Needs to be >= 2,
            since each outfit's negative is built by borrowing an item
            from another outfit in the same batch.
        checkpoint_path: if given, saves the trained model's state_dict
            here after training. If None (default), nothing is saved
            to disk, only returned in memory.
        num_workers: DataLoader subprocesses for image loading/decoding.

    Returns:
        (model, losses): the trained model (already in memory) and the
        loss recorded at each step.
    """
    if outfits_per_batch < 2:
        raise ValueError(f"outfits_per_batch must be >= 2, got {outfits_per_batch}")

    model = CompatibilityModel(embedding_dim=embedding_dim, freeze_until_block=freeze_until_block)

    dataset = PolyvoreOutfitDataset(split="train", transform=model.preprocess_train)
    logger.info(f"Loaded {len(dataset)} outfits (split='train')")

    # Variable-length outfits don't stack into one tensor like
    # same-shape images do, so the batch stays a plain list of
    # (item_images, item_ids, category_labels, outfit_id) tuples --
    # unpacked per-outfit in the loop below, not collated into a
    # single tensor.
    loader = DataLoader(
        dataset, batch_size=outfits_per_batch, shuffle=True, num_workers=num_workers,
        collate_fn=_identity_collate,
    )
    steps_per_epoch = len(loader)
    logger.info(
        f"outfits_per_batch={outfits_per_batch} -> {steps_per_epoch} steps per epoch "
        f"(requested num_steps={num_steps}, {num_steps / steps_per_epoch:.2f} epochs)"
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"CompatibilityModel: freeze_until_block={freeze_until_block}, "
        f"{trainable:,} / {total:,} trainable parameters ({trainable / total:.1%})"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    losses_recorded = []
    data_iter = iter(loader)
    epoch = 1
    for step in range(1, num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"-- epoch {epoch} starts (batches reshuffled) --")
            data_iter = iter(loader)
            batch = next(data_iter)

        step_loss = torch.tensor(0.0)
        for i, (item_images, _item_ids, _category_labels, _outfit_id) in enumerate(batch):
            real_embeddings = model(torch.stack(item_images))
            real_score = model.score_outfit(real_embeddings)

            fake_images = _build_negative_outfit(item_images, batch, own_index=i)
            fake_embeddings = model(torch.stack(fake_images))
            fake_score = model.score_outfit(fake_embeddings)

            target = torch.ones(1)
            step_loss = step_loss + F.margin_ranking_loss(
                real_score.unsqueeze(0), fake_score.unsqueeze(0), target, margin=margin
            )
        loss = step_loss / len(batch)

        optimizer.zero_grad()
        loss.backward()
        # Total gradient norm across every trainable parameter, not one
        # hardcoded layer, since there's no single ".weight" that
        # stands in for "the model."
        grad_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in trainable_params if p.grad is not None)
        ).item()
        optimizer.step()

        losses_recorded.append(loss.item())
        logger.info(
            f"step {step}/{num_steps}: {len(batch)} outfits -> "
            f"loss={loss.item():.4f} | grad_norm={grad_norm:.4f} | weights updated"
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
        checkpoint_path=MODELS_DIR / "compatibility_first_run.pt"
    )
    print(f"Training loop losses: {recorded_losses}")
