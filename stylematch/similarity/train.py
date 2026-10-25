"""Stub training loop for the similarity engine.

Proves the training loop plumbing works end-to-end: data loads, a forward
pass runs, a loss computes, backward() + optimizer.step() update weights.

Uses a throwaway placeholder model defined here, not the real
SimilarityModel in model.py. Loading the real Marqo-FashionCLIP backbone
and building the real model are separate, later PRs (Week 2 checklist:
"Load Marqo-FashionCLIP pretrained weights", "Build SimilarityModel in
similarity/model.py, replaces NotImplementedError stub"). The loss here is
also a placeholder for the same reason, real triplet/contrastive loss
wiring (via pytorch-metric-learning) is a separate Week 2 item too, and
needs the batch sampler + online miner described in dataset.py's
docstring, neither of which exist yet.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from stylematch.similarity.dataset import MOCK_IMAGE_SIZE, DeepFashionCAPDataset


class _PlaceholderModel(nn.Module):
    """Throwaway stand-in for SimilarityModel (model.py), used only to
    prove the training loop runs end-to-end. Flattens each image and
    projects it down to a small embedding with a single linear layer, not
    meant to produce meaningful embeddings. Replaced by the real backbone
    and projection head in Week 2."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        input_dim = 3 * MOCK_IMAGE_SIZE * MOCK_IMAGE_SIZE
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened = images.flatten(start_dim=1)
        return self.projection(flattened)


def run_stub_training_loop(
    num_steps: int = 3,
    batch_size: int = 4,
    sample_size: int = 16,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Runs a handful of steps end-to-end against the mock dataset.

    Returns the loss recorded at each step, mainly so a caller or test can
    confirm it's actually changing (gradients are flowing through the
    whole pipeline), not that it's meaningfully decreasing. The loss
    itself isn't real yet.
    """
    dataset = DeepFashionCAPDataset(split="train", sample_size=sample_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _PlaceholderModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    data_iter = iter(loader)
    for _ in range(num_steps):
        try:
            images, _category_labels, _attribute_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _category_labels, _attribute_labels = next(data_iter)

        embeddings = model(images)

        # Placeholder loss: pulls embeddings toward zero, just needs to
        # depend on the model's output so backward() has something to
        # compute gradients for. Stands in for the real triplet loss.
        loss = embeddings.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


if __name__ == "__main__":
    recorded_losses = run_stub_training_loop()
    print(f"Stub training loop losses: {recorded_losses}")
