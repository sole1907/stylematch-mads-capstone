"""Smoke tests for CompatibilityModel: output shape, normalization,
freeze strategy, and score_outfit's pairwise-similarity aggregation.
Uses synthetic image tensors, not real Polyvore samples, since these
tests check the model's own architecture/scoring logic in isolation,
not the data pipeline. Downloads Marqo-FashionCLIP's pretrained weights
on first run, cached locally afterward.
"""

import pytest
import torch

from stylematch.compatibility.model import CompatibilityModel


def test_forward_pass_shape_and_normalization():
    model = CompatibilityModel(embedding_dim=256)
    model.eval()
    dummy_images = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        embeddings = model(dummy_images)

    assert embeddings.shape == (4, 256)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_freeze_strategy_freezes_correct_blocks():
    """Checks the actual intent, not just internal gradient
    consistency: blocks before freeze_until_block must be frozen,
    from it onward trainable."""
    model = CompatibilityModel(freeze_until_block=8)

    frozen_blocks = model.backbone.transformer.resblocks[:8]
    trainable_blocks = model.backbone.transformer.resblocks[8:]

    for i, block in enumerate(frozen_blocks):
        for param in block.parameters():
            assert not param.requires_grad, f"block {i} should be frozen"

    for i, block in enumerate(trainable_blocks, start=8):
        for param in block.parameters():
            assert param.requires_grad, f"block {i} should be trainable"


@pytest.mark.parametrize("bad_value", [-1, 13])
def test_invalid_freeze_until_block_raises(bad_value):
    with pytest.raises(ValueError):
        CompatibilityModel(freeze_until_block=bad_value)


def test_score_outfit_identical_items_scores_near_one():
    """Sanity check on the scoring math itself, not the backbone: if
    every item embedding is identical, every pairwise cosine
    similarity is exactly 1.0, so the mean must be too."""
    model = CompatibilityModel(embedding_dim=8)
    one_embedding = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    item_embeddings = one_embedding.repeat(5, 1)  # 5 identical items

    score = model.score_outfit(item_embeddings)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-5)


def test_score_outfit_orthogonal_items_scores_near_zero():
    """Two embeddings at 90 degrees have cosine similarity 0 -- checks
    the aggregation isn't accidentally counting the diagonal (which
    would pull the mean toward 1.0 even for unrelated items)."""
    model = CompatibilityModel(embedding_dim=2)
    item_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    score = model.score_outfit(item_embeddings)
    assert torch.isclose(score, torch.tensor(0.0), atol=1e-5)


def test_score_outfit_requires_at_least_two_items():
    model = CompatibilityModel(embedding_dim=8)
    single_item = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)

    with pytest.raises(ValueError):
        model.score_outfit(single_item)


if __name__ == "__main__":
    model = CompatibilityModel(freeze_until_block=8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.1%})")

    dummy_images = torch.randn(4, 3, 224, 224)
    embeddings = model(dummy_images)
    print(f"Output shape: {tuple(embeddings.shape)}")
    print(f"Outfit compatibility score: {model.score_outfit(embeddings).item():.4f}")
