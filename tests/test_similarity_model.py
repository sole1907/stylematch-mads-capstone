"""Smoke tests for SimilarityModel: output shape, normalization, and
whether the freeze strategy actually works, not just whether it runs.
Uses synthetic image tensors, not real DeepFashion samples, since these
tests check the model's own architecture and freeze logic in isolation,
not the data pipeline. Downloads Marqo-FashionCLIP's pretrained weights
from Hugging Face on first run (cached locally afterward); needs network
access the first time it's run.
"""

import pytest
import torch

from stylematch.similarity.model import SimilarityModel


def test_forward_pass_shape_and_normalization():
    model = SimilarityModel(embedding_dim=256)
    model.eval()
    dummy_images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embeddings = model(dummy_images)

    assert embeddings.shape == (2, 256)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_freeze_strategy_gradients():
    """Confirms .grad is internally consistent with .requires_grad after a
    backward pass: anything marked trainable actually received a gradient
    (catches dead/disconnected parameters, e.g. a new layer defined but
    never called in forward()), and anything marked frozen never did.

    This does NOT confirm the freeze was applied to the *correct* blocks,
    a reversed slice (freezing 8-11 instead of 0-7) would still pass this
    test, since .grad would stay consistent with whatever .requires_grad
    ended up being. See test_freeze_strategy_freezes_correct_blocks for
    that check.
    """
    model = SimilarityModel(freeze_until_block=8)
    dummy_images = torch.randn(2, 3, 224, 224)

    embeddings = model(dummy_images)
    embeddings.sum().backward()

    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} is trainable but got no gradient"
        else:
            assert param.grad is None, f"{name} is frozen but got a gradient"

    for name, param in model.projection_head.named_parameters():
        assert param.grad is not None, f"{name} is the new head and should always get a gradient"


def test_freeze_strategy_freezes_correct_blocks():
    """Checks the actual intent, not just internal consistency: blocks
    before freeze_until_block must be frozen, blocks from freeze_until_block
    onward must be trainable. Hard-codes the expected answer by block
    index, so unlike test_freeze_strategy_gradients, this would catch a
    reversed slice (e.g. [:n] instead of [n:]) in _apply_freeze.
    """
    model = SimilarityModel(freeze_until_block=8)

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
        SimilarityModel(freeze_until_block=bad_value)


if __name__ == "__main__":
    model = SimilarityModel(freeze_until_block=8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.1%})")

    dummy_images = torch.randn(2, 3, 224, 224)
    embeddings = model(dummy_images)
    print(f"Output shape: {tuple(embeddings.shape)}")
    print(f"Output norms: {embeddings.norm(dim=-1).tolist()}")
