"""Tests for the zero-shot raw-backbone baseline wrapper. Requires
the DeepFashion CAP subset to already be downloaded; skipped
otherwise.
"""

import pytest
import torch

from stylematch.baseline import RawBackboneModel
from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_raw_backbone_output_is_512_dim_and_normalized():
    model = RawBackboneModel()
    dataset = DeepFashionCAPDataset(split="val", transform=model.preprocess_val, sample_size=4)
    images = torch.stack([dataset[i][0] for i in range(4)])

    embeddings = model(images)

    assert embeddings.shape == (4, 512)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(4), atol=1e-4)


def test_raw_backbone_score_outfit_matches_compatibility_model_formula():
    """Cross-checks against the real CompatibilityModel.score_outfit
    method directly, not a re-derived copy of the same formula."""
    from stylematch.compatibility.model import CompatibilityModel

    torch.manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(4, 512), dim=-1)

    raw_model = RawBackboneModel()
    real_model = CompatibilityModel(embedding_dim=512)

    assert raw_model.score_outfit(embeddings).item() == pytest.approx(
        real_model.score_outfit(embeddings).item()
    )


def test_raw_backbone_score_outfit_below_two_items_raises():
    model = RawBackboneModel()
    with pytest.raises(ValueError):
        model.score_outfit(torch.randn(1, 512))
