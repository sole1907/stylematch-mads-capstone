"""Raw, pretrained Marqo-FashionCLIP backbone -- no projection head,
no fine-tuning -- wrapped to satisfy the same interface
SimilarityModel/CompatibilityModel expose (.eval(), .preprocess_val,
callable forward, .score_outfit()), so the exact same evaluation
functions (evaluate_retrieval_accuracy, evaluate_compatibility) work
unmodified against it.

Exists to answer a real question neither sweep's "before" baseline
actually answered: every sweep's accuracy_before was computed on a
freshly-constructed model that ALREADY has a new, randomly-initialized
projection head on top of the backbone -- not the backbone's own raw
512-dim output. A random projection can scramble whatever useful
structure the pretrained space already had, so "before" was never a
clean "fine-tuning vs. the untouched base model" comparison.
"""

import open_clip
import torch
from torch import Tensor

BACKBONE_NAME = "hf-hub:Marqo/marqo-fashionCLIP"


class RawBackboneModel:
    """Duck-types SimilarityModel/CompatibilityModel's eval interface,
    using the backbone's own output directly -- no new head, no
    training. L2-normalized, same convention as both real models."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(BACKBONE_NAME)
        self.backbone = backbone.visual
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.backbone.to(self.device)

    def eval(self) -> "RawBackboneModel":
        self.backbone.eval()
        return self

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Tensor:
        images = images.to(self.device)
        embedding = self.backbone(images)
        return torch.nn.functional.normalize(embedding, dim=-1)

    def score_outfit(self, item_embeddings: Tensor) -> Tensor:
        """Identical formula to CompatibilityModel.score_outfit --
        mean cosine similarity across every distinct pair -- copied
        rather than imported, since that method doesn't use any
        model-specific state and this avoids a real-model dependency
        for a baseline that's deliberately not using that model."""
        num_items = item_embeddings.shape[0]
        if num_items < 2:
            raise ValueError(f"an outfit needs >= 2 items to score compatibility, got {num_items}")
        similarity_matrix = item_embeddings @ item_embeddings.T
        pair_mask = torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1)
        return similarity_matrix[pair_mask].mean()
