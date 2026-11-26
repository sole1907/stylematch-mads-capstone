"""Similarity embedding model: Marqo-FashionCLIP's visual backbone
(partially frozen) plus a new, fully-trainable projection head, fine-tuned
for garment-to-garment similarity.
"""

import open_clip
import torch
from torch import nn, Tensor

BACKBONE_NAME = "hf-hub:Marqo/marqo-fashionCLIP"
BACKBONE_OUTPUT_DIM = 512  # confirmed via smoke test (tests/test_marqo_backbone.py)


class SimilarityModel(nn.Module):
    """Marqo-FashionCLIP's visual tower (partially frozen) + a new trainable
    projection head, producing a single L2-normalized embedding per image.
    Early blocks stay frozen; later blocks, final norm, and backbone
    projection are fine-tuned alongside the new head."""

    def __init__(
        self,
        embedding_dim: int = 256,
        freeze_until_block: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Args:
            embedding_dim: output dimensionality of the new projection head.
            freeze_until_block: transformer blocks before this index stay
                frozen; range 0-12 (0 = nothing frozen, 12 = all frozen).
            device: None picks CUDA if available, else CPU.
        """
        super().__init__()
        if not 0 <= freeze_until_block <= 12:
            raise ValueError(f"freeze_until_block must be 0-12, got {freeze_until_block}")

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            BACKBONE_NAME
        )
        self.backbone = backbone.visual  # only the image tower; text tower unused
        # Exposed so callers (e.g. train.py's Dataset) can match this
        # backbone's required preprocessing without a second, redundant
        # open_clip.create_model_and_transforms() call.
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self._apply_freeze(freeze_until_block)

        self.projection_head = nn.Sequential(
            nn.Linear(BACKBONE_OUTPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )
        self.to(self.device)

    def _apply_freeze(self, freeze_until_block: int) -> None:
        """Freezes the whole backbone, then unfreezes from freeze_until_block
        onward (later blocks, final norm, backbone projection). Text tower
        stays frozen throughout -- unused for image-to-image similarity."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.transformer.resblocks[freeze_until_block:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.ln_post.parameters():
            param.requires_grad = True
        self.backbone.proj.requires_grad = True

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: float tensor (batch, 3, 224, 224), preprocessed to
                match the backbone (open_clip's preprocessing, not
                dataset.py's placeholder transform).

        Returns:
            L2-normalized embedding, shape (batch, embedding_dim).
        """
        images = images.to(self.device)
        backbone_embedding = self.backbone(images)            # (batch, 512) — full backbone, incl. its own proj
        embedding = self.projection_head(backbone_embedding)  # (batch, embedding_dim) — new head
        return nn.functional.normalize(embedding, dim=-1)
