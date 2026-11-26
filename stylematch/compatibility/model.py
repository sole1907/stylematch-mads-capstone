"""Compatibility scoring model: same Marqo-FashionCLIP backbone as the
similarity engine, its own separately fine-tuned projection head.
Compatibility is a function of embeddings' pairwise similarity, not a
separate learned "outfit" representation. Independent from
SimilarityModel for now (own weights); reusing its embeddings instead
is a later comparison experiment."""

import open_clip
import torch
from torch import nn, Tensor

BACKBONE_NAME = "hf-hub:Marqo/marqo-fashionCLIP"
BACKBONE_OUTPUT_DIM = 512  # confirmed via smoke test (tests/test_marqo_backbone.py)


class CompatibilityModel(nn.Module):
    """Marqo-FashionCLIP's visual tower (partially frozen) + a new,
    separately-trained projection head, producing one L2-normalized
    embedding per item. Compatibility is the mean pairwise cosine
    similarity across every distinct pair in a candidate outfit --
    this project's simplified analogue of Vasileva et al.'s
    pairwise-distance aggregation, without type-conditioned masking."""

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
        """Freezes the whole backbone, then unfreezes from
        freeze_until_block onward: the later transformer blocks, the
        final norm, and the backbone's own projection."""
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
                match the backbone.

        Returns:
            L2-normalized embedding, shape (batch, embedding_dim).
            Batch is just "however many item images"; turning these
            into one compatibility score is score_outfit's job.
        """
        images = images.to(self.device)
        backbone_embedding = self.backbone(images)
        embedding = self.projection_head(backbone_embedding)
        return nn.functional.normalize(embedding, dim=-1)

    def score_outfit(self, item_embeddings: Tensor) -> Tensor:
        """Scores one outfit's compatibility from its items' embeddings.

        Args:
            item_embeddings: shape (num_items, embedding_dim), already
                L2-normalized (from forward()). Takes embeddings
                directly since outfits vary in size and don't stack
                into one same-shape batch.

        Returns:
            Scalar tensor: mean cosine similarity (dot product, since
            L2-normalized) across every distinct pair of items.
        """
        num_items = item_embeddings.shape[0]
        if num_items < 2:
            raise ValueError(f"an outfit needs >= 2 items to score compatibility, got {num_items}")
        similarity_matrix = item_embeddings @ item_embeddings.T  # (num_items, num_items)
        # Each distinct pair counted once: upper triangle, excluding the
        # diagonal (an item's similarity to itself is always 1.0, not a
        # real compatibility signal).
        pair_mask = torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1)
        return similarity_matrix[pair_mask].mean()
