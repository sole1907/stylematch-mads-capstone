"""Similarity engine model architecture (backbone + embedding head)."""

import torch.nn as nn


class SimilarityModel(nn.Module):
    """Backbone + projection head producing embeddings for retrieval.

    TODO: wire up the chosen backbone (FashionCLIP or rainalexotl's repo)
    and a projection head trained with pytorch-metric-learning.
    """

    def __init__(self):
        super().__init__()
        # ---- REPLACE WITH ACTUAL BACKBONE + HEAD ----
        raise NotImplementedError
        # -----------------------------------------------

    def forward(self, x):
        raise NotImplementedError
