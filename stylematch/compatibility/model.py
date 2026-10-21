"""Compatibility engine model architecture.

Predicts/scores compatibility between items in an outfit. May optionally
accept the similarity engine's embeddings as input, per the project's
architecture choice (independent training, optional embedding chaining).
"""

import torch.nn as nn


class CompatibilityModel(nn.Module):
    """TODO: wire up architecture. Optionally accepts similarity-engine
    embeddings as input per the project's architecture choice.
    """

    def __init__(self):
        super().__init__()
        # ---- REPLACE WITH ACTUAL ARCHITECTURE ----
        raise NotImplementedError
        # ---------------------------------------------

    def forward(self, x):
        raise NotImplementedError
