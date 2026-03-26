"""
Perception Encoder branch — Meta PE-Core-L14-336 (1024-d).

Input:  (B, T, 1024)
Output: (B, hidden_dim)

Mean-pools over time, then 2-layer MLP.
"""

import torch.nn as nn


class PEBranch(nn.Module):

    INPUT_DIM = 1024

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1, **kw):
        super().__init__()
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x.mean(dim=1))
