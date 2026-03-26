"""
Paper (DFD-FCG) branch encoder.

Input:  (B, 1280)  — syno_s (1024) || syno_t (256)
Output: (B, hidden_dim)
"""

import torch.nn as nn


class PaperBranch(nn.Module):

    INPUT_DIM = 1280

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1, **kw):
        super().__init__()
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)
