"""
CoTracker branch — facial landmark trajectories.

Input:  (B, T, N, C) — kinematic features per keypoint per frame
Output: (B, hidden_dim)

Flattens keypoints per frame (T, N*C), mean-pools over time, then MLP.
Supports variable N and C for different feature variants.
"""

import torch.nn as nn


class CoTrackerBranch(nn.Module):

    INPUT_DIM = 8

    def __init__(self, hidden_dim: int = 128, n_keypoints: int = 68,
                 feat_dim: int = 8, dropout: float = 0.1, **kw):
        super().__init__()
        self.output_dim = hidden_dim
        flat_dim = feat_dim * n_keypoints
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        B, T, N, C = x.shape
        return self.net(x.reshape(B, T, N * C).mean(dim=1))
