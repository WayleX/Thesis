"""Attention fusion: project each branch to a token, self-attention, mean-pool."""

import torch
import torch.nn as nn


class CrossAttnFusion(nn.Module):
    """Multi-head self-attention over one token per branch, then mean-pool.

    For two branches this acts as a learned interaction between DFD-FCG and
    the second modality (depth or CoTracker) before aggregation.
    """

    def __init__(self, branch_dims, hidden_dim=128, n_heads=4, **kwargs):
        super().__init__()
        nh = int(n_heads) if n_heads is not None else 4
        nh = max(1, nh)
        while nh > 1 and hidden_dim % nh != 0:
            nh -= 1
        self.n_heads = nh
        self.names = list(branch_dims.keys())
        self.output_dim = hidden_dim
        self.projs = nn.ModuleDict(
            {n: nn.Linear(d, hidden_dim) for n, d in branch_dims.items()})
        self.attn = nn.MultiheadAttention(
            hidden_dim, self.n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, branch_embs):
        tokens = torch.stack(
            [self.projs[n](branch_embs[n]) for n in self.names], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        h = self.norm1(tokens + attended)
        h2 = self.ff(h)
        h = self.norm2(h + h2)
        return h.mean(dim=1)
