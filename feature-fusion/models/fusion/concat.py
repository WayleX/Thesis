import torch
import torch.nn as nn


class ConcatFusion(nn.Module):

    def __init__(self, branch_dims, **kw):
        super().__init__()
        self.output_dim = sum(branch_dims.values())

    def forward(self, branch_embs):
        return torch.cat(list(branch_embs.values()), dim=-1)
