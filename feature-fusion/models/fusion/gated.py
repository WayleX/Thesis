import torch
import torch.nn as nn


class GatedFusion(nn.Module):

    def __init__(self, branch_dims, hidden_dim=128, **kw):
        super().__init__()
        self.branch_names = list(branch_dims.keys())
        n = len(self.branch_names)
        total = sum(branch_dims.values())
        self.output_dim = hidden_dim

        self.gate_net = nn.Sequential(
            nn.Linear(total, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n),
        )
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in branch_dims.items()
        })

    def forward(self, branch_embs):
        cat = torch.cat(list(branch_embs.values()), dim=-1)
        gates = torch.softmax(self.gate_net(cat), dim=-1)
        out = torch.zeros(cat.shape[0], self.output_dim, device=cat.device)
        for i, name in enumerate(self.branch_names):
            proj = self.projections[name](branch_embs[name])
            out = out + gates[:, i : i + 1] * proj
        return out
