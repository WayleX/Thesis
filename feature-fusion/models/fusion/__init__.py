from models.fusion.concat import ConcatFusion
from models.fusion.gated import GatedFusion

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated":  GatedFusion,
}


def build_fusion(fusion_type, branch_dims, hidden_dim=128):
    return FUSION_REGISTRY[fusion_type](branch_dims, hidden_dim=hidden_dim)
