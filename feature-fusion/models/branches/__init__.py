from models.branches.paper import PaperBranch
from models.branches.depth import DepthBranch
from models.branches.pe import PEBranch
from models.branches.cotracker import CoTrackerBranch

BRANCH_REGISTRY = {
    "paper": PaperBranch,
    "depth": DepthBranch,
    "pe":    PEBranch,
    "ct":    CoTrackerBranch,
}


def build_branch(name, **kwargs):
    return BRANCH_REGISTRY[name](**kwargs)
