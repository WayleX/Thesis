"""
Central path registry.

TRAIN  = FF++ {real, DF, F2F, FS, NT} at c23.  FaceShifter held out.
VAL    = FaceShifter (FF++), DFDC, DeepFakeDataset v1.1 (8 generators).

Depth uses DINOv2-Base (768-d) features stored as depth_base_features.
"""

from pathlib import Path

DATASETS = Path("/extra_space2/shykula/gg/datasets")

FFPP = DATASETS / "ffpp"
FFPP_C23 = "c23"
FFPP_TRAIN_METHODS = ["real", "DF", "F2F", "FS", "NT"]

DFDC = DATASETS / "dfdc"

DFD_V11 = DATASETS / "deepfake_v1.1"

DFD_GENERATORS = {
    "Grok":       "Grok_imagine",
    "HunyuanI2V": "HunyuanVideo_1.5_14b_i2v",
    "HunyuanT2V": "HunyuanVideo_1.5_14b_t2v",
    "Kling":      "Kling_3.0",
    "Veo":        "Veo3.1",
    "Wan22I2V":   "Wan_2.2_14b_i2v",
    "Wan22T2V":   "Wan_2.2_14b_t2v",
    "Wan26":      "Wan_2.6",
}


def feature_dirs(dataset):
    if dataset == "ffpp":
        base = FFPP
    elif dataset == "dfdc":
        base = DFDC
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return {
        "paper": base / "paper_features",
        "depth": base / "depth_base_features",
        "pe":    base / "pe_features_pe_real",
        "ct":    base / "cotracker_features",
    }


def dfd_dirs(generator_key):
    gen_dir = DFD_GENERATORS[generator_key]
    base = DFD_V11
    return {
        "real_paper": base / "paper_features" / "real",
        "real_depth": base / "depth_base_features" / "real",
        "real_pe":    base / "pe_features_pe_real" / "real",
        "real_ct":    base / "cotracker_features" / "real",
        "fake_paper": base / "paper_features" / "fake" / gen_dir,
        "fake_depth": base / "depth_base_features" / "fake" / gen_dir,
        "fake_pe":    base / "pe_features_pe_real" / "fake" / gen_dir,
        "fake_ct":    base / "cotracker_features" / "fake" / gen_dir,
        "fake_aux":   base / "aux_features" / "fake" / gen_dir,
        "real_aux":   base / "aux_features" / "real",
    }
