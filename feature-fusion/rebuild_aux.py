#!/usr/bin/env python3
"""
Rebuild aux_features for DeepFakeDataset v1.1 from actual feature files.

Strategy: take the intersection of all 5 feature types for each category.
A sample is included only if ALL feature types exist for it.
"""
import shutil
import torch
from pathlib import Path

DFD = Path("/extra_space2/shykula/gg/datasets/deepfake_v1.1")
FEAT_DIRS = {
    "paper": "paper_features",
    "depth": "depth_features",
    "pe":    "pe_features_pe_real",
    "ct":    "cotracker_features",
    "ct_grid": "cotracker_grid_features",
}


def get_stems(base_dir):
    if not base_dir.exists():
        return set()
    return {p.stem for p in base_dir.glob("*.pt")}


def rebuild():
    shutil.rmtree(DFD / "aux_features", ignore_errors=True)

    real_sets = {}
    for feat_name, feat_dir in FEAT_DIRS.items():
        stems = get_stems(DFD / feat_dir / "real")
        real_sets[feat_name] = stems
        print(f"  Real {feat_name}: {len(stems)}")

    real_stems = set.intersection(*real_sets.values()) if real_sets else set()
    print(f"  Real intersection (all 5): {len(real_stems)}")

    real_aux = DFD / "aux_features" / "real"
    real_aux.mkdir(parents=True, exist_ok=True)
    for stem in sorted(real_stems):
        torch.save({"marker": True}, real_aux / f"{stem}.pt")

    fake_base = DFD / "paper_features" / "fake"
    if not fake_base.exists():
        print("No fake paper_features found - trying pe_features_pe_real/fake")
        fake_base = DFD / "pe_features_pe_real" / "fake"
    if not fake_base.exists():
        print("No fake features found at all!")
        return

    all_gen_dirs = set()
    for feat_dir in FEAT_DIRS.values():
        fd = DFD / feat_dir / "fake"
        if fd.exists():
            for d in fd.iterdir():
                if d.is_dir():
                    all_gen_dirs.add(d.name)

    print(f"\nGenerators found: {sorted(all_gen_dirs)}")

    for gen_name in sorted(all_gen_dirs):
        gen_sets = {}
        for feat_name, feat_dir in FEAT_DIRS.items():
            stems = get_stems(DFD / feat_dir / "fake" / gen_name)
            gen_sets[feat_name] = stems
            print(f"  {gen_name} {feat_name}: {len(stems)}")

        gen_stems = set.intersection(*gen_sets.values()) if gen_sets else set()
        print(f"  {gen_name} intersection: {len(gen_stems)}")

        fake_aux = DFD / "aux_features" / "fake" / gen_name
        fake_aux.mkdir(parents=True, exist_ok=True)
        for stem in sorted(gen_stems):
            torch.save({"marker": True}, fake_aux / f"{stem}.pt")

    print("\n=== VERIFICATION ===")
    real_count = len(list((DFD / "aux_features" / "real").glob("*.pt")))
    print(f"  real: {real_count}")
    total = real_count
    for d in sorted((DFD / "aux_features" / "fake").iterdir()):
        if d.is_dir():
            c = len(list(d.glob("*.pt")))
            print(f"  {d.name}: {c}")
            total += c
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    rebuild()
