#!/usr/bin/env python3
"""
Extract Perception Encoder features for DeepFakeDataset v1.1.

REQUIRES pe_env conda environment:
    cd /extra_space2/shykula/perception_models
    conda run -n pe_env python <this_script> --gpu 0

Usage:
  python extract_pe_v11.py --gpu 0 --shard 0 --num_shards 3
"""

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/extra_space2/shykula/perception_models")
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

DATASETS = Path("/extra_space2/shykula/gg/datasets")
DFD_V11 = DATASETS / "deepfake_v1.1"
CROPPED = DFD_V11 / "cropped" / "videos"


def load_pe_model(device="cuda"):
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
    model = model.to(device).eval()
    preprocess = transforms.get_image_transform(model.image_size)
    return model, preprocess


def read_frames(path, n_frames=16):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return []
    if len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[i] for i in idx]
    return frames


@torch.no_grad()
def extract_pe_features(model, preprocess, frames, device):
    from PIL import Image
    tokens = []
    for frame in frames:
        img = Image.fromarray(frame)
        x = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(x)
        tokens.append(feat.cpu())
    return torch.cat(tokens, dim=0)


def get_all_cropped_videos():
    """Collect face-cropped videos (150x150) to match FFPP training domain."""
    videos = []
    real_dir = CROPPED / "real"
    if real_dir.exists():
        for v in sorted(real_dir.glob("*.avi")):
            videos.append((v, "real", "real"))
    fake_dir = CROPPED / "fake"
    if fake_dir.exists():
        for gen_dir in sorted(fake_dir.iterdir()):
            if gen_dir.is_dir():
                for v in sorted(gen_dir.glob("*.avi")):
                    videos.append((v, "fake", gen_dir.name))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--n_frames", type=int, default=16)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model, preprocess = load_pe_model(device)

    videos = get_all_cropped_videos()
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    print(f"[Shard {args.shard}/{args.num_shards}] Processing {len(videos)} videos on GPU {args.gpu}")

    for vid_path, label_type, gen_name in tqdm(videos, desc="PE"):
        if label_type == "real":
            out_dir = DFD_V11 / "pe_features_pe_real" / "real"
        else:
            out_dir = DFD_V11 / "pe_features_pe_real" / "fake" / gen_name

        out_path = out_dir / f"{vid_path.stem}.pt"
        if out_path.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        frames = read_frames(vid_path, args.n_frames)
        if not frames:
            continue

        features = extract_pe_features(model, preprocess, frames, device)
        torch.save({"pe": features.half()}, out_path)

    print(f"[Shard {args.shard}] Done.")


if __name__ == "__main__":
    main()
