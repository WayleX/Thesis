#!/usr/bin/env python3
"""
Extract Depth Anything V2 BASE features for DeepFakeDataset v1.1.

Uses DINOv2-Base backbone (768-dim) from Depth Anything V2-Base.
Extracts CLS tokens from the last 4 hidden layers, averaged -> (T, 768).

Usage:
  python extract_depth_base_v11.py --gpu 0 --shard 0 --num_shards 3
"""

import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATASETS = Path("/extra_space2/shykula/gg/datasets")
DFD_V11 = DATASETS / "deepfake_v1.1"
CROPPED = DFD_V11 / "cropped" / "videos"


def load_depth_backbone(device):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf")
    backbone = model.backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def load_video_tensor(path, max_frames=10):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]
    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def extract_cls_features(backbone, frames_tensor, device, num_layers=4):
    batch = frames_tensor.to(device)
    batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False)
    with torch.amp.autocast("cuda"):
        out = backbone(batch, output_hidden_states=True)
    cls_tokens = [h[:, 0, :] for h in out.hidden_states[-num_layers:]]
    stacked = torch.stack(cls_tokens, dim=1)
    return stacked.mean(dim=1).cpu().half()


def get_all_cropped_videos():
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
    parser.add_argument("--max_frames", type=int, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    backbone = load_depth_backbone(device)

    videos = get_all_cropped_videos()
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    print(f"[Shard {args.shard}/{args.num_shards}] Processing {len(videos)} videos on GPU {args.gpu}")

    for vid_path, label_type, gen_name in tqdm(videos, desc="Depth-Base"):
        if label_type == "real":
            out_dir = DFD_V11 / "depth_base_features" / "real"
        else:
            out_dir = DFD_V11 / "depth_base_features" / "fake" / gen_name

        out_path = out_dir / f"{vid_path.stem}.pt"
        if out_path.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        frames = load_video_tensor(vid_path, args.max_frames)
        if frames is None:
            continue

        features = extract_cls_features(backbone, frames, device)
        torch.save({"depth": features, "num_frames": len(frames)}, out_path)

    print(f"[Shard {args.shard}] Done.")


if __name__ == "__main__":
    main()
