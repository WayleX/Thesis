#!/usr/bin/env python3
"""
Extract Depth Anything V2 BASE features for FFPP and DFDC.

Usage:
  python extract_depth_base_ffpp.py --dataset ffpp --gpu 0 --shard 0 --num_shards 3
  python extract_depth_base_ffpp.py --dataset dfdc --gpu 0
"""

import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATASETS = Path("/extra_space2/shykula/gg/datasets")


def load_depth_backbone(device):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf")
    backbone = model.backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def load_video_tensor(path, max_frames=None):
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
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]
    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def extract_cls_features(backbone, frames_tensor, device, num_layers=4, batch_size=16):
    all_feats = []
    for i in range(0, len(frames_tensor), batch_size):
        batch = frames_tensor[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False)
        with torch.amp.autocast("cuda"):
            out = backbone(batch, output_hidden_states=True)
        cls_tokens = [h[:, 0, :] for h in out.hidden_states[-num_layers:]]
        stacked = torch.stack(cls_tokens, dim=1)
        all_feats.append(stacked.mean(dim=1).cpu().half())
        del batch, out, stacked
    return torch.cat(all_feats, dim=0)


def collect_ffpp_videos():
    ffpp = DATASETS / "ffpp"
    videos = []
    for method in ["real", "DF", "F2F", "FS", "NT", "FSh"]:
        vid_dir = ffpp / method / "c23" / "videos"
        out_base = ffpp / "depth_base_features" / method / "c23"
        for v in sorted(vid_dir.glob("*.avi")):
            videos.append((v, out_base))
    return videos


def collect_dfdc_videos():
    dfdc = DATASETS / "dfdc"
    videos = []
    for label in ["real", "fake"]:
        vid_dir = dfdc / label
        if not vid_dir.exists():
            continue
        out_base = dfdc / "depth_base_features" / label
        for v in sorted(vid_dir.glob("*")):
            if v.suffix in (".avi", ".mp4"):
                videos.append((v, out_base))
    return videos


def collect_dfdc_cropped_videos():
    dfdc = Path("/extra_space2/shykula/gg/dfdc")
    cropped_dir = dfdc / "cropped" / "videos"
    videos = []
    if not cropped_dir.exists():
        return videos
    out_base = dfdc / "depth_base_features"
    for v in sorted(cropped_dir.glob("*.avi")):
        videos.append((v, out_base))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ffpp", "dfdc"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    backbone = load_depth_backbone(device)

    if args.dataset == "ffpp":
        videos = collect_ffpp_videos()
    else:
        videos = collect_dfdc_cropped_videos()

    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    print(f"[{args.dataset} shard {args.shard}] {len(videos)} videos on GPU {args.gpu}")

    for vid_path, out_base in tqdm(videos, desc=f"Depth-Base-{args.dataset}"):
        out_path = out_base / f"{vid_path.stem}.pt"
        if out_path.exists():
            continue
        out_base.mkdir(parents=True, exist_ok=True)
        frames = load_video_tensor(vid_path, max_frames=args.max_frames)
        if frames is None:
            continue
        features = extract_cls_features(backbone, frames, device)
        torch.save({"depth": features, "num_frames": len(frames)}, out_path)

    print(f"[{args.dataset} shard {args.shard}] Done.")


if __name__ == "__main__":
    main()
