#!/usr/bin/env python3
"""
Extract CoTracker3 features for DeepFakeDataset v1.1.

Supports both landmark (68 facial keypoints) and grid (8x8) initialization.
Uses face-cropped videos (150x150) to match FFPP training domain.

Usage:
  python extract_cotracker_v11.py --mode landmark --gpu 0 --shard 0 --num_shards 3
  python extract_cotracker_v11.py --mode grid --gpu 0 --shard 0 --num_shards 3
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATASETS = Path("/extra_space2/shykula/gg/datasets")
DFD_V11 = DATASETS / "deepfake_v1.1"
CROPPED = DFD_V11 / "cropped" / "videos"


def load_cotracker(device):
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device).eval()
    return model


def read_video_tensor(path, max_frames=100):
    """Read video frames.  max_frames caps the number of frames fed to
    CoTracker to avoid GPU OOM.  For DFD v1.1 videos (125 frames @ 25 fps),
    100 frames covers 4 of the 5 seconds; for FF++ / DFDC the original
    extraction already used full-length tracks."""
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

    video = np.stack(frames)
    return torch.from_numpy(video).permute(0, 3, 1, 2).float().unsqueeze(0)


def get_landmarks(frame):
    """Get 68 facial landmarks using face_alignment."""
    try:
        import face_alignment
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False)
        preds = fa.get_landmarks(frame)
        if preds is not None and len(preds) > 0:
            return torch.from_numpy(preds[0]).float()
    except Exception:
        pass
    return None


def get_grid_queries(h, w, grid_size=8):
    """Generate regular grid query points."""
    ys = torch.linspace(h * 0.1, h * 0.9, grid_size)
    xs = torch.linspace(w * 0.1, w * 0.9, grid_size)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 2)
    t_col = torch.zeros(grid.shape[0], 1)
    return torch.cat([t_col, grid[:, [1, 0]]], dim=1).unsqueeze(0)


@torch.no_grad()
def extract_tracks(model, video, mode, device):
    video = video.to(device)
    B, T, C, H, W = video.shape

    if mode == "landmark":
        first_frame = video[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        landmarks = get_landmarks(first_frame)
        if landmarks is None:
            return None, None
        t_col = torch.zeros(landmarks.shape[0], 1)
        queries = torch.cat([t_col, landmarks], dim=1).unsqueeze(0).to(device)
        pred = model(video, queries=queries)
    else:
        queries = get_grid_queries(H, W).to(device)
        pred = model(video, queries=queries)

    tracks = pred[0].cpu()
    visibility = pred[1].cpu()
    return tracks[0], visibility[0]


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
    parser.add_argument("--mode", choices=["landmark", "grid"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    feat_dir_name = "cotracker_features" if args.mode == "landmark" else "cotracker_grid_features"
    device = f"cuda:{args.gpu}"
    model = load_cotracker(device)

    videos = get_all_cropped_videos()
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    print(f"[{args.mode} shard {args.shard}/{args.num_shards}] {len(videos)} videos on GPU {args.gpu}")

    for vid_path, label_type, gen_name in tqdm(videos, desc=f"CT-{args.mode}"):
        if label_type == "real":
            out_dir = DFD_V11 / feat_dir_name / "real"
        else:
            out_dir = DFD_V11 / feat_dir_name / "fake" / gen_name

        out_path = out_dir / f"{vid_path.stem}.pt"
        if out_path.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        video = read_video_tensor(vid_path)
        if video is None:
            continue

        tracks, visibility = extract_tracks(model, video, args.mode, device)
        if tracks is None:
            continue

        torch.save({"tracks": tracks, "visibility": visibility}, out_path)

    print(f"[Shard {args.shard}] Done.")


if __name__ == "__main__":
    main()
