#!/usr/bin/env python3
"""
Extract Paper (DFD-FCG) features for DeepFakeDataset v1.1.

Requires face-cropped .avi videos and DFD-FCG model weights.
Extracts syno_s (1024-d spatial) + syno_t (256-d temporal) features.

Usage:
  conda run -n dfd-fcg python extract_paper_v11.py --gpu 0
"""
import sys
import inspect
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

DFD_FCG_ROOT = Path("/extra_space2/shykula/gg/Repos/DFD-FCG")
sys.path.insert(0, str(DFD_FCG_ROOT))

DATASETS = Path("/extra_space2/shykula/gg/datasets")
DFD_V11 = DATASETS / "deepfake_v1.1"


def load_paper_model(ckpt_path, device):
    from src.model.clip.svl import FFGSynoVideoLearner
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    valid_params = set(inspect.signature(FFGSynoVideoLearner.__init__).parameters.keys())
    valid_params.discard("self")
    filtered = {k: v for k, v in hp.items() if k in valid_params}
    model = FFGSynoVideoLearner(**filtered)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    return model


def read_video_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    n = min(max_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return torch.stack(frames) if frames else None


@torch.no_grad()
def extract_features(model, frames, device):
    frames = frames.to(device)
    if frames.shape[-1] != model.n_px:
        frames = F.interpolate(
            frames, size=(model.n_px, model.n_px),
            mode="bilinear", align_corners=False,
        )
    T = 10
    if frames.shape[0] > T:
        idx = torch.linspace(0, frames.shape[0] - 1, T).long()
        frames = frames[idx]
    elif frames.shape[0] < T:
        pad = frames[-1:].repeat(T - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)

    x = frames.unsqueeze(0)
    x = model.transform(x)

    with torch.cuda.amp.autocast():
        output = model(x)

    feats = {}
    if "syno_s" in output:
        feats["syno_s"] = output["syno_s"].float().cpu().squeeze(0)
    if "syno_t" in output:
        feats["syno_t"] = output["syno_t"].float().cpu().squeeze(0)
    if "logits" in output:
        feats["logits"] = output["logits"].float().cpu().squeeze(0)
    return feats


def get_cropped_videos():
    cropped = DFD_V11 / "cropped" / "videos"
    videos = []
    real_dir = cropped / "real"
    if real_dir.exists():
        for v in sorted(real_dir.glob("*.avi")):
            videos.append((v, "real", "real"))
    fake_dir = cropped / "fake"
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
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(DFD_FCG_ROOT / "checkpoint" / "weights.ckpt"),
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model = load_paper_model(args.ckpt, device)

    videos = get_cropped_videos()
    if not videos:
        print("No cropped videos found. Run face cropping first.")
        return

    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    print(f"[Shard {args.shard}/{args.num_shards}] {len(videos)} cropped videos on GPU {args.gpu}")

    for vid_path, label_type, gen_name in tqdm(videos, desc="Paper"):
        if label_type == "real":
            out_dir = DFD_V11 / "paper_features" / "real"
        else:
            out_dir = DFD_V11 / "paper_features" / "fake" / gen_name

        out_path = out_dir / f"{vid_path.stem}.pt"
        if out_path.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        frames = read_video_frames(str(vid_path))
        if frames is None:
            continue

        feats = extract_features(model, frames, device)
        if feats:
            torch.save(feats, out_path)

    print(f"[Shard {args.shard}] Done.")


if __name__ == "__main__":
    main()
