#!/usr/bin/env python3
"""
End-to-end deepfake detection inference.

Pipeline: raw video → face crop → feature extraction → fusion model → fake probability

Supports single video or folder of videos. Videos should be uncropped —
faces are automatically detected and cropped before feature extraction.

Usage:
  # Single video
  python infer.py --input video.mp4 --config configs/10_all_concat_s3.yaml

  # Folder of videos
  python infer.py --input /path/to/videos/ --config configs/10_all_concat_s3.yaml

  # Specify checkpoint explicitly
  python infer.py --input video.mp4 --config configs/03_pe.yaml \
      --ckpt checkpoints/03_pe/best-epoch=12.ckpt

  # Verify that model+checkpoint reproduce saved evaluation scores
  python infer.py --verify --config configs/10_all_concat_s3.yaml

  # Use pre-cropped videos (skip face detection)
  python infer.py --input video.mp4 --config configs/03_pe.yaml --cropped
"""

import argparse
import inspect
import sys
import json
import cv2
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

DFD_FCG_ROOT = Path("/extra_space2/shykula/gg/Repos/DFD-FCG")
DFD_FCG_CKPT = DFD_FCG_ROOT / "checkpoint" / "weights.ckpt"
PE_ROOT = Path("/extra_space2/shykula/perception_models")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ─── Utilities ────────────────────────────────────────────────────────────────

def read_all_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]
    return frames


def sample_frames(frames, n):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idx]


def uniform_sample_tensor(feat, n):
    t = feat.shape[0]
    if t >= n:
        idx = torch.linspace(0, t - 1, n).long()
        return feat[idx]
    pad = torch.zeros(n - t, *feat.shape[1:], dtype=feat.dtype)
    return torch.cat([feat, pad])


def frames_to_tensor(frames):
    arr = np.stack(frames)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def frames_to_tensor_imagenet(frames):
    tensor = frames_to_tensor(frames)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def collect_videos(path):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        vids = []
        for ext in VIDEO_EXTS:
            vids.extend(sorted(p.glob(f"*{ext}")))
            vids.extend(sorted(p.glob(f"**/*{ext}")))
        return sorted(set(vids))
    raise FileNotFoundError(f"Not found: {path}")


# ─── Face Detection & Cropping ───────────────────────────────────────────────

class FaceCropper:
    """Detect and crop faces using DFD-FCG alignment pipeline.

    Uses face_alignment (SFD) for landmark detection, then DFD-FCG's
    affine-alignment crop to match the training preprocessing exactly.
    Falls back to simple bbox crop if DFD-FCG repo is unavailable.
    """

    _98_TO_68 = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
        77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95,
    ]

    def __init__(self, device="cuda", crop_size=150, target_size=256):
        import face_alignment
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector="sfd",
            device=str(device),
            flip_input=False,
        )
        self.crop_size = crop_size
        self.target_size = target_size

        self._use_dfd_crop = False
        try:
            sys.path.insert(0, str(DFD_FCG_ROOT / "src" / "preprocess"))
            from crop_main_face import get_main_face_data, crop_patch
            self._get_main_face_data = get_main_face_data
            self._crop_patch = crop_patch
            mean_face_path = str(DFD_FCG_ROOT / "misc" / "20words_mean_face.npy")
            self._reference = np.load(mean_face_path)
            self._use_dfd_crop = True
        except Exception:
            print("  WARNING: DFD-FCG crop unavailable, using simple bbox crop")

    def _detect_all_landmarks(self, frames, max_res=800):
        frame_faces = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = min(1.0, max_res / max(h, w))
            inp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
            try:
                results = self.fa.get_landmarks_from_batch(inp_t, return_bboxes=True)
                lms_list, _, bbs_list = results
                if len(lms_list) > 0 and lms_list[0] is not None and len(lms_list[0]) > 0:
                    lms_68 = lms_list[0].reshape(-1, 68, 2) / scale
                    bbs = [bb[:-1] / scale for bb in bbs_list[0]]
                    frame_faces.append({
                        "landmarks": [lm for lm in lms_68],
                        "bboxes": bbs,
                    })
                else:
                    frame_faces.append(None)
            except Exception:
                frame_faces.append(None)
        return frame_faces

    def crop_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames_bgr = []
        frames_rgb = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames_rgb:
            return None

        if self._use_dfd_crop:
            return self._crop_aligned(frames_bgr, frames_rgb, fps)
        return self._crop_simple(frames_rgb)

    def _crop_aligned(self, frames_bgr, frames_rgb, fps):
        """Full DFD-FCG style alignment crop."""
        face_data = self._detect_all_landmarks(frames_rgb)

        frame_landmarks, frame_bboxes = [], []
        for fd in face_data:
            if fd is None:
                frame_landmarks.append([])
                frame_bboxes.append([])
            else:
                lms = [
                    (lm[self._98_TO_68] if len(lm) == 98 else lm)
                    for lm in fd["landmarks"]
                ]
                bbs = [
                    bb.reshape(2, 2) if len(bb.shape) == 1 else bb
                    for bb in fd["bboxes"]
                ]
                frame_landmarks.append(lms)
                frame_bboxes.append(bbs)

        try:
            landmarks, bboxes, indices = self._get_main_face_data(
                frame_landmarks=frame_landmarks,
                frame_bboxes=frame_bboxes,
                d_rate=0.65,
                max_paddings=fps * 3,
            )
        except Exception:
            return self._crop_simple(frames_rgb)

        if len(landmarks) < len(frames_bgr) * 0.5:
            return self._crop_simple(frames_rgb)

        try:
            crop_frames, _, _ = self._crop_patch(
                frames_bgr, landmarks, bboxes, indices, self._reference,
                window_margin=12, start_idx=15, stop_idx=68,
                crop_size=self.crop_size, target_size=self.target_size,
            )
        except Exception:
            return self._crop_simple(frames_rgb)

        return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in crop_frames]

    def _crop_simple(self, frames_rgb):
        """Fallback: simple bbox crop around face."""
        box = None
        for i in range(min(10, len(frames_rgb))):
            preds = self.fa.get_landmarks(frames_rgb[i])
            if preds is not None and len(preds) > 0:
                lms = preds[0]
                x_min, y_min = lms.min(axis=0)
                x_max, y_max = lms.max(axis=0)
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                size = max(x_max - x_min, y_max - y_min) * 1.5
                half = size / 2
                box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
                break
        if box is None:
            return None

        x1, y1, x2, y2 = box
        size = x2 - x1
        cropped = []
        for frame in frames_rgb:
            h, w = frame.shape[:2]
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            sx1, sy1 = max(0, x1), max(0, y1)
            sx2, sy2 = min(w, x2), min(h, y2)
            dx1, dy1 = sx1 - x1, sy1 - y1
            crop[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = frame[sy1:sy2, sx1:sx2]
            cropped.append(cv2.resize(crop, (self.target_size, self.target_size)))
        return cropped


# ─── Feature Extractors ──────────────────────────────────────────────────────

class PaperExtractor:
    """DFD-FCG spatial+temporal synopsis → (1280,) vector."""

    def __init__(self, device, ckpt_path=None):
        import os
        if not DFD_FCG_ROOT.is_dir():
            raise RuntimeError(
                f"DFD-FCG repo not found at {DFD_FCG_ROOT}.\n"
                "Paper branch requires it. Use a config without use_paper, "
                "or clone the repo.")
        sys.path.insert(0, str(DFD_FCG_ROOT))
        ckpt = ckpt_path or str(DFD_FCG_CKPT)

        prev_cwd = os.getcwd()
        os.chdir(str(DFD_FCG_ROOT))
        try:
            from src.model.clip.svl import FFGSynoVideoLearner
            raw = torch.load(ckpt, map_location="cpu")
            hp = raw["hyper_parameters"]
            valid = set(inspect.signature(FFGSynoVideoLearner.__init__).parameters) - {"self"}
            self.model = FFGSynoVideoLearner(**{k: v for k, v in hp.items() if k in valid})
            self.model.load_state_dict(raw["state_dict"], strict=False)
            self.model = self.model.to(device).eval()
        finally:
            os.chdir(prev_cwd)
        self.device = device

    @torch.no_grad()
    def __call__(self, cropped_frames):
        tensor = frames_to_tensor(cropped_frames).to(self.device)
        if tensor.shape[-1] != self.model.n_px:
            tensor = F.interpolate(tensor, size=(self.model.n_px, self.model.n_px),
                                   mode="bilinear", align_corners=False)
        T = 10
        if tensor.shape[0] > T:
            idx = torch.linspace(0, tensor.shape[0] - 1, T).long()
            tensor = tensor[idx]
        elif tensor.shape[0] < T:
            tensor = torch.cat([tensor, tensor[-1:].repeat(T - tensor.shape[0], 1, 1, 1)])

        x = self.model.transform(tensor.unsqueeze(0))
        with torch.amp.autocast("cuda"):
            out = self.model(x)

        syno_s = out.get("syno_s", torch.zeros(1, 1024)).float().cpu().squeeze(0)
        syno_t = out.get("syno_t", torch.zeros(1, 256)).float().cpu().squeeze(0)
        return torch.cat([syno_s.flatten(), syno_t.flatten()])


class DepthExtractor:
    """Depth-Anything-V2-Base DINOv2 backbone → (T, 768) CLS features."""

    def __init__(self, device):
        from transformers import AutoModelForDepthEstimation
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf")
        self.backbone = model.backbone.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def __call__(self, cropped_frames, n_frames=10):
        sampled = sample_frames(cropped_frames, n_frames)
        tensor = frames_to_tensor_imagenet(sampled)
        tensor = F.interpolate(tensor, size=(518, 518), mode="bilinear", align_corners=False)
        tensor = tensor.to(self.device)
        with torch.amp.autocast("cuda"):
            out = self.backbone(tensor, output_hidden_states=True)
        cls_tokens = [h[:, 0, :] for h in out.hidden_states[-4:]]
        features = torch.stack(cls_tokens, dim=1).mean(dim=1).float().cpu()
        return uniform_sample_tensor(features, n_frames)


class PEExtractor:
    """Meta Perception Encoder (PE-Core-L14-336) → (T, 1024) embeddings."""

    def __init__(self, device):
        if not PE_ROOT.is_dir():
            raise RuntimeError(
                f"Perception models repo not found at {PE_ROOT}.\n"
                "PE branch requires it. Use a config without use_pe, "
                "or set up the repo.")
        sys.path.insert(0, str(PE_ROOT))
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        self.model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        self.model = self.model.to(device).eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.device = device

    @torch.no_grad()
    def __call__(self, cropped_frames, n_frames=16):
        from PIL import Image
        sampled = sample_frames(cropped_frames, n_frames)
        tokens = []
        for frame in sampled:
            x = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
            tokens.append(self.model.encode_image(x).float().cpu())
        features = torch.cat(tokens, dim=0)
        return uniform_sample_tensor(features, n_frames)


class CTExtractor:
    """CoTracker3 landmark tracking → (tracks, visibility) raw tensors."""

    def __init__(self, device):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.model = self.model.to(device).eval()
        self.device = device
        import face_alignment
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False)

    @torch.no_grad()
    def __call__(self, cropped_frames, max_frames=100):
        sampled = sample_frames(cropped_frames, max_frames)
        video = torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2).float().unsqueeze(0)

        first = sampled[0]
        preds = self.fa.get_landmarks(first)
        if preds is None or len(preds) == 0:
            return None, None

        landmarks = torch.from_numpy(preds[0]).float()
        t_col = torch.zeros(landmarks.shape[0], 1)
        queries = torch.cat([t_col, landmarks], dim=1).unsqueeze(0).to(self.device)

        pred = self.model(video.to(self.device), queries=queries)
        return pred[0][0].cpu(), pred[1][0].cpu()


# ─── CT Feature Processing (matches data/dataset.py) ─────────────────────────

def compute_kinematics(pos, vis):
    T = pos.shape[0]
    vel = torch.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = torch.zeros_like(pos)
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    centroid = pos.mean(dim=1, keepdim=True)
    cdelta = (pos - centroid).norm(dim=-1, keepdim=True)
    cdelta_dt = torch.zeros_like(cdelta)
    cdelta_dt[1:] = cdelta[1:] - cdelta[:-1]
    return torch.cat([pos, vel, acc, cdelta_dt, vis.unsqueeze(-1)], dim=-1)


def process_ct_features(tracks, visibility, cfg):
    """Convert raw CoTracker output to model-ready tensor (matching _load_ct)."""
    ct_n_windows = cfg.get("ct_n_windows", 0)
    ct_window_size = cfg.get("ct_window_size", 10)
    ct_stride = cfg.get("ct_stride", 1)
    ct_max_track_frames = cfg.get("ct_max_track_frames", 0)
    n_frames = cfg.get("n_frames", 10)

    raw_tracks = tracks.float()
    raw_vis = visibility.float()

    if ct_max_track_frames > 0 and raw_tracks.shape[0] > ct_max_track_frames:
        raw_tracks = raw_tracks[:ct_max_track_frames]
        raw_vis = raw_vis[:ct_max_track_frames]

    pos_norm = raw_tracks / 75.0 - 1.0

    if ct_n_windows > 0:
        T = pos_norm.shape[0]
        raw_span = ct_window_size * ct_stride
        needed = max(raw_span, 1)

        if T < needed:
            pad_len = needed - T
            pos_norm = torch.cat([pos_norm, torch.zeros(pad_len, *pos_norm.shape[1:])])
            raw_vis = torch.cat([raw_vis, torch.zeros(pad_len, *raw_vis.shape[1:])])
            T = needed

        total_raw = ct_n_windows * raw_span
        if T <= total_raw:
            anchors = torch.linspace(0, max(T - raw_span, 0), ct_n_windows).long()
        else:
            anchors = torch.linspace(0, T - raw_span, ct_n_windows).long()

        window_feats = []
        for a in anchors:
            idx = torch.arange(ct_window_size) * ct_stride + a
            window_feats.append(compute_kinematics(pos_norm[idx], raw_vis[idx]))
        return torch.cat(window_feats, dim=0)
    else:
        sampled_pos = uniform_sample_tensor(pos_norm, n_frames)
        sampled_vis = uniform_sample_tensor(raw_vis, n_frames)
        return compute_kinematics(sampled_pos, sampled_vis)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_detector(cfg, ckpt_path, device):
    from train_exp import ExpDetector
    try:
        m = ExpDetector.load_from_checkpoint(str(ckpt_path), cfg=cfg)
        return m.eval().to(device)
    except Exception:
        pass
    from models.detector import DeepfakeDetector
    return DeepfakeDetector.load_from_checkpoint(
        str(ckpt_path), cfg=cfg).eval().to(device)


def find_checkpoint(cfg):
    name = cfg["name"]
    ckpt_dir = BASE / "checkpoints" / name
    ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.is_dir() else []
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. "
            f"Train first or pass --ckpt explicitly.")
    return ckpts[0]


# ─── Main Inference Pipeline ─────────────────────────────────────────────────

class DeepfakeInference:
    """End-to-end deepfake inference from raw (uncropped) videos."""

    def __init__(self, cfg, ckpt_path=None, device="cuda:0", paper_ckpt=None):
        self.cfg = cfg
        self.device = device
        self.branches = [b for b in ["paper", "depth", "pe", "ct"]
                         if cfg.get(f"use_{b}")]
        self.n_frames = cfg.get("n_frames", 10)
        self.pe_frames = cfg.get("pe_frames", 16)

        print(f"Active branches: {self.branches}")
        print(f"Loading extraction models...")

        self.extractors = {}
        if "paper" in self.branches:
            print("  Loading Paper (DFD-FCG)...")
            self.extractors["paper"] = PaperExtractor(device, paper_ckpt)
            self._offload("paper")
        if "depth" in self.branches:
            print("  Loading Depth (DINOv2-Base)...")
            self.extractors["depth"] = DepthExtractor(device)
            self._offload("depth")
        if "pe" in self.branches:
            print("  Loading PE (Perception Encoder)...")
            self.extractors["pe"] = PEExtractor(device)
            self._offload("pe")
        if "ct" in self.branches:
            print("  Loading CoTracker3...")
            self.extractors["ct"] = CTExtractor(device)
            self._offload("ct")

        ckpt = ckpt_path or find_checkpoint(cfg)
        print(f"Loading detector from {ckpt}...")
        self.model = load_detector(cfg, ckpt, device)
        print("Ready.\n")

    def _offload(self, name):
        """Move an extractor to CPU to free GPU memory."""
        ext = self.extractors.get(name)
        if ext is None:
            return
        if hasattr(ext, "model"):
            ext.model.cpu()
        if hasattr(ext, "backbone"):
            ext.backbone.cpu()
        torch.cuda.empty_cache()

    def _reload(self, name):
        """Move an extractor back to GPU."""
        ext = self.extractors.get(name)
        if ext is None:
            return
        if hasattr(ext, "model"):
            ext.model.to(self.device)
        if hasattr(ext, "backbone"):
            ext.backbone.to(self.device)

    @torch.no_grad()
    def predict(self, cropped_frames):
        """Run inference on pre-cropped face frames. Returns fake probability."""
        feats = {}

        if "paper" in self.branches:
            self._reload("paper")
            feats["paper"] = self.extractors["paper"](cropped_frames)
            self._offload("paper")

        if "depth" in self.branches:
            self._reload("depth")
            feats["depth"] = self.extractors["depth"](cropped_frames, self.n_frames)
            self._offload("depth")

        if "pe" in self.branches:
            self._reload("pe")
            feats["pe"] = self.extractors["pe"](cropped_frames, self.pe_frames)
            self._offload("pe")

        if "ct" in self.branches:
            self._reload("ct")
            tracks, vis = self.extractors["ct"](cropped_frames)
            self._offload("ct")
            if tracks is not None:
                feats["ct"] = process_ct_features(tracks, vis, self.cfg)
            else:
                ct_out = self.cfg.get("ct_n_windows", 0) * self.cfg.get("ct_window_size", 10)
                ct_out = ct_out if ct_out > 0 else self.n_frames
                feats["ct"] = torch.zeros(ct_out, 68, 8)

        batch = {k: v.unsqueeze(0).to(self.device) for k, v in feats.items()}
        logits = self.model(batch)
        prob = F.softmax(logits, dim=-1)[0, 1].item()
        return prob

    def run(self, video_paths, cropper=None):
        """Run on a list of video paths. Returns list of (path, prob) tuples."""
        results = []
        for vp in video_paths:
            vp = Path(vp)
            print(f"Processing {vp.name}...", end=" ", flush=True)

            if cropper is not None:
                frames = cropper.crop_video(vp)
            else:
                frames = read_all_frames(vp)

            if frames is None or len(frames) == 0:
                print("SKIPPED (no frames/face)")
                results.append((str(vp), None))
                continue

            prob = self.predict(frames)
            label = "FAKE" if prob > 0.5 else "REAL"
            print(f"{prob:.4f} ({label})")
            results.append((str(vp), prob))
        return results


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_scores(cfg_path, device="cuda:0"):
    """Re-run evaluation using pre-extracted features and compare to saved results."""
    from evaluate import load_and_score, compute_metrics

    print(f"Running evaluation for {cfg_path}...")
    scores, _, name = load_and_score(cfg_path, device)
    metrics = compute_metrics(scores)

    results_path = BASE / "results" / "final_summary.json"
    if not results_path.exists():
        print(f"No saved results at {results_path}")
        return metrics

    with open(results_path) as f:
        saved = json.load(f)

    if name not in saved:
        print(f"No saved results for '{name}'")
        return metrics

    saved_m = saved[name]
    print(f"\n{'Dataset':<15} {'Computed':>10} {'Saved':>10} {'Delta':>8}")
    print("-" * 48)
    all_match = True
    for key in sorted(metrics):
        if isinstance(metrics[key], dict):
            computed = metrics[key]["auc"]
            saved_val = saved_m[key]["auc"]
        else:
            computed = metrics[key]
            saved_val = saved_m[key]
        delta = abs(computed - saved_val)
        ok = delta < 0.005
        all_match = all_match and ok
        mark = "OK" if ok else "DIFF"
        print(f"  {key:<13} {computed:>10.4f} {saved_val:>10.4f} {delta:>7.4f} {mark}")

    print(f"\n{'ALL SCORES MATCH' if all_match else 'MISMATCH DETECTED'}")
    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end deepfake detection inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str,
                        help="Path to a video file or directory of videos")
    parser.add_argument("--config", type=str, default=str(BASE / "configs" / "10_all_concat_s3.yaml"),
                        help="Model config YAML (default: 10_all_concat_s3)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (auto-detected from config name if omitted)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cropped", action="store_true",
                        help="Input videos are already face-cropped (skip detection)")
    parser.add_argument("--paper-ckpt", type=str, default=None,
                        help="DFD-FCG checkpoint path (for paper branch)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verify", action="store_true",
                        help="Verify scores against saved results (uses pre-extracted features)")
    parser.add_argument("--verify-all", action="store_true",
                        help="Verify all configs against saved results")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.verify_all:
        results_path = BASE / "results" / "final_summary.json"
        with open(results_path) as f:
            saved = json.load(f)
        for cfg_path in sorted((BASE / "configs").glob("*.yaml")):
            try:
                verify_scores(str(cfg_path), device)
                print()
            except Exception as e:
                print(f"SKIP {cfg_path.stem}: {e}\n")
        return

    if args.verify:
        verify_scores(args.config, device)
        return

    if not args.input:
        parser.error("--input is required for inference (or use --verify)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    videos = collect_videos(args.input)
    if not videos:
        print(f"No videos found at {args.input}")
        return

    print(f"Config: {cfg['name']}")
    print(f"Videos: {len(videos)}")
    print()

    pipeline = DeepfakeInference(
        cfg, ckpt_path=args.ckpt, device=device, paper_ckpt=args.paper_ckpt)

    cropper = None
    if not args.cropped:
        print("Initializing face detector...")
        cropper = FaceCropper(device=device)

    results = pipeline.run(videos, cropper=cropper)

    n_fake = sum(1 for _, p in results if p is not None and p > args.threshold)
    n_real = sum(1 for _, p in results if p is not None and p <= args.threshold)
    n_skip = sum(1 for _, p in results if p is None)
    print(f"\nSummary: {n_fake} fake, {n_real} real, {n_skip} skipped "
          f"(threshold={args.threshold})")

    if args.output:
        out_data = []
        for path, prob in results:
            entry = {"video": path, "fake_probability": prob}
            if prob is not None:
                entry["prediction"] = "FAKE" if prob > args.threshold else "REAL"
            out_data.append(entry)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
