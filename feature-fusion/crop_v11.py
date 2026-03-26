#!/usr/bin/env python3
"""
Face cropping for DeepFakeDataset v1.1 using DFD-FCG pipeline.

Two-phase pipeline:
  Phase 1: Extract landmarks + bboxes (face_alignment) → frame_data/*.pickle
  Phase 2: Crop faces → cropped/videos/*.avi

Supports GPU sharding for Phase 1 (GPU-bound face detection).

Usage:
  python crop_v11.py --phase landmarks --gpu 0 --shard 0 --num_shards 3
  python crop_v11.py --phase crop --workers 8
  python crop_v11.py --phase both --gpu 0  # single GPU, both phases
"""
import os
import sys
import cv2
import math
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob

DFD_V11 = Path("/extra_space2/shykula/gg/datasets/deepfake_v1.1")
DFD_FCG = Path("/extra_space2/shykula/gg/Repos/DFD-FCG")
MEAN_FACE = str(DFD_FCG / "misc" / "20words_mean_face.npy")

sys.path.insert(0, str(DFD_FCG / "src" / "preprocess"))


def get_all_videos():
    videos = []
    real_dir = DFD_V11 / "videos" / "real"
    if real_dir.exists():
        for v in sorted(real_dir.glob("*.mp4")):
            videos.append(v)
    fake_dir = DFD_V11 / "videos" / "fake"
    if fake_dir.exists():
        for gen_dir in sorted(fake_dir.iterdir()):
            if gen_dir.is_dir():
                for v in sorted(gen_dir.glob("*.mp4")):
                    videos.append(v)
    return videos


def video_to_fdata_path(video_path):
    rel = video_path.relative_to(DFD_V11 / "videos")
    return DFD_V11 / "frame_data" / rel.with_suffix(".pickle")


def video_to_crop_path(video_path):
    rel = video_path.relative_to(DFD_V11 / "videos")
    return DFD_V11 / "cropped" / "videos" / rel.with_suffix(".avi")


def video_to_crop_fdata_path(video_path):
    rel = video_path.relative_to(DFD_V11 / "videos")
    return DFD_V11 / "cropped" / "frame_data" / rel.with_suffix(".pickle")


@torch.inference_mode()
def extract_landmarks_batch(videos, gpu, batch_size=1, max_res=800):
    """Phase 1: face_alignment landmark extraction."""
    import face_alignment

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector="sfd",
        dtype=torch.float16,
        flip_input=False,
        device="cuda",
    )

    def driver(x):
        return model.get_landmarks_from_batch(x, return_bboxes=True)

    for video_path in tqdm(videos, desc=f"[GPU {gpu}] Landmarks"):
        fdata_path = video_to_fdata_path(video_path)
        if fdata_path.exists():
            continue

        try:
            cap = cv2.VideoCapture(str(video_path))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            scale = min(1.0, max_res / max(height, width))

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale < 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)
            cap.release()

            if not frames:
                print(f"  SKIP (no frames): {video_path}")
                continue

            frame_faces = [None] * len(frames)
            batch_frames, batch_indices = [], []

            for i, frame in enumerate(frames):
                batch_frames.append(frame)
                batch_indices.append(i)
                if len(batch_frames) == batch_size or i == len(frames) - 1:
                    results = driver(
                        torch.tensor(np.stack(batch_frames).transpose((0, 3, 1, 2)))
                    )
                    for idx, lms, bbs in zip(
                        batch_indices, results[0], results[2]
                    ):
                        if len(lms) > 0:
                            lms_68 = lms.reshape(-1, 68, 2) / scale
                            frame_faces[idx] = {
                                "landmarks": [lm for lm in lms_68],
                                "bboxes": [bb[:-1] / scale for bb in bbs],
                            }
                    batch_frames.clear()
                    batch_indices.clear()

            fdata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fdata_path, "wb") as f:
                pickle.dump(frame_faces, f)

        except Exception as e:
            print(f"  ERROR landmarks {video_path.name}: {e}")


def run_cropping(videos, workers=8):
    """Phase 2: crop faces using DFD-FCG crop_main_face logic."""
    from crop_main_face import (
        get_video_frames,
        get_main_face_data,
        crop_patch,
        save_video,
    )

    reference = np.load(MEAN_FACE)
    crop_size = 150
    target_size = 256
    start_idx = 15
    stop_idx = 68
    window_margin = 12
    d_rate = 0.65
    max_pad_secs = 3
    min_crop_rate = 0.9

    _98_to_68 = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
        77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95,
    ]

    success, fail = 0, 0
    for video_path in tqdm(videos, desc="Cropping"):
        fdata_path = video_to_fdata_path(video_path)
        crop_video_path = video_to_crop_path(video_path)
        crop_fdata_path = video_to_crop_fdata_path(video_path)

        if crop_video_path.exists():
            success += 1
            continue

        if not fdata_path.exists():
            fail += 1
            continue

        try:
            fps, frames = get_video_frames(str(video_path))
            with open(fdata_path, "rb") as f:
                raw_fdata = pickle.load(f)

            if len(frames) != len(raw_fdata):
                print(f"  MISMATCH {video_path.name}: {len(frames)} frames vs {len(raw_fdata)} fdata")
                fail += 1
                continue

            frame_landmarks = []
            frame_bboxes = []
            for fd in raw_fdata:
                if fd is None:
                    frame_landmarks.append([])
                    frame_bboxes.append([])
                else:
                    lms = fd["landmarks"]
                    bbs = fd["bboxes"]
                    lms = [
                        (lm[_98_to_68] if len(lm) == 98 else lm) for lm in lms
                    ]
                    bbs = [
                        bb.reshape(2, 2) if len(bb.shape) == 1 else bb
                        for bb in bbs
                    ]
                    frame_landmarks.append(lms)
                    frame_bboxes.append(bbs)

            landmarks, bboxes, indices = get_main_face_data(
                frame_landmarks=frame_landmarks,
                frame_bboxes=frame_bboxes,
                d_rate=d_rate,
                max_paddings=fps * max_pad_secs,
            )

            if len(landmarks) < len(frames) * min_crop_rate:
                print(f"  LOW-TRACK {video_path.name}: {len(landmarks)}/{len(frames)}")
                fail += 1
                continue

            crop_frames, crop_lms, crop_bbs = crop_patch(
                frames, landmarks, bboxes, indices, reference,
                window_margin=window_margin,
                start_idx=start_idx,
                stop_idx=stop_idx,
                crop_size=crop_size,
                target_size=target_size,
            )

            crop_video_path.parent.mkdir(parents=True, exist_ok=True)
            save_video(str(crop_video_path), crop_frames, fps)

            crop_fdata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(crop_fdata_path, "wb") as f:
                pickle.dump(
                    [
                        dict(landmarks=[lm], bboxes=[bb])
                        for lm, bb in zip(crop_lms, crop_bbs)
                    ],
                    f,
                )
            success += 1

        except Exception as e:
            print(f"  ERROR crop {video_path.name}: {e}")
            fail += 1

    print(f"\nCropping done: {success} success, {fail} fail")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["landmarks", "crop", "both"], default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    all_videos = get_all_videos()
    print(f"Total videos: {len(all_videos)}")

    if args.phase in ("landmarks", "both"):
        shard_size = math.ceil(len(all_videos) / args.num_shards)
        shard_videos = all_videos[args.shard * shard_size : (args.shard + 1) * shard_size]
        print(f"Shard {args.shard}/{args.num_shards}: {len(shard_videos)} videos")
        extract_landmarks_batch(shard_videos, args.gpu)

    if args.phase in ("crop", "both"):
        run_cropping(all_videos, args.workers)


if __name__ == "__main__":
    main()
