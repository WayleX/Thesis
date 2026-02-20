"""
Video file discovery and frame extraction utilities.

Provides three frame-extraction strategies for different consumers:
  - BGR numpy arrays  → text/face detection (OpenCV-native)
  - RGB numpy arrays  → MTCNN (expects RGB input)
  - PIL Images        → Gemini API (serialised to JPEG bytes)
  - Save-to-disk      → final dataset frames output
"""

import os
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config import VIDEO_EXTENSIONS


# ── Discovery ─────────────────────────────────────────────────────────────────

def find_videos(root: str, extensions: set[str] | None = None) -> list[Path]:
    """Recursively find all video files under *root*, sorted by path."""
    exts = extensions or VIDEO_EXTENSIONS
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                videos.append(Path(dirpath) / fname)
    videos.sort()
    return videos


# ── Frame extraction ──────────────────────────────────────────────────────────

def _compute_indices_evenly_spaced(
    total_frames: int, n_frames: int, max_frame: int | None = None,
) -> list[int]:
    """Compute evenly-spaced frame indices within the usable range.

    With *max_frame* set, indices are restricted to [0, max_frame).
    Uses (n+1)-way division so the first and last indices sit inside the range
    rather than on the boundary.
    """
    limit = min(total_frames, max_frame) if max_frame else total_frames
    if limit <= 0:
        return []
    step = max(1, limit // (n_frames + 1))
    indices = [step * (i + 1) for i in range(n_frames)]
    return [idx for idx in indices if idx < limit]


def _compute_indices_fixed_positions(total_frames: int, n_frames: int = 4) -> list[int]:
    """Compute fixed-position indices: start, 1/3, 2/3, end.

    Falls back to evenly-spaced when n_frames != 4.
    """
    if n_frames == 4:
        return [0, total_frames // 3, 2 * total_frames // 3, max(0, total_frames - 1)]
    step = max(1, (total_frames - 1) // (n_frames - 1))
    return [min(step * i, total_frames - 1) for i in range(n_frames)]


def extract_frames_bgr(
    video_path: str,
    n_frames: int = 5,
    max_seconds: float | None = None,
) -> list[np.ndarray]:
    """Extract evenly-spaced BGR frames (for text/face detection).

    When *max_seconds* is set, only the initial window is sampled.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        return []

    max_frame = int(fps * max_seconds) if max_seconds else None
    indices = _compute_indices_evenly_spaced(total, n_frames, max_frame)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def extract_frames_rgb(
    video_path: str,
    n_frames: int = 5,
    max_seconds: float | None = None,
) -> list[np.ndarray]:
    """Extract evenly-spaced RGB frames (e.g. for MTCNN face detection)."""
    bgr = extract_frames_bgr(video_path, n_frames, max_seconds)
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr]


def extract_frames_pil(
    video_path: str, n_frames: int = 4,
) -> list[Image.Image]:
    """Extract fixed-position frames as PIL Images (for Gemini API).

    Samples at: beginning, 1/3, 2/3, end (when n_frames == 4).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = _compute_indices_fixed_positions(total, n_frames)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def extract_and_save_frames(
    video_path: str, out_dir: str, n_frames: int = 4,
) -> list[str]:
    """Extract key frames and save as JPEG. Returns saved file paths.

    Uses fixed-position sampling (start, 1/3, 2/3, end) for consistency
    with the Gemini prompt inputs and generation pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = _compute_indices_fixed_positions(total, n_frames)
    os.makedirs(out_dir, exist_ok=True)

    saved: list[str] = []
    stem = Path(video_path).stem
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fpath = os.path.join(out_dir, f"{stem}_frame{i}.jpg")
        cv2.imwrite(fpath, frame)
        saved.append(fpath)
    cap.release()
    return saved


# ── Image serialisation ───────────────────────────────────────────────────────

def pil_to_bytes(
    img: Image.Image, fmt: str = "JPEG", quality: int = 85,
) -> bytes:
    """Convert a PIL Image to JPEG bytes for API transmission."""
    buf = BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()
