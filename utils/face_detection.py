"""
MTCNN-based GPU-accelerated face detection for video frames.

Determines whether a video contains exactly one person by sampling
several frames and applying a majority-vote heuristic.
"""

import logging
from collections import Counter

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from config import FACE_MIN_DETECTION_CONFIDENCE

log = logging.getLogger(__name__)

SINGLE_FACE_THRESHOLD = 0.8


def init_detector(device: str | None = None) -> MTCNN:
    """Initialise MTCNN face detector on the specified device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading MTCNN face detector on %s …", device.upper())
    return MTCNN(keep_all=True, device=device, post_process=False)


def count_faces_in_frame(
    detector: MTCNN,
    frame_rgb: np.ndarray,
    min_confidence: float = FACE_MIN_DETECTION_CONFIDENCE,
) -> int:
    """Count high-confidence faces in a single RGB frame."""
    pil_img = Image.fromarray(frame_rgb)
    boxes, probs = detector.detect(pil_img)
    if boxes is None:
        return 0
    return int((probs >= min_confidence).sum())


def classify_face_count(
    detector: MTCNN,
    frames_rgb: list[np.ndarray],
    min_confidence: float = FACE_MIN_DETECTION_CONFIDENCE,
) -> int:
    """Determine the dominant face count across sampled frames.
    """
    if not frames_rgb:
        return -1

    counts = [count_faces_in_frame(detector, f, min_confidence) for f in frames_rgb]

    if any(c >= 2 for c in counts):
        return max(counts)

    if sum(1 for c in counts if c == 1) / len(counts) >= SINGLE_FACE_THRESHOLD:
        return 1

    dominant, _ = Counter(counts).most_common(1)[0]
    return dominant
