"""
EasyOCR-based text overlay detection for video frames.

Detects persistent overlays such as news tickers, subtitles, lower
thirds, and watermarks.  Used both in the main filter pipeline and
in the standalone detect_text / resample_clean scripts.
"""

import logging

import easyocr
import numpy as np

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3
MIN_WORD_LENGTH = 2


def init_reader(
    languages: list[str] | None = None, gpu: bool = True,
) -> easyocr.Reader:
    """Initialise and return an EasyOCR reader."""
    langs = languages or ["en"]
    log.info("Loading EasyOCR (languages=%s, gpu=%s) …", langs, gpu)
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    log.info("EasyOCR ready.")
    return reader


def count_text_chars(reader: easyocr.Reader, frame: np.ndarray) -> int:
    """Count recognised text characters in a single frame.

    Only counts words with confidence >= 0.3 and length >= 2 to filter
    out false positives from noise or simple patterns.
    """
    results = reader.readtext(frame, detail=1, paragraph=False)
    return sum(
        len(text.strip())
        for _, text, conf in results
        if conf >= CONFIDENCE_THRESHOLD and len(text.strip()) >= MIN_WORD_LENGTH
    )


def has_text_in_frames(
    reader: easyocr.Reader,
    frames: list[np.ndarray],
    min_chars: int = 3,
    min_frames: int = 1,
) -> bool:
    """Return True if text overlays are detected in enough sampled frames.

    A frame is flagged when it contains >= *min_chars* of recognised text.
    The video is flagged when >= *min_frames* are individually flagged.
    """
    flagged = 0
    for frame in frames:
        if count_text_chars(reader, frame) >= min_chars:
            flagged += 1
            if flagged >= min_frames:
                return True
    return False


def check_video_for_text(
    reader: easyocr.Reader,
    frames: list[np.ndarray],
    min_chars: int = 3,
    min_frames: int = 1,
) -> dict:
    """Produce a detailed text-detection report for one video's frames.

    Returns a dict with keys: has_text, frames_checked,
    frames_with_text, total_chars, detected_texts.
    """
    if not frames:
        return {
            "has_text": False, "frames_checked": 0,
            "frames_with_text": 0, "total_chars": 0, "detected_texts": "",
        }

    frames_with_text = 0
    total_chars = 0
    all_texts: list[str] = []

    for frame in frames:
        results = reader.readtext(frame, detail=1, paragraph=False)
        recognised = [
            (text.strip(), conf)
            for _, text, conf in results
            if conf >= CONFIDENCE_THRESHOLD and len(text.strip()) >= MIN_WORD_LENGTH
        ]
        chars = sum(len(t) for t, _ in recognised)
        total_chars += chars
        if chars >= min_chars:
            frames_with_text += 1
            all_texts.extend(t for t, _ in recognised)

    unique_texts = list(dict.fromkeys(all_texts))[:10]
    return {
        "has_text": frames_with_text >= min_frames,
        "frames_checked": len(frames),
        "frames_with_text": frames_with_text,
        "total_chars": total_chars,
        "detected_texts": " | ".join(unique_texts),
    }
