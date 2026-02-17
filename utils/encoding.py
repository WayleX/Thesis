"""
FFmpeg video probing and encoding with NVENC GPU acceleration.

Encoding strategy (in order of preference):
  1. Stream-copy   – when only trimming (no resize needed)
  2. h264_nvenc    – GPU-accelerated NVIDIA encoder
  3. libx264       – CPU fallback with Lanczos downscaling
"""

import json
import logging
import subprocess
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def get_video_info(video_path: str) -> tuple[float, int, int]:
    """Extract duration, width, and height via ffprobe.

    Returns (duration_seconds, width, height).
    Returns (0.0, 0, 0) on any failure.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return 0.0, 0, 0

        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        width, height = 0, 0
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                if duration == 0 and "duration" in stream:
                    duration = float(stream["duration"])
                break
        return duration, width, height
    except Exception:
        return 0.0, 0, 0


def encode_video(
    src: str,
    dst: str,
    trim_duration: float | None = None,
    resize: tuple[int, int] | None = None,
) -> bool:
    """Encode a video with optional trim and resize. Returns True on success."""
    needs_reencode = resize is not None
    try:
        if not needs_reencode and trim_duration is not None:
            cmd = [
                "ffmpeg", "-y", "-i", src,
                "-t", str(trim_duration),
                "-c", "copy", "-avoid_negative_ts", "make_zero", dst,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return True
            needs_reencode = True

        encoders = [
            ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"]),
            ("libx264",    ["-preset", "fast", "-crf", "18"]),
        ]
        for enc, extra in encoders:
            cmd = ["ffmpeg", "-y", "-i", src]
            if trim_duration is not None:
                cmd += ["-t", str(trim_duration)]
            if resize is not None:
                w, h = resize
                scale = f"scale={w}:{h}"
                if enc == "libx264":
                    scale += ":flags=lanczos"
                cmd += ["-vf", scale]
            cmd += ["-c:v", enc] + extra + ["-c:a", "aac", "-b:a", "192k", dst]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return True

        return False
    except Exception:
        return False


def copy_video(src: str, dst: str) -> None:
    """Simple file copy when no re-encoding is needed."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
