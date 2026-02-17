#!/usr/bin/env python3
"""
Filter videos by resolution, duration, text overlays, and face count.
This are main specs that are needed to generate a final dataset for deepfake detection research.
Meaning we need one person in the video, and the video needs to be at least 5 seconds long.
We also need to remove any videos with text as it is can be processed after.
We also need to remove any videos that are not in the allowed resolutions.
Copies elligible video to the output directory.

Usage:
    python filter_videos.py 
    --input /path/to/source_videos 
    --output /path/to/filtered_output 
    --trim 
    --trim-duration 5 
    --resize 1280x720 
    --report filter_report.csv
    --probe-workers 16 # adjust based on your system pls
    --encode-workers 4 # adjust based on your system pls
"""

import sys
import csv
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from config import (
    ALLOWED_RESOLUTIONS, MIN_DURATION_SEC, REQUIRED_PERSON_COUNT,
    FACE_SAMPLE_FRAMES, TEXT_MIN_CHARS_PER_FRAME, TEXT_MIN_FRAMES_WITH_TEXT,
)
from utils.video import find_videos, extract_frames_rgb
from utils.encoding import get_video_info, encode_video, copy_video
from utils.text_detection import init_reader, has_text_in_frames
from utils.face_detection import init_detector, classify_face_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Filter videos by resolution, duration, text, and face count.",
    )
    parser.add_argument("--input", required=True, help="Source video directory")
    parser.add_argument("--output", required=True, help="Filtered output directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip encoding step")
    parser.add_argument("--trim", action="store_true", help="Trim to --trim-duration")
    parser.add_argument("--trim-duration", type=float, default=8.0)
    parser.add_argument("--resize", type=str, default=None, metavar="WxH",
                        help="Resize output (e.g. 1280x720)")
    parser.add_argument("--report", default="filter_report.csv")
    parser.add_argument("--probe-workers", type=int, default=15)
    parser.add_argument("--encode-workers", type=int, default=6)
    args = parser.parse_args()

    target_resize: tuple[int, int] | None = None
    if args.resize:
        try:
            parts = args.resize.lower().split("x")
            target_resize = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            parser.error(f"Invalid --resize format '{args.resize}'.")

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        log.error("Input directory does not exist: %s", input_root)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Input       : %s", input_root)
    log.info("Output      : %s", output_root)
    log.info("Resolutions : %s", ALLOWED_RESOLUTIONS)
    log.info("Min duration: %.1f s", MIN_DURATION_SEC)
    log.info("Persons     : %d", REQUIRED_PERSON_COUNT)
    log.info("Trim        : %s", f"{args.trim_duration}s" if args.trim else "off")
    log.info("Resize      : %s",
             f"{target_resize[0]}x{target_resize[1]}" if target_resize else "original")
    log.info("Text detect : EasyOCR (min %d chars)", TEXT_MIN_CHARS_PER_FRAME)
    log.info("Face detect : MTCNN on %s", device.upper())
    log.info("Dry run     : %s", args.dry_run)

    t0 = time.time()

    all_videos = find_videos(str(input_root))
    log.info("Found %d video files.", len(all_videos))

    log.info("Phase 1/3: probing metadata (%d threads) …", args.probe_workers)
    
    metadata: dict[str, tuple[float, int, int]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.probe_workers) as pool:
        futures = {pool.submit(get_video_info, str(v)): v for v in all_videos}
        for fut in as_completed(futures):
            dur, w, h = fut.result()
            path = str(futures[fut])
            metadata[path] = (dur, w, h)
            done += 1
            if done % 500 == 0 or done == len(all_videos):
                log.info("  probed %d / %d", done, len(all_videos))

    candidates: list[Path] = []
    report_rows: list[dict] = []
    stats = {
        "total": len(all_videos), "pass_resolution": 0, "pass_duration": 0,
        "pass_text": 0, "pass_people": 0, "copied": 0, "failed": 0,
    }

    for vpath in all_videos:
        dur, w, h = metadata[str(vpath)]
        rel = str(vpath.relative_to(input_root))
        row = {
            "file": rel, "duration": f"{dur:.2f}", "width": w, "height": h,
            "people": "", "resolution_ok": False, "duration_ok": False,
            "text_ok": False, "people_ok": False, "copied": False, "reason": "",
        }

        if (w, h) not in ALLOWED_RESOLUTIONS:
            row["reason"] = f"resolution {w}x{h}"
            report_rows.append(row)
            continue
        row["resolution_ok"] = True
        stats["pass_resolution"] += 1

        if dur < MIN_DURATION_SEC:
            row["reason"] = f"duration {dur:.2f}s < {MIN_DURATION_SEC}s"
            report_rows.append(row)
            continue
        row["duration_ok"] = True
        stats["pass_duration"] += 1
        candidates.append(vpath)

    log.info(
        "Phase 1 done in %.0fs: %d pass res, %d pass dur → %d candidates.",
        time.time() - t0, stats["pass_resolution"],
        stats["pass_duration"], len(candidates),
    )

    t2 = time.time()
    log.info("Phase 2/3: text + face detection on %d videos …", len(candidates))

    ocr_reader = init_reader(gpu=torch.cuda.is_available())
    detector = init_detector(device)
    check_seconds = args.trim_duration if args.trim else None

    to_encode: list[Path] = []
    for idx, vpath in enumerate(candidates):
        key = str(vpath)
        rel = str(vpath.relative_to(input_root))
        dur, w, h = metadata[key]
        base_row = {
            "file": rel, "duration": f"{dur:.2f}", "width": w, "height": h,
            "people": "", "resolution_ok": True, "duration_ok": True,
            "text_ok": False, "people_ok": False, "copied": False, "reason": "",
        }

        frames = extract_frames_rgb(key, FACE_SAMPLE_FRAMES, check_seconds)
        if not frames:
            base_row["reason"] = "could not extract frames"
            report_rows.append(base_row)
            continue

        if has_text_in_frames(ocr_reader, frames,
                              TEXT_MIN_CHARS_PER_FRAME, TEXT_MIN_FRAMES_WITH_TEXT):
            base_row["reason"] = "text_detected"
            report_rows.append(base_row)
            if (idx + 1) % 200 == 0:
                log.info("  checked %d / %d", idx + 1, len(candidates))
            continue
        base_row["text_ok"] = True
        stats["pass_text"] += 1

        face_count = classify_face_count(detector, frames)
        base_row["people"] = face_count
        if face_count == REQUIRED_PERSON_COUNT:
            base_row["people_ok"] = True
            stats["pass_people"] += 1
            to_encode.append(vpath)
        else:
            base_row["reason"] = f"people={face_count}, need {REQUIRED_PERSON_COUNT}"
            report_rows.append(base_row)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(candidates):
            log.info(
                "  checked %d / %d  (text_ok=%d, face_ok=%d)",
                idx + 1, len(candidates), stats["pass_text"], stats["pass_people"],
            )

    del detector
    torch.cuda.empty_cache()
    log.info(
        "Phase 2 done in %.0fs: %d pass text, %d pass face → %d to encode.",
        time.time() - t2, stats["pass_text"], stats["pass_people"], len(to_encode),
    )

    t3 = time.time()
    need_ffmpeg = args.trim or target_resize is not None

    if args.dry_run:
        log.info("Phase 3/3: dry run — skipping encoding.")
        for vpath in to_encode:
            rel = str(vpath.relative_to(input_root))
            dur, w, h = metadata[str(vpath)]
            extras = []
            if args.trim:
                extras.append(f"trimmed {args.trim_duration}s")
            if target_resize:
                extras.append(f"resized {target_resize[0]}x{target_resize[1]}")
            report_rows.append({
                "file": rel, "duration": f"{dur:.2f}", "width": w, "height": h,
                "people": 1, "resolution_ok": True, "duration_ok": True,
                "text_ok": True, "people_ok": True, "copied": True,
                "reason": "PASS" + (f" ({', '.join(extras)})" if extras else ""),
            })
            stats["copied"] += 1
    elif need_ffmpeg:
        log.info(
            "Phase 3/3: encoding %d videos (%d threads) …",
            len(to_encode), args.encode_workers,
        )
        encode_jobs: list[tuple[str, str]] = []
        for vpath in to_encode:
            rel = vpath.relative_to(input_root)
            dest = output_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            encode_jobs.append((str(vpath), str(dest)))

        done = 0
        with ThreadPoolExecutor(max_workers=args.encode_workers) as pool:
            futures = {
                pool.submit(
                    encode_video, src, dst,
                    trim_duration=args.trim_duration if args.trim else None,
                    resize=target_resize,
                ): src
                for src, dst in encode_jobs
            }
            for fut in as_completed(futures):
                success = fut.result()
                src_path = futures[fut]
                done += 1
                rel = str(Path(src_path).relative_to(input_root))
                dur, w, h = metadata[src_path]
                if success:
                    extras = []
                    if args.trim:
                        extras.append(f"trimmed {args.trim_duration}s")
                    if target_resize:
                        extras.append(f"resized {target_resize[0]}x{target_resize[1]}")
                    report_rows.append({
                        "file": rel, "duration": f"{dur:.2f}", "width": w, "height": h,
                        "people": 1, "resolution_ok": True, "duration_ok": True,
                        "text_ok": True, "people_ok": True, "copied": True,
                        "reason": "PASS" + (f" ({', '.join(extras)})" if extras else ""),
                    })
                    stats["copied"] += 1
                else:
                    report_rows.append({
                        "file": rel, "duration": f"{dur:.2f}", "width": w, "height": h,
                        "people": 1, "resolution_ok": True, "duration_ok": True,
                        "text_ok": True, "people_ok": True, "copied": False,
                        "reason": "ENCODE_FAILED",
                    })
                    stats["failed"] += 1
                if done % 100 == 0 or done == len(encode_jobs):
                    log.info("  encoded %d / %d", done, len(encode_jobs))
    else:
        for vpath in to_encode:
            rel = vpath.relative_to(input_root)
            dest = output_root / rel
            copy_video(str(vpath), str(dest))
            dur, w, h = metadata[str(vpath)]
            report_rows.append({
                "file": str(rel), "duration": f"{dur:.2f}", "width": w, "height": h,
                "people": 1, "resolution_ok": True, "duration_ok": True,
                "text_ok": True, "people_ok": True, "copied": True, "reason": "PASS",
            })
            stats["copied"] += 1

    log.info("Phase 3 done in %.0fs.", time.time() - t3)

    report_path = Path(args.report)
    fieldnames = [
        "file", "duration", "width", "height", "people",
        "resolution_ok", "duration_ok", "text_ok", "people_ok", "copied", "reason",
    ]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    log.info("Report: %s", report_path)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.0fs  (%.1f min)", elapsed, elapsed / 60)
    log.info("Total scanned      : %d", stats["total"])
    log.info("Passed resolution  : %d", stats["pass_resolution"])
    log.info("Passed duration    : %d", stats["pass_duration"])
    log.info("Passed text        : %d", stats["pass_text"])
    log.info("Passed face (=%d)  : %d", REQUIRED_PERSON_COUNT, stats["pass_people"])
    log.info("Copied             : %d", stats["copied"])
    log.info("Failed encode      : %d", stats["failed"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
