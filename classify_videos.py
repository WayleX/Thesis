#!/usr/bin/env python3
"""
Classify videos using Gemini 2.5 Flash Lite.

For each video, extracts 4 key frames and asks Gemini to produce:
  - A classification label (Official Statement / Studio Interview /
    Direct-to-Camera Casual / Other)
  - A confidence score (0.0 – 1.0)
  - Demographics: gender, age_range, ethnicity

Features:
  - Exponential backoff on 503 / rate-limit errors (up to 5 retries)
  - Auto-resume: reads existing JSONL and skips already-classified videos
  - Parallel workers for throughput

Usage:
    python classify_videos.py \\
        --input /path/to/filtered_videos \\
        --output classification.csv \\
        --output-json classification.jsonl \\
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from utils.video import find_videos, extract_frames_pil
from utils.io import load_done_files, DualWriter
from utils.gemini import create_client, call_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Gemini system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a video classification expert. You will be given 4 frames extracted \
from a single video clip at evenly spaced intervals (beginning, one-third, \
two-thirds, and end).

Your tasks:

1. **Classification**: Classify the video into exactly ONE of these categories:
   - "Official Statement" — formal government, corporate, or institutional \
announcements delivered at podiums, press rooms, or official settings.
   - "Studio Interview" — news broadcasts, talk-show segments, press-style \
interviews in a professional studio environment.
   - "Direct-to-Camera / Casual" — webcam/phone-call communication, job interviews, \
social-media content, vlogging, and informal speaking scenarios recorded with \
personal equipment or simple setups.
   - "Other" — anything that does not fit the above three categories.

2. **Confidence**: Provide a confidence score between 0.0 and 1.0 for your classification.

3. **Demographics**: Identify the person's apparent:
   - "gender": "Male" or "Female"
   - "age_range": approximate age range, e.g. "25-35", "40-50", "60+"
   - "ethnicity": one of "White", "Black", "Asian", "Latino/Hispanic", \
"Middle Eastern", "Indian/South Asian", or "Other"

Respond ONLY with valid JSON in this exact schema (no markdown, no extra text):
{
  "classification": "...",
  "classification_confidence": 0.0,
  "gender": "...",
  "age_range": "...",
  "ethnicity": "..."
}
"""

CSV_FIELDNAMES = [
    "file", "classification", "classification_confidence",
    "gender", "age_range", "ethnicity"
]


def main():
    parser = argparse.ArgumentParser(
        description="Classify videos with Gemini (retry + auto-resume).",
    )
    parser.add_argument("--input", required=True, help="Root directory of videos")
    parser.add_argument("--output", default="classification_results.csv")
    parser.add_argument("--output-json", default=None,
                        help="JSONL output (also used for auto-resume)")
    parser.add_argument("--limit", type=int, default=0, help="Process first N only")
    parser.add_argument("--workers", type=int, default=5, help="Parallel API threads")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("No API key. Set GEMINI_API_KEY env var.")
        sys.exit(1)

    client = create_client(api_key)
    input_root = Path(args.input).resolve()
    if not input_root.is_dir():
        log.error("Input directory does not exist: %s", input_root)
        sys.exit(1)

    all_videos = find_videos(str(input_root))
    log.info("Found %d video files in %s", len(all_videos), input_root)

    done: set[str] = set()
    if args.output_json and os.path.exists(args.output_json):
        done = load_done_files(args.output_json)
        if done:
            log.info("Auto-resume: %d already classified, skipping.", len(done))

    work = [v for v in all_videos
            if str(v.relative_to(input_root)) not in done]
    if args.limit > 0:
        work = work[: args.limit]
    if not work:
        log.info("Nothing to classify.")
        return

    total_work = len(work)
    log.info("Videos to classify: %d  |  Workers: %d", total_work, args.workers)

    resuming = len(done) > 0
    writer = DualWriter(args.output, CSV_FIELDNAMES, args.output_json, resuming)
    stats = {"success": 0, "error": 0, "done": 0}
    t0 = time.time()

    def process_one(vpath: Path) -> None:
        rel = str(vpath.relative_to(input_root))
        frames = extract_frames_pil(str(vpath))
        if len(frames) < 2:
            stats["error"] += 1
            stats["done"] += 1
            log.warning("Skipping %s — could not extract frames", rel)
            return

        try:
            result = call_with_retry(
                client, frames, SYSTEM_PROMPT,
                "Analyze these 4 frames from a video clip and respond "
                "with the JSON as instructed.",
                temperature=0.2,
            )
        except Exception as e:
            stats["error"] += 1
            stats["done"] += 1
            log.error("Failed for %s: %s", rel, e)
            return

        csv_row = {
            "file": rel,
            "classification": result.get("classification", "Other"),
            "classification_confidence": result.get("classification_confidence", 0.0),
            "gender": result.get("gender", ""),
            "age_range": result.get("age_range", ""),
            "ethnicity": result.get("ethnicity", ""),
        }
        result["file"] = rel
        writer.write(csv_row, result)

        stats["success"] += 1
        stats["done"] += 1
        d = stats["done"]
        if d % 25 == 0 or d == total_work:
            elapsed = time.time() - t0
            rate = stats["success"] / elapsed if elapsed > 0 else 0
            log.info(
                "[%d/%d] classified=%d  errors=%d  (%.1f vid/s)",
                d, total_work, stats["success"], stats["error"], rate,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, v): v for v in work}
        for fut in as_completed(futures):
            fut.result()

    writer.close()

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.1f s", elapsed)
    log.info("Total       : %d", len(all_videos))
    log.info("Already done: %d", len(done))
    log.info("Classified  : %d", stats["success"])
    log.info("Errors      : %d", stats["error"])
    log.info("CSV         : %s", args.output)
    if args.output_json:
        log.info("JSONL       : %s", args.output_json)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
