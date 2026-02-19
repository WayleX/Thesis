#!/usr/bin/env python3
"""
Generate deepfake video-generation prompts.

For each video, extracts 4 key frames and asks Gemini to write a detailed
prompt suitable for AI video generation models (Wan2.2, HunyuanVideo, etc.).

Supports a --model-context flag to inject model-specific instructions
so the prompt style matches what the target model expects.

Features:
  - Exponential backoff on transient errors
  - Auto-resume via JSONL
  - Parallel workers

Usage:
    python generate_prompts.py \\
        --input /path/to/curated_dataset \\
        --output prompts.csv \\
        --output-json prompts.jsonl \\
        --model-context "Kling 2.1: short 1-2 sentence prompts, focus on motion"
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
dotenv.load_dotenv()

from utils.video import find_videos, extract_frames_pil
from utils.io import load_done_files, DualWriter
from utils.gemini import create_client, call_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert at writing prompts for AI video generation models. \
You will receive 4 frames extracted from a real video clip at evenly spaced \
intervals (beginning, one-third, two-thirds, end). The video shows a single \
person speaking.

Your task is to write a prompt that an AI video generation model would use \
to recreate this video as a realistic deepfake — a synthetic video of a \
person speaking that looks as real as possible.

**Requirements for the prompt:**
1. Describe the person's physical appearance in detail: hair, skin tone, \
facial features, clothing, accessories.
2. Describe the setting / background: room, studio, lighting, colors, objects.
3. Describe the camera: angle, framing (close-up, medium shot, etc.), \
whether the camera is static or moving.
4. ALWAYS include that **the person is speaking / talking** — describe \
their mouth movement, gestures, head movements, facial expressions while talking.
5. Describe the lighting: natural, studio, warm/cool, shadows, etc.
6. The prompt should be 2-5 sentences, vivid, and specific enough that \
an AI model could generate a convincing video from it alone.
7. Do NOT mention frame numbers, still images, or that you are analyzing frames.
8. Write the prompt as if you are describing a real video to recreate.
{model_context_block}
Respond ONLY with valid JSON in this exact schema (no markdown, no extra text):
{{
  "prompt": "...",
  "scene_description": "brief 1-sentence summary of the scene",
  "speaking_style": "one of: formal, casual, animated, calm, emotional, monotone",
  "camera_type": "one of: static, slow_pan, slight_movement, handheld"
}}
"""

CSV_FIELDNAMES = [
    "file", "class", "prompt",
    "scene_description", "speaking_style", "camera_type",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate deepfake video prompts using Gemini.",
    )
    parser.add_argument("--input", required=True, help="Root dir of curated videos")
    parser.add_argument("--output", default="prompts.csv")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--frames-dir", default=None, help="If provided, extract the first frame into this directory preserving folder structure")
    parser.add_argument("--model-context", default=None,
                        help="Model-specific prompt instructions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("No API key. set GEMINI_API_KEY env var.")
        sys.exit(1)

    if args.model_context:
        ctx = (
            f"\n**Target model instructions**: {args.model_context}\n"
            "Adapt the prompt style, length, and focus to match what this "
            "specific model performs best with.\n"
        )
    else:
        ctx = ""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(model_context_block=ctx)

    client = create_client(api_key)
    input_root = Path(args.input).resolve()
    if not input_root.is_dir():
        log.error("Input directory does not exist: %s", input_root)
        sys.exit(1)

    all_videos = find_videos(str(input_root))
    log.info("Found %d video files.", len(all_videos))
    if args.model_context:
        log.info("Model context: %s", args.model_context)

    done: set[str] = set()
    if args.output_json and os.path.exists(args.output_json):
        done = load_done_files(args.output_json)
        if done:
            log.info("Auto-resume: %d already done.", len(done))

    work = [v for v in all_videos if str(v.relative_to(input_root)) not in done]
    if args.limit > 0:
        work = work[: args.limit]
    if not work:
        log.info("Nothing to process.")
        return

    total_work = len(work)
    log.info("Videos to process: %d  |  Workers: %d", total_work, args.workers)

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

        if args.frames_dir:
            out_frame_path = Path(args.frames_dir) / rel
            out_frame_path = out_frame_path.with_suffix(".png")
            out_frame_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(out_frame_path, format="PNG")

        try:
            result = call_with_retry(
                client, frames, system_prompt,
                "Analyze these 4 frames from a video of a person speaking. "
                "Write a video generation prompt as instructed.",
                temperature=0.4,
            )
        except Exception as e:
            stats["error"] += 1
            stats["done"] += 1
            log.error("Failed %s: %s", rel, e)
            return

        video_class = vpath.parent.name
        csv_row = {
            "file": rel, "class": video_class,
            "prompt": result.get("prompt", ""),
            "scene_description": result.get("scene_description", ""),
            "speaking_style": result.get("speaking_style", ""),
            "camera_type": result.get("camera_type", ""),
        }
        result["file"] = rel
        result["class"] = video_class
        writer.write(csv_row, result)

        stats["success"] += 1
        stats["done"] += 1
        d = stats["done"]
        if d % 20 == 0 or d == total_work:
            elapsed = time.time() - t0
            rate = stats["success"] / elapsed if elapsed > 0 else 0
            log.info(
                "[%d/%d] ok=%d  err=%d  (%.1f vid/s)",
                d, total_work, stats["success"], stats["error"], rate,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, v): v for v in work}
        for fut in as_completed(futures):
            fut.result()

    writer.close()
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.1fs  |  OK: %d  |  Errors: %d",
             elapsed, stats["success"], stats["error"])
    log.info("CSV: %s", args.output)
    if args.output_json:
        log.info("JSONL: %s", args.output_json)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
