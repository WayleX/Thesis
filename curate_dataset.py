#!/usr/bin/env python3
"""
Curate a balanced dataset from classified videos.

Selection rules:
  1. Cap at --per-class videos per classification category (default 100).
  2. talkingcelebs: only allowed in the Official Statement class.
  3. All classes: balance ethnicities via round-robin.

Outputs:
  - <output>/ClassName/video.mp4    – copied videos in class folders
  - <output>/selected_entries.jsonl  – full metadata for selected videos
  - curation_stats.csv               – ethnicity / gender / source breakdowns

Usage:
    python curate_dataset.py \\
        --input /path/to/filtered_videos \\
        --jsonl classification_results.jsonl \\
        --output /path/to/curated_output \\
        --per-class 100
"""

import sys
import json
import random
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from utils.io import load_jsonl, write_jsonl
from utils.curation import (
    get_source, deduplicate_speaker5m, balance_ethnicities,
    get_folder_name, write_stats_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def select_for_class(
    entries: list[dict], class_name: str, target: int,
) -> list[dict]:
    """Select up to *target* videos for one class.

    Official Statement gets no ethnicity balancing (just random shuffle);
    all others are ethnicity-balanced via round-robin.
    """
    if class_name == "Official Statement":
        random.shuffle(entries)
        return entries[:target]
    return balance_ethnicities(entries, target)


def main():
    parser = argparse.ArgumentParser(description="Curate a balanced dataset.")
    parser.add_argument("--input", required=True, help="Root dir of filtered videos")
    parser.add_argument("--jsonl", default="classification_results.jsonl",
                        help="JSONL classification file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-csv", default=None,
                        help="Path for stats CSV (default: <output>/curation_stats.csv)")
    args = parser.parse_args()

    random.seed(args.seed)
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    target = args.per_class

    # Load all classifications
    all_entries = load_jsonl(args.jsonl)
    log.info("Loaded %d classifications from %s", len(all_entries), args.jsonl)

    # Group by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in all_entries:
        by_class[e.get("classification", "Other")].append(e)

    log.info("Class distribution (before filtering):")
    for cls in sorted(by_class):
        log.info("  %-30s : %d", cls, len(by_class[cls]))

    # Apply selection rules per class
    final_selections: dict[str, list[dict]] = {}

    for cls, entries in by_class.items():
        log.info("--- Processing class: %s (%d candidates) ---", cls, len(entries))

        # Rule: talkingcelebs only allowed in Official Statement
        if cls != "Official Statement":
            before = len(entries)
            entries = [e for e in entries if get_source(e["file"]) != "talkingcelebs"]
            removed = before - len(entries)
            if removed:
                log.info("  Removed %d talkingcelebs videos", removed)

        # Rule: Speaker5m — one video per speaker
        before = len(entries)
        entries = deduplicate_speaker5m(entries)
        deduped = before - len(entries)
        if deduped:
            log.info("  Speaker5m dedup: %d → %d (-%d)", before, len(entries), deduped)

        # Verify source files exist on disk
        entries = [e for e in entries if (input_root / e["file"]).exists()]
        log.info("  Candidates after filtering: %d", len(entries))

        selected = select_for_class(entries, cls, target)
        final_selections[cls] = selected
        log.info("  Selected: %d / %d target", len(selected), target)

    # Copy files to class folders
    log.info("=" * 60)
    log.info("Copying files to %s …", output_root)
    total_copied = 0
    all_selected: list[dict] = []

    for cls, selected in final_selections.items():
        folder = get_folder_name(cls)
        dest_dir = output_root / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        for entry in selected:
            src = input_root / entry["file"]
            dest = dest_dir / src.name

            # Handle duplicate filenames
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(str(src), str(dest))
            entry["_dest"] = str(dest.relative_to(output_root))
            all_selected.append(entry)
            total_copied += 1

    log.info("Copied %d videos total.", total_copied)

    # Write stats CSV
    stats_path = (
        Path(args.stats_csv) if args.stats_csv
        else output_root / "curation_stats.csv"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    write_stats_csv(str(stats_path), final_selections, all_selected)
    log.info("Stats: %s", stats_path)

    # Write selected entries JSONL
    sel_jsonl = output_root / "selected_entries.jsonl"
    write_jsonl(str(sel_jsonl), all_selected)
    log.info("Selected entries JSONL: %s", sel_jsonl)

    # Final summary
    log.info("=" * 60)
    log.info("CURATION COMPLETE")
    for cls in sorted(final_selections):
        ec = Counter(
            e.get("ethnicity", "Other") or "Other" for e in final_selections[cls]
        )
        eth_str = ", ".join(f"{k}:{v}" for k, v in sorted(ec.items()))
        log.info("  %-30s : %3d  [%s]", get_folder_name(cls),
                 len(final_selections[cls]), eth_str)
    log.info("  %-30s : %3d", "TOTAL", total_copied)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
