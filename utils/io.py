"""
File I/O helpers for JSONL, CSV, and pipeline resume support.

Every LLM-based pipeline step (classify, label, generate_prompts) writes
both CSV and JSONL.  The JSONL doubles as a checkpoint: on re-run the
set of already-processed file keys is loaded and those items are skipped.
"""

import os
import json
import csv
import threading
from pathlib import Path


# ── JSONL ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file. Returns [] if file is missing."""
    entries: list[dict] = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_done_files(jsonl_path: str, key: str = "file") -> set[str]:
    """Load set of already-processed identifiers from JSONL for resume.

    Reads an existing JSONL output and collects values of *key* from each
    record.  Used to skip items that were completed in a previous run.
    """
    done: set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key in obj:
                    done.add(obj[key])
            except json.JSONDecodeError:
                continue
    return done


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of records to a JSONL file (overwrite)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Thread-safe dual CSV + JSONL writer ───────────────────────────────────────

class DualWriter:
    """Thread-safe writer that appends rows to both CSV and JSONL files.

    Designed for parallel Gemini API workers that produce results
    concurrently.  Each write is protected by a threading lock.
    """

    def __init__(
        self,
        csv_path: str,
        csv_fieldnames: list[str],
        jsonl_path: str | None = None,
        resuming: bool = False,
    ):
        mode = "a" if resuming else "w"
        self._csv_file = open(csv_path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=csv_fieldnames)
        if not resuming:
            self._writer.writeheader()

        self._jsonl_file = None
        if jsonl_path:
            self._jsonl_file = open(jsonl_path, mode, encoding="utf-8", buffering=1)

        self._lock = threading.Lock()

    def write(self, csv_row: dict, jsonl_record: dict | None = None) -> None:
        """Write a single result row (thread-safe)."""
        with self._lock:
            self._writer.writerow(csv_row)
            self._csv_file.flush()
            if self._jsonl_file and jsonl_record:
                self._jsonl_file.write(
                    json.dumps(jsonl_record, ensure_ascii=False) + "\n"
                )
                self._jsonl_file.flush()

    def close(self) -> None:
        self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()
