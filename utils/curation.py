"""
Dataset curation helpers: ethnicity balancing and stats.
"""

import csv
import random
from collections import defaultdict, Counter

from config import FOLDER_MAP, ETHNICITY_ORDER


def get_source(file_path: str) -> str:
    """Extract top-level data source from a relative path.
    """
    return file_path.split("/")[0]

def get_folder_name(class_name: str) -> str:
    """Map a classification label to a filesystem-safe folder name."""
    return FOLDER_MAP.get(class_name, class_name.replace("/", "_").replace(" ", "_"))

def balance_ethnicities(entries: list[dict], target: int) -> list[dict]:
    """Select target entries with ethnicities as balanced as possible.
    """
    by_eth: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        eth = e.get("ethnicity", "Other") or "Other"
        if eth in ("?", ""):
            eth = "Other"
        by_eth[eth].append(e)

    for eth in by_eth:
        random.shuffle(by_eth[eth])

    selected: list[dict] = []
    while len(selected) < target:
        made_progress = False
        for eth in ETHNICITY_ORDER:
            if len(selected) >= target:
                break
            if by_eth.get(eth):
                selected.append(by_eth[eth].pop())
                made_progress = True
        # Handle any ethnicity not in the canonical list
        for eth in list(by_eth.keys()):
            if eth not in ETHNICITY_ORDER:
                if len(selected) >= target:
                    break
                if by_eth[eth]:
                    selected.append(by_eth[eth].pop())
                    made_progress = True
        if not made_progress:
            break

    return selected


def write_stats_csv(
    path: str,
    final_selections: dict[str, list[dict]],
    all_selected: list[dict],
) -> None:
    """Write a comprehensive curation statistics CSV.

    Sections produced:
      1. Per-class video count summary
      2. Ethnicity breakdown by class
      3. Gender breakdown by class
      4. Data-source breakdown by class
      5. Age-range breakdown by class
    """
    total = len(all_selected)

    with open(path, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)

        sw.writerow(["=== PER-CLASS SUMMARY ==="])
        sw.writerow(["class", "count"])
        for cls in sorted(final_selections):
            sw.writerow([cls, len(final_selections[cls])])
        sw.writerow(["TOTAL", total])
        sw.writerow([])

        sw.writerow(["=== ETHNICITY BY CLASS ==="])
        sw.writerow(["class"] + ETHNICITY_ORDER + ["Unknown", "Total"])
        for cls in sorted(final_selections):
            ec = Counter(
                e.get("ethnicity", "Other") or "Other"
                for e in final_selections[cls]
            )
            row = [cls] + [ec.get(eth, 0) for eth in ETHNICITY_ORDER]
            row += [ec.get("?", 0) + ec.get("", 0), len(final_selections[cls])]
            sw.writerow(row)
        sw.writerow([])

        sw.writerow(["=== GENDER BY CLASS ==="])
        sw.writerow(["class", "Male", "Female", "Unknown", "Total"])
        for cls in sorted(final_selections):
            gc = Counter(
                e.get("gender", "Unknown") or "Unknown"
                for e in final_selections[cls]
            )
            sw.writerow([
                cls, gc.get("Male", 0), gc.get("Female", 0),
                gc.get("Unknown", 0) + gc.get("", 0),
                len(final_selections[cls]),
            ])
        sw.writerow([])

        sw.writerow(["=== SOURCE BY CLASS ==="])
        all_sources = sorted({get_source(e["file"]) for e in all_selected})
        sw.writerow(["class"] + all_sources + ["Total"])
        for cls in sorted(final_selections):
            sc = Counter(get_source(e["file"]) for e in final_selections[cls])
            sw.writerow(
                [cls] + [sc.get(s, 0) for s in all_sources]
                + [len(final_selections[cls])]
            )
        sw.writerow([])

        sw.writerow(["=== AGE RANGE BY CLASS ==="])
        all_ages = sorted({e.get("age_range", "?") or "?" for e in all_selected})
        sw.writerow(["class"] + all_ages + ["Total"])
        for cls in sorted(final_selections):
            ac = Counter(
                e.get("age_range", "?") or "?" for e in final_selections[cls]
            )
            sw.writerow(
                [cls] + [ac.get(a, 0) for a in all_ages]
                + [len(final_selections[cls])]
            )
