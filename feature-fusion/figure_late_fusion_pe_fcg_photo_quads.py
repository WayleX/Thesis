#!/usr/bin/env python3
"""
Photo PDF: 4 rows × 4 frames. Rows (top → bottom): fp, tp, tn, fn.

Late fusion: ½·01_paper + ½·03_pe, threshold 0.5, DFD v1.1 (no I2V buckets).

Example selection:
  fp — real with highest fused score (worst false positive)
  tp — fake with highest fused score (clearest true positive)
  tn — real with highest fused score among true negatives (hardest real still real)
  fn — fake with lowest fused score (clearest false negative)

One text block per row on the left (ylabel); no text on the frame images; no top title.

Usage:
  conda run -n dfd-fcg python figure_late_fusion_pe_fcg_photo_quads.py --gpu 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from failure_analysis import (
    DFD_GENERATORS,
    dfd_generators_for_figures,
    extract_frames,
    find_dfd_video,
    late_fuse_stems,
    score_with_stems,
    FIGS,
    FONT_PLACEHOLDER,
    FONT_ROW_LABEL,
)

N_FRAMES = 4

# (key, phrase, color, pick: "max" | "min" | "tn_max")
ROWS = [
    ("fp", "Real classified as fake", "orange", "max"),
    ("tp", "Fake classified as fake", "green", "max"),
    ("tn", "Real classified as real", "steelblue", "tn_max"),
    ("fn", "Fake classified as real", "red", "min"),
]


def collect_all(fused, thresh=0.5):
    tp, fp, tn, fn = [], [], [], []
    for gen in dfd_generators_for_figures():
        for stem, sc, lab in fused.get(gen, []):
            pred = 1 if sc > thresh else 0
            t = (sc, gen, stem, lab)
            if lab == 1 and pred == 1:
                tp.append(t)
            elif lab == 0 and pred == 1:
                fp.append(t)
            elif lab == 0 and pred == 0:
                tn.append(t)
            elif lab == 1 and pred == 0:
                fn.append(t)
    return tp, fp, tn, fn


def pick_example(cands, mode):
    if not cands:
        return None
    if mode == "max":
        return max(cands, key=lambda x: x[0])
    if mode == "min":
        return min(cands, key=lambda x: x[0])
    if mode == "tn_max":
        return max(cands, key=lambda x: x[0])
    raise ValueError(mode)


def pick_all_quadrants(tp, fp, tn, fn):
    out = {}
    for key, _phrase, _color, mode in ROWS:
        if key == "fp":
            out[key] = pick_example(fp, mode)
        elif key == "tp":
            out[key] = pick_example(tp, mode)
        elif key == "tn":
            out[key] = pick_example(tn, mode)
        elif key == "fn":
            out[key] = pick_example(fn, mode)
    return out


def left_row_text(tag, phrase, sc, gen, stem, lab):
    true = "REAL" if lab == 0 else "FAKE"
    pred = "fake" if sc > 0.5 else "real"
    sub = DFD_GENERATORS.get(gen, gen) if lab == 1 else "real"
    s = stem if len(stem) <= 42 else stem[:41] + "…"
    return (
        f"{tag} — {phrase}\n"
        f"{gen} · {sub}\n"
        f"id: {s}\n"
        f"fused score={sc:.3f}  (true={true}, pred={pred})"
    )


def build_figure(examples, out_path: Path, n_frames=N_FRAMES):
    n_rows = len(ROWS)
    fig, axes = plt.subplots(
        n_rows,
        n_frames,
        figsize=(n_frames * 3.0, n_rows * 3.5),
    )

    for i, (tag, phrase, color, _mode) in enumerate(ROWS):
        spec = examples.get(tag)
        if spec is None:
            for j in range(n_frames):
                ax = axes[i, j]
                ax.text(
                    0.5,
                    0.5,
                    "no example",
                    ha="center",
                    va="center",
                    fontsize=FONT_PLACEHOLDER,
                    transform=ax.transAxes,
                )
                ax.axis("off")
            continue

        sc, gen, stem, lab = spec
        video = find_dfd_video(stem, gen, lab)
        frames = extract_frames(video, n_frames) if video else []

        for j in range(n_frames):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=FONT_PLACEHOLDER,
                    transform=ax.transAxes,
                )
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(
                    left_row_text(tag, phrase, sc, gen, stem, lab),
                    fontsize=FONT_ROW_LABEL,
                    color=color,
                    rotation=0,
                    labelpad=155,
                    ha="right",
                    va="center",
                )

    plt.tight_layout(rect=[0.02, 0, 0.98, 1.0])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(FIGS / "late_fusion_PE_FCG_fp_tp_tn_fn.pdf"),
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"

    res_paper = score_with_stems(str(BASE / "configs" / "01_paper.yaml"), device)
    res_pe = score_with_stems(str(BASE / "configs" / "03_pe.yaml"), device)
    fused = late_fuse_stems(res_paper, res_pe)

    tp, fp, tn, fn = collect_all(fused)
    ex = pick_all_quadrants(tp, fp, tn, fn)

    for key, _p, _c, _m in ROWS:
        e = ex.get(key)
        if e:
            print(f"{key}: {e[1]} {e[2]} score={e[0]:.4f} lab={e[3]}")
        else:
            print(f"{key}: (empty)")

    out = build_figure(ex, Path(args.out))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
