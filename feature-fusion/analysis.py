"""
Generate all analysis figures for the thesis.

Usage:
  python analysis.py --results-dir results/ --output-dir figures/
"""

import argparse, torch, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from data.paths import DFD_GENERATORS

DFD_GENS = list(DFD_GENERATORS.keys())
CORE = ["FSh", "DFDC"]

DISPLAY_NAMES = {
    "01_paper": "DFD-FCG",
    "02_depth": "Depth",
    "03_pe": "PE",
    "04_ct_s3": "CoTracker",
    "05_depth_pe_concat": "D+PE (concat)",
    "06_paper_depth_concat": "FCG+D (concat)",
    "07_paper_pe_concat": "FCG+PE (concat)",
    "08_paper_ct_s3": "FCG+CT (concat)",
    "09_dpc_concat_s3": "D+PE+CT (concat)",
    "10_all_concat_s3": "All (concat)",
    "11_depth_pe_gated": "D+PE (gated)",
    "12_paper_depth_gated": "FCG+D (gated)",
    "13_depth_pe_ct_gated_s3": "D+PE+CT (gated)",
    "14_all_gated_s3": "All (gated)",
    "paper_ct_gated": "FCG+CT (gated)",
    "pe_ct_concat": "PE+CT (concat)",
    "pe_ct_gated": "PE+CT (gated)",
    "paper_pe_gated": "FCG+PE (gated)",
    "depth_ct_concat": "D+CT (concat)",
    "depth_ct_gated": "D+CT (gated)",
    "paper_pe_ct_concat": "FCG+PE+CT (concat)",
    "paper_pe_ct_gated": "FCG+PE+CT (gated)",
    "paper_pe_depth_concat": "FCG+PE+D (concat)",
    "paper_pe_depth_gated": "FCG+PE+D (gated)",
    "paper_depth_ct_concat": "FCG+D+CT (concat)",
    "paper_depth_ct_gated": "FCG+D+CT (gated)",
}


def dn(name):
    return DISPLAY_NAMES.get(name, name)


def load_results(results_dir):
    data = {}
    for pt in sorted(results_dir.glob("*_scores.pt")):
        name = pt.stem.replace("_scores", "")
        d = torch.load(pt, map_location="cpu", weights_only=False)
        data[name] = d
    return data


def _compute_mean_auc(scores):
    fsh = roc_auc_score(scores["FSh"][1], scores["FSh"][0]) if "FSh" in scores else 0
    dfdc = roc_auc_score(scores["DFDC"][1], scores["DFDC"][0]) if "DFDC" in scores else 0
    dfd_aucs = [roc_auc_score(scores[g][1], scores[g][0])
                for g in DFD_GENS if g in scores]
    dfd = np.mean(dfd_aucs) if dfd_aucs else 0
    return (fsh + dfdc + dfd) / 3


def plot_roc_curves(data, names, output_dir, datasets=None):
    if datasets is None:
        datasets = CORE + ["DFD_mean"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for name in names:
            if name not in data:
                continue
            scores = data[name]["scores"]
            if ds == "DFD_mean":
                all_s = np.concatenate([scores[g][0] for g in DFD_GENS if g in scores])
                all_l = np.concatenate([scores[g][1] for g in DFD_GENS if g in scores])
            elif ds in scores:
                all_s, all_l = scores[ds]
            else:
                continue
            fpr, tpr, _ = roc_curve(all_l, all_s)
            auc = roc_auc_score(all_l, all_s)
            ax.plot(fpr, tpr, label=f"{dn(name)} ({auc:.3f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        title = "DFD v1.1 (combined)" if ds == "DFD_mean" else ds
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("FPR", fontsize=11)
        ax.set_ylabel("TPR", fontsize=11)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved roc_curves.pdf")


def plot_heatmap(data, names, output_dir):
    all_ds = CORE + DFD_GENS
    matrix = np.zeros((len(names), len(all_ds)))
    for i, name in enumerate(names):
        if name not in data:
            continue
        scores = data[name]["scores"]
        for j, ds in enumerate(all_ds):
            if ds in scores:
                try:
                    matrix[i, j] = roc_auc_score(scores[ds][1], scores[ds][0])
                except ValueError:
                    matrix[i, j] = 0.5
    fig, ax = plt.subplots(figsize=(max(14, len(all_ds) * 1.4), max(6, len(names) * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.3, vmax=1.0,
                xticklabels=all_ds, yticklabels=[dn(n) for n in names], ax=ax,
                annot_kws={"fontsize": 8}, linewidths=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved heatmap.pdf")


def plot_tsne(data, names, output_dir, dataset="FSh"):
    n = len(names)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        if name not in data or "embeddings" not in data[name]:
            ax.set_visible(False)
            continue
        embs = data[name]["embeddings"]
        if dataset not in embs or dataset not in data[name]["scores"]:
            ax.set_visible(False)
            continue
        X = embs[dataset]
        labels = data[name]["scores"][dataset][1]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        Z = tsne.fit_transform(X)
        for lab, color, lbl in [(0, "tab:blue", "Real"), (1, "tab:red", "Fake")]:
            mask = labels == lab
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.4, s=12, label=lbl)
        ax.set_title(f"{dn(name)} ({dataset})", fontsize=12)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"tsne_{dataset}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved tsne_{dataset}.pdf")


def plot_score_distributions(data, names, output_dir, dataset="FSh"):
    n = len(names)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        if name not in data:
            ax.set_visible(False)
            continue
        scores = data[name]["scores"]
        if dataset not in scores:
            ax.set_visible(False)
            continue
        s, l = scores[dataset]
        ax.hist(s[l == 0], bins=50, alpha=0.6, label="Real", color="tab:blue", density=True)
        ax.hist(s[l == 1], bins=50, alpha=0.6, label="Fake", color="tab:red", density=True)
        ax.set_title(f"{dn(name)} ({dataset})", fontsize=12)
        ax.set_xlabel("P(fake)", fontsize=10)
        ax.legend(fontsize=9)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"score_dist_{dataset}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved score_dist_{dataset}.pdf")


def plot_fusion_progression(data, output_dir):
    stages = [
        ("01_paper", "DFD-FCG\n(baseline)"),
        ("06_paper_depth_concat", "FCG+D\n(concat)"),
        ("03_pe", "PE\n(solo)"),
        ("13_depth_pe_ct_gated_s3", "D+PE+CT\n(gated)"),
    ]
    bars, labels = [], []
    for name, label in stages:
        if name in data:
            bars.append(_compute_mean_auc(data[name]["scores"]))
            labels.append(label)

    late_names = ["03_pe", "paper_pe_depth_concat"]
    if all(n in data for n in late_names):
        combined = {}
        for vn in data[late_names[0]]["scores"]:
            s0 = data[late_names[0]]["scores"][vn][0]
            s1 = data[late_names[1]]["scores"][vn][0]
            l = data[late_names[0]]["scores"][vn][1]
            combined[vn] = (0.5 * s0 + 0.5 * s1, l)
        bars.append(_compute_mean_auc(combined))
        labels.append("Late Fusion\n(best)")

    if not bars:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(bars) * 1.5), 5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"][:len(bars)]
    x = range(len(bars))
    ax.bar(x, bars, color=colors, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean AUC", fontsize=12)
    ax.set_ylim(0.55, 0.9)
    for i, v in enumerate(bars):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "fusion_progression.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fusion_progression.pdf")


def plot_confusion_at_eer(data, names, output_dir, dataset="FSh"):
    n = len(names)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        if name not in data or dataset not in data[name]["scores"]:
            ax.set_visible(False)
            continue
        s, l = data[name]["scores"][dataset]
        fpr, tpr, thresholds = roc_curve(l, s)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        thresh = thresholds[eer_idx]
        preds = (s >= thresh).astype(int)
        cm = confusion_matrix(l, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    annot_kws={"fontsize": 14})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{dn(name)} @ EER", fontsize=12)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_{dataset}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion_{dataset}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    data = load_results(results_dir)
    if not data:
        print("No results found. Run evaluate.py --save first.")
        return

    THESIS_MODELS = [
        "01_paper", "02_depth", "03_pe", "04_ct_s3",
        "05_depth_pe_concat", "06_paper_depth_concat",
        "07_paper_pe_concat", "08_paper_ct_s3",
        "pe_ct_concat", "pe_ct_gated",
        "paper_ct_gated", "paper_pe_gated",
        "depth_ct_concat", "depth_ct_gated",
        "09_dpc_concat_s3", "10_all_concat_s3",
        "11_depth_pe_gated", "12_paper_depth_gated",
        "13_depth_pe_ct_gated_s3", "14_all_gated_s3",
        "paper_pe_ct_concat", "paper_pe_ct_gated",
        "paper_pe_depth_concat", "paper_pe_depth_gated",
        "paper_depth_ct_concat", "paper_depth_ct_gated",
    ]
    base_names = [n for n in THESIS_MODELS if n in data]
    print(f"Found {len(data)} total results, {len(base_names)} thesis models")

    key_models = [n for n in ["01_paper", "02_depth", "03_pe", "04_ct_s3",
                               "13_depth_pe_ct_gated_s3", "paper_pe_depth_concat"] if n in data]
    if not key_models:
        key_models = base_names[:6]

    plot_roc_curves(data, key_models, output_dir)
    plot_heatmap(data, base_names, output_dir)
    plot_fusion_progression(data, output_dir)

    for ds in ["FSh", "DFDC"]:
        plot_tsne(data, key_models[:4], output_dir, ds)
        plot_score_distributions(data, key_models[:4], output_dir, ds)
        plot_confusion_at_eer(data, key_models[:4], output_dir, ds)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
