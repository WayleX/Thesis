"""
Detailed failure case analysis with frame extraction for thesis figures.
Identifies misclassified samples, extracts face-crop frames, creates panels.
"""
import torch, yaml, json, os, sys
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGS = BASE / "figures" / "failures"
FIGS.mkdir(parents=True, exist_ok=True)

# PE vs FCG comparison figure (large slide / print typography)
FONT_PEVSFCG_TITLE = 30
FONT_PEVSFCG_SUPTITLE = 38

DFD_V11 = Path("/extra_space2/shykula/gg/datasets/deepfake_v1.1")
DFDC_DIR = Path("/extra_space2/shykula/gg/datasets/dfdc")
FFPP_DIR = Path("/extra_space2/shykula/gg/datasets/ffpp")

DFD_GENERATORS = {
    "Grok": "Grok_imagine",
    "HunyuanI2V": "HunyuanVideo_1.5_14b_i2v",
    "HunyuanT2V": "HunyuanVideo_1.5_14b_t2v",
    "Kling": "Kling_3.0",
    "Veo": "Veo3.1",
    "Wan22I2V": "Wan_2.2_14b_i2v",
    "Wan22T2V": "Wan_2.2_14b_t2v",
    "Wan26": "Wan_2.6",
}

OPEN_SOURCE = {"HunyuanI2V", "HunyuanT2V", "Wan22I2V", "Wan22T2V", "Wan26"}
CLOSED_SOURCE = {"Grok", "Kling", "Veo"}


def extract_frames(video_path, n_frames=4):
    """Extract n uniformly spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def find_dfd_video(stem, generator_key, label):
    """Find face-cropped video for a DFD sample."""
    gen_dir = DFD_GENERATORS.get(generator_key, "")
    if label == 0:
        paths = [
            DFD_V11 / "cropped" / "videos" / "real" / f"{stem}.avi",
            DFD_V11 / "videos" / "real" / f"{stem}.mp4",
        ]
    else:
        paths = [
            DFD_V11 / "cropped" / "videos" / "fake" / gen_dir / f"{stem}.avi",
            DFD_V11 / "videos" / "fake" / gen_dir / f"{stem}.mp4",
        ]
    for p in paths:
        if p.exists():
            return p
    return None


def score_with_stems(cfg_path, device="cuda:0"):
    """Run model inference returning per-sample (stem, score, label)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from data.dataset import build_dataloaders, DFDDataset, BinaryDataset, FFppDataset
    _, val_names, val_loaders = build_dataloaders(cfg)
    cfg["val_names"] = val_names

    ckpt = list((BASE / "checkpoints" / cfg["name"]).glob("*.ckpt"))
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint for {cfg['name']}")

    from train_exp import ExpDetector
    try:
        model = ExpDetector.load_from_checkpoint(str(ckpt[0]), cfg=cfg).eval().to(device)
    except Exception:
        from models.detector import DeepfakeDetector
        model = DeepfakeDetector.load_from_checkpoint(str(ckpt[0]), cfg=cfg).eval().to(device)

    all_results = {}
    with torch.no_grad():
        for vn, vl in zip(val_names, val_loaders):
            ds = vl.dataset
            stem_list = [s[0] for s in ds.samples]
            scores, labels = [], []
            batch_idx = 0
            for batch in vl:
                feats, labs, _ = batch
                feats = {k: v.to(device) for k, v in feats.items()}
                probs = F.softmax(model(feats), dim=-1)[:, 1]
                scores.append(probs.cpu().numpy())
                labels.append(labs.numpy())
                batch_idx += labs.shape[0]

            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            stems = stem_list[:len(scores)]
            all_results[vn] = list(zip(stems, scores, labels))
    del model
    torch.cuda.empty_cache()
    return all_results


def late_fuse_stems(results_a, results_b):
    """Late fuse two result dicts (stem-level)."""
    combined = {}
    for vn in results_a:
        if vn not in results_b:
            continue
        map_b = {stem: sc for stem, sc, _ in results_b[vn]}
        fused = []
        for stem, sc_a, lab in results_a[vn]:
            if stem in map_b:
                fused.append((stem, 0.5 * sc_a + 0.5 * map_b[stem], lab))
        combined[vn] = fused
    return combined


def get_worst_samples(results, gen_key, n=5, mode="fn"):
    """Get n worst false negatives (mode='fn') or false positives (mode='fp')."""
    samples = results.get(gen_key, [])
    if not samples:
        return []
    if mode == "fn":
        fakes = [(stem, sc, lab) for stem, sc, lab in samples if lab == 1]
        fakes.sort(key=lambda x: x[1])
        return fakes[:n]
    else:
        reals = [(stem, sc, lab) for stem, sc, lab in samples if lab == 0]
        reals.sort(key=lambda x: -x[1])
        return reals[:n]


def create_failure_panel(samples, gen_key, model_name, mode, n_frames=4):
    """Create a panel figure showing frames from misclassified videos."""
    n_samples = len(samples)
    if n_samples == 0:
        return None

    fig, axes = plt.subplots(n_samples, n_frames, figsize=(n_frames * 2.5, n_samples * 2.5))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    type_label = "False Negatives (fake scored low)" if mode == "fn" else "False Positives (real scored high)"
    fig.suptitle(f"{model_name} — {gen_key} — {type_label}", fontsize=13, y=1.02)

    for i, (stem, score, label) in enumerate(samples):
        true_label = "FAKE" if label == 1 else "REAL"
        video = find_dfd_video(stem, gen_key, label)
        frames = extract_frames(video, n_frames) if video else []

        for j in range(n_frames):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12,
                        transform=ax.transAxes)
            ax.axis("off")
            if j == 0:
                color = "red" if mode == "fn" else "green"
                ax.set_ylabel(f"{stem[:20]}\nscore={score:.3f}\n({true_label})",
                              fontsize=8, color=color, rotation=0, labelpad=80,
                              ha="right", va="center")

    plt.tight_layout()
    return fig


def create_comparison_panel(results_a, results_b, name_a, name_b, gen_key, n=4, n_frames=4):
    """Show samples where model A succeeds but model B fails (and vice versa)."""
    map_a = {stem: (sc, lab) for stem, sc, lab in results_a.get(gen_key, [])}
    map_b = {stem: (sc, lab) for stem, sc, lab in results_b.get(gen_key, [])}

    common_stems = set(map_a.keys()) & set(map_b.keys())

    a_right_b_wrong = []
    b_right_a_wrong = []
    for stem in common_stems:
        sc_a, lab = map_a[stem]
        sc_b, _ = map_b[stem]
        pred_a = 1 if sc_a > 0.5 else 0
        pred_b = 1 if sc_b > 0.5 else 0
        if pred_a == lab and pred_b != lab:
            a_right_b_wrong.append((stem, sc_a, sc_b, lab))
        elif pred_b == lab and pred_a != lab:
            b_right_a_wrong.append((stem, sc_a, sc_b, lab))

    a_right_b_wrong.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    b_right_a_wrong.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)

    samples = a_right_b_wrong[:n] + b_right_a_wrong[:n]
    if not samples:
        return None

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Model Disagreement — {gen_key}\n"
                 f"Top: {name_a} correct, {name_b} wrong | "
                 f"Bottom: {name_b} correct, {name_a} wrong",
                 fontsize=11, y=1.04)

    for i, (stem, sc_a, sc_b, lab) in enumerate(samples):
        true_label = "FAKE" if lab == 1 else "REAL"
        video = find_dfd_video(stem, gen_key, lab)
        frames = extract_frames(video, n_frames) if video else []

        section = "A>B" if i < len(a_right_b_wrong[:n]) else "B>A"

        for j in range(n_frames):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"{stem[:18]}\n{name_a}:{sc_a:.2f}\n{name_b}:{sc_b:.2f}\n({true_label})",
                              fontsize=7, rotation=0, labelpad=85, ha="right", va="center")

    plt.tight_layout()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print("=" * 60)
    print("SCORING MODELS WITH SAMPLE TRACKING")
    print("=" * 60)

    configs = {
        "D+PE+CT (gated)": "configs/13_depth_pe_ct_gated_s3.yaml",
        "PE (solo)": "configs/03_pe.yaml",
        "FCG (solo)": "configs/01_paper.yaml",
        "FCG+PE+D (concat)": "configs/paper_pe_depth_concat.yaml",
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"\nScoring: {name}")
        all_results[name] = score_with_stems(cfg, device)

    print("\nComputing late fusion: FCG+PE+D(c) + PE")
    all_results["Late: F+PE+D + PE"] = late_fuse_stems(
        all_results["FCG+PE+D (concat)"], all_results["PE (solo)"])

    print("\n" + "=" * 60)
    print("FAILURE CASE ANALYSIS")
    print("=" * 60)

    gens_open = ["Wan22T2V", "Wan22I2V", "HunyuanI2V"]
    gens_closed = ["Veo", "Kling", "Grok"]
    models_to_analyze = ["D+PE+CT (gated)", "Late: F+PE+D + PE"]

    for model_name in models_to_analyze:
        results = all_results[model_name]
        safe_name = model_name.replace(" ", "_").replace(":", "").replace("+", "_")

        print(f"\n--- {model_name} ---")
        for gen in gens_open + gens_closed:
            if gen not in results:
                continue
            samples = results[gen]
            scores = np.array([s for _, s, _ in samples])
            labels = np.array([l for _, _, l in samples])
            if len(np.unique(labels)) < 2:
                continue
            auc = roc_auc_score(labels, scores)

            fn_worst = get_worst_samples(results, gen, n=3, mode="fn")
            fp_worst = get_worst_samples(results, gen, n=3, mode="fp")

            source = "open-source" if gen in OPEN_SOURCE else "closed-source"
            print(f"  {gen} ({source}): AUC={auc:.3f}")
            print(f"    Worst FN (fake scored as real):")
            for stem, sc, lab in fn_worst:
                print(f"      {stem}: score={sc:.4f}")
            print(f"    Worst FP (real scored as fake):")
            for stem, sc, lab in fp_worst:
                print(f"      {stem}: score={sc:.4f}")

            fig = create_failure_panel(fn_worst, gen, model_name, "fn")
            if fig:
                fig.savefig(FIGS / f"{safe_name}_{gen}_fn.pdf", bbox_inches="tight", dpi=150)
                fig.savefig(FIGS / f"{safe_name}_{gen}_fn.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

            fig = create_failure_panel(fp_worst, gen, model_name, "fp")
            if fig:
                fig.savefig(FIGS / f"{safe_name}_{gen}_fp.pdf", bbox_inches="tight", dpi=150)
                fig.savefig(FIGS / f"{safe_name}_{gen}_fp.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (where one succeeds, other fails)")
    print("=" * 60)

    comparisons = [
        ("PE (solo)", "FCG (solo)"),
        ("D+PE+CT (gated)", "FCG (solo)"),
        ("D+PE+CT (gated)", "PE (solo)"),
    ]

    for name_a, name_b in comparisons:
        for gen in ["Wan22T2V", "Wan26", "Veo", "Grok"]:
            fig = create_comparison_panel(
                all_results[name_a], all_results[name_b],
                name_a, name_b, gen, n=3)
            if fig:
                safe_a = name_a.replace(" ", "").replace("(", "").replace(")", "")
                safe_b = name_b.replace(" ", "").replace("(", "").replace(")", "")
                fig.savefig(FIGS / f"compare_{safe_a}_vs_{safe_b}_{gen}.pdf",
                            bbox_inches="tight", dpi=150)
                fig.savefig(FIGS / f"compare_{safe_a}_vs_{safe_b}_{gen}.png",
                            bbox_inches="tight", dpi=150)
                plt.close(fig)
                print(f"  Saved comparison: {name_a} vs {name_b} on {gen}")

    print("\n" + "=" * 60)
    print("COMPOSITE FAILURE FIGURE FOR THESIS")
    print("=" * 60)

    for model_name in models_to_analyze:
        results = all_results[model_name]
        safe_name = model_name.replace(" ", "_").replace(":", "").replace("+", "_")

        easy_gen = "Wan26"
        hard_gen = "Wan22T2V"

        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.15)

        sections = [
            (easy_gen, "fn", f"{easy_gen} — Fake missed (FN)", 0),
            (easy_gen, "fp", f"{easy_gen} — Real flagged (FP)", 1),
            (hard_gen, "fn", f"{hard_gen} — Fake missed (FN)", 2),
            (hard_gen, "fp", f"{hard_gen} — Real flagged (FP)", 3),
        ]

        for gen, mode, title, row in sections:
            worst = get_worst_samples(results, gen, n=1, mode=mode)
            if not worst:
                continue
            stem, score, label = worst[0]
            video = find_dfd_video(stem, gen, label)
            frames = extract_frames(video, 4) if video else []
            true_str = "FAKE" if label == 1 else "REAL"
            pred_str = "fake" if score > 0.5 else "real"

            for j in range(4):
                ax = fig.add_subplot(gs[row, j])
                if j < len(frames):
                    ax.imshow(frames[j])
                ax.axis("off")
                if j == 0:
                    color = "red" if (mode == "fn") else "orange"
                    ax.set_title(f"{title}\n{stem[:25]}\nscore={score:.3f} (true={true_str})",
                                 fontsize=8, color=color, loc="left")

        fig.suptitle(f"Failure Cases — {model_name}", fontsize=14, y=1.01)
        fig.savefig(FIGS / f"composite_{safe_name}.pdf", bbox_inches="tight", dpi=150)
        fig.savefig(FIGS / f"composite_{safe_name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved composite: {safe_name}")

    pe_res = all_results["PE (solo)"]
    fcg_res = all_results["FCG (solo)"]

    fig = plt.figure(figsize=(22, 17))
    gs = gridspec.GridSpec(3, 4, hspace=0.78, wspace=0.18)
    row = 0

    for gen in ["Wan26", "Grok", "Wan22T2V"]:
        map_pe = {stem: (sc, lab) for stem, sc, lab in pe_res.get(gen, [])}
        map_fcg = {stem: (sc, lab) for stem, sc, lab in fcg_res.get(gen, [])}
        common = set(map_pe.keys()) & set(map_fcg.keys())

        pe_right = [(s, map_pe[s][0], map_fcg[s][0], map_pe[s][1])
                     for s in common
                     if map_pe[s][1] == 1 and map_pe[s][0] > 0.5 and map_fcg[s][0] < 0.5]
        pe_right.sort(key=lambda x: x[1] - x[2], reverse=True)

        if pe_right:
            stem, sc_pe, sc_fcg, lab = pe_right[0]
            video = find_dfd_video(stem, gen, lab)
            frames = extract_frames(video, 4) if video else []
            for j in range(4):
                ax = fig.add_subplot(gs[row, j])
                if j < len(frames):
                    ax.imshow(frames[j])
                ax.axis("off")
                if j == 0:
                    ax.set_title(
                        f"{gen}: PE={sc_pe:.2f} FCG={sc_fcg:.2f}\n"
                        f"(FAKE — PE correct, FCG wrong)",
                        fontsize=FONT_PEVSFCG_TITLE,
                        color="blue",
                        loc="left",
                    )
        row += 1

    fig.suptitle(
        "PE Detects What FCG Misses — Example Fakes Across Generators",
        fontsize=FONT_PEVSFCG_SUPTITLE,
        y=1.02,
    )
    fig.savefig(FIGS / "pe_vs_fcg_comparison.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(FIGS / "pe_vs_fcg_comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved pe_vs_fcg_comparison")

    print(f"\nAll failure figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
