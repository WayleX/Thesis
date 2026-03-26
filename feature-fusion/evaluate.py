"""
Evaluation and late fusion script.

Usage:
  python evaluate.py --config configs/03_pe.yaml                # single model
  python evaluate.py --configs configs/01_paper.yaml configs/03_pe.yaml --late  # late fusion
  python evaluate.py --all-late                                  # all pairwise late fusions
"""

import argparse, yaml, json, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from data.dataset import build_dataloaders
from data.paths import DFD_GENERATORS

BASE = Path(__file__).resolve().parent
DFD_GENS = list(DFD_GENERATORS.keys())


def _eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    try:
        return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        return 0.5


def _load_model(ckpt_path, cfg, device):
    """Try loading as ExpDetector first, then DeepfakeDetector."""
    from train_exp import ExpDetector
    try:
        m = ExpDetector.load_from_checkpoint(str(ckpt_path), cfg=cfg)
        return m.eval().to(device)
    except Exception:
        pass
    from models.detector import DeepfakeDetector
    return DeepfakeDetector.load_from_checkpoint(str(ckpt_path), cfg=cfg).eval().to(device)


def load_and_score(cfg_path, device="cuda:0"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _, val_names, val_loaders = build_dataloaders(cfg)
    cfg["val_names"] = val_names
    ckpt = list((BASE / "checkpoints" / cfg["name"]).glob("*.ckpt"))
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint for {cfg['name']}")
    model = _load_model(ckpt[0], cfg, device)
    scores = {}
    embeddings = {}
    with torch.no_grad():
        for vn, vl in zip(val_names, val_loaders):
            all_s, all_l, all_e = [], [], []
            for batch in vl:
                feats, labels, _ = batch
                feats = {k: v.to(device) for k, v in feats.items()}
                probs = F.softmax(model(feats), dim=-1)[:, 1]
                emb = model.get_fused(feats)
                all_s.append(probs.cpu())
                all_l.append(labels)
                all_e.append(emb.cpu())
            scores[vn] = (torch.cat(all_s).numpy(), torch.cat(all_l).numpy())
            embeddings[vn] = torch.cat(all_e).numpy()
    del model
    torch.cuda.empty_cache()
    return scores, embeddings, cfg["name"]


def compute_metrics(scores):
    results = {}
    for vn, (s, l) in scores.items():
        try:
            auc = roc_auc_score(l, s)
        except ValueError:
            auc = 0.5
        eer = _eer(l, s)
        results[vn] = {"auc": auc, "eer": eer}
    fsh = results.get("FSh", {}).get("auc", 0)
    dfdc = results.get("DFDC", {}).get("auc", 0)
    dfd = np.mean([results[g]["auc"] for g in DFD_GENS if g in results])
    results["DFD_mean"] = float(dfd)
    results["Mean_AUC"] = float((fsh + dfdc + dfd) / 3)
    return results


def late_fuse(score_list, weights=None):
    if weights is None:
        weights = [1.0 / len(score_list)] * len(score_list)
    combined = {}
    for vn in score_list[0]:
        s = sum(w * m[vn][0] for w, m in zip(weights, score_list))
        combined[vn] = (s, score_list[0][vn][1])
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--configs", nargs="+", type=str)
    parser.add_argument("--late", action="store_true")
    parser.add_argument("--all-late", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    if args.all_late:
        cfgs = sorted((BASE / "configs").glob("*.yaml"))
        all_scores = {}
        for cfg_path in cfgs:
            try:
                scores, embs, name = load_and_score(cfg_path, device)
                all_scores[name] = scores
                metrics = compute_metrics(scores)
                print(f"{name:<30} FSh={metrics['FSh']['auc']:.3f}  "
                      f"DFDC={metrics['DFDC']['auc']:.3f}  "
                      f"DFD={metrics['DFD_mean']:.3f}  "
                      f"Mean={metrics['Mean_AUC']:.3f}")
                if args.save:
                    out = BASE / "results" / f"{name}_scores.pt"
                    out.parent.mkdir(exist_ok=True)
                    torch.save({"scores": scores, "embeddings": embs}, out)
            except Exception as e:
                print(f"SKIP {cfg_path.stem}: {e}")

        names = list(all_scores.keys())
        print(f"\n{'='*60}\nALL PAIRWISE LATE FUSIONS\n{'='*60}")
        pairs = []
        for n1, n2 in combinations(names, 2):
            fused = late_fuse([all_scores[n1], all_scores[n2]])
            m = compute_metrics(fused)
            pairs.append((f"{n1}+{n2}", m))
        pairs.sort(key=lambda x: -x[1]["Mean_AUC"])
        for combo, m in pairs:
            print(f"{combo:<50} Mean={m['Mean_AUC']:.3f}")

    elif args.late and args.configs:
        score_list = []
        for cfg_path in args.configs:
            scores, embs, name = load_and_score(cfg_path, device)
            score_list.append(scores)
            print(f"  Loaded {name}")
        fused = late_fuse(score_list)
        metrics = compute_metrics(fused)
        print(f"\nLate fusion: Mean={metrics['Mean_AUC']:.3f}")

    elif args.config:
        scores, embs, name = load_and_score(args.config, device)
        metrics = compute_metrics(scores)
        print(f"\n{name}")
        for vn in sorted(metrics):
            v = metrics[vn]
            if isinstance(v, dict):
                print(f"  {vn}: AUC={v['auc']:.3f}  EER={v['eer']:.3f}")
            else:
                print(f"  {vn}: {v:.3f}")
        if args.save:
            out = BASE / "results" / f"{name}_scores.pt"
            out.parent.mkdir(exist_ok=True)
            torch.save({"scores": scores, "embeddings": embs, "metrics": metrics}, out)


if __name__ == "__main__":
    main()
