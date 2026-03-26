"""
DeepfakeDetector — main LightningModule.

Composes branch encoders + fusion + classifier.
Supports mixup and label smoothing.
Mean AUC = avg(FSh, DFDC, DFD_mean) where DFD_mean averages 8 generators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models.branches import build_branch
from models.fusion import build_fusion, ConcatFusion


def _eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    try:
        return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        return 0.5


class DeepfakeDetector(pl.LightningModule):

    CORE_DATASETS = {"FSh", "DFDC"}

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-3))

        self.use_mixup = cfg.get("use_mixup", False)
        self.mixup_alpha = float(cfg.get("mixup_alpha", 0.4))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))

        hidden_dim = cfg.get("hidden_dim", 128)
        dropout = float(cfg.get("dropout", 0.1))

        all_branch_keys = ["paper", "depth", "pe", "ct"]
        self.branch_names = [b for b in all_branch_keys if cfg.get(f"use_{b}")]
        assert self.branch_names, "Enable at least one branch"

        ct_variant = cfg.get("ct_variant", "full")
        ct_n_kp = cfg.get("ct_n_keypoints", 68)
        ct_feat_dim = cfg.get("ct_feat_dim", 8)
        from data.dataset import CT_VARIANT_DIMS, MOUTH_EYES_IDX, DIST_LANDMARKS
        if ct_variant == "mouth_eyes":
            ct_n_kp = len(MOUTH_EYES_IDX)
        if ct_variant == "inter_dist":
            n_dist = len(DIST_LANDMARKS)
            ct_n_kp = 1
            ct_feat_dim = n_dist * (n_dist - 1) // 2
        elif ct_variant in CT_VARIANT_DIMS and CT_VARIANT_DIMS[ct_variant] is not None:
            ct_feat_dim = CT_VARIANT_DIMS[ct_variant]

        self.encoders = nn.ModuleDict()
        branch_dims = {}
        for name in self.branch_names:
            extra_kw = {}
            if name == "ct":
                extra_kw = dict(n_keypoints=ct_n_kp, feat_dim=ct_feat_dim)
            enc = build_branch(name, hidden_dim=hidden_dim, dropout=dropout, **extra_kw)
            self.encoders[name] = enc
            branch_dims[name] = enc.output_dim

        fusion_type = cfg.get("fusion", "concat")
        if len(branch_dims) == 1:
            self.fusion = ConcatFusion(branch_dims)
        else:
            self.fusion = build_fusion(
                fusion_type, branch_dims, hidden_dim=hidden_dim)

        fused_dim = self.fusion.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(float(cfg.get("head_dropout", 0.3))),
            nn.Linear(fused_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.val_names = []
        self._val_buf = {}

    def encode(self, feats):
        return {name: self.encoders[name](feats[name]) for name in self.encoders}

    def forward(self, feats):
        embs = self.encode(feats)
        fused = self.fusion(embs)
        return self.classifier(fused)

    def get_fused(self, feats):
        embs = self.encode(feats)
        return self.fusion(embs)

    def training_step(self, batch, batch_idx):
        feats, labels, _ = batch

        if self.use_mixup and np.random.random() < 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            perm = torch.randperm(labels.size(0), device=labels.device)
            mixed = {k: lam * v + (1 - lam) * v[perm] for k, v in feats.items()}
            logits = self(mixed)
            oh = F.one_hot(labels, 2).float()
            target = lam * oh + (1 - lam) * oh[perm]
            loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        else:
            logits = self(feats)
            loss = F.cross_entropy(
                logits, labels, label_smoothing=self.label_smoothing)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        feats, labels, _ = batch
        probs = F.softmax(self(feats), dim=-1)[:, 1]
        key = f"v{dataloader_idx}"
        if key not in self._val_buf:
            self._val_buf[key] = {"p": [], "l": []}
        self._val_buf[key]["p"].append(probs.cpu())
        self._val_buf[key]["l"].append(labels.cpu())

    def on_validation_epoch_end(self):
        aucs, eers = {}, {}
        for i, name in enumerate(self.val_names):
            key = f"v{i}"
            if key not in self._val_buf:
                continue
            p = torch.cat(self._val_buf[key]["p"]).numpy()
            l = torch.cat(self._val_buf[key]["l"]).numpy()
            try:
                auc = roc_auc_score(l, p)
            except ValueError:
                auc = 0.5
            eer = _eer(l, p)
            aucs[name] = auc
            eers[name] = eer
            is_core = name in self.CORE_DATASETS
            self.log(f"val/{name}/auc", auc, prog_bar=is_core)
            self.log(f"val/{name}/eer", eer)

        gen_names = [n for n in self.val_names if n not in self.CORE_DATASETS]
        gen_aucs = [aucs[n] for n in gen_names if n in aucs]
        dfd_mean = sum(gen_aucs) / len(gen_aucs) if gen_aucs else 0.0
        self.log("val/dfd_mean_auc", dfd_mean)

        groups = [aucs.get("FSh", 0.5), aucs.get("DFDC", 0.5), dfd_mean]
        mean = sum(groups) / len(groups)
        self.log("val/mean_auc", mean, prog_bar=True)

        parts = [f"FSh={aucs.get('FSh', 0):.3f}", f"DFDC={aucs.get('DFDC', 0):.3f}"]
        print(f"  [Ep {self.current_epoch:02d}] {' | '.join(parts)}"
              f" | DFD={dfd_mean:.3f} | mean={mean:.3f}")
        self._val_buf = {}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]
