"""
Experimental training with toolkit concepts:
- Temporal augmentation (frame drop/jitter) for PE
- Supervised contrastive loss
- Variable hidden size
- L2 normalization on embeddings

Usage:
  python train_exp.py --config configs/03_pe.yaml --gpu 0 --hidden_dim 256
  python train_exp.py --config configs/03_pe.yaml --gpu 0 --temp_aug --supcon 0.1
"""

import os, argparse, yaml, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from data.dataset import build_dataloaders
from models.branches import build_branch
from models.fusion import build_fusion, ConcatFusion


def _eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    try:
        return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        return 0.5


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.t()) / self.temperature
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)
        pos_count = mask.sum(dim=1).clamp(min=1)
        log_prob = sim - torch.logsumexp(sim.masked_fill(
            torch.eye(len(labels), device=sim.device).bool(), -1e9), dim=1, keepdim=True)
        loss = -(log_prob * mask.float()).sum(dim=1) / pos_count
        return loss.mean()


class ExpDetector(pl.LightningModule):
    CORE_DATASETS = {"FSh", "DFDC"}

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-3))

        hidden_dim = cfg.get("hidden_dim", 128)
        dropout = float(cfg.get("dropout", 0.1))
        self.temp_aug = cfg.get("temp_aug", False)
        self.temp_drop = float(cfg.get("temp_drop", 0.2))
        self.temp_jitter = float(cfg.get("temp_jitter", 0.3))
        self.supcon_weight = float(cfg.get("supcon_weight", 0.0))
        self.l2_norm = cfg.get("l2_norm", False)

        all_branch_keys = ["paper", "depth", "pe", "ct"]
        self.branch_names = [b for b in all_branch_keys if cfg.get(f"use_{b}")]
        assert self.branch_names

        ct_variant = cfg.get("ct_variant", "full")
        self.encoders = nn.ModuleDict()
        branch_dims = {}
        for name in self.branch_names:
            kw = dict(hidden_dim=hidden_dim, dropout=dropout)
            if name == "ct":
                from data.dataset import CT_VARIANT_DIMS, MOUTH_EYES_IDX, DIST_LANDMARKS
                if ct_variant == "mouth_eyes":
                    kw["n_keypoints"] = len(MOUTH_EYES_IDX)
                elif ct_variant == "inter_dist":
                    n_dist = len(DIST_LANDMARKS)
                    kw["n_keypoints"] = 1
                    kw["feat_dim"] = n_dist * (n_dist - 1) // 2
                elif ct_variant in CT_VARIANT_DIMS and CT_VARIANT_DIMS[ct_variant] is not None:
                    kw["feat_dim"] = CT_VARIANT_DIMS[ct_variant]
            enc = build_branch(name, **kw)
            self.encoders[name] = enc
            branch_dims[name] = enc.output_dim

        fusion_type = cfg.get("fusion", "concat")
        if len(branch_dims) == 1:
            self.fusion = ConcatFusion(branch_dims)
        else:
            self.fusion = build_fusion(fusion_type, branch_dims, hidden_dim=hidden_dim)

        fused_dim = self.fusion.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(float(cfg.get("head_dropout", 0.3))),
            nn.Linear(fused_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        if self.supcon_weight > 0:
            self.supcon_loss = SupConLoss(temperature=0.07)

        self.val_names = []
        self._val_buf = {}

    def _temporal_augment(self, feats):
        if not self.training or not self.temp_aug:
            return feats
        out = {}
        for k, v in feats.items():
            if v.dim() == 3:
                B, T, D = v.shape
                if np.random.random() < self.temp_drop:
                    keep = torch.rand(B, T, device=v.device) > 0.2
                    keep[:, 0] = True
                    v = v * keep.unsqueeze(-1).float()
                    counts = keep.float().sum(dim=1, keepdim=True).clamp(min=1)
                    v = v * (T / counts)
                if np.random.random() < self.temp_jitter:
                    perm = torch.stack([torch.randperm(T) for _ in range(B)])
                    v = v[torch.arange(B).unsqueeze(1), perm]
            out[k] = v
        return out

    def encode(self, feats):
        return {name: self.encoders[name](feats[name]) for name in self.encoders}

    def forward(self, feats):
        embs = self.encode(feats)
        fused = self.fusion(embs)
        return self.classifier(fused)

    def get_fused(self, feats):
        embs = self.encode(feats)
        fused = self.fusion(embs)
        if self.l2_norm:
            fused = F.normalize(fused, dim=1)
        return fused

    def training_step(self, batch, batch_idx):
        feats, labels, _ = batch
        feats = self._temporal_augment(feats)
        logits = self(feats)
        loss = F.cross_entropy(logits, labels)

        if self.supcon_weight > 0:
            embs = self.encode(feats)
            fused = self.fusion(embs)
            sc_loss = self.supcon_loss(fused, labels)
            loss = loss + self.supcon_weight * sc_loss
            self.log("train/supcon", sc_loss)

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
        aucs = {}
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
            aucs[name] = auc
            self.log(f"val/{name}/auc", auc, prog_bar=(name in self.CORE_DATASETS))

        gen_names = [n for n in self.val_names if n not in self.CORE_DATASETS]
        gen_aucs = [aucs[n] for n in gen_names if n in aucs]
        dfd_mean = sum(gen_aucs) / len(gen_aucs) if gen_aucs else 0.0
        self.log("val/dfd_mean_auc", dfd_mean)
        groups = [aucs.get("FSh", 0.5), aucs.get("DFDC", 0.5), dfd_mean]
        mean = sum(groups) / len(groups)
        self.log("val/mean_auc", mean, prog_bar=True)
        print(f"  [Ep {self.current_epoch:02d}] FSh={aucs.get('FSh',0):.3f} | DFDC={aucs.get('DFDC',0):.3f}"
              f" | DFD={dfd_mean:.3f} | mean={mean:.3f}")
        self._val_buf = {}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]


def _load_wandb_key():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "wandb_api_key" in line.lower() and "=" in line:
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--temp_aug", action="store_true")
    parser.add_argument("--supcon", type=float, default=0.0)
    parser.add_argument("--l2_norm", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
    if args.temp_aug:
        cfg["temp_aug"] = True
    if args.supcon > 0:
        cfg["supcon_weight"] = args.supcon
    if args.l2_norm:
        cfg["l2_norm"] = True
    if args.lr is not None:
        cfg["lr"] = args.lr
    epochs = args.epochs or cfg.get("epochs", 25)

    base_name = cfg.get("name", Path(args.config).stem)
    suffix = ""
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    run_name = args.run_name or f"{base_name}{suffix}"
    cfg["name"] = run_name

    train_loader, val_names, val_loaders = build_dataloaders(cfg)
    cfg["val_names"] = val_names
    model = ExpDetector(cfg)
    model.val_names = val_names

    import os
    if os.environ.get("WANDB_MODE") != "disabled":
        _load_wandb_key()
        try:
            import wandb
            wandb.login()
            logger = pl.loggers.WandbLogger(
                project="Fusion-Final", name=run_name, entity="waylex-ucu", config=cfg)
        except Exception:
            logger = False
    else:
        logger = False

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="gpu", devices=[args.gpu],
        precision="16-mixed", logger=logger,
        default_root_dir=str(Path(__file__).resolve().parent / "logs"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir), monitor="val/mean_auc",
                mode="max", save_top_k=1, filename="best-{epoch:02d}"),
            pl.callbacks.EarlyStopping(
                monitor="val/mean_auc", patience=10, mode="max"),
        ],
        gradient_clip_val=1.0, num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loaders)
    print(f"\nBest checkpoint: {ckpt_dir}")
    print(f"Best val/mean_auc: {trainer.callback_metrics.get('val/mean_auc', 0):.4f}")


if __name__ == "__main__":
    main()
