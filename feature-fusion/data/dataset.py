"""
Dataset classes for training and evaluation.

Train: FF++ c23 (real, DF, F2F, FS, NT)
Eval:  FaceShifter, DFDC, DeepFakeDataset v1.1 (8 generators)

Four feature branches: paper (1280-d), depth (768-d), pe (1024-d), ct (T,68,8).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from data.paths import (
    DATASETS, FFPP, FFPP_C23, FFPP_TRAIN_METHODS,
    DFDC, DFD_V11, DFD_GENERATORS, feature_dirs, dfd_dirs,
)

TYPE_MAP = {"real": 0, "DF": 1, "F2F": 2, "FS": 3, "NT": 4, "FSh": 5, "fake": 6}
ALL_BRANCHES = ["paper", "depth", "pe", "ct"]


def _uniform_sample(feat, n):
    t = feat.shape[0]
    if t >= n:
        idx = torch.linspace(0, t - 1, n).long()
        return feat[idx]
    pad = torch.zeros(n - t, *feat.shape[1:], dtype=feat.dtype)
    return torch.cat([feat, pad])


def _find_file(base, type_tag, stem, label):
    if base is None:
        return None
    candidates = [
        base / type_tag / FFPP_C23 / f"{stem}.pt",
        base / type_tag / f"{stem}.pt",
        base / ("real" if label == 0 else "fake") / f"{stem}.pt",
        base / f"{stem}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_paper(path):
    if path is None or not path.is_file():
        return torch.zeros(1280)
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d.get("syno_s", torch.zeros(1024)).float().flatten()
    t = d.get("syno_t", torch.zeros(256)).float().flatten()
    return torch.cat([s, t])


def _load_depth(path, n_frames):
    if path is None or not path.is_file():
        return torch.zeros(n_frames, 768)
    d = torch.load(path, map_location="cpu", weights_only=False)
    return _uniform_sample(d["depth"].float(), n_frames)


def _load_pe(path, n_frames):
    if path is None or not path.is_file():
        return torch.zeros(n_frames, 1024)
    d = torch.load(path, map_location="cpu", weights_only=False)
    return _uniform_sample(d["pe"].float(), n_frames)


CT_VARIANT_DIMS = {
    "full": 8,
    "raw_pos": 2,
    "vel_acc": 4,
    "pos_vel": 4,
    "mouth_eyes": 8,
    "inter_dist": None,
}

MOUTH_EYES_IDX = list(range(36, 61))
DIST_LANDMARKS = [0, 8, 16, 17, 21, 22, 26, 27, 30, 33, 36, 45]


def _compute_kinematics(pos, vis, ct_variant):
    """Compute kinematic features from consecutive position frames.

    Velocity and acceleration are frame-to-frame finite differences, so they
    are only meaningful when *pos* contains truly consecutive frames.
    """
    T = pos.shape[0]
    vel = torch.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = torch.zeros_like(pos)
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    centroid = pos.mean(dim=1, keepdim=True)
    cdelta = (pos - centroid).norm(dim=-1, keepdim=True)
    cdelta_dt = torch.zeros_like(cdelta)
    cdelta_dt[1:] = cdelta[1:] - cdelta[:-1]

    if ct_variant == "raw_pos":
        return pos
    elif ct_variant == "vel_acc":
        return torch.cat([vel, acc], dim=-1)
    elif ct_variant == "pos_vel":
        return torch.cat([pos, vel], dim=-1)
    elif ct_variant == "inter_dist":
        idx = [i for i in DIST_LANDMARKS if i < pos.shape[1]]
        sub = pos[:, idx]  # (T, n_dist, 2)
        n = len(idx)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append((sub[:, i] - sub[:, j]).norm(dim=-1))
        return torch.stack(dists, dim=-1).unsqueeze(1)  # (T, 1, n_pairs)
    else:
        return torch.cat([pos, vel, acc, cdelta_dt, vis.unsqueeze(-1)], dim=-1)


def _extract_windows(data, n_windows, window_size):
    """Extract n_windows consecutive windows of window_size from data.

    Anchor points are spaced uniformly across the temporal extent.
    If the track is shorter than n_windows * window_size, windows overlap;
    if shorter than window_size, the data is zero-padded.
    """
    T = data.shape[0]
    total = n_windows * window_size
    if T < window_size:
        pad = torch.zeros(window_size - T, *data.shape[1:], dtype=data.dtype)
        data = torch.cat([data, pad])
        T = window_size

    if T <= total:
        anchors = torch.linspace(0, max(T - window_size, 0), n_windows).long()
    else:
        anchors = torch.linspace(0, T - window_size, n_windows).long()

    windows = []
    for a in anchors:
        windows.append(data[a : a + window_size])
    return torch.cat(windows, dim=0)


def _load_ct(path, n_frames, n_keypoints=68, ct_variant="full",
             ct_n_windows=0, ct_window_size=10, ct_stride=1,
             ct_max_track_frames=0):
    """Load CoTracker tracks and compute kinematic features.

    Sampling modes (controlled by ct_n_windows):
      ct_n_windows=0  – legacy uniform sampling to n_frames.
      ct_n_windows>0  – extract ct_n_windows windows from uniformly-spaced
                        anchor points.  Within each window, frames are taken
                        every ct_stride-th raw frame.  Kinematics (velocity,
                        acceleration) are computed on these strided frames so
                        that each finite difference spans a fixed, known
                        temporal interval regardless of total video length.
                        Total output frames = ct_n_windows * ct_window_size.

    The stride eliminates video-length bias: a stride of 3 at 25 fps always
    gives velocity over 120 ms, whether the video is 2 s or 30 s long.

    ct_max_track_frames>0 caps the raw track data to the first N frames,
    discarding later frames where tracking has drifted from the initial
    landmark positions or where chunk-boundary artefacts occur.
    """
    if ct_variant == "mouth_eyes":
        n_kp = len(MOUTH_EYES_IDX)
    elif ct_variant == "inter_dist":
        n_kp = 1
    else:
        n_kp = n_keypoints

    if ct_variant == "inter_dist":
        n_d = len(DIST_LANDMARKS)
        feat_dim = n_d * (n_d - 1) // 2
    else:
        feat_dim = CT_VARIANT_DIMS.get(ct_variant, 8)
    out_frames = ct_n_windows * ct_window_size if ct_n_windows > 0 else n_frames

    if path is None or not path.is_file():
        return torch.zeros(out_frames, n_kp, feat_dim)

    d = torch.load(path, map_location="cpu", weights_only=False)
    raw_tracks = d["tracks"].float()
    raw_vis = d["visibility"].float()

    if ct_max_track_frames > 0 and raw_tracks.shape[0] > ct_max_track_frames:
        raw_tracks = raw_tracks[:ct_max_track_frames]
        raw_vis = raw_vis[:ct_max_track_frames]

    N = raw_tracks.shape[1]

    if ct_variant == "mouth_eyes":
        kp_idx = [i for i in MOUTH_EYES_IDX if i < N]
        raw_tracks = raw_tracks[:, kp_idx]
        raw_vis = raw_vis[:, kp_idx]
        N = len(kp_idx)
        n_kp = N

    pos_norm = raw_tracks / 75.0 - 1.0

    if ct_n_windows > 0:
        T = pos_norm.shape[0]
        raw_span = ct_window_size * ct_stride
        needed = max(raw_span, 1)

        if T < needed:
            pad_len = needed - T
            pos_norm = torch.cat([pos_norm,
                                  torch.zeros(pad_len, *pos_norm.shape[1:])])
            raw_vis = torch.cat([raw_vis,
                                 torch.zeros(pad_len, *raw_vis.shape[1:])])
            T = needed

        total_raw = ct_n_windows * raw_span
        if T <= total_raw:
            anchors = torch.linspace(0, max(T - raw_span, 0),
                                     ct_n_windows).long()
        else:
            anchors = torch.linspace(0, T - raw_span,
                                     ct_n_windows).long()

        window_feats = []
        for a in anchors:
            idx = torch.arange(ct_window_size) * ct_stride + a
            w_pos = pos_norm[idx]
            w_vis = raw_vis[idx]
            window_feats.append(_compute_kinematics(w_pos, w_vis, ct_variant))
        feats = torch.cat(window_feats, dim=0)
    else:
        sampled_pos = _uniform_sample(pos_norm, n_frames)
        sampled_vis = _uniform_sample(raw_vis, n_frames)
        feats = _compute_kinematics(sampled_pos, sampled_vis, ct_variant)

    if N != n_kp:
        out = torch.zeros(out_frames, n_kp, feats.shape[-1])
        out[:, :N, :] = feats
        return out
    return feats


_LOADERS = {
    "paper": lambda path, nf, pf, ctf, **kw: _load_paper(path),
    "depth": lambda path, nf, pf, ctf, **kw: _load_depth(path, nf),
    "pe":    lambda path, nf, pf, ctf, **kw: _load_pe(path, pf),
    "ct":    lambda path, nf, pf, ctf, **kw: _load_ct(
        path, ctf, ct_variant=kw.get("ct_variant", "full"),
        ct_n_windows=kw.get("ct_n_windows", 0),
        ct_window_size=kw.get("ct_window_size", 10),
        ct_stride=kw.get("ct_stride", 1),
        ct_max_track_frames=kw.get("ct_max_track_frames", 0)),
}


class FFppDataset(Dataset):
    def __init__(self, methods, feat_paths, branches, n_frames=10, pe_frames=16,
                 ct_frames=None, ct_variant="full",
                 ct_n_windows=0, ct_window_size=10, ct_stride=1,
                 ct_max_track_frames=0):
        self.branches = branches
        self.n_frames = n_frames
        self.pe_frames = pe_frames
        self.ct_frames = ct_frames or n_frames
        self.feat_paths = feat_paths
        self.ct_variant = ct_variant
        self.ct_n_windows = ct_n_windows
        self.ct_window_size = ct_window_size
        self.ct_stride = ct_stride
        self.ct_max_track_frames = ct_max_track_frames
        self.samples = []
        for method in methods:
            label = 0 if method == "real" else 1
            aux_dir = FFPP / "aux_features" / method / FFPP_C23
            if not aux_dir.is_dir():
                continue
            for pt in sorted(aux_dir.glob("*.pt")):
                self.samples.append((pt.stem, label, method))
        print(f"  FFpp [{','.join(methods)}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label, method = self.samples[idx]
        feats = {}
        for branch in self.branches:
            path = _find_file(self.feat_paths.get(branch), method, stem, label)
            feats[branch] = _LOADERS[branch](
                path, self.n_frames, self.pe_frames, self.ct_frames,
                ct_variant=self.ct_variant,
                ct_n_windows=self.ct_n_windows,
                ct_window_size=self.ct_window_size,
                ct_stride=self.ct_stride,
                ct_max_track_frames=self.ct_max_track_frames)
        return feats, label, TYPE_MAP.get(method, 6)


class BinaryDataset(Dataset):
    """For DFDC: real/fake binary split with pre-extracted features."""
    def __init__(self, name, feat_paths, branches, n_frames=10, pe_frames=16,
                 ct_frames=None, ct_variant="full",
                 ct_n_windows=0, ct_window_size=10, ct_stride=1,
                 ct_max_track_frames=0):
        self.branches = branches
        self.n_frames = n_frames
        self.pe_frames = pe_frames
        self.ct_frames = ct_frames or n_frames
        self.feat_paths = feat_paths
        self.ct_variant = ct_variant
        self.ct_n_windows = ct_n_windows
        self.ct_window_size = ct_window_size
        self.ct_stride = ct_stride
        self.ct_max_track_frames = ct_max_track_frames
        self.samples = []
        base_map = {"dfdc": DFDC}
        aux_root = base_map[name] / "aux_features"
        for sub, label in [("real", 0), ("fake", 1)]:
            d = aux_root / sub
            if not d.is_dir():
                continue
            for pt in sorted(d.glob("*.pt")):
                self.samples.append((pt.stem, label, sub))
        print(f"  {name}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label, sub = self.samples[idx]
        feats = {}
        for branch in self.branches:
            path = _find_file(self.feat_paths.get(branch), sub, stem, label)
            feats[branch] = _LOADERS[branch](
                path, self.n_frames, self.pe_frames, self.ct_frames,
                ct_variant=self.ct_variant,
                ct_n_windows=self.ct_n_windows,
                ct_window_size=self.ct_window_size,
                ct_stride=self.ct_stride,
                ct_max_track_frames=self.ct_max_track_frames)
        return feats, label, TYPE_MAP.get(sub, 6)


class DFDDataset(Dataset):
    """DeepFakeDataset v1.1 — per-generator evaluation."""
    def __init__(self, generator, branches, n_frames=10, pe_frames=16,
                 ct_frames=None, ct_variant="full",
                 ct_n_windows=0, ct_window_size=10, ct_stride=1,
                 ct_max_track_frames=0):
        self.branches = branches
        self.n_frames = n_frames
        self.pe_frames = pe_frames
        self.ct_frames = ct_frames or n_frames
        self.ct_variant = ct_variant
        self.ct_n_windows = ct_n_windows
        self.ct_window_size = ct_window_size
        self.ct_stride = ct_stride
        self.ct_max_track_frames = ct_max_track_frames
        self.samples = []

        dirs = dfd_dirs(generator)
        real_aux = dirs["real_aux"]
        fake_aux = dirs["fake_aux"]

        if real_aux.is_dir():
            for pt in sorted(real_aux.glob("*.pt")):
                paths = {b: dirs[f"real_{b}"] / f"{pt.stem}.pt" for b in ALL_BRANCHES}
                self.samples.append((pt.stem, 0, paths))
        if fake_aux.is_dir():
            for pt in sorted(fake_aux.glob("*.pt")):
                paths = {b: dirs[f"fake_{b}"] / f"{pt.stem}.pt" for b in ALL_BRANCHES}
                self.samples.append((pt.stem, 1, paths))

        print(f"  DFD {generator}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label, paths = self.samples[idx]
        feats = {}
        for branch in self.branches:
            p = paths.get(branch)
            if p is not None and not p.is_file():
                p = None
            feats[branch] = _LOADERS[branch](
                p, self.n_frames, self.pe_frames, self.ct_frames,
                ct_variant=self.ct_variant,
                ct_n_windows=self.ct_n_windows,
                ct_window_size=self.ct_window_size,
                ct_stride=self.ct_stride,
                ct_max_track_frames=self.ct_max_track_frames)
        return feats, label, 6


def _collate(batch):
    feats = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    types = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return feats, labels, types


def build_dataloaders(cfg):
    branches = [b for b in ALL_BRANCHES if cfg.get(f"use_{b}")]
    n_frames = cfg.get("n_frames", 10)
    pe_frames = cfg.get("pe_frames", 16)
    ct_frames = cfg.get("ct_frames", n_frames)
    bs = cfg.get("batch_size", 64)
    ct_variant = cfg.get("ct_variant", "full")
    ct_n_windows = cfg.get("ct_n_windows", 0)
    ct_window_size = cfg.get("ct_window_size", 10)
    ct_stride = cfg.get("ct_stride", 1)
    ct_max_track_frames = cfg.get("ct_max_track_frames", 0)
    fkw = dict(n_frames=n_frames, pe_frames=pe_frames, ct_frames=ct_frames,
               ct_variant=ct_variant,
               ct_n_windows=ct_n_windows, ct_window_size=ct_window_size,
               ct_stride=ct_stride, ct_max_track_frames=ct_max_track_frames)

    ffpp_feat = feature_dirs("ffpp")

    print("Building training dataset...")
    train_ds = FFppDataset(FFPP_TRAIN_METHODS, ffpp_feat, branches, **fkw)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=_collate)

    val_names, val_loaders = [], []

    print("Building FSh validation...")
    fsh_ds = FFppDataset(["real", "FSh"], ffpp_feat, branches, **fkw)
    val_names.append("FSh")
    val_loaders.append(DataLoader(
        fsh_ds, batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=_collate))

    print("Building DFDC validation...")
    dfdc_feat = feature_dirs("dfdc")
    dfdc_ds = BinaryDataset("dfdc", dfdc_feat, branches, **fkw)
    val_names.append("DFDC")
    val_loaders.append(DataLoader(
        dfdc_ds, batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=_collate))

    print("Building DeepFakeDataset v1.1 validation (per-generator)...")
    for gen_key in DFD_GENERATORS:
        ds = DFDDataset(gen_key, branches, **fkw)
        if len(ds) > 0:
            val_names.append(gen_key)
            val_loaders.append(DataLoader(
                ds, batch_size=bs, shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=_collate))

    return train_loader, val_names, val_loaders
