"""Sequence-correct window dataset for Scenario 31 blockage prediction (camera + radar).

v2 loading architecture (see ML_Proj_Vault/plan/improvement-plan.md):
  * **dense anchoring** — every native frame is a window-end (not `frames[::3]`), so we keep all of
    the rare positives instead of discarding ~2/3 of them;
  * **timestamp-based frame selection** — the W input frames and K horizon labels are chosen by the
    real `time_stamp` (closest frame to anchor ± j*300ms), not a fixed row stride, so each step is a
    true ~300ms despite irregular native spacing (~92ms median, with dropped-frame doubles);
  * **timestamp coherence guard** — a window is rejected if any target time has no frame within a
    tolerance, so a window never silently spans a time gap / scene seam;
  * windows never cross a `seq_index` (scene) boundary;
  * **train-set radar normalization** — pass per-channel stats fit on train (instance-norm fallback).

Each item:
  camera : (W, 3, H, H) float32   (if "camera" in modalities)
  radar  : (W, 8, 256, 64) float32 (if "radar" in modalities)
  label  : (K,) float32  binary blockage at the K timestamped horizon steps t+1..t+K
  meta   : dict (seq_index, anchor index)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .radar_features import radar_raw_to_features, add_gaussian_noise


def _basename_join(data_root: Path, csv_path_entry: str) -> Path:
    rel = str(csv_path_entry).lstrip("./").replace("\\", "/")
    return data_root / rel


def parse_time_seconds(ts: pd.Series) -> np.ndarray:
    """'HH:MM:SS-microseconds' -> float seconds."""
    hms = ts.str.split("-").str[0]
    us = ts.str.split("-").str[1].astype(float) / 1e6
    h = hms.str.split(":").str[0].astype(int)
    m = hms.str.split(":").str[1].astype(int)
    s = hms.str.split(":").str[2].astype(int)
    return (h * 3600 + m * 60 + s).to_numpy() + us.to_numpy()


def _nearest(t_arr: np.ndarray, target: float) -> tuple[int, float]:
    """Index of the timestamp closest to `target` (t_arr sorted asc) + the abs error."""
    pos = int(np.searchsorted(t_arr, target))
    best, besterr = -1, np.inf
    for k in (pos - 1, pos):
        if 0 <= k < len(t_arr):
            e = abs(t_arr[k] - target)
            if e < besterr:
                best, besterr = k, e
    return best, besterr


def build_windows(
    df: pd.DataFrame,
    keep_seqs: list[int],
    W: int,
    K: int,
    step_s: float = 0.30,
    tol_s: float = 0.13,
    anchor_stride: int = 1,
    seq_col: str = "seq_index",
    index_col: str = "index",
    time_col: str = "_t",
) -> list[dict]:
    """Dense, timestamp-based windows that never cross a sequence boundary or a time gap.

    For each anchor frame (every `anchor_stride`-th frame), inputs are the frames closest to
    anchor_time - j*step_s (j=W-1..0) and labels are the frames closest to anchor_time + j*step_s
    (j=1..K). The window is kept only if every target has a frame within `tol_s`.
    """
    windows: list[dict] = []
    for seq in keep_seqs:
        g = df[df[seq_col] == seq].sort_values(index_col)
        rows = g.index.to_numpy()
        t = g[time_col].to_numpy()
        n = len(rows)
        for i in range(0, n, anchor_stride):
            ta = t[i]
            win, hor, ok = [], [], True
            for j in range(W - 1, -1, -1):                 # inputs: earliest -> anchor
                k, err = _nearest(t, ta - j * step_s)
                if err > tol_s:
                    ok = False
                    break
                win.append(rows[k])
            if not ok:
                continue
            for j in range(1, K + 1):                       # horizon labels
                k, err = _nearest(t, ta + j * step_s)
                if err > tol_s:
                    ok = False
                    break
                hor.append(rows[k])
            if ok:
                windows.append({"seq": seq, "win": win, "hor": hor, "anchor": int(g.at[rows[i], index_col])})
    return windows


class BlockageWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        keep_seqs: list[int],
        W: int = 5,
        K: int = 5,
        step_s: float = 0.30,
        tol_s: float = 0.13,
        modalities: tuple[str, ...] = ("camera", "radar"),
        image_size: int = 256,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
        augment: bool = False,
        radar_noise_sigma: float = 0.0,
        radar_norm: tuple[np.ndarray, np.ndarray] | None = None,
        label_col: str = "label",
        positive: str = "blocked",
        seq_col: str = "seq_index",
    ):
        df = df.copy()
        if "_t" not in df.columns:
            df["_t"] = parse_time_seconds(df["time_stamp"])
        self.df = df
        self.data_root = Path(data_root)
        self.W, self.K = W, K
        self.modalities = tuple(modalities)
        self.augment = augment
        self.radar_noise_sigma = radar_noise_sigma
        self.radar_norm = radar_norm
        self._rng = np.random.default_rng(0)

        self.y = (df[label_col] == positive).astype(np.float32)
        self.windows = build_windows(df, keep_seqs, W, K, step_s, tol_s, seq_col=seq_col)

        tfm = [transforms.Resize((image_size, image_size))]
        if augment:
            tfm += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
                # day/night robustness: scenarios 31/32 are daytime, 33/34 night (see
                # cross-scenario-investigation.md). Cheap stand-in for MIRNet enhancement.
                transforms.ColorJitter(brightness=0.4, contrast=0.3),
            ]
        tfm += [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
        self.cam_tfm = transforms.Compose(tfm)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_camera(self, row_labels) -> torch.Tensor:
        frames = []
        for r in row_labels:
            p = _basename_join(self.data_root, self.df.at[r, "unit1_rgb"])
            with Image.open(p) as im:
                frames.append(self.cam_tfm(im.convert("RGB")))
        return torch.stack(frames, 0)

    def _load_radar(self, row_labels) -> torch.Tensor:
        frames = []
        for r in row_labels:
            p = _basename_join(self.data_root, self.df.at[r, "unit1_radar"])
            raw = np.load(p)
            if self.augment and self.radar_noise_sigma > 0:
                raw = add_gaussian_noise(raw, self.radar_noise_sigma, self._rng)
            feats = radar_raw_to_features(raw, stats=self.radar_norm,
                                          normalize=(self.radar_norm is None))
            frames.append(torch.from_numpy(feats))
        return torch.stack(frames, 0)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        out: dict[str, object] = {}
        if "camera" in self.modalities:
            out["camera"] = self._load_camera(w["win"])
        if "radar" in self.modalities:
            out["radar"] = self._load_radar(w["win"])
        out["label"] = torch.tensor([self.y[r] for r in w["hor"]], dtype=torch.float32)
        out["meta"] = {"seq": w["seq"], "anchor": w["anchor"]}
        return out


def compute_pos_weight(dataset: BlockageWindowDataset) -> torch.Tensor:
    """Per-horizon pos_weight = N0/N1 over this dataset's windows (for weighted BCE)."""
    Y = np.stack([np.array([dataset.y[r] for r in w["hor"]]) for w in dataset.windows])
    n1 = Y.sum(0)
    n0 = Y.shape[0] - n1
    return torch.tensor(n0 / np.maximum(n1, 1.0), dtype=torch.float32)


def make_loaders(
    csv: str | Path,
    data_root: str | Path,
    seqs_by_split: dict[str, list[int]],
    W: int = 5,
    K: int = 5,
    step_s: float = 0.30,
    tol_s: float = 0.13,
    modalities=("camera", "radar"),
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: int = 256,
    radar_noise_sigma: float = 0.01,
    radar_norm: tuple[np.ndarray, np.ndarray] | None = None,
    persistent_workers: bool = False,
    balanced_sampler: bool = False,
    **kw,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from a sequence-level split assignment.

    `persistent_workers` (config-driven) keeps the TRAIN workers alive across epochs (big speedup on
    Windows where spawning re-imports torch each epoch). Val/test loaders never persist, so workers
    don't accumulate at eval time (which previously exhausted the paging file).
    """
    df = pd.read_csv(csv)
    df["_t"] = parse_time_seconds(df["time_stamp"])
    pin = torch.cuda.is_available()
    loaders: dict[str, DataLoader] = {}
    for split, seqs in seqs_by_split.items():
        ds = BlockageWindowDataset(
            df=df, data_root=data_root, keep_seqs=seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
            modalities=modalities, image_size=image_size,
            augment=(split == "train"),
            radar_noise_sigma=(radar_noise_sigma if split == "train" else 0.0),
            radar_norm=radar_norm,
            **kw,
        )
        persist = persistent_workers and split == "train" and num_workers > 0
        sampler = None
        if balanced_sampler and split == "train":
            # equalize pos/neg windows per batch (window positive if any horizon step blocked);
            # replaces extreme pos_weight in the loss, which wrecked probability calibration.
            wy = np.array([float(any(ds.y[r] for r in w["hor"])) for w in ds.windows])
            class_w = {0.0: 1.0 / max((wy == 0).sum(), 1), 1.0: 1.0 / max((wy == 1).sum(), 1)}
            weights = torch.tensor([class_w[v] for v in wy], dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ds),
                                                             replacement=True)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=persist,
            prefetch_factor=(2 if num_workers > 0 else None),
        )
    return loaders
