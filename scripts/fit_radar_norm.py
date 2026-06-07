"""Fit per-channel radar normalization statistics on the TRAIN split.

Why: per-sample instance norm strips the absolute magnitude level that encodes blockage
(radar AUC 0.41 in the first run). Global train-set stats fix this. See improvement-plan.md.

Run:  .venv/Scripts/python.exe -m scripts.fit_radar_norm
Output: outputs/radar_norm.npz  (mean (8,), std (8,))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.splits import split_from_config
from src.data.radar_features import channel_stats

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    rcfg = yaml.safe_load((ROOT / "configs" / "radar.yaml").read_text())
    n_samples = int(rcfg.get("norm", {}).get("fit_samples", 800))
    out_path = ROOT / rcfg.get("norm", {}).get("path", "outputs/radar_norm.npz")

    csv = ROOT / dcfg["paths"]["csv"]
    data_root = ROOT / dcfg["paths"]["data_root"]
    df = pd.read_csv(csv)
    lcol = dcfg["label"]["column"]; seqc = dcfg.get("seq_col", "seq_index")
    res = split_from_config(df, dcfg)
    train_rows = df[df[seqc].isin(res.seqs["train"])]

    # sample frames (include all positives so blocked magnitude is represented)
    pos = train_rows[train_rows[lcol] == dcfg["label"]["positive"]]
    neg = train_rows[train_rows[lcol] != dcfg["label"]["positive"]]
    rng = np.random.default_rng(dcfg["split"]["seed"])
    n_neg = max(0, n_samples - len(pos))
    neg_s = neg.sample(min(n_neg, len(neg)), random_state=dcfg["split"]["seed"])
    sample = pd.concat([pos, neg_s])
    print(f"fitting radar norm on {len(sample)} train frames ({len(pos)} blocked + {len(neg_s)} clear)")

    csum = np.zeros(8); csumsq = np.zeros(8); npix = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        raw = np.load(data_root / str(row["unit1_radar"]).lstrip("./"))
        s, sq, n = channel_stats(raw)
        csum += s; csumsq += sq; npix += n
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sample)}")

    mean = csum / npix
    std = np.sqrt(np.maximum(csumsq / npix - mean ** 2, 1e-12))
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path, mean=mean, std=std)
    print("per-channel mean:", np.round(mean, 4))
    print("per-channel std :", np.round(std, 4))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
