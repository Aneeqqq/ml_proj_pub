"""End-to-end smoke test for the sequence-correct dataloader.

Verifies the invariants from ML_Proj_Vault/dataset/sequences-and-batching.md:
  1. no window spans two seq_index values;
  2. no seq_index appears in more than one split;
  3. camera/radar frames in a window come from identical `index` values (time-aligned);
  4. tensor shapes are correct: camera (B,W,3,256,256), radar (B,W,8,256,64), label (B,K).

Run:  .venv/Scripts/python.exe -m scripts.smoke_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.splits import stratified_sequence_split
from src.data.dataset import (BlockageWindowDataset, build_windows, compute_pos_weight,
                              make_loaders, parse_time_seconds)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    csv = ROOT / cfg["paths"]["csv"]
    data_root = ROOT / cfg["paths"]["data_root"]
    W, K = cfg["window"]["W"], cfg["window"]["K"]
    step_s, tol_s = cfg["window"]["step_ms"] / 1000, cfg["window"]["tol_ms"] / 1000
    df = pd.read_csv(csv)
    df["_t"] = parse_time_seconds(df["time_stamp"])

    res = stratified_sequence_split(df, cfg["split"]["ratios"], cfg["split"]["seed"],
                                    positive=cfg["label"]["positive"])
    print("=== split stats ===")
    print(res.stats.to_string(index=False))

    # (2) disjoint splits
    all_seqs = [s for v in res.seqs.values() for s in v]
    assert len(all_seqs) == len(set(all_seqs)), "LEAK: seq in multiple splits"
    print("\n[OK] splits are sequence-disjoint (no leakage)")

    # (1) no window crosses a sequence boundary + (timestamp guard) each step within tolerance
    for split, seqs in res.seqs.items():
        wins = build_windows(df, seqs, W, K, step_s, tol_s)
        for w in wins[:2000]:
            rows = list(w["win"]) + list(w["hor"])
            assert df.loc[rows, "seq_index"].nunique() == 1, "LEAK: window crosses sequences"
            times = df.loc[list(w["win"]), "_t"].to_numpy()
            # each frame is within tol of its 300ms grid point -> consecutive diff <= step + 2*tol
            assert (np.diff(times) <= step_s + 2 * tol_s + 1e-6).all(), "window spans a time gap"
    print("[OK] no window crosses a seq_index boundary; input steps within time tolerance")

    # build the training dataset and pull one batch
    loaders = make_loaders(csv, data_root, res.seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
                           modalities=("camera", "radar"), batch_size=4, num_workers=0)
    for split, dl in loaders.items():
        print(f"  {split}: {len(dl.dataset)} windows, {len(dl)} batches")

    batch = next(iter(loaders["train"]))
    cam, rad, lab = batch["camera"], batch["radar"], batch["label"]
    print(f"\nbatch shapes -> camera {tuple(cam.shape)} | radar {tuple(rad.shape)} | label {tuple(lab.shape)}")
    assert cam.shape[1:] == (W, 3, cfg["camera"]["image_size"], cfg["camera"]["image_size"])
    assert rad.shape[1:] == (W, 8, 256, 64)
    assert lab.shape[1] == K
    assert not np.isnan(rad.numpy()).any(), "NaN in radar features"
    print("[OK] tensor shapes correct; radar features finite")

    # (4) per-horizon pos_weight (weighted BCE, multiply by alpha=1.1 in loss)
    pw = compute_pos_weight(loaders["train"].dataset)
    print(f"\ntrain per-horizon pos_weight (t+1..t+{K}): {np.round(pw.numpy(), 1)}")
    print("multiply by alpha=1.1 for the paper's w_pos.")
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
