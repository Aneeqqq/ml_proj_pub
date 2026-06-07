"""Visualize what actually gets fed to the models for one window.

Produces:
  figs/input_camera.png  -- the 5-frame camera window (W,3,256,256), de-normalized for viewing
  figs/input_radar.png   -- the 8 radar feature-map channels (8,256,64) for the anchor frame

Picks a window whose horizon contains a blockage (most illustrative). See modalities/{camera,radar}.md.
Run:  .venv/Scripts/python.exe -m scripts.visualize_inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.splits import stratified_sequence_split
from src.data.dataset import BlockageWindowDataset

ROOT = Path(__file__).resolve().parents[1]
RADAR_CH = ["magnitude", "phase", "Doppler-FFT", "mean-mag", "std-mag",
            "entropy", "Doppler mean-vel", "spectral-width"]
IM_MEAN = np.array([0.485, 0.456, 0.406]); IM_STD = np.array([0.229, 0.224, 0.225])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ahead", "during"], default="ahead",
                    help="'ahead': blockage in the horizon (clear inputs); "
                         "'during': blockage present in the input frames")
    args = ap.parse_args()

    d = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    csv = ROOT / d["paths"]["csv"]; data_root = ROOT / d["paths"]["data_root"]
    W, K = d["window"]["W"], d["window"]["K"]
    step_s, tol_s = d["window"]["step_ms"] / 1000, d["window"]["tol_ms"] / 1000
    df = pd.read_csv(csv)

    # load train-set radar norm if available (so radar matches what the model is fed)
    norm_path = ROOT / "outputs" / "radar_norm.npz"
    radar_norm = None
    if norm_path.exists():
        z = np.load(norm_path); radar_norm = (z["mean"], z["std"])

    lcol = d.get("label", {}).get("column", "label")
    seqc = d.get("seq_col", "seq_index")
    # pick a sequence that contains blockage
    per = df.assign(p=(df[lcol] == "blocked").astype(int)).groupby(seqc).p.sum()
    blocked_seqs = per[per > 0].index.tolist()
    seq = blocked_seqs[len(blocked_seqs) // 2]

    ds = BlockageWindowDataset(df, data_root, keep_seqs=[seq], W=W, K=K, step_s=step_s, tol_s=tol_s,
                               modalities=("camera", "radar"), augment=False, radar_norm=radar_norm,
                               label_col=lcol, positive=d["label"]["positive"], seq_col=seqc)
    # pick the window to show
    if args.case == "during":
        # prefer a window where the anchor (last input frame) is blocked; else any input-frame blocked
        idx = next((i for i, w in enumerate(ds.windows) if ds.y[w["win"][-1]] == 1),
                   next((i for i, w in enumerate(ds.windows)
                         if any(ds.y[r] for r in w["win"])), 0))
    else:
        # blockage in the horizon, inputs clear (proactive case)
        idx = next((i for i, w in enumerate(ds.windows)
                    if any(ds.y[r] for r in w["hor"])), 0)
    item = ds[idx]
    cam = item["camera"].numpy()        # (W,3,256,256) normalized
    rad = item["radar"].numpy()         # (W,8,256,64)
    label = item["label"].numpy().astype(int)
    win_lab = [int(ds.y[r]) for r in ds.windows[idx]["win"]]
    anchor = ds.windows[idx]["anchor"]
    print(f"seq {seq}, anchor index {anchor} | input-frame blockage {win_lab} | horizon label t+1..t+5 {label.tolist()}")

    # ---- camera: 5 frames ----
    fig, ax = plt.subplots(1, W, figsize=(3 * W, 3.4))
    for i in range(W):
        img = np.clip(cam[i].transpose(1, 2, 0) * IM_STD + IM_MEAN, 0, 1)
        ax[i].imshow(img); ax[i].axis("off")
        ax[i].set_title(f"t-{W-1-i}  ({'BLOCKED' if win_lab[i] else 'clear'})",
                        color=("crimson" if win_lab[i] else "black"), fontsize=10)
    fig.suptitle(f"Camera input window (seq {seq}) — 5x 256x256 @300ms | "
                 f"horizon t+1..t+5 = {label.tolist()}", fontsize=11)
    cam_out = ROOT / "figs" / f"input_camera_{args.case}.png"
    fig.tight_layout(); fig.savefig(cam_out, dpi=130); plt.close(fig)

    # ---- radar: 8 channels of the anchor (last input) frame ----
    last = rad[-1]                       # (8,256,64)
    fig, ax = plt.subplots(2, 4, figsize=(13, 8))
    for c in range(8):
        a = ax[c // 4, c % 4]
        im = a.imshow(last[c], aspect="auto", cmap="viridis")
        a.set_title(f"ch{c}: {RADAR_CH[c]}", fontsize=10)
        a.set_xlabel("Doppler (64)"); a.set_ylabel("range (256)")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Radar input — 8 feature maps (8x256x64), anchor frame (seq {seq}) "
                 f"{'[BLOCKED]' if win_lab[-1] else '[clear]'}", fontsize=12)
    rad_out = ROOT / "figs" / f"input_radar_{args.case}.png"
    fig.tight_layout(); fig.savefig(rad_out, dpi=110); plt.close(fig)

    print(f"wrote {cam_out.name} and {rad_out.name}")


if __name__ == "__main__":
    main()
