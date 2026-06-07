"""Camera + radar late fusion evaluation (paper's best config, §III-F).

Loads the trained camera & radar models, runs both over the SAME windows, computes
softmax-over-validation-F1 fusion weights (per horizon), fuses probabilities, tunes per-horizon
thresholds on val, and reports test F1/AUC. Compare to results.md (camera+radar ≈97.2% F1@t+5).

Run:  .venv/Scripts/python.exe -m scripts.fuse_eval
Prereqs: outputs/camera_best.pt and outputs/radar_best.pt (from train_camera / train_radar).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.splits import split_from_config
from src.data.dataset import make_loaders
from src.models.camera import CameraBlockageModel
from src.models.radar import RadarBlockageModel
from src.train.metrics import per_horizon_metrics, tune_thresholds
from src.fusion.late_fusion import predict_probs, softmax_f1_weights, fuse_probs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature for F1 weights (1.0 = paper; <1 sharpens)")
    ap.add_argument("--per-horizon-weights", action="store_true",
                    help="compute fusion weights per horizon instead of one scalar")
    args = ap.parse_args()

    dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ccfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text())["model"]
    rcfg = yaml.safe_load((ROOT / "configs" / "radar.yaml").read_text())["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    csv = ROOT / dcfg["paths"]["csv"]
    data_root = ROOT / dcfg["paths"]["data_root"]
    W, K = dcfg["window"]["W"], dcfg["window"]["K"]
    step_s, tol_s = dcfg["window"]["step_ms"] / 1000, dcfg["window"]["tol_ms"] / 1000

    # radar normalization must match training
    rtrain = yaml.safe_load((ROOT / "configs" / "radar.yaml").read_text())
    norm_path = ROOT / rtrain.get("norm", {}).get("path", "outputs/radar_norm.npz")
    radar_norm = None
    if norm_path.exists():
        z = np.load(norm_path); radar_norm = (z["mean"], z["std"])

    df = pd.read_csv(csv)
    lcol = dcfg["label"]["column"]; seqc = dcfg.get("seq_col", "seq_index")
    res = split_from_config(df, dcfg)
    # ONE combined loader so camera & radar windows are aligned
    loaders = make_loaders(csv, data_root, res.seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
                           modalities=("camera", "radar"), batch_size=8, num_workers=0,
                           image_size=dcfg["camera"]["image_size"], radar_norm=radar_norm,
                           label_col=lcol, positive=dcfg["label"]["positive"], seq_col=seqc)

    cam = CameraBlockageModel(horizon=ccfg["horizon"], lstm_hidden=ccfg["lstm_hidden"],
                              fc_hidden=ccfg["fc_hidden"], dropout=ccfg["dropout"],
                              pretrained=False).to(device)
    rad = RadarBlockageModel(in_channels=rcfg["in_channels"], horizon=rcfg["horizon"],
                             lstm_hidden=rcfg["lstm_hidden"], fc_hidden=rcfg["fc_hidden"],
                             dropout=rcfg["dropout"], conv_channels=tuple(rcfg["conv_channels"])).to(device)
    cam.load_state_dict(torch.load(ROOT / "outputs" / "camera_best.pt", map_location=device)["model"])
    rad.load_state_dict(torch.load(ROOT / "outputs" / "radar_best.pt", map_location=device)["model"])
    models = {"camera": cam, "radar": rad}

    # --- validation: per-modality F1 -> fusion weights ---
    v_probs, v_labels = predict_probs(models, loaders["val"], device)
    if args.per_horizon_weights:
        f1s = {k: np.array(per_horizon_metrics(np.log(v_probs[k] / (1 - v_probs[k] + 1e-12)),
                                               v_labels)["f1_per_horizon"]) for k in models}
    else:
        f1s = {k: per_horizon_metrics(np.log(v_probs[k] / (1 - v_probs[k] + 1e-12)),
                                      v_labels)["f1_mean"] for k in models}
    weights = softmax_f1_weights(f1s, temperature=args.temperature)
    print("val F1 (for weights):", {k: np.round(v, 4).tolist() for k, v in f1s.items()})
    print("fusion weights:", {k: np.round(v, 4).tolist() for k, v in weights.items()})

    # tune thresholds on the fused VALIDATION probabilities
    v_fused = fuse_probs(v_probs, weights)
    v_fused_logits = np.log(v_fused / (1 - v_fused + 1e-12))
    thr = tune_thresholds(v_fused_logits, v_labels)

    # --- test: fuse and score ---
    t_probs, t_labels = predict_probs(models, loaders["test"], device)
    t_fused = fuse_probs(t_probs, weights)
    t_fused_logits = np.log(t_fused / (1 - t_fused + 1e-12))

    fused_metrics = per_horizon_metrics(t_fused_logits, t_labels, thresholds=thr)
    cam_metrics = per_horizon_metrics(np.log(t_probs["camera"] / (1 - t_probs["camera"] + 1e-12)), t_labels, thr)
    rad_metrics = per_horizon_metrics(np.log(t_probs["radar"] / (1 - t_probs["radar"] + 1e-12)), t_labels, thr)

    print("\n=== TEST (tuned thresholds) ===")
    for name, m in [("camera", cam_metrics), ("radar", rad_metrics), ("FUSED", fused_metrics)]:
        print(f"{name:7s} | f1_mean {m['f1_mean']:.4f} | f1@t+5 {m['f1_t5']:.4f} | "
              f"auc_mean {m['auc_mean']:.4f}")
    print("fused per-horizon F1:", fused_metrics["f1_per_horizon"])

    out = {"weights": {k: np.asarray(v).tolist() for k, v in weights.items()},
           "thresholds": thr.tolist(),
           "camera": cam_metrics, "radar": rad_metrics, "fused": fused_metrics}
    (ROOT / "outputs" / "fusion_test.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'outputs' / 'fusion_test.json'}")


if __name__ == "__main__":
    main()
