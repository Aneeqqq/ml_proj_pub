"""Threshold-tuned evaluation of a trained camera checkpoint (t+1 / single horizon).

For the train/val-only experiments (e.g. Scenario 2+5) there is no held-out test split, so we
report the *operating-point* F1: tune one threshold on val (maximising F1) and report val metrics
at 0.5 vs that threshold, plus the full confusion matrix. The tuned number is IN-SAMPLE (threshold
chosen on the same val set) -> treat it as an optimistic operating point, not a generalisation
estimate. With only a couple of independent blockage events in val it is inherently high-variance;
the point here is to see what decision the model actually makes, not to certify a number.

Run: .venv/Scripts/python.exe -m scripts.eval_tuned --data-config data_s2s5.yaml --run-name s2s5_r2p1d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.data.splits import split_from_config
from src.data.dataset import make_loaders
from src.models.camera import build_camera_model, KINETICS_MEAN, KINETICS_STD
from src.train.engine import predict
from src.train.metrics import tune_thresholds, _sigmoid

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="data_s2s5.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--arch", default=None, help="override model.arch to match the checkpoint")
    args = ap.parse_args()

    dcfg = yaml.safe_load((ROOT / "configs" / args.data_config).read_text())
    ccfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text())
    if args.run_name:
        dcfg["run_name"] = args.run_name
    run = dcfg.get("run_name", "run")
    outdir = ROOT / "outputs" / run
    mcfg, tcfg = ccfg["model"], ccfg["train"]
    if args.arch:
        mcfg["arch"] = args.arch

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv = ROOT / dcfg["paths"]["csv"]; data_root = ROOT / dcfg["paths"]["data_root"]
    W, K = dcfg["window"]["W"], dcfg["window"]["K"]
    step_s, tol_s = dcfg["window"]["step_ms"] / 1000, dcfg["window"]["tol_ms"] / 1000

    df = pd.read_csv(csv)
    lcol = dcfg["label"]["column"]; seqc = dcfg.get("seq_col", "seq_index")
    res = split_from_config(df, dcfg)

    arch = mcfg.get("arch", "resnet18_lstm")
    if arch == "r2plus1d_18":
        image_size, norm_mean, norm_std = mcfg.get("image_size", 112), KINETICS_MEAN, KINETICS_STD
    else:
        image_size = dcfg["camera"]["image_size"]
        norm_mean = tuple(dcfg["camera"]["norm_mean"]); norm_std = tuple(dcfg["camera"]["norm_std"])

    loaders = make_loaders(
        csv, data_root, {"val": res.seqs["val"]}, W=W, K=K, step_s=step_s, tol_s=tol_s,
        modalities=("camera",), batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
        image_size=image_size, norm_mean=norm_mean, norm_std=norm_std,
        label_col=lcol, positive=dcfg["label"]["positive"], seq_col=seqc,
    )

    model = build_camera_model(mcfg).to(device)
    ckpt = torch.load(outdir / "camera_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {outdir/'camera_best.pt'} (epoch {ckpt.get('epoch','?')})")

    logits, labels, _ = predict(model, loaders["val"], device, "camera")
    probs = _sigmoid(logits)[:, 0]
    y = labels[:, 0].astype(int)
    thr = tune_thresholds(logits, labels)[0]

    def report(tag, t):
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"\n{tag} (threshold={t:.3f})")
        print(f"  F1 {f1_score(y,pred,zero_division=0):.3f} | "
              f"precision {precision_score(y,pred,zero_division=0):.3f} | "
              f"recall {recall_score(y,pred,zero_division=0):.3f}")
        print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}   (positives total {tp+fn})")

    from sklearn.metrics import roc_auc_score
    print(f"\nval windows {len(y)} | positives {y.sum()} | AUC {roc_auc_score(y, probs):.4f}")
    report("@0.5      ", 0.5)
    report("@tuned(val)", thr)
    print("\nNOTE: tuned threshold is chosen on val itself -> optimistic operating point.")


if __name__ == "__main__":
    main()
