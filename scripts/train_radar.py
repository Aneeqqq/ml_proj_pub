"""Train the radar blockage model (3xConv2D + LSTM) with per-horizon weighted BCE.

Run (full):   .venv/Scripts/python.exe -m scripts.train_radar
Run (smoke):  .venv/Scripts/python.exe -m scripts.train_radar --smoke

Mirrors scripts/train_camera.py but for the radar modality. See modalities/radar.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.data.splits import stratified_sequence_split
from src.data.dataset import make_loaders, compute_pos_weight
from src.models.radar import RadarBlockageModel
from src.train.engine import train_one_epoch, evaluate, predict
from src.train.metrics import tune_thresholds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, few batches")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    rcfg = yaml.safe_load((ROOT / "configs" / "radar.yaml").read_text())
    mcfg, tcfg = rcfg["model"], rcfg["train"]
    if args.smoke:                              # CPU-safe quick check regardless of GPU config
        tcfg["num_workers"], tcfg["batch_size"] = 0, 2

    torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    csv = ROOT / dcfg["paths"]["csv"]
    data_root = ROOT / dcfg["paths"]["data_root"]
    W, K = dcfg["window"]["W"], dcfg["window"]["K"]
    step_s, tol_s = dcfg["window"]["step_ms"] / 1000, dcfg["window"]["tol_ms"] / 1000

    # train-set radar normalization stats (run scripts.fit_radar_norm first)
    norm_path = ROOT / rcfg.get("norm", {}).get("path", "outputs/radar_norm.npz")
    radar_norm = None
    if norm_path.exists():
        z = np.load(norm_path); radar_norm = (z["mean"], z["std"])
        print(f"using train-set radar norm from {norm_path}")
    else:
        print(f"[WARN] {norm_path} not found -> falling back to per-sample instance norm "
              f"(run: python -m scripts.fit_radar_norm)")

    df = pd.read_csv(csv)
    lcol = dcfg["label"]["column"]
    res = stratified_sequence_split(df, dcfg["split"]["ratios"], dcfg["split"]["seed"],
                                    label_col=lcol, positive=dcfg["label"]["positive"])
    loaders = make_loaders(
        csv, data_root, res.seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
        modalities=("radar",),                          # radar-only: skip camera loading
        batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
        radar_noise_sigma=tcfg["radar_noise_sigma"], radar_norm=radar_norm,
        label_col=lcol, positive=dcfg["label"]["positive"],
    )
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(loaders[s].dataset)} windows")

    pos_weight = (compute_pos_weight(loaders["train"].dataset) * tcfg["alpha"]).to(device)
    print(f"pos_weight (xalpha): {pos_weight.cpu().numpy().round(1)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = RadarBlockageModel(
        in_channels=mcfg["in_channels"], horizon=mcfg["horizon"],
        lstm_hidden=mcfg["lstm_hidden"], fc_hidden=mcfg["fc_hidden"],
        dropout=mcfg["dropout"], conv_channels=tuple(mcfg["conv_channels"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.param_groups(tcfg["lr"]), weight_decay=tcfg["weight_decay"])

    use_amp = (device == "cuda") and bool(tcfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("mixed precision (AMP) enabled")

    epochs = args.epochs or (1 if args.smoke else tcfg["epochs"])
    max_batches = 4 if args.smoke else None
    best_auc, best_path, history, patience = -1.0, ROOT / "outputs" / "radar_best.pt", [], 0

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                                  "radar", max_batches, scaler=scaler, use_amp=use_amp)
        val = evaluate(model, loaders["val"], criterion, device, "radar", max_batches)
        print(f"epoch {ep:02d} | train_loss {tr_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"val_auc_mean {val['auc_mean']:.4f} | val_f1_mean {val['f1_mean']:.4f} | "
              f"val_f1@t+5 {val['f1_t5']:.4f}")
        history.append({"epoch": ep, "train_loss": tr_loss, **val})
        if val["auc_mean"] > best_auc:
            best_auc = val["auc_mean"]
            torch.save({"model": model.state_dict(), "val": val, "epoch": ep}, best_path)
            patience = 0
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"early stop at epoch {ep} (no val AUC improvement for {patience})")
                break

    (ROOT / "outputs" / "radar_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val auc_mean: {best_auc:.4f}  ->  {best_path}")

    if not args.smoke:
        import traceback
        try:
            print("=== TEST EVAL (tuning thresholds on val) ===", flush=True)
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            v_logits, v_labels, _ = predict(model, loaders["val"], device, "radar")
            thr = tune_thresholds(v_logits, v_labels)
            test05 = evaluate(model, loaders["test"], criterion, device, "radar")
            testT = evaluate(model, loaders["test"], criterion, device, "radar", thresholds=thr)
            print(f"TEST @0.5   | f1_mean {test05['f1_mean']:.4f} | f1@t+5 {test05['f1_t5']:.4f} | "
                  f"auc_mean {test05['auc_mean']:.4f}", flush=True)
            print(f"TEST @tuned | f1_mean {testT['f1_mean']:.4f} | f1@t+5 {testT['f1_t5']:.4f} | "
                  f"auc_mean {testT['auc_mean']:.4f}", flush=True)
            print("tuned thresholds:", testT["thresholds"])
            print("per-horizon F1 (tuned):", testT["f1_per_horizon"])
            torch.save({**ckpt, "thresholds": thr.tolist()}, best_path)
            (ROOT / "outputs" / "radar_test.json").write_text(
                json.dumps({"test_0.5": test05, "test_tuned": testT}, indent=2))
            print("wrote outputs/radar_test.json", flush=True)
        except Exception:
            print("!!! TEST EVAL FAILED — traceback below !!!", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
