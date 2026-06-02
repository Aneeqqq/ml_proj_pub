"""Train the camera blockage model (ResNet-18 + LSTM) with per-horizon weighted BCE.

Run (full):   .venv/Scripts/python.exe -m scripts.train_camera
Run (smoke):  .venv/Scripts/python.exe -m scripts.train_camera --smoke

Saves best (by val f1_mean) to outputs/camera_best.pt and a metrics log to outputs/.
See ML_Proj_Vault/modalities/camera.md and concepts/class-imbalance.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn

from src.data.splits import stratified_sequence_split
from src.data.dataset import make_loaders, compute_pos_weight
from src.models.camera import CameraBlockageModel
from src.train.engine import train_one_epoch, evaluate, predict
from src.train.metrics import tune_thresholds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, few batches, no pretrained DL wait")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ccfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text())
    mcfg, tcfg = ccfg["model"], ccfg["train"]
    if args.smoke:                              # CPU-safe quick check regardless of GPU config
        tcfg["num_workers"], tcfg["batch_size"] = 0, 2

    torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    csv = ROOT / dcfg["paths"]["csv"]
    data_root = ROOT / dcfg["paths"]["data_root"]
    W, K = dcfg["window"]["W"], dcfg["window"]["K"]
    step_s, tol_s = dcfg["window"]["step_ms"] / 1000, dcfg["window"]["tol_ms"] / 1000

    df = pd.read_csv(csv)
    res = stratified_sequence_split(df, dcfg["split"]["ratios"], dcfg["split"]["seed"],
                                    positive=dcfg["label"]["positive"])
    loaders = make_loaders(
        csv, data_root, res.seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
        modalities=("camera",),                       # camera-only: skip radar loading
        batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
        image_size=dcfg["camera"]["image_size"],
    )
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(loaders[s].dataset)} windows")

    # weighted BCE: pos_weight = alpha * (N0/N1) per horizon (computed on train)
    pos_weight = compute_pos_weight(loaders["train"].dataset) * tcfg["alpha"]
    pos_weight = pos_weight.to(device)
    print(f"pos_weight (xalpha): {pos_weight.cpu().numpy().round(1)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = CameraBlockageModel(
        horizon=mcfg["horizon"], lstm_hidden=mcfg["lstm_hidden"], fc_hidden=mcfg["fc_hidden"],
        dropout=mcfg["dropout"],
        pretrained=(mcfg["pretrained"] and not args.smoke),  # skip weight DL during smoke
        freeze_backbone=mcfg["freeze_backbone"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.param_groups(tcfg["backbone_lr"], tcfg["head_lr"]),
        weight_decay=tcfg["weight_decay"],
    )

    use_amp = (device == "cuda") and bool(tcfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("mixed precision (AMP) enabled")

    epochs = args.epochs or (1 if args.smoke else tcfg["epochs"])
    max_batches = 4 if args.smoke else None
    best_auc, best_path, history = -1.0, ROOT / "outputs" / "camera_best.pt", []
    patience = 0

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                                  "camera", max_batches, scaler=scaler, use_amp=use_amp)
        val = evaluate(model, loaders["val"], criterion, device, "camera", max_batches)
        print(f"epoch {ep:02d} | train_loss {tr_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"val_auc_mean {val['auc_mean']:.4f} | val_f1_mean {val['f1_mean']:.4f} | "
              f"val_f1@t+5 {val['f1_t5']:.4f}")
        history.append({"epoch": ep, "train_loss": tr_loss, **val})
        if val["auc_mean"] > best_auc:                  # select/early-stop on AUC (calibration-free)
            best_auc = val["auc_mean"]
            torch.save({"model": model.state_dict(), "val": val, "epoch": ep}, best_path)
            patience = 0
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"early stop at epoch {ep} (no val AUC improvement for {patience})")
                break

    (ROOT / "outputs" / "camera_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val auc_mean: {best_auc:.4f}  ->  {best_path}")

    if not args.smoke:
        import traceback
        try:
            print("=== TEST EVAL (tuning thresholds on val) ===", flush=True)
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            v_logits, v_labels, _ = predict(model, loaders["val"], device, "camera")
            thr = tune_thresholds(v_logits, v_labels)
            test05 = evaluate(model, loaders["test"], criterion, device, "camera")
            testT = evaluate(model, loaders["test"], criterion, device, "camera", thresholds=thr)
            print(f"TEST @0.5   | f1_mean {test05['f1_mean']:.4f} | f1@t+5 {test05['f1_t5']:.4f} | "
                  f"auc_mean {test05['auc_mean']:.4f}", flush=True)
            print(f"TEST @tuned | f1_mean {testT['f1_mean']:.4f} | f1@t+5 {testT['f1_t5']:.4f} | "
                  f"auc_mean {testT['auc_mean']:.4f}", flush=True)
            print("tuned thresholds:", testT["thresholds"])
            print("per-horizon F1 (tuned):", testT["f1_per_horizon"])
            torch.save({**ckpt, "thresholds": thr.tolist()}, best_path)   # for fusion reuse
            (ROOT / "outputs" / "camera_test.json").write_text(
                json.dumps({"test_0.5": test05, "test_tuned": testT}, indent=2))
            print("wrote outputs/camera_test.json", flush=True)
        except Exception:
            print("!!! TEST EVAL FAILED — traceback below !!!", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
