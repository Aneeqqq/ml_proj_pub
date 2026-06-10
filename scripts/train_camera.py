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

from src.data.splits import split_from_config
from src.data.dataset import make_loaders, compute_pos_weight
from src.models.camera import build_camera_model, KINETICS_MEAN, KINETICS_STD
from src.train.engine import train_one_epoch, evaluate, predict
from src.train.metrics import tune_thresholds
from src.train.tracking import get_writer, log_scalars, close_writer, append_run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, few batches, no pretrained DL wait")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--arch", default=None, help="override model.arch (resnet18_lstm | r2plus1d_18)")
    ap.add_argument("--run-name", default=None, help="override data.yaml run_name")
    args = ap.parse_args()

    dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ccfg = yaml.safe_load((ROOT / "configs" / "camera.yaml").read_text())
    if args.run_name:
        dcfg["run_name"] = args.run_name
    outdir = ROOT / "outputs" / dcfg.get("run_name", "run"); outdir.mkdir(parents=True, exist_ok=True)
    print(f"run_name: {dcfg.get('run_name','run')} -> {outdir}")
    mcfg, tcfg = ccfg["model"], ccfg["train"]
    if args.arch:
        mcfg["arch"] = args.arch
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
    lcol = dcfg["label"]["column"]; seqc = dcfg.get("seq_col", "seq_index")
    res = split_from_config(df, dcfg)
    print(res.stats.to_string(index=False))

    arch = mcfg.get("arch", "resnet18_lstm")
    if arch == "r2plus1d_18":                 # video model: native 112² + Kinetics stats
        image_size = mcfg.get("image_size", 112)
        norm_mean, norm_std = KINETICS_MEAN, KINETICS_STD
    else:
        image_size = dcfg["camera"]["image_size"]
        norm_mean = tuple(dcfg["camera"]["norm_mean"]); norm_std = tuple(dcfg["camera"]["norm_std"])
    print(f"arch: {arch} | image_size: {image_size}")

    balanced = bool(tcfg.get("balanced_sampler", False))
    loaders = make_loaders(
        csv, data_root, res.seqs, W=W, K=K, step_s=step_s, tol_s=tol_s,
        modalities=("camera",),                       # camera-only: skip radar loading
        batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
        image_size=image_size, norm_mean=norm_mean, norm_std=norm_std,
        label_col=lcol, positive=dcfg["label"]["positive"], seq_col=seqc,
        persistent_workers=tcfg.get("persistent_workers", False),
        balanced_sampler=balanced,
    )
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(loaders[s].dataset)} windows")

    if balanced:
        # batches are class-balanced by the sampler -> no loss re-weighting on top
        pos_weight = torch.ones(K, device=device)
        print("balanced sampler ON -> pos_weight = 1")
    else:
        pos_weight = (compute_pos_weight(loaders["train"].dataset) * tcfg["alpha"]).to(device)
        print(f"pos_weight (xalpha): {pos_weight.cpu().numpy().round(1)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_camera_model(mcfg, smoke=args.smoke).to(device)

    optimizer = torch.optim.Adam(
        model.param_groups(tcfg["backbone_lr"], tcfg["head_lr"]),
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"], eta_min=tcfg["head_lr"] * 0.01)

    use_amp = (device == "cuda") and bool(tcfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("mixed precision (AMP) enabled")

    epochs = args.epochs or (1 if args.smoke else tcfg["epochs"])
    max_batches = 4 if args.smoke else None
    best_auc, best_path, history = -1.0, outdir / "camera_best.pt", []
    patience = 0
    writer = get_writer(outdir)

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                                  "camera", max_batches, scaler=scaler, use_amp=use_amp)
        val = evaluate(model, loaders["val"], criterion, device, "camera", max_batches)
        print(f"epoch {ep:02d} | train_loss {tr_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"val_auc(t+1) {val['auc_per_horizon'][0]:.4f} | val_f1(t+1) {val['f1_per_horizon'][0]:.4f}",
              flush=True)
        scheduler.step()
        log_scalars(writer, ep, train_loss=tr_loss, val_loss=val["loss"],
                    val_auc=val["auc_per_horizon"][0], val_f1=val["f1_per_horizon"][0],
                    lr=optimizer.param_groups[0]["lr"])
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

    (outdir / "camera_history.json").write_text(json.dumps(history, indent=2))
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
            print(f"TEST @0.5   | f1(t+1) {test05['f1_per_horizon'][0]:.4f} | "
                  f"auc(t+1) {test05['auc_per_horizon'][0]:.4f}", flush=True)
            print(f"TEST @tuned | f1(t+1) {testT['f1_per_horizon'][0]:.4f} | "
                  f"auc(t+1) {testT['auc_per_horizon'][0]:.4f}", flush=True)
            print("tuned thresholds:", testT["thresholds"])
            print("per-horizon F1 (tuned):", testT["f1_per_horizon"])
            torch.save({**ckpt, "thresholds": thr.tolist()}, best_path)   # for fusion reuse

            # per-scenario breakdown on the test split (does day vs night drag the average?)
            per_scn = {}
            if "scenario" in df.columns:
                test_seqs = res.seqs["test"]
                for scn in sorted(df["scenario"].astype(str).unique()):
                    seqs_s = [s for s in test_seqs
                              if df.loc[df[seqc] == s, "scenario"].astype(str).iloc[0] == scn]
                    if not seqs_s:
                        continue
                    ld = make_loaders(csv, data_root, {"test": seqs_s}, W=W, K=K,
                                      step_s=step_s, tol_s=tol_s, modalities=("camera",),
                                      batch_size=tcfg["batch_size"], num_workers=0,
                                      image_size=image_size, norm_mean=norm_mean, norm_std=norm_std,
                                      label_col=lcol, positive=dcfg["label"]["positive"],
                                      seq_col=seqc)["test"]
                    m = evaluate(model, ld, criterion, device, "camera", thresholds=thr)
                    per_scn[scn] = {"auc": round(m["auc_per_horizon"][0], 4),
                                    "f1": round(m["f1_per_horizon"][0], 4),
                                    "n_windows": len(ld.dataset)}
                    print(f"  test scenario {scn}: auc {per_scn[scn]['auc']} | "
                          f"f1 {per_scn[scn]['f1']} | {per_scn[scn]['n_windows']} windows", flush=True)

            (outdir / "camera_test.json").write_text(
                json.dumps({"test_0.5": test05, "test_tuned": testT,
                            "per_scenario": per_scn}, indent=2))
            print("wrote outputs/camera_test.json", flush=True)
            append_run({
                "run_name": dcfg.get("run_name", "run"), "modality": "camera", "arch": arch,
                "protocol": dcfg["split"].get("protocol"),
                "alpha": ("sampler" if balanced else tcfg["alpha"]),
                "freeze_backbone": mcfg.get("freeze_backbone", False),
                "batch_size": tcfg["batch_size"],
                "K": K, "epochs_ran": ep, "best_val_auc": round(best_auc, 4),
                "test_auc": round(testT["auc_per_horizon"][0], 4),
                "test_f1_tuned": round(testT["f1_per_horizon"][0], 4),
                "test_f1_0p5": round(test05["f1_per_horizon"][0], 4),
            })
        except Exception:
            print("!!! TEST EVAL FAILED — traceback below !!!", flush=True)
            traceback.print_exc()
    close_writer(writer)


if __name__ == "__main__":
    main()
