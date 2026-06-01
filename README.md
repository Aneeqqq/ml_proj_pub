# Proactive mmWave Blockage Prediction — Replication (DeepSense 6G, Scenario 31)

A from-scratch replication of **"Multi-Modal Sensor Fusion for Proactive Blockage Prediction in
mmWave Vehicular Networks"** (Nazar, Celik, Selim, Abdallah, Qiao, Eltawil — arXiv:2507.15769v1),
focused on the **camera** and **radar** modalities, on the **DeepSense 6G** dataset (Scenario 31).

Predict whether the mmWave RSU↔vehicle link will be **blocked** at each of the next 5 steps
(t+1…t+5, 300 ms apart → up to 1.5 s ahead) from a 5-frame window of past sensor observations.

## What's here
```
src/
  data/      splits.py          stratified, sequence-level train/val/test (no scene leakage)
             dataset.py         dense, timestamp-based windowing (W=5, K=5, 300ms) + coherence guard
             radar_features.py  complex (4,256,250) -> 8-channel (8,256,64); train-set normalization
  models/    camera.py          ResNet-18 (ImageNet) + LSTM(128) + FC head
             radar.py           3x Conv2D (8->128) + LSTM(64) + FC head
  train/     engine.py          train/eval loops (AMP-ready)
             metrics.py         per-horizon F1 / AUC-ROC + threshold tuning
  fusion/    late_fusion.py     softmax-over-validation-F1 weighted probability fusion
scripts/     make_splits, fit_radar_norm, smoke_test, train_camera, train_radar, fuse_eval
configs/     data.yaml, camera.yaml, radar.yaml
ML_Proj_Vault/   an LLM-maintained knowledge base (the paper, dataset, decisions, lessons, plans)
SETUP_GPU.md     how to run on a CUDA GPU
```

## The dataset is NOT included
The **DeepSense 6G** dataset (Scenario 31, ~22 GB) is not redistributable — get it from
<https://www.deepsense6g.net/>. Place it at `scenario31_new/scenario31/` and add a `label` column
(`blocked`/`not_blocked`) to the dev CSV (the challenge scenarios ship no native blockage label).
Paths are configured in `configs/data.yaml`.

## Quick start (GPU)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-gpu.txt   # or requirements.txt for CPU
python -m scripts.make_splits
python -m scripts.fit_radar_norm        # required before radar
python -m scripts.train_camera
python -m scripts.train_radar
python -m scripts.fuse_eval
```
See [SETUP_GPU.md](SETUP_GPU.md) for details. Results write to `outputs/` (per-horizon F1@tuned + AUC).

## Status & notes
Pipeline is verified end-to-end; convergence runs need a GPU. Known next steps (sequence-level
cross-validation, focal loss, tolerance labels) are tracked in `ML_Proj_Vault/plan/improvement-plan.md`.
This is an independent research replication; all credit for the method to the original authors, and
for the dataset to the DeepSense 6G team.
