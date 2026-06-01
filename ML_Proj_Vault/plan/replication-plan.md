---
title: Replication Plan
tags: [plan, roadmap]
updated: 2026-06-01
status: verified
---

# Replication Plan — phased

Focus: **camera + radar** (the paper's best configs). Build the pipeline so other modalities and
scenarios 32–34 slot in later. Resolve the 🔴 critical unknowns ([[abnormalities]]) **before** training.

## Phase 0 — Resolve blockers (do FIRST)
- [x] ✅ **Blockage label pinned** — `scenario31_dev_labelled.csv`, `label`∈{blocked,not_blocked},
  200 pos (2.85%), N0/N1=34. → [[blockage-label]]
- [ ] **Fix the sub-sampling step `s`** (likely 3 → 300 ms) and the window/horizon convention (W=5,
  K=5). → [[problem-formulation]]
- [ ] **Define the STRATIFIED sequence-level split** — only 15/52 seqs have blockage; distribute
  those across train/val/test (whole seqs), verify positive windows per horizon. → [[sequences-and-batching]]

## Phase 1 — Data pipeline (the make-or-break layer)  → built in `ML_Proj_Claude/src/data/`
- [x] CSV loader on `scenario31_dev_labelled.csv`, group by `seq_index`, sort by `index`. (`dataset.py`)
- [x] Sequence-aware windower: sub-sample step=3, W=5 within each sequence, K=5 future labels;
  **never crosses sequence boundaries**. (`build_windows`) → [[sequences-and-batching]]
- [x] Stratified sequence-level split (positives distributed). (`splits.py`, `scripts/make_splits.py`)
- [x] Sanity asserts in `scripts/smoke_test.py` (no cross-seq window, disjoint splits, shapes).
- [x] Camera reader (jpg→256, ImageNet norm, aug) + radar reader (npy complex→(8,256,64)). (`radar_features.py`)
- [x] `compute_pos_weight` (per-horizon N0/N1 ×α=1.1). → [[class-imbalance]]
- [ ] (TODO) fit radar/gps normalizers on **train split only** (currently radar uses instance-norm). → [[legacy-code-audit]]
- [x] `smoke_test` PASSED: train=1353/val=315/test=217 windows; shapes & invariants OK; pos_weight
  ≈30–34. Split: train 37seq/11pos, val 8/2, test 7/2 (pos rate ~2–3%).

## Phase 2 — Camera model (highest ROI) → [[camera]]  → built in `src/models/camera.py`
- [x] Preprocess in dataloader: resize 256², ImageNet-normalize, train-time augment (flip/rotate/blur).
- [x] Model: ResNet-18 (pretrained, FC removed) shared over 5 frames → LSTM(128) → 2-FC head
  (dropout 0.4) → 5 horizon logits. (`CameraBlockageModel`)
- [x] Train with weighted BCE (`w_pos=1.1·N0/N1`); Adam split LRs; early-stop on val f1_mean.
  (`scripts/train_camera.py`, `configs/camera.yaml`) → [[class-imbalance]]
- [x] Per-horizon F1/AUC metrics (`src/train/metrics.py`); smoke + bounded run verified.
- [ ] (TODO, needs GPU) full training to convergence; compare F1@t+5 to ≈97.1% target. → [[results]]

## Phase 3 — Radar model → [[radar]]  → built in `src/models/radar.py`
- [x] Preprocess: complex `(4,256,250)` → 8 feature maps → `(8,256,64)`. (`src/data/radar_features.py`)
- [x] Model: 3×Conv2D(8→128, BN+ReLU) → adaptive avg pool → LSTM(64) → 2-FC head (dropout 0.3).
  (`RadarBlockageModel`); train-time Gaussian noise σ=0.01. (`configs/radar.yaml`, `scripts/train_radar.py`)
- [x] Smoke run verified loop + shapes on CPU.
- [ ] (TODO, needs GPU) train to convergence; target ≈ 93.5% F1 @ t+5. → [[results]]

## Phase 4 — Fusion (camera+radar first) → [[fusion]]  → built in `src/fusion/late_fusion.py`
- [x] Per-modality validation F1 → weights = softmax(F1) (scalar or per-horizon). (`softmax_f1_weights`)
- [x] `P_fused = Σ w_i P_i`; per-horizon **threshold tuning** on val; evaluate test. (`scripts/fuse_eval.py`)
- [x] Generic over modality subsets (dict-driven). Verified on CPU with existing checkpoints.
- [ ] (TODO, needs GPU) run on converged models → target camera+radar ≈ 97.2% F1 @ t+5. → [[results]]

## Phase 1b — Evaluation hygiene (NEW, done)
- [x] Per-horizon **threshold tuning** (`tune_thresholds`): pick F1-max threshold on val, apply to
  test. Verified f1_mean 0.315→0.602 on synthetic data. Train scripts now report F1@0.5 AND F1@tuned
  and save thresholds into the checkpoint. → [[lessons-learned]]

## Phase 5 — Extend (optional / later)
- [ ] GPS (18-D handcrafted features → 2-LSTM) and LiDAR (BEV stack → ResNet-18-15ch). → [[gps]] [[lidar]]
- [ ] Full 15-permutation sweep → reproduce Table I & II. → [[results]]
- [ ] Generalize to Scenarios 32–34.
- [ ] Inference-time benchmarking against the 300 ms budget.

## Repo layout (suggested)
```
src/
  data/   csv_index.py, sequences.py (windower+splits), readers/ (camera.py, radar.py)
  models/ camera_lstm.py, radar_cnn_lstm.py, heads.py
  fusion/ late_fusion.py
  train/  train_modality.py, eval.py, metrics.py (F1, AUC, timing)
configs/  camera.yaml, radar.yaml, fusion.yaml   # record s, W, K, τ, split here
```

## Definition of done (per phase)
A phase is done when: code runs end-to-end on Scenario 31, the sanity asserts pass, and metrics are
logged per horizon and compared to [[results]] (directional, pending [[blockage-label]]).

Related: [[open-questions]] · [[lessons-learned]] · [[abnormalities]]
