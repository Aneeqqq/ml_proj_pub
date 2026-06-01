---
title: Improvement Plan v1 (post first results)
tags: [plan, improvement, dataloader, time, critical]
updated: 2026-06-02
status: active
---

# Improvement Plan v1 — after the first GPU run

First run (RTX 4060, 9 epochs each, early-stopped) underperformed badly. This page captures the
**thorough diagnosis** and the **fix plan**, leading with the **loading / time architecture** as
directed. Companion: [[sequences-and-batching]], [[radar]], [[results]].

## What the results said (📊 `outputs/*_test.json`, `*_history.json`)
- **Camera:** test **AUC 0.957→0.802** across t+1..t+5 (mean 0.90) — *ranks well* — but **F1 ≈ 0**
  at 0.5; tuned only **0.42 @t+1, 0.21 @t+2, 0 @t+3..t+5**. Early-stopped at epoch 9.
- **Radar:** **AUC ≈ 0.41 (worse than random)**, F1 ≈ 0.02 — *not learning*. But a trivial
  Doppler-FFT-mean feature scores **AUC 0.725** and mag.mean 0.654 → **radar carries signal; the bug
  is in normalization/training, not the data.** `radar_test.json`/`fusion_test.json` never written.
- **Both undertrained** (early stop at 9, not 30).

## Root causes (ranked)

### 🔴 1. The loading decimation destroys the (already rare) positives  ← #1 issue
- Pipeline does `frames[::step]` (step=3) **then** windows → it **discards 68% of positive frames**
  (only **64 of 200** survive as anchors/labels).
- Result: **test has ~6 positive samples per horizon** (val ~6–8, train ~40). F1 on 6 positives is
  noise; training signal is tiny.
- 📊 Dense anchoring (every native frame is a window-end; inputs/labels sampled at 300 ms spacing)
  gives **3× positives**: train 124–137/horizon, val 18–26, **test 17/horizon** — same 300 ms timing,
  far more supervision. This is the single biggest fix.

### 🔴 2. Radar per-sample instance normalization removes the blockage signal
- `radar_features.radar_raw_to_features(normalize=True)` z-scores **each map of each frame
  independently** → strips the **absolute magnitude level**, which is exactly what encodes a blocker
  (mag.mean AUC 0.654). Must normalize with **train-set statistics**, not per-instance.

### 🟠 3. Early-stop / checkpoint on F1@0.5
- F1@0.5 is ~0 and noisy (uncalibrated under heavy `pos_weight`), so "best epoch" is essentially
  random and training halts early. Select/stop on **val AUC** (or val loss) instead.

### 🟠 4. `pos_weight ≈ 37` over-predicts positives → precision ≈ 0 → F1@0.5 ≈ 0.
### 🟠 5. Single tiny test split (2 positive sequences) → unstable metrics. Need CV.
### 🟡 6. Far-horizon difficulty: episodes ~0.8 s; predicting exact frame 1.5 s ahead is hard.

## The fix plan

### A. Loading & time architecture  (do FIRST — addresses #1, #2, the paper's timing)
- **A1 Dense anchoring.** Every native frame `e` (within a sequence) is a window-end. Inputs =
  frames at 300 ms steps back from `e`; labels = blockage at 300 ms steps forward. Keeps the paper's
  ΔT=1.5 s / 300 ms semantics; ~3× supervision. Replaces decimate-then-window.
- **A2 Timestamp-based frame selection** (use the `time_stamp` column, not a fixed row step). From
  anchor time `t_e`, pick the frame whose timestamp is **closest to** `t_e − 300 ms, −600 ms, …`
  (inputs) and `t_e + 300 ms … +1500 ms` (labels). Native spacing is ~92 ms median but has ~184 ms
  doubles and dropped frames; timestamp selection makes every step a true ~300 ms, unlike `::3`.
- **A3 Timestamp coherence guard.** Reject a window if any selected neighbour is missing within a
  tolerance (e.g., no frame within ±120 ms of the target time) — so windows never silently span a
  time gap / scene seam. This is "use timestamps to switch scenes appropriately."
  (📊 timing facts: within-seq dt median 0.092 s, p95 0.184, max 0.547; 41 duplicate stamps; only
  **1** internal gap >0.5 s (seq 51); sequences are 16–438 s apart → seq_index is a clean scene key.)
- **A4 Keep `seq_index` scene grouping** (verified clean) + never cross it (already enforced).
- **A5 Train-set normalization.** Fit per-channel radar (and any) stats on the **train split**,
  persist to `outputs/`, apply to val/test. Camera keeps ImageNet stats.

### B. Training & loss
- **B1** Early-stop + checkpoint on **val AUC mean** (calibration-free), not F1@0.5.
- **B2** Tame imbalance: try **focal loss** (γ≈2) or a milder weight (`pos_weight=√(N0/N1)`),
  instead of raw α·N0/N1≈37.
- **B3** Train to real convergence (more positives → more useful epochs); raise patience.
- **B4** Keep per-horizon **threshold tuning** on val (already implemented).

### C. Label / task (optional, document any deviation)
- **C1** Tolerance labels: count `y_{t+i}=1` if a blockage falls within ±1 step of `t+i` — eases
  exact-frame alignment at far horizons.
- **C2** Auxiliary "any blockage within next K" head.

### D. Evaluation methodology
- **D1** **Sequence-level k-fold CV** over the 15 positive sequences for stable F1/AUC (replace the
  one tiny test split). → [[sequences-and-batching]]
- **D2** Report **AUC primarily** + tuned-threshold F1; per-horizon.

### E. Radar specifics
- **E1** Apply A5 (train-set norm) — likely flips AUC from 0.41 to >0.7.
- **E2** Doppler-FFT carries the most signal (AUC 0.725); consider DeepSense-standard **range-angle
  (RA)** and **range-velocity (RV)** 2D-FFT maps. → [[deepsense-hardware]]
- **E3** Floor check: a logistic baseline on Doppler-FFT-mean should reach AUC ≈ 0.7; use as a sanity
  gate before trusting the CNN.

## Implementation status (2026-06-02)
- ✅ **A1–A4 done** — `src/data/dataset.py` rewritten: dense anchoring + timestamp-based frame
  selection (`build_windows` via `_nearest`/`searchsorted`) + coherence guard (`tol_ms`). Config
  `window.step_ms/tol_ms`. Windows now **train 4032 / val 948 / test 660** (was 1353/315/217).
- ✅ **A5/E1 done** — `radar_features.radar_raw_to_features(stats=...)` for train-set global norm;
  `scripts/fit_radar_norm.py` fits per-channel stats on train. Verified global norm **preserves
  cross-frame signal**: per-channel frame-mean AUC ch6=0.787, ch3=0.663, ch2=0.644 (instance-norm
  destroyed this → old radar AUC 0.41).
- ✅ **B1 done** — both train scripts early-stop + checkpoint on **val AUC mean** (not F1@0.5).
- ✅ threshold tuning retained (`tune_thresholds`), reported as F1@0.5 + F1@tuned.
- ⬜ **TODO next:** B2 (focal/milder weight), C1 (tolerance labels), D1 (sequence-level CV), E2
  (RA/RV maps). Re-run on GPU and compare.

## Suggested order
1. ✅ A1–A5 (windower rewrite + train-set norm).  2. ✅ B1 (AUC early-stop) + E1 (radar norm).
3. **Re-run on GPU** (`fit_radar_norm` → `train_camera` → `train_radar` → `fuse_eval`); expect radar
   AUC > 0.7 and camera F1 up at near horizons. 4. D1 (CV) for trustworthy numbers; then B2/C1.

Related: [[sequences-and-batching]] · [[radar]] · [[camera]] · [[results]] · [[lessons-learned]] · [[replication-plan]]
