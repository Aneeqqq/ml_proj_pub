---
title: Radar Modality (PRIORITY)
tags: [modality, radar, priority]
updated: 2026-06-01
status: verified
source: paper §III-A6, §III-E, Fig. 2
priority: high
---

# Radar Blockage Model  ★ PRIORITY MODALITY

Strong cheap modality. Radar-only = 93.5% F1 @ 92.6 ms; adding radar to camera gives the best
overall config (97.2% @ 95.7 ms) for ~6 ms extra. This page must stay crystal clear.

## Raw data (📊 DATA — Scenario 31)
- Path per sample: `unit1/radar_data/radar_data_<index>.npy` (column `unit1_radar`).
- **Native array: shape `(4, 256, 250)`, dtype `complex64`.** 7012 files.
- **Correct interpretation (📄 WEB — DeepSense official, see [[deepsense-hardware]]):**
  `(4, 256, 250) = (# RX antennas, # samples per chirp, # chirps per frame)`.
  - Dim 0 = **4 RX antennas** (sensor is **TI AWR2243**, 4 RX × 1 TX, FMCW **76–81 GHz**, 750 MHz BW
    — a *different band* from the 60 GHz comms link).
  - Dim 1 = **256 range bins** (ADC samples per chirp; range-FFT axis; res ≈ 0.2 m, max ≈ 45 m).
  - Dim 2 = **250 Doppler bins** (chirps; Doppler-FFT axis). ⚠️ **250**, not 256 — drives the
    trim/pad step below.
  - ⚠️ The paper §III-A6 calls dim-0 "azimuth". It is **antennas**, not azimuth — azimuth/angle only
    appears after an **angle-FFT across the 4 RX**. Standard DeepSense pipeline makes range-angle
    (RA) and range-velocity (RV) maps via 2D FFTs. Keep this straight when building feature maps.

## Preprocessing → 8-channel `(8, 256, 64)` tensor (📄 PAPER §III-A6 + Fig. 2 ①)
Goal: turn the complex raw cube into **8 real feature maps**, each normalized to spatial size
**256 × 64**, then stacked → **`(8, 256, 64)`** per frame.

Pipeline (Fig. 2: Magnitude&Phase → FFT → Statistical Mean/Std/Entropy → Doppler features → Normalize):
1. **Magnitude & Phase extraction** from complex data → amplitude and phase maps.
2. **1-D FFT along the Doppler dimension** → spectral energy across velocity bins.
3. **Statistical descriptors** across the 4 antenna channels:
   - **mean** of magnitude, **std** of magnitude,
   - **entropy** map from normalized magnitude (proxy for scene complexity / target diversity).
4. **Doppler features** from the power spectrum: **mean velocity** and **spectral width**.
5. **Normalize**, and unify spatial resolution to **256 × 64** by **trimming or zero-padding** the
   variable Doppler axis (250 → 64). 1-D descriptors are **repeated across antenna channels** to
   keep shapes consistent.
6. **Stack the 8 feature maps** → final per-frame tensor **`(8, 256, 64)`**.

The "8 features" (📄 names them as): magnitude, phase, FFT/spectral, mean-magnitude, std-magnitude,
entropy, mean-velocity (Doppler), spectral-width (Doppler).
- ⚠️ The paper lists categories that sum to 8 but the exact 8-way assignment is slightly
  under-specified (e.g., how FFT output maps to a single channel). Pin the exact 8 in code and
  record it. ❓ → [[open-questions]]

Augmentation (train only, §III-A1): **add random Gaussian noise to raw samples** (sensor uncertainty).

Per window: **5 frames** → `(5, 8, 256, 64)`; batched `(B, 5, 8, 256, 64)`.

## Model (📄 PAPER §III-E + Fig. 2 ②)
**Per-frame 2D CNN → adaptive pool → LSTM → FC head.**

1. **3× Conv2D**, each with **ReLU + BatchNorm**, progressively **reducing spatial resolution** and
   **increasing channels 8 → … → 128** (e.g. 8→32→64→128).
2. **Adaptive average pooling** per frame → a feature vector per frame.
3. **Single-layer LSTM, hidden = 64** over the 5-frame sequence; take the **final hidden state**.
4. **FC head: 2 linear layers**, ReLU, **dropout p = 0.3** → blockage logits (k=5 for multi-horizon).

```
(B,5,8,256,64)
  → [3× (Conv2D→BN→ReLU), 8→32→64→128, shared over 5 frames] → adaptive avg pool → (B,5,128)
  → [LSTM(128→64, 1 layer), take h_T] → (B,64)
  → [FC(64→h) → ReLU → Dropout(0.3) → FC(h→5)] → (B,5) logits → sigmoid
```

## Training notes
- Loss: weighted BCE, `w_pos = 1.1·N0/N1`. → [[class-imbalance]]
- Gaussian-noise augmentation on raw radar (train only).
- Inference budget < 300 ms; radar-only ≈ 92.6 ms, prep ≈ 12.4 ms. → [[results]]
- complex64 handling: keep precision through magnitude/phase/FFT; cast to float32 for the net.

## 🔴 BUG found in first run — radar AUC 0.41 (not learning); fix in [[improvement-plan]]
First GPU run: radar **AUC ≈ 0.41 (worse than random)**, F1 ≈ 0.02. But trivial features carry
signal — **Doppler-FFT-mean AUC 0.725**, mag.mean 0.654. So the data is fine; the cause is
**per-sample instance normalization** in `radar_features.radar_raw_to_features(normalize=True)`,
which z-scores each map per frame and **strips the absolute magnitude level** that encodes a blocker.
**Fix:** normalize with **train-set statistics** (fit on train, persist, apply to val/test). Also the
decimation positive-loss (see [[sequences-and-batching]]) starved radar of positives. Sanity floor:
a logistic baseline on Doppler-FFT-mean should reach AUC ≈ 0.7. Full plan: [[improvement-plan]].

## ✅ Implementation (`src/models/radar.py`, `scripts/train_radar.py`)
- `RadarBlockageModel`: per-frame `3× (Conv2d s2 → BN → ReLU)` 8→32→64→128 (256×64→32×8) →
  `AdaptiveAvgPool2d(1)` → `(B,5,128)` → `LSTM(128→64, 1 layer)` → last hidden →
  `FC(64→64)→ReLU→Dropout(0.3)→FC(64→5)` logits.
- Training mirrors camera: `BCEWithLogitsLoss(pos_weight=α·N0/N1)`, α=1.1; Adam lr 1e-3; train-time
  raw-cube Gaussian noise σ=0.01 (paper §III-A1); early-stop on val f1_mean. Config `configs/radar.yaml`.
- Reuses `src/train/{engine,metrics}.py` (modality-agnostic). Smoke run verified loop + shapes on CPU.
- ⚠️ Radar loading is heavy (np.load + FFT/feature build per frame) → slowest modality on CPU.

## ❓ OPEN for radar
- Exact channel count per conv layer and kernel/stride (only "8→128" + "3 conv layers" stated).
- Precise definition + ordering of the 8 feature maps.
- Doppler trim-vs-pad rule to reach 64 (center-crop vs take-first-64). Document the choice.
- Whether FFT is applied before or independent of magnitude/phase channels.

Related: [[camera]] (best partner) · [[architecture]] · [[fusion]] · [[lstm-temporal]] · [[sequences-and-batching]]
