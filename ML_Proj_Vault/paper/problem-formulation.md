---
title: Problem Formulation (windowing, labels, math)
tags: [paper, problem-formulation, dataset]
updated: 2026-06-01
status: verified
source: paper §II eqs (1)-(3), §IV
---

# Problem Formulation — §II

## Inputs, window, horizon (📄 PAPER §II)
- Discrete time steps `t ∈ {t1,...,tn}`.
- **Input** = the latest **5 sensor observations**:
  `X_t = { x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}, x_t }`, with `x_i ∈ R^{C×H×W}`.   (Eq. 1)
- Window: **ΔT = 1.5 s**, step **t = 300 ms**.
- **Output** = blockage probability vector over the next `k` steps:
  `p_t = [p_{t+1},...,p_{t+k}]`, `p_{t+i} ∈ [0,1]`.   (Eq. 2)
- Trained as a **binary classifier per future step**; labels `y_{t+i} ∈ {0,1}`.
- `p_t = f_θ(X_t)`.   (Eq. 3)
- Evaluation horizons reported: **t+1 … t+5** (so **k = 5**). t+5 = **1.5 s** ahead.

## ⚠️ ABNORMALITY / ❓ OPEN — the 300 ms vs ~100 ms sampling gap (CRITICAL for the dataloader)
- 📊 DATA: Scenario 31 is sampled at **~100 ms (≈10 Hz)** within each sequence (measured mean
  within-sequence Δt ≈ 100.8 ms; see [[sequences-and-batching]]). ✅ Corroborated by DeepSense
  official spec: the 64-beam codebook is swept at **10 Hz** ([[deepsense-hardware]]).
- 📄 PAPER: operates at **300 ms steps**. So the paper must **sub-sample ≈ every 3rd frame**
  (100 ms × 3 ≈ 300 ms) to build its 5-frame windows and 5-step horizon.
- Two self-consistent readings of "5 obs, ΔT=1.5s, t=300ms":
  - **(A)** 5 input frames at 300 ms spacing span 4×300 = **1.2 s**; predict 5 steps ahead → t+5 =
    **1.5 s** ahead. (Horizon = 1.5 s; window = 1.2 s.) ← most consistent with "up to 1.5 s in advance".
  - **(B)** Window itself = 1.5 s (≈ frames at 375 ms, or 5 obs over native 1.5 s ≈ every 3 frames).
- **Decision to make & document:** sub-sample factor `s` (likely **3**) used to convert native
  10 Hz to 300 ms steps, applied **within each sequence**. This choice changes every window. See
  [[sequences-and-batching]] and log the chosen value. ❓ → [[open-questions]]

## ⚠️ ABNORMALITY / ❓ OPEN — the label `y_{t+i}` is NOT in the provided CSV
The CSV has `unit1_beam` (1..64) and `unit1_max_pwr`, but **no blockage column**. The binary
blockage ground truth must be **derived or fetched** before any training is possible. This is the
single highest-risk unknown. Full treatment in [[blockage-label]].

## Loss & imbalance (📄 PAPER §III-A2)
- Heavy class imbalance (far more `y=0` than `y=1`).
- Positive class weight: `w_pos = α · N0/N1`, with **α = 1.1**, `N0`/`N1` = #neg/#pos.
- Used in a weighted BCE-style loss. → [[class-imbalance]]

## Metrics (📄 PAPER §IV)
- **F1-score**, **AUC-ROC**, **inference time (ms)**. Reported per horizon t+1..t+5.
- Real-time constraint: each model must infer **within the 300 ms** inter-input interval. → [[results]]

Related: [[architecture]] · [[sequences-and-batching]] · [[blockage-label]] · [[replication-plan]]
