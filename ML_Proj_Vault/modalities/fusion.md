---
title: Late Fusion (Stage 3)
tags: [fusion, architecture]
updated: 2026-06-01
status: verified
source: paper §III-F, eqs (5)-(6), Fig. 2 ③④
---

# Multi-Modal Late Fusion — §III-F

## Mechanism (📄 PAPER)
- Each modality model independently outputs a blockage probability `P_i` (camera, radar, lidar, gps).
- **Weighted average** at the decision level (Eq. 5):
  `P_fused = Σ_{i} w_i · P_i`.
- **Weights from validation F1** via softmax (Eq. 6): with `s = [s_LiDAR, s_Radar, s_Camera, ...]`
  the per-modality validation F1-scores,
  `w = softmax(s)`, i.e. `w_i = exp(s_i) / Σ_j exp(s_j)`.
- Higher-performing modalities get more influence; avoids manual weight bias; adapts to model
  strengths. (📄 §III-F)
- Final: threshold `P_fused` → binary blockage (Stage ④).

## Implementation guidance
- Train each modality model **independently** to convergence first; compute each one's **validation
  F1** (per horizon t+i). Then freeze and combine — fusion has **no learned parameters** beyond the
  F1-derived weights.
- For our focus, the key config is **camera+radar**: `w = softmax([F1_cam, F1_radar])`,
  `P = w_cam·P_cam + w_radar·P_radar`. → [[camera]] · [[radar]]
- Weights can be computed **per horizon** (t+1..t+5) since F1 varies by horizon. Document whether
  you use one global weight set or per-horizon.

## ✅ Implementation (`src/fusion/late_fusion.py`, `scripts/fuse_eval.py`)
- `predict_probs(models, loader)` runs camera & radar over **one combined loader** so windows are
  aligned across modalities (required for fusion).
- `softmax_f1_weights(f1_by_modality, temperature)` → `w = softmax(F1/T)`; supports scalar or
  per-horizon F1. `fuse_probs(probs, weights)` → `Σ w_i·P_i`.
- `scripts/fuse_eval.py`: loads `camera_best.pt` + `radar_best.pt`, computes weights from **val F1**,
  tunes per-horizon thresholds on **fused val** probs, scores **test** (camera, radar, FUSED).
- **Threshold tuning** (`src/train/metrics.py::tune_thresholds`) is essential: with heavy pos_weight,
  F1@0.5 is pessimistic. Verified on synthetic data f1_mean **0.315→0.602** after tuning.
- ✅ Verified: T=1 weights are near-uniform (cam 0.509 / radar 0.491) — the flat-softmax abnormality
  below is real; use `--temperature <1` or `--per-horizon-weights` to sharpen.

## ⚠️ ABNORMALITIES (mirrored to [[abnormalities]])
- **Generic over subsets:** §IV evaluates **15 permutations**, so fusion code must accept **any
  subset** of the 4 modalities, not a fixed 3/4. Eq. (5) text only lists LiDAR/Radar/Camera.
- **"Sigmoid Probability Fusion" (Fig. 2) vs weighted average (Eq. 5):** reconcile as — each head's
  logits → sigmoid → probability; fuse probabilities by F1-softmax weighted average; threshold.
- **softmax over F1 is very flat:** F1 values (e.g. 0.971 vs 0.935) are close, so `softmax` gives
  near-equal weights (≈0.509 vs 0.491 for two close scores). The "confidence weighting" is therefore
  mild. Consider whether the paper scaled F1 (e.g. ×10) before softmax — not stated. ❓ → [[open-questions]]

Related: [[architecture]] · [[results]] · [[class-imbalance]]
