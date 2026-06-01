---
title: Architecture (Fig. 2)
tags: [paper, architecture]
updated: 2026-06-01
status: verified
source: paper §III, Fig. 2 (../_fig2.png)
---

# Architecture — §III / Fig. 2 (the 4-stage pipeline)

Rendered diagram: `../_fig2.png`. Four stages (📄 PAPER §III):

- **Stage ①  Preprocessing** — modality-specific; raw sensor → spatially/temporally consistent tensor.
- **Stage ②  Per-modality blockage models** — independent deep nets, each outputs a blockage prob.
- **Stage ③  Weighted probability fusion** — scale each prediction by estimated reliability.
- **Stage ④  Blockage classification** — final binary output.

Design rationale (📄): late fusion → **modularity, interpretability, real-time feasibility,
resilience to per-sensor failure/noise**, avoids end-to-end multi-modal complexity.

## Per-modality blocks exactly as drawn in Fig. 2

> Read the dashed boxes top→bottom. ① = preprocessing block, ② = predictor block.

### Camera  → full detail in [[camera]]
- ① Preprocessing: **Resize → Augment → Normalize**
- ② Predictor: **Conv2D → BatchNorm → ReLU → MaxPool → ResNet → GlobalAvgPool → LSTM →
  2× Fully Connected → Dropout**
  (= ResNet-18 backbone per frame, then LSTM over the 5-frame sequence, then FC head.)

### GPS  → [[gps]]
- ① Preprocessing: **Temporal Feature Extraction** (Displacements, Speeds, Angle Changes,
  Accelerations, Angular Velocity, Curvature) **→ Normalize**
- ② Predictor: **Normalization → LSTM → ReLU → Fully Connected → Dropout**

### LiDAR  → [[lidar]]
- ① Preprocessing: **Voxelization → Normalization → ROI Crop → BEV Transformation**
- ② Predictor: **Conv2D → BatchNorm → ReLU → MaxPool → ResNet → GlobalAvgPool → Fully Connected**
  (= ResNet-18 with first conv modified to 15 channels; **no explicit LSTM** — temporal info is
  early-fused by stacking 5 BEV frames on the channel axis.)

### Radar  → full detail in [[radar]]
- ① Preprocessing: **Magnitude & Phase Extraction → FFT → Statistical (Mean, Std, Entropy) →
  Doppler Feature Extraction → Normalize**
- ② Predictor: **3× Conv2D → GlobalAvgPool → LSTM → 2× Fully Connected → Dropout**

### Fusion (Stage ③) → [[fusion]]
- **Apply Weights → Sigmoid Probability Fusion** → **Stage ④ Blockage Prediction**
- Weights `w_i = softmax(F1_val,i)`; `P_fused = Σ_i w_i · P_i`. (Eq. 5–6)

## ⚠️ ABNORMALITY — fusion set vs evaluation set
Fig. 2 and Eq. (5) describe fusion over **{Camera, GPS, LiDAR, Radar}** (Eq. 5 text lists only
LiDAR/Radar/Camera). But §IV evaluates **15 unordered permutations** of the 4 modalities. So the
fusion code must be **generic over any subset**, not hard-wired to 3 or 4 modalities. → [[fusion]]

## ⚠️ ABNORMALITY — "Sigmoid Probability Fusion" vs weighted average
The diagram labels Stage ③ "Sigmoid Probability Fusion", while the text (Eq. 5) is a plain
**weighted average** of probabilities. Each modality head already outputs a probability (via
sigmoid). Treat fusion as: per-head sigmoid → weighted average → threshold. → [[fusion]]

Related: [[problem-formulation]] · [[results]] · [[class-imbalance]]
