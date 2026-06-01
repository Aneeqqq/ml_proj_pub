---
title: Camera Modality (PRIORITY)
tags: [modality, camera, priority]
updated: 2026-06-01
status: verified
source: paper §III-A3, §III-B, Fig. 2
priority: high
---

# Camera Blockage Model  ★ PRIORITY MODALITY

The paper's **best** modality. Camera-only = 97.1% F1 @ 89.8 ms; camera+radar = 97.2% @ 95.7 ms.
This page must stay crystal clear and complete.

## Raw data (📊 DATA — Scenario 31)
- Path per sample: `unit1/camera_data/image_<index>.jpg` (column `unit1_rgb`).
- **Native resolution: 960 × 540, RGB JPEG**, base **30 fps** (sub-sampled to dataset rate). Road-
  facing camera on the stationary base station (Unit 1). → [[deepsense-hardware]]
- 7012 images (one per CSV row). Scene-continuous within each `seq_index`. → [[scenario31-structure]]

## Preprocessing (📄 PAPER §III-A3 + Fig. 2 ①)
1. **Resize** to **256 × 256**.
2. **Augment** (train only): **random horizontal flip**, **small-angle rotation**, **Gaussian blur**
   — simulate perspective/lighting variation. (§III-A1)
3. **Normalize** pixel values. → use **ImageNet mean/std** (backbone is ImageNet-pretrained ResNet-18).
   - ⚠️ The paper says only "normalized"; ImageNet stats are the faithful choice for a pretrained
     ResNet-18. Document if you deviate.

Per window: **5 frames** → tensor `(5, 3, 256, 256)` (sub-sampled to 300 ms steps, see
[[problem-formulation]]). Batched: `(B, 5, 3, 256, 256)`.

## Model (📄 PAPER §III-B + Fig. 2 ②)
**Per-frame spatial backbone → temporal LSTM → FC head.**

1. **Backbone: ResNet-18, ImageNet-pretrained, final FC removed.** Outputs a 512-d embedding per
   frame. (Fig. 2 draws the ResNet stem explicitly: Conv2D → BatchNorm → ReLU → MaxPool → ResNet
   blocks → GlobalAvgPool.)
   - Apply the backbone to **each of the 5 frames** independently (share weights). Reshape
     `(B,5,3,256,256) → (B·5,3,256,256) → backbone → (B·5,512) → (B,5,512)`.
2. **Temporal: single-layer LSTM, hidden = 128.** Input the 5 frame embeddings; take the **final
   hidden state** `(B,128)`.
3. **Head: 2-layer fully-connected classifier**, ReLU activation, **dropout p = 0.4**, → blockage
   logits. For multi-horizon (t+1..t+5) output **5 logits** (one per future step) → sigmoid.
   - ⚠️ The paper describes a single blockage probability output but evaluates t+1..t+5. Faithful
     replication: head outputs **k=5** logits (multi-label), each trained with weighted BCE. → [[problem-formulation]]

```
(B,5,3,256,256)
  → [ResNet-18 (no FC), shared over 5 frames] → (B,5,512)
  → [LSTM(512→128, 1 layer), take h_T] → (B,128)
  → [FC(128→h) → ReLU → Dropout(0.4) → FC(h→5)] → (B,5) logits → sigmoid
```

## Training notes
- Loss: weighted BCE with `w_pos = 1.1·N0/N1`. → [[class-imbalance]]
- Augment train split only; never augment val/test.
- Pretrained backbone → small LR for backbone, larger for LSTM/head is a sensible default (paper
  doesn't state LRs — ❓ [[open-questions]]).
- Inference budget: < 300 ms; paper hits 89.8 ms (camera-only). → [[results]]

## Why camera wins (📄 PAPER §IV)
Visual stream carries dense spatiotemporal context — approaching vehicles, occlusion onset, object
trajectories — directly observable before the LOS actually breaks. Radar adds cheap motion/depth on
top → [[radar]]; LiDAR/GPS add redundancy/noise.

## ✅ Implementation (`src/models/camera.py`, `scripts/train_camera.py`)
- `CameraBlockageModel`: ResNet-18 (ImageNet, `fc`→Identity) shared over 5 frames → `(B,5,512)` →
  `LSTM(512→128, 1 layer)` → last hidden → `FC(128→128)→ReLU→Dropout(0.4)→FC(128→5)` logits.
- Training: `BCEWithLogitsLoss(pos_weight = α·N0/N1)`, **α=1.1** (per-horizon pos_weight ≈ 33–37);
  Adam with split LRs (backbone 1e-4, LSTM+head 1e-3), early-stop on val `f1_mean`.
- Config `configs/camera.yaml`. Metrics: per-horizon F1 + AUC-ROC (`src/train/metrics.py`).
- ⚠️ **CPU-only box** → real training is slow (~10+ min/epoch); use a GPU to reach paper epochs.
  Smoke + bounded runs verified the loop (loss decreases, metrics compute).

## ❓ OPEN for camera (defaults chosen, flagged)
- Normalization = ImageNet stats (backbone is ImageNet-pretrained).
- FC head hidden width = 128; output = **5** horizon logits (multi-label), not 1.
- Augmentation magnitudes: hflip 0.5, ±5° rotation, GaussianBlur σ∈[0.1,1.5] — guesses (unspecified).
- LR/optimizer/epochs not in paper → Adam, 1e-4/1e-3, 30 epochs, early-stop (documented defaults).

Related: [[architecture]] · [[radar]] · [[fusion]] · [[lstm-temporal]] · [[sequences-and-batching]]
