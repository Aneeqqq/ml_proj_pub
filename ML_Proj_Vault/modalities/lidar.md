---
title: LiDAR Modality
tags: [modality, lidar]
updated: 2026-06-01
status: verified
source: paper §III-A5, §III-D, Fig. 2
priority: low
---

# LiDAR Blockage Model

Detailed 3D structure but **hurts** strong configs (overfitting/redundancy) and is the heaviest to
preprocess. lidar-only ≈ 87.3% F1; camera+lidar < camera-only. Secondary for our focus. → [[results]]

## Raw data (📊 DATA — Scenario 31)
- Path: `unit1/lidar_data/lidar_data_<index>.ply` (column `unit1_lidar`). 7012 point-cloud files.

## Preprocessing — multi-step → BEV stack (📄 PAPER §III-A5 + Fig. 2 ①)
1. **Voxel-grid downsampling** (reduce density, preserve structure).
2. **Statistical outlier removal**: per-point mean distance to k nearest neighbors; drop points
   beyond a std-factor threshold.
3. **Ground removal** via **RANSAC** plane fitting (drop points within fixed distance of ground).
4. **DBSCAN** object clustering: **ε = 0.75**, **min_samples = 5** (ε from k-distance-graph max
   curvature); drop sub-minimum clusters as noise.
5. **BEV projection** onto grid **X∈[-50,50] m, Y∈[-50,50] m, Z∈[-2.5,15] m**, resolution
   **Δr = 0.25 m/cell**. Each cell → **3 channels**: max-Z (height map), log-count (density map),
   height-variance (roughness).
6. BEV frame resolution **(H,W) = (700, 1200)**, 3 channels/frame.
7. **Stack 5 sequential BEV frames on the channel axis → `(700, 1200, 15)`** (early temporal fusion).
8. Normalize: height min-max within Z-range; density log-scaled; variance standardized by max.

## Model (📄 PAPER §III-D + Fig. 2 ②)
- **ResNet-18** with **first conv modified to accept 15 channels**.
- Residual hierarchy → final FC modified to output blockage prob.
- **No explicit LSTM** — temporal info is **implicit** via the 15-channel early fusion (unlike
  [[camera]]/[[radar]] which use LSTMs).

## Notes
- ⚠️ Heaviest preprocessing (~38–56 ms) and inference (~117 ms+). Real-time marginal. → [[results]]
- ⚠️ BEV (700×1200×15) is large; watch memory in the dataloader.

Related: [[architecture]] · [[camera]] · [[fusion]]
