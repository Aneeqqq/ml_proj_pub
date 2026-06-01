---
title: Results (Tables I & II)
tags: [paper, results, benchmarks]
updated: 2026-06-01
status: verified
source: paper §IV, Table I & Table II
---

# Results — §IV (targets to reproduce)

15 unordered permutations of {camera, gps, lidar, radar} evaluated. KPIs: **F1, AUC-ROC, inference time**.
Each model must infer within the **300 ms** budget.

## Table I — F1 / AUC-ROC by horizon (📄 PAPER)
Sorted by t+5 F1 (best→worst).

| Config | t+1 F1 | t+2 F1 | t+3 F1 | t+4 F1 | **t+5 F1** | t+5 AUC |
|---|---|---|---|---|---|---|
| **camera radar** | 98.4 | 98.0 | 97.9 | 97.5 | **97.2** | 0.968 |
| **camera only** | 98.1 | 97.8 | 97.5 | 97.2 | **97.1** | 0.963 |
| camera lidar | 96.2 | 96.1 | 95.7 | 95.5 | 95.5 | 0.933 |
| camera gps | 94.4 | 94.1 | 93.9 | 93.9 | 93.8 | 0.924 |
| camera radar lidar | 94.0 | 93.9 | 93.8 | 93.7 | 93.7 | 0.920 |
| **radar only** | 93.7 | 93.7 | 93.6 | 93.5 | **93.5** | 0.915 |
| camera gps radar lidar | 93.1 | 93.0 | 92.7 | 92.2 | 92.0 | 0.909 |
| camera gps radar | 92.3 | 92.2 | 92.0 | 91.8 | 91.7 | 0.898 |
| radar lidar | 91.8 | 91.6 | 91.5 | 91.4 | 91.3 | 0.891 |
| camera gps lidar | 91.2 | 91.1 | 91.1 | 90.8 | 90.8 | 0.890 |
| gps lidar radar | 89.9 | 89.8 | 89.5 | 89.4 | 89.0 | 0.879 |
| gps radar | 89.3 | 89.2 | 89.0 | 88.9 | 88.6 | 0.873 |
| lidar only | 87.9 | 87.9 | 87.7 | 87.4 | 87.3 | 0.865 |
| gps lidar | 84.1 | 84.0 | 83.6 | 83.2 | 83.0 | 0.831 |
| gps only | 61.7 | 61.4 | 60.9 | 60.7 | 60.6 | 0.588 |

(Values are %; AUC is a fraction. F1 degrades gracefully as horizon grows — expected.)

## Table II — Timings (ms) (📄 PAPER)
| Config | Preprocess | Inference |
|---|---|---|
| camera gps radar lidar | 56.0 | 137.2 |
| camera radar lidar | 54.3 | 128.9 |
| camera gps lidar | 43.3 | 127.5 |
| gps lidar radar | 51.8 | 125.0 |
| radar lidar | 50.7 | 124.1 |
| camera lidar | 42.4 | 121.8 |
| gps lidar | 38.4 | 121.6 |
| camera gps radar | 18.3 | 121.2 |
| lidar only | 37.9 | 117.3 |
| camera gps | 4.96 | 109.4 |
| gps radar | 13.5 | 96.1 |
| **camera radar** | 17.8 | **95.7** |
| **radar only** | 12.4 | 92.6 |
| **camera only** | 4.11 | **89.8** |
| gps only | <1 | 37.3 |

## Takeaways (📄 PAPER §IV–V)
- **Camera dominates**: rich spatiotemporal cues; camera-only is the efficiency/accuracy sweet spot.
- **Radar helps cheaply**: +0.1 F1 over camera-only for ~6 ms — motion/depth perception. → [[radar]]
- **LiDAR hurts in strong configs**: high dimensionality/redundancy → overfitting risk; +latency.
- **GPS is weak/negative**: lacks environmental awareness (gps-only F1 ≈ 60%).
- Conclusion: **lightweight vision-centric configs** are most deployable. Best single number: **98.4%
  F1 (camera+radar, t+1)**.

## ❓ OPEN — reproducibility caveats
These targets assume the paper's (undisclosed-in-detail) **blockage label** and **split**. Our
reproduced numbers are only comparable if [[blockage-label]] and the **sequence-level split**
([[sequences-and-batching]]) match the paper's intent. Treat the table as **directional targets**,
not exact goals, until the label is pinned down.

Related: [[fusion]] · [[problem-formulation]] · [[replication-plan]]
