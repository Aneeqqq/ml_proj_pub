---
title: Overview
tags: [overview, entry-point]
updated: 2026-06-01
status: verified
---

# Overview — Proactive mmWave Blockage Prediction (replication)

**Goal:** Replicate the paper *"Multi-Modal Sensor Fusion for Proactive Blockage Prediction in
mmWave Vehicular Networks"* (Nazar, Celik, Selim, Abdallah, Qiao, Eltawil — Iowa State + KAUST,
arXiv:2507.15769v1, 21 Jul 2025) on the **DeepSense 6G dataset, Scenario 31**, focusing on the
**camera** and **radar** modalities.

## The thesis in one paragraph

mmWave vehicular links break when the line-of-sight (LOS) between a roadside unit (RSU) and a
vehicle is **blocked** by another vehicle/pedestrian. The paper predicts blockage events **up to
1.5 s in advance** so the network can react proactively (beam steering, rerouting, RIS). It runs a
**separate deep model per sensor modality** (camera, GPS, LiDAR, radar), each emitting a blockage
probability, then **fuses them late** with softmax-over-validation-F1 weights. Finding:
**camera-only** is the best standalone trade-off (97.1% F1, 89.8 ms); **camera+radar** is the best
overall (97.2% F1, 95.7 ms). GPS and LiDAR add little or hurt. See [[results]].

## Map of the vault

- **Paper, faithfully captured:** [[paper-summary]] · [[system-model]] (Fig.1) ·
  [[architecture]] (Fig.2) · [[problem-formulation]] · [[results]]
- **Modalities (camera & radar are priority):** [[camera]] · [[radar]] · [[gps]] · [[lidar]] · [[fusion]]
- **Dataset (make-or-break):** [[scenario31-structure]] · [[sequences-and-batching]] ·
  [[blockage-label]] · [[abnormalities]]
- **Concepts:** [[blockage-prediction]] · [[late-fusion]] · [[lstm-temporal]] · [[class-imbalance]]
- **Doing the work:** [[replication-plan]] · [[open-questions]] · [[lessons-learned]]

## The three things most likely to sink this replication

1. ✅ **Blockage label RESOLVED** — provided in `scenario31_dev_labelled.csv` (`label` ∈
   {blocked, not_blocked}, 200 positives = 2.85%). New twist: positives are scarce & lumpy
   (15/52 sequences) → the sequence-level split must be **stratified**. See [[blockage-label]].
2. **Data is organized into scenes (`seq_index`).** Windowing and train/val/test splits must respect
   sequence boundaries, or you leak future into past. See [[sequences-and-batching]].
3. **Native sampling is ~10 Hz (~100 ms); the paper operates at 300 ms steps.** The window/horizon
   math depends on a sub-sampling decision the paper states ambiguously. See [[problem-formulation]].
