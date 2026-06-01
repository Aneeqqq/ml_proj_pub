---
title: Concept — Proactive Blockage Prediction
tags: [concept]
updated: 2026-06-01
status: verified
---

# Concept — Proactive mmWave Blockage Prediction

**Problem:** mmWave links (here 60 GHz) need near line-of-sight. A moving obstacle (bus, truck,
pedestrian) crossing between RSU and vehicle blocks the beam → sudden link loss. Reactive recovery
(re-scan beams) is too slow for fast traffic.

**Proactive idea:** use **sensors** (camera/radar/lidar/gps) to *see the blocker coming* and predict
the blockage **before** it happens (here up to 1.5 s ahead), so the network can pre-emptively beam-steer,
reroute, or switch to a RIS path. → [[system-model]]

**Formulation:** sliding window of past sensor observations → binary blockage probability for each of
the next k steps. → [[problem-formulation]]

**Why ML / temporal models:** blockage onset is a spatiotemporal pattern (object approaching the LOS
corridor). CNNs extract spatial cues per frame; LSTMs model the motion across frames. → [[lstm-temporal]]

Prior art (📄 refs): [8] analytical highway blockage probability; [9] radar object-tracking for
blockage (Demirhan & Alkhateeb); [10] LiDAR+GNN for RIS beamforming. This paper's novelty = unified
**multi-modal late fusion** with confidence weighting on real DeepSense data. → [[fusion]]

Related: [[paper-summary]] · [[results]]
