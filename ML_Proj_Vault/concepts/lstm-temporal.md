---
title: Concept — Temporal Modeling (LSTM) vs early fusion
tags: [concept, lstm, temporal]
updated: 2026-06-01
status: verified
---

# Concept — Temporal Modeling in this paper

Two distinct strategies for capturing the **5-frame** temporal window:

- **Explicit (LSTM):** [[camera]] and [[radar]] run a per-frame spatial encoder (ResNet-18 /
  3×Conv2D), producing a 5-step embedding sequence, then an **LSTM** consumes it and the **final
  hidden state** feeds the classifier. Camera LSTM = 1 layer, hidden 128. Radar LSTM = 1 layer,
  hidden 64. GPS = 2-layer LSTM, hidden 128 over handcrafted motion features. → [[gps]]
- **Implicit (early/channel fusion):** [[lidar]] stacks the 5 BEV frames on the channel axis
  (3×5 = 15 channels) and lets a single ResNet-18 learn temporal structure implicitly — no LSTM.

**Takeaway:** the window length is **5** everywhere; only the *mechanism* differs. Keep the 5-frame
window time-aligned across modalities (same `index` set) → [[sequences-and-batching]].

Related: [[architecture]] · [[problem-formulation]]
