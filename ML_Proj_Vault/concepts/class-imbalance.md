---
title: Concept — Class Imbalance Handling
tags: [concept, training, imbalance]
updated: 2026-06-01
status: verified
source: paper §III-A2 eq (4)
---

# Concept — Class Imbalance

Blockage is **rare**: many `y=0`, few `y=1`. Naive training → predicts "no blockage" always.

**Paper's fix (📄 §III-A2, Eq. 4):** weighted loss with a **positive class weight**
`w_pos = α · N0 / N1`, where N0 = #negatives, N1 = #positives, and **α = 1.1** (tunable).
Use it as the `pos_weight` in a BCE-with-logits loss so positives are penalized ~`w_pos`× harder.

**Actual numbers (📊 `scenario31_dev_labelled.csv`, whole-dataset):** N1=200 blocked (2.85%),
N0=6812 → N0/N1 = 34.06 → **w_pos ≈ 37.5** at α=1.1. (Recompute on the *training split only* — and
note the per-window positive rate at each horizon will differ from the per-sample 2.85%.) → [[blockage-label]]

**Augmentation also helps (📄 §III-A1):** per-modality augmentation (image flips/rotation/blur;
radar Gaussian noise; lidar flips/rotation/scaling) increases effective positive diversity.

**Replication notes:**
- Compute `N0/N1` on the **training split only**, after the [[blockage-label]] is defined and after
  sequence-level splitting. → [[sequences-and-batching]]
- Because metrics are **F1 / AUC-ROC** (not accuracy), imbalance is handled at both loss and metric
  level — accuracy would be misleadingly high. → [[results]]
- The label-threshold τ (Option B in [[blockage-label]]) directly sets N0/N1 — pick τ so positives
  stay the minority, consistent with the paper's premise.

Related: [[blockage-label]] · [[problem-formulation]] · [[results]]
