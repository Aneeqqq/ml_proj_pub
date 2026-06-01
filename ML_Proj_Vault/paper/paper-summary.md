---
title: Paper Summary
tags: [paper, source-summary]
updated: 2026-06-01
status: verified
source: ../RLC_Bockage.pdf (arXiv:2507.15769v1)
---

# Paper Summary — Multi-Modal Sensor Fusion for Proactive Blockage Prediction in mmWave Vehicular Networks

**Authors:** Ahmad M. Nazar, Abdulkadir Celik, Mohamed Y. Selim, Asmaa Abdallah, Daji Qiao,
Ahmed M. Eltawil (Iowa State University + KAUST). **Venue/ID:** arXiv:2507.15769v1 [cs.LG], 21 Jul 2025.
**Length:** 5 pages, IEEE-letter format.

## Abstract claims (📄 PAPER)
- Proactive blockage prediction for mmWave I2V (infrastructure-to-vehicle) using **camera, GPS,
  LiDAR, radar**.
- **Modality-specific** deep models → **softmax-weighted ensemble** (late fusion) by validation F1.
- Predicts **up to 1.5 s ahead**.
- **Camera-only:** F1 **97.1%**, inference **89.8 ms** (best standalone trade-off).
- **Camera+radar:** F1 **97.2%**, **95.7 ms** (best overall).
- Index terms: Blockage Prediction, LSTM, Multi-Modal, ISAC.

## Section map
- **§I Introduction** — motivation; mmWave LOS fragility; prior art is analytical/reactive
  ([8] highway blockage prob.; [9] radar object-tracking; [10] LiDAR+GNN for RIS). Contributions:
  (1) formulate I2V blockage as multi-modal prediction on DeepSense6G; (2) per-modality models +
  confidence-weighted fusion; (3) evaluate 15 sensor configs. → [[blockage-prediction]]
- **§II System Model & Problem Formulation** — RSU with sensor suite serves a vehicle; predict
  future blockage from a window of past observations. → [[system-model]] · [[problem-formulation]]
- **§III Proposed Architecture** — 4 stages: preprocess → per-modality model → weighted fusion →
  classify. → [[architecture]] · per-modality: [[camera]] [[gps]] [[lidar]] [[radar]] [[fusion]]
- **§IV Evaluation** — Table I (F1/AUC-ROC at t+1..t+5), Table II (timings). → [[results]]
- **§V Conclusion** — vision-centric, lightweight configs are best for deployment; future work:
  online adaptation, uncertainty, integrate with beam selection.

## Headline numbers to reproduce (📄 PAPER, t+5 horizon)
| Config | F1 (t+5) | AUC (t+5) | Infer ms |
|---|---|---|---|
| camera+radar | **97.2%** | 0.968 | 95.7 |
| camera only | 97.1% | 0.963 | 89.8 |
| radar only | 93.5% | 0.915 | 92.6 |
| gps only | 60.6% | 0.588 | 37.3 |

Full tables in [[results]]. Best F1 anywhere is **98.4%** (camera+radar at t+1).

## Key methodological facts (anchor for replication)
- **Dataset:** DeepSense 6G, **Scenarios 31–34** (we have **Scenario 31** so far). → [[scenario31-structure]]
- **Input window:** latest **5 observations**, step **t = 300 ms**, **ΔT = 1.5 s**. → [[problem-formulation]]
- **Output:** blockage probability vector `p_t = [p_{t+1}..p_{t+k}]`, **k = 5**; binary classifier
  per future step. → [[problem-formulation]]
- **Class imbalance:** positive weight `w_pos = α·N0/N1`, **α = 1.1**. → [[class-imbalance]]
- **Fusion:** weighted average of per-modality probs; weights = `softmax(F1_val)`. → [[fusion]]

## Provenance notes
- Full extracted text: `../_pdf_text.txt`. Figures: `../_fig1.png` (system model), `../_fig2.png`
  (architecture). The paper text was extracted and figures rendered at 4× zoom on 2026-06-01.
- ⚠️ Several gaps between paper and the provided data (esp. the **blockage label**, **300 ms vs
  native ~100 ms sampling**, and **which unit holds the antenna array**). See [[abnormalities]].
