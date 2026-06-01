---
title: Concept — Late (Decision-Level) Fusion
tags: [concept, fusion]
updated: 2026-06-01
status: verified
---

# Concept — Late Fusion

**Late / decision-level fusion:** each modality is a *complete* classifier producing a probability;
combine the **probabilities**, not intermediate features. Opposite of **early fusion** (concatenate
raw/feature tensors and train one big net).

**Why the paper chose it (📄 §III-F):** modularity (swap/drop a sensor freely), interpretability
(per-modality scores), real-time feasibility (parallelizable, no giant fused net), resilience to a
single sensor failing or being noisy.

**This paper's flavor:** weighted average with **softmax-over-validation-F1** weights — a data-driven
"confidence weighting". Implementation + caveats (flat weights, generic subsets) → [[fusion]].

Contrast: [[lidar]] uses *early* fusion **within** its own temporal window (stacking BEV frames), but
across modalities the system is strictly *late* fusion.

Related: [[architecture]] · [[results]]
