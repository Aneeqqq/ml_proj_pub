---
title: Cross-Scenario Failure — Full Investigation
tags: [investigation, generalization, critical]
updated: 2026-06-08
status: verified
---

# Why cross-scenario blockage prediction fails — top-to-bottom investigation

First cross-scenario run (train 32/33/34, test unseen 31, t+1) gave **camera AUC 0.46 (below random),
radar AUC 0.54, fused 0.46**, and radar `train_loss` was frozen. Investigated layer by layer.

## L1 — The label is consistent (NOT the problem) ✅
Per scenario: blocked 4.9–9.3%, episode length 5–7 frames, and the **fade depth is ~−4 dB in every
scenario**. "Blocked" means the same thing everywhere. (Raw power level differs a lot — 31 med 0.18 vs
33/34 ~0.6 — but the label is *relative* to a per-seq envelope, so it normalises that out.)

## L2 — Camera failure = extreme domain shift 🔴 (root cause for camera)
The four scenarios are **four different intersections**, and:
- **31 & 32 = daytime** (mean brightness ~100); **33 & 34 = night** (~27). (figs/L2_camera_scenes.png)

So the camera trained on 32/33/34 = **1 day + 2 night** scenes and was tested on **31 = a different
daytime intersection it never saw**. Since blockage is *visually invisible* (see [[blockage-label]])
and the camera relies on **scene-geometry context** ("car here in this scene → fade soon"), that cue
is unique to each location+lighting and **cannot transfer**. AUC 0.46 (below random → the cue is
irrelevant/inverted in 31) is exactly expected.

## L3 — Radar barely carries the signal at all 🔴 (root cause for radar)
Trivial logistic on 7 radar features (mag/Doppler/phase stats):
- train (32/33/34) self-AUC **0.573**; cross → unseen 31 **0.511**; leave-one-scenario-out 0.53–0.58.

The **76 GHz radar barely predicts the derived label**, because the label is a **60 GHz comms-link
fade** — a *different sensor*. The radar does not directly observe the 60 GHz blockage. (Earlier we
saw radar Doppler AUC ~0.72 — but that was vs the *hand-label/current power*, not the derived
60 GHz-fade label.)

## L4 — The radar MODEL is fine (flat loss = no signal, not a bug) ✅
"Overfit one batch" test: the `RadarBlockageModel` drives a 48-window batch to **loss 0.0002,
train_acc 1.00**. Architecture/optimizer/gradients all work. So the frozen full-train loss is because
there is **no consistent radar→label signal to fit** across the diverse training set, not a code bug.

## Root cause (synthesis)
The task is **predicting a 60 GHz comms-link power fade from camera/radar sensors that don't directly
observe it**. That's only possible via **scene-specific correlations** (where/when fades occur in each
location), which:
- **camera:** are scene-geometry+lighting specific → don't transfer to a new intersection;
- **radar:** are too weak (radar ≈ uninformative for 60 GHz fades).

⇒ **Cross-scenario generalization is fundamentally near-impossible** for this task/label/sensor set.
It is NOT a pipeline bug. The paper's high numbers (up to 98%) are almost certainly **within-scenario
(pooled)**, where the model memorises each scene's fade-geometry.

## Implications / recommendations
1. **Pooled protocol is the fair replication** of the paper (each scene seen in training). Expect
   camera to recover to ~0.9 there; radar to stay weak (~0.57) — consistent with the paper's finding
   that **camera-only is best and radar adds little**.
2. **Cross-scenario is an honest negative result** worth reporting, not a bug to chase.
3. Don't invest in "fixing radar" for this label — there's no signal to find (L3/L4). Radar's value,
   if any, is within-scenario scene-memorisation.
4. The deeper truth: blockage here is an **RF event invisible to the sensors**; any model only learns
   per-scene correlations.

## Status of the two "bugs" from the run
- Camera below-random on 31 → **expected** (domain shift), not a bug.
- Radar flat train_loss → **expected** (no signal), model verified healthy.

Related: [[blockage-label]] · [[multi-scenario]] · [[results]] · [[camera]] · [[radar]]
