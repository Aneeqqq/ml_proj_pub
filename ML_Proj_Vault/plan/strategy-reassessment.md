---
title: Strategy Reassessment v2 — camera-first
tags: [plan, strategy, architecture-review, critical]
updated: 2026-06-13
status: active
---

# Strategy Reassessment — why we're at 0.68 and what actually works

Full stop-and-rethink after the recovery. Covers: root causes ranked, external calibration,
architecture review, and the corrected plan (camera first; radar later).

## 0. External calibration — our expectations were WRONG, not just our model

The DeepSense team's own vision-aided blockage work (Charan & Alkhateeb, Globecom'22,
[arXiv:2203.01907](https://arxiv.org/abs/2203.01907), [repo](https://github.com/gourangc/Vision_Aided_Blockage_prediction)):
- CNN-LSTM on RGB sequences, **~90% accuracy at 0.1 s ahead, only ~80% at ~1 s ahead** — on
  **scenarios 17–22**, which are DeepSense's *dedicated blockage scenarios with real blockage labels*.
- They needed **MIRNet image enhancement** for night scenes.

⇒ Two consequences:
1. **The 97% F1 in our target paper is not reproducible as stated** — it repurposes beam-prediction
   scenarios (31–34) with an undisclosed label. The field's realistic number at 1 s is ~80% acc.
2. Our pooled camera **AUC 0.68 (frozen backbone, fuzzy derived label) is "low-normal," not broken.**
   The gap to close is 0.68 → ~0.85, not 0.68 → 0.97.

## 1. Ranked causes of our underperformance

| # | Cause | Evidence | Fixable? |
|---|-------|----------|----------|
| 1 | **Frozen ImageNet backbone** — generic features can't encode scene-specific geometry cues | fine-tuned single-scene run hit val AUC 0.90; frozen pooled = 0.75 | ✅ unfreeze (run was pending) |
| 2 | **Imbalance handled by loss weight only** — pos_weight 5–37 distorts calibration, F1 collapses | F1@0.5 ≈ 0 in every run despite decent AUC | ✅ balanced batch sampler, pos_weight→1 |
| 3 | **Day/night domain mix with no lighting augmentation** — 33/34 are night, 31/32 day | L2 investigation; Charan needed MIRNet | ✅ brightness/contrast aug (cheap MIRNet proxy) |
| 4 | **Motion is only implicit** — per-frame CNN then LSTM; the cue is *vehicle motion toward the LOS corridor*, invisible in any single frame | blocked vs clear frames pixel-identical (investigation) | ✅ video model with pretrained spatiotemporal kernels, or frame-differencing |
| 5 | **Label fuzziness** — derived power-fade label has loose event boundaries; per-frame F1 punished | hand label was worse; fade −3dB is physical but boundaries ±2–3 frames | ⚠️ partially (tolerance/event-level eval) |
| 6 | **Small data** (~13k windows, ~850 pos) — large models overfit, small ones underfit | val AUC noisy across epochs | ✅ pretrained models + aug, not bigger models |
| 7 | Fixed LR, patience 7 — early-stops on noise | several runs stopped while improving | ✅ cosine schedule, patience 10 |

(Cross-scenario generalization remains fundamentally hard — separate issue, parked. Pooled is the
paper-comparable protocol.)

## 2. Architecture review (camera-only, future-blockage, ~13k windows)

| Architecture | Pros | Cons | Verdict |
|---|---|---|---|
| **ResNet18+LSTM** (paper, ours) | faithful to paper; simple | motion implicit; ImageNet-only pretraining | keep as baseline, **unfrozen** |
| **R(2+1)D-18, Kinetics-400 pretrained** (`torchvision.models.video.r2plus1d_18`) | *pretrained spatiotemporal* kernels = motion features for free; in torchvision (fully replicable); right size (31M); native short-clip input (T=5 ok, 112²) | heavier than ResNet18; needs low LR | **★ primary recommendation** |
| MC3-18 / S3D (torchvision, Kinetics) | same idea, lighter | slightly weaker on benchmarks | fallback if VRAM tight |
| Frame-differencing channels (RGB+Δ) on ResNet18+LSTM | explicit motion, nearly free | hand-crafted; less principled than pretrained video | cheap Phase-A add-on, optional |
| Two-stream optical flow | classic motion | flow compute per frame; complexity | skip (R(2+1)D supersedes) |
| **Object-detector boxes (YOLO) → GRU** (Charan-style family) | scene-robust (coordinates not pixels); tiny model; interpretable; best bet for *cross-scenario* | extra detector stage; loses subtle context | **Phase C** — the cross-scenario weapon |
| TimeSformer / VideoMAE | SOTA video | data-hungry; 13k windows ≪ needed | skip |
| ConvLSTM | end-to-end spatiotemporal | no pretraining available | skip |

**Recommendation:** **R(2+1)D-18 (Kinetics-pretrained, fine-tuned)** as the new primary camera
model — it directly attacks cause #4 (motion) and #6 (pretraining), is one-line replicable from
torchvision, and is sized for our data. Keep unfrozen ResNet18+LSTM as the paper-faithful baseline.
If cross-scenario ever becomes the goal again: detector-boxes+GRU (Phase C).

## 3. Plan of action

**Phase A — training fixes (applies to all archs):**
- `WeightedRandomSampler` → balanced batches; `pos_weight → 1.0` (no more 37× distortion)
- `ColorJitter(brightness/contrast)` augmentation (day/night robustness)
- Cosine LR schedule, `early_stop_patience: 10`
- Unfrozen backbone (the pending experiment, folded in)
- **Per-scenario test breakdown** in eval (see whether night scenes drag the average)

**Phase B — architecture:** config-selectable `arch: resnet18_lstm | r2plus1d_18`; run R(2+1)D-18
at 112², Kinetics normalization. Compare via experiments registry.

**Phase C (parked):** YOLO-boxes + GRU for cross-scenario; tolerance/event-level metrics for the
label-fuzziness ceiling.

**Success criteria:** Phase A unfrozen ResNet ≥ 0.80 test AUC pooled; Phase B R(2+1)D ≥ Phase A;
report per-scenario.

Related: [[cross-scenario-investigation]] · [[improvement-plan]] · [[camera]] · [[results]] · [[blockage-label]]
