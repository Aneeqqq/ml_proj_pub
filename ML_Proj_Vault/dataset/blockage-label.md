---
title: Blockage Label — RESOLVED
tags: [dataset, label, resolved]
updated: 2026-06-01
status: verified
priority: critical
---

# Blockage Label — `y ∈ {0,1}`  ✅ RESOLVED (2026-06-01)

The human labelled the dataset. The label now lives in **`scenario31_dev_labelled.csv`** (use this
file, not the original `scenario31_dev.csv`). The former #1 risk is closed; what remains is a
**split-stratification** challenge (positives are scarce & lumpy — see below).

> 📄 WEB context: Scenarios 31–34 are the **Multi-Modal Beam Prediction Challenge** scenarios and
> ship **beam/power vectors, not a native blockage label** ([[deepsense-hardware]]). So labelling
> blockage by hand was **necessary and correct** — there was no official column to fetch (Option A
> from the earlier version is moot). The paper's authors must have labelled blockage themselves too.

## The label (📊 measured from `scenario31_dev_labelled.csv`)
- New file: `../scenario31_new/scenario31/scenario31_dev_labelled.csv` — **same 7012 rows / 20
  columns as the original, plus one column `label`** (21 total). Row order & `index`/`seq_index`
  unchanged. → [[scenario31-structure]]
- **`label` is categorical strings: `"blocked"` / `"not_blocked"`.** Map to binary:
  `y = 1 if label=="blocked" else 0`.
- It is a **per-sample, current-time** blockage flag (blockage state at that sample's `index`).

## Class balance (📊)
- **blocked = 200 (2.85%)**, **not_blocked = 6812 (97.15%)**.
- **N0/N1 = 34.06** → positive class weight `w_pos = α·N0/N1 = 1.1 × 34.06 ≈ 37.5`. → [[class-imbalance]]
- Power sanity check: blocked samples have lower `unit1_max_pwr` (mean 0.180 vs 0.230) but the
  distributions **overlap heavily** — so the human label is more informative than a naive power
  threshold would be. (Validates not using Option B.)

## Episodic structure (📊 — matters for windowing & splitting)
- Blockage occurs as **23 contiguous episodes**, each inside a single sequence.
- Episode length: **min 5, median ~8, max 21 samples** (~0.5–2.1 s at native ~100 ms).
- **Only 15 of 52 sequences contain any blockage; 37 sequences are entirely clear.**
- Per-sequence positive fraction ranges 3%–18% (for the 15 affected sequences).

## How it feeds the model (current label → future target)
The paper predicts **future** blockage `y_{t+i}`, i=1..K. The per-sample `label` is exactly the
signal: for a window ending at time `t`, the target vector is the label read at the (sub-sampled)
future steps `t+1 … t+K`. So:
```
y_window = [ y[t+1], y[t+2], ..., y[t+K] ]   # K=5, each 0/1 from `label`
```
A window is a "positive" example (for horizon i) iff a blockage episode overlaps step `t+i`. → [[problem-formulation]]

## ⚠️ NEW RISK (moved to top of [[open-questions]]) — scarce, lumpy positives vs sequence-level split
Only **15/52 sequences** and **23 episodes** carry all 200 positives. A naive sequence-level split
([[sequences-and-batching]]) could starve val/test of positives → unstable/undefined F1.
**Mitigation:** stratify the sequence-level split so positive episodes are distributed across
train/val/test (e.g. allocate the 15 positive sequences ~10/2/3 while keeping whole sequences
intact). Verify each split has enough positive *windows* at every horizon t+1..t+5. → [[sequences-and-batching]]

## 🔬 FINDING 2026-06-02 — blockage is NOT visually obvious (investigated seq 32)
Investigated the seq-32 episode (indices 4313–4333, 21 frames):
- **Camera: a "blocked" frame is visually ≈ identical to "clear" frames.** clear(4310) vs
  blocked(4318) mean pixel diff = **7.66/255**; two *clear* frames (4308 vs 4310) = **7.84/255** —
  the blocked-vs-clear change is *smaller* than clear-vs-clear. Diff map shows only ambient traffic,
  no blocker crossing the LOS. (figs/investigate_seq32.png, figs/investigate_diff.png)
- **Label does NOT track instantaneous power**: onset 4313 (pwr 0.199) is labeled blocked while the
  preceding clear 4307 is *lower* (0.151); mid-episode dips to 0.147 then **recovers to 0.44–0.49
  while still labeled blocked**. The hand-label brackets a link-dip *event* loosely, not per-frame.
- **Implication:** blockage = a 60 GHz mmWave *link* event, not a camera-visible occlusion. A camera
  model can't "see the blocker"; its signal (val AUC ~0.90) is **scene/trajectory context** — in a
  fixed RSU scene, blockages recur at specific geometry (this episode is the drive's last 21 frames).
  Real but scene-specific. Fuzzy label boundaries cap per-frame F1 → supports **tolerance labels**
  ([[improvement-plan]] C1). May also warrant re-checking the labeling pass.

## 🔬 FINDING 2026-06-05 — hand-label is unreliable; deriving a clean power label
Thorough power investigation (figs/power_label_analysis.png, figs/derived_label.png):
- **Hand-label correlates with nothing strongly**: AUC vs hand = abs power 0.648, per-seq z 0.617,
  speed 0.599, relative dB-drop 0.580, beam 0.512. Invisible in camera. The labeling tool
  (`DataLabel2/label.py`) is a **manual GUI** (3 dB-drop is only a *reminder*), so the label is
  subjective human judgment → not reliable ground truth.
- **Relative-median drop is the WRONG model**: 3 dB drops = 0.3% of frames, don't match hand-label.
- **LOS-envelope model WORKS**: envelope = per-seq rolling q90 of `max_pwr` (~4 s); blocked = power
  ≥X dB below it for ≥3 frames. Lands on genuine dips. Trade-off (min_dur 3): −2 dB→13.3%, **−3 dB→
  6.3%/51 epi**, −4 dB→2.7%/27 epi, −5 dB→0.8%. ~0% overlap with hand-label (different events).
- **DECISION (user 2026-06-05): replace hand-label with this self-consistent power-derived label**;
  frame the project as a self-defined-blockage task, applied consistently across scenarios.
- ✅ **IMPLEMENTED** — `src/data/derive_label.py` + `scripts/derive_label.py` write
  `scenario31_dev_derived.csv` with a **`label_derived`** column. Params (configs/data.yaml `label:`):
  env=rolling q90 win 41, **thr −3 dB**, min_dur 3, **deep override −4.5 dB**. Verified on examples
  (figs/verify_derived.png): positives are genuine deep fades (−4 to −6 dB), borderline −2..−3 dB
  correctly excluded. Results: **6.3% positive (443 frames), 32/52 sequences** (hand was 15/52; only
  11 frames overlap). Splits now have 5 positive seqs + 59 positive frames each in val/test (was 2/17)
  → far more stable; pos_weight ~14 (was ~33). Pipeline (`label.column: label_derived`) reruns clean.
  Hand `label` preserved in the original CSV for comparison.

## Decision log
- 2026-06-01: Label RESOLVED via `scenario31_dev_labelled.csv` (`label` ∈ {blocked, not_blocked}).
  Switch all pipeline code to this file. Option A/B (fetch/derive) no longer needed.

Related: [[problem-formulation]] · [[sequences-and-batching]] · [[class-imbalance]] · [[results]]
