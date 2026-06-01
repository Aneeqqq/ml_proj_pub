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

## Decision log
- 2026-06-01: Label RESOLVED via `scenario31_dev_labelled.csv` (`label` ∈ {blocked, not_blocked}).
  Switch all pipeline code to this file. Option A/B (fetch/derive) no longer needed.

Related: [[problem-formulation]] · [[sequences-and-batching]] · [[class-imbalance]] · [[results]]
