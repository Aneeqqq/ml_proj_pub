---
title: Sequences & Batching (CRITICAL)
tags: [dataset, batching, sequences, critical]
updated: 2026-06-01
status: verified
source: measured from scenario31_dev.csv on 2026-06-01
priority: critical
---

# Sequences & Batching  ★ GET THIS RIGHT OR EVERYTHING DOWNSTREAM BREAKS

The human's intuition was correct: **the images/data are "scene-like" — organized into sequences.**
The grouping key is the **`seq_index`** column (timestamps corroborate it). Mishandling sequence
boundaries silently corrupts windows, leaks future→past, and inflates metrics. Everything here is
📊 **measured** from `scenario31_dev.csv`.

## The scene/sequence structure (📊)
- **52 distinct sequences**, `seq_index ∈ {2..63}` (with gaps — not all ids present).
- **Sequence lengths range 66 → 281 samples**, total **7012**.
- **Within every sequence, `index` is perfectly contiguous** (Δindex = 1). Verified for all 52.
- **CSV rows are in global `index` order**, and sequences appear as contiguous blocks. So a sequence
  = a contiguous run of rows sharing one `seq_index`.
- Each sequence is **one continuous drive/scene**: the vehicle passes the RSU, the best beam index
  rises monotonically (1→64) as it crosses the field of view.

### Per-sequence facts (selected; full set from /tmp/seq.py analysis)
- Longest: seq 3 (n=281, idx 265–545); seq 62 (216); seq 9 (218); seq 7 (207); seq 53 (204).
- Shortest: seq 10 (n=66), seq 52 (74), seq 14/40/59 (81).
- All 52 are index-contiguous internally.

## Timing (📊) — native ~10 Hz, NOT the paper's 300 ms
- Within-sequence mean Δt ≈ **100.8 ms** (≈10 Hz nominal).
- BUT timing is **irregular**: some Δt = 0 ms (duplicate timestamps), many ≈ 90 ms, some ≈ 180 ms,
  occasional gaps up to ~550 ms. Do **not** assume a perfectly uniform clock.
- Paper operates at **300 ms steps** → it must **sub-sample (~every 3rd frame)** within sequences.
  See [[problem-formulation]] for the 300-ms-vs-100-ms abnormality.

## ⚠️ THE RULES (non-negotiable for the dataloader)

1. **Never build a window across a sequence boundary.** A window of 5 frames + a horizon of 5
   future labels must lie entirely inside ONE `seq_index`. Group by `seq_index`, then slide within.
2. **Split train/val/test at the SEQUENCE level, not the sample level.** If frames from one drive
   land in both train and test, the model memorizes the scene → leaked, inflated F1. Assign whole
   `seq_index` values to one split.
   - ⚠️ **STRATIFY by blockage presence.** Positives are scarce & lumpy: **only 15 of 52 sequences
     contain any blockage** (23 episodes, 200 positive samples — see [[blockage-label]]). A naive
     random sequence split can starve val/test of positives → unstable/undefined F1. Distribute the
     **15 positive sequences** across splits (e.g. ~10/2/3) while keeping each sequence whole, then
     verify every split has positive *windows* at each horizon t+1..t+5.
3. **Fit all scalers/normalizers on the training split only**, apply to val/test (GPS scaler,
   radar/lidar normalization stats). → [[gps]] [[radar]] [[lidar]]
4. **Sub-sample within sequences, consistently.** Decide the step `s` (likely 3 → ~300 ms) and apply
   it inside each sequence before windowing. Document `s` and stick to it across all modalities so
   frames stay time-aligned across camera/radar/etc.
5. **Keep modalities time-aligned by `index`.** All modalities for a given sample share the same
   `index`; build windows on the index axis so camera frame t and radar frame t are the same instant.

## ⚠️ LOADING FLAW found after first results (see [[improvement-plan]])
The v1 loader did `frames[::step]` (decimate) **then** windowed → it **discarded 68% of positive
frames** (64 of 200), leaving the **test split with ~6 positives/horizon** (noise). Fix = **dense
anchoring**: every native frame is a window-end; inputs/labels selected at ~300 ms spacing **by
timestamp** (not fixed row step), with a **timestamp-coherence guard** so a window never spans a time
gap. Gives ~3× positives at the same 300 ms timing. Timing facts that motivate timestamp selection:
within-seq dt median **0.092 s**, p95 **0.184 s** (dropped-frame doubles), max 0.547 s, 41 duplicate
stamps; only **1** internal gap >0.5 s (seq 51); sequences are **16–438 s** apart (seq_index = clean
scene key). Full plan: [[improvement-plan]].

## Windowing math (📊, W=5 input frames, K=5 horizon, no sub-sampling)
- A usable window needs 5 past + 5 future ⇒ sequence must have ≥ 10 samples (all 52 qualify).
- Usable windows per sequence ≈ `n − (W−1) − K = n − 9`.
- **Total usable windows ≈ 6544** (raw, at native rate, no sub-sampling).
- **With sub-sampling by s=3** the effective per-sequence length drops to ~n/3, so usable windows
  shrink substantially (rough order ~2k). Recompute exactly once `s` is fixed. ❓ [[open-questions]]

## Recommended dataloader shape
```
for seq in sequences (grouped by seq_index, sorted by index):
    rows = subsample(seq.rows, step=s)          # s≈3 → 300ms
    for i in range(len(rows) - W - K + 1):       # +1 if labels at t+1..t+K
        X = rows[i : i+W]                        # 5 input frames (per chosen modalities)
        y = blockage_label[ rows[i+W : i+W+K] ]  # K future binary labels  (see blockage-label)
        yield X, y
# splits: assign whole seq_index to train/val/test BEFORE this loop
```

## Sanity checks to run before trusting any training
- Assert no window spans two `seq_index` values.
- Assert no `seq_index` appears in more than one split.
- Assert camera/radar/etc. for each window come from identical `index` values.
- Plot `unit1_max_pwr` over a sequence to eyeball blockage dips before/after labeling.

Related: [[scenario31-structure]] · [[blockage-label]] · [[problem-formulation]] · [[lessons-learned]]
