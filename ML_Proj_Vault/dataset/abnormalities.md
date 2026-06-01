---
title: Abnormalities & Inconsistencies (hub)
tags: [abnormalities, risks, hub]
updated: 2026-06-01
status: verified
---

# Abnormalities & Inconsistencies — hub

Every `⚠️` flag from across the vault, collected. Ordered by impact on replication.

## 🔴 Critical (block or distort training)
1. ✅ **RESOLVED — blockage label.** Now provided in `scenario31_dev_labelled.csv` (`label` ∈
   {blocked, not_blocked}, 200 positives / 2.85%). → [[blockage-label]]
1b. **Scarce, lumpy positives.** Only **15/52 sequences** (23 episodes) carry all positives →
   sequence-level splits must be **stratified** or val/test gets no positives. → [[sequences-and-batching]]
2. **Scene/sequence leakage risk.** Data is grouped by `seq_index`; windows must not cross
   boundaries and splits must be sequence-level. → [[sequences-and-batching]]
3. **300 ms (paper) vs ~100 ms (native ~10 Hz) sampling.** Requires a sub-sampling decision (likely
   step 3) that reshapes every window. → [[problem-formulation]]

## 🟠 Moderate (affect fidelity / numbers)
4. **Antenna-side mismatch (paper prose wrong).** Paper: vehicle has the ULA, RSU single-antenna.
   DeepSense official + data: **Unit 1 (stationary BS) holds the 16-element 60 GHz array + 64-beam
   codebook**; Unit 2 (car) is GPS-only. Trust data/DeepSense. → [[system-model]] / [[deepsense-hardware]]
4b. **Radar dim-0 is RX antennas, not "azimuth".** Paper §III-A6 mislabels it. `(4,256,250)` =
   (4 RX, 256 range-samples, 250 Doppler-chirps); azimuth needs an angle-FFT. → [[radar]] / [[deepsense-hardware]]
4c. **DATA QUALITY — 440 rows (6.3%) NaN in unit2 GPS telemetry** (speed/sats/pdop/etc.); core
   modalities clean. DeepSense also notes ~200 NaN samples across scen 31–34 from sensor errors.
   Handle in the GPS path. → [[gps]] / [[deepsense-hardware]]
5. **Fusion described over 3–4 fixed modalities but evaluated over 15 subsets.** Fusion must be
   generic over any modality subset. → [[fusion]]
6. **"Sigmoid Probability Fusion" (Fig.2) vs weighted average (Eq.5).** Reconcile: sigmoid per head
   → F1-softmax weighted average → threshold. → [[fusion]]
7. **softmax-over-F1 weights are nearly uniform** (F1 values are close), so "confidence weighting"
   is mild; paper may have scaled F1 first (unstated). → [[fusion]]
8. **Radar Doppler 250 → 64** via trim/pad — rule (center vs first-64) unspecified. → [[radar]]
9. **Window vs horizon ambiguity** in "5 obs, ΔT=1.5s, t=300ms": window ≈1.2 s, horizon t+5 =1.5 s
   is the most consistent reading. → [[problem-formulation]]

## 🟡 Minor (note and move on)
10. **Irregular timestamps**: duplicate (Δt=0) and large-gap (>500 ms) samples exist within
    sequences. Don't assume a uniform clock. → [[sequences-and-batching]]
11. **`unit2_interpolated_position` flag**: some GPS fixes are interpolated (lower reliability). → [[gps]]
12. **`index` is global with gaps (173–8535)** but only 7012 rows / 52 seqs present — this dev CSV is
    a subset of the global capture. Don't infer missing indices exist locally. → [[scenario31-structure]]
13. **Only Scenario 31 provided so far**; paper uses 31–34. Plan to generalize the pipeline across
    scenarios. → [[replication-plan]]
14. **PDF text quirks**: extraction shows `t` for the step symbol and merged tokens; trust the
    rendered figures (`../_fig2.png`) over raw text where they disagree.

All numbered items above should also have a home on their source page. When one is resolved, update
both here and the source page, and append a `lint`/`decision` entry to [[log]].

Related: [[open-questions]] · [[lessons-learned]] · [[replication-plan]]
