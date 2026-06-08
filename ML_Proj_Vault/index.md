---
title: Index
tags: [index, catalog]
updated: 2026-06-01
---

# Index — ML_Proj_Vault catalog

Content catalog for the blockage-prediction replication. Read [[00_overview]] first; conventions in
[[CLAUDE]] (schema). Start every query here.

## Entry points
- [[00_overview]] — thesis, vault map, the 3 biggest risks.
- [[CLAUDE]] — vault schema, conventions, workflows (ingest/query/lint).

## Paper (faithful capture)
- [[paper-summary]] — full paper summary, headline numbers, section map.
- [[system-model]] — Fig. 1; RSU + sensors serving a vehicle; the blockage scene.
- [[architecture]] — Fig. 2; the 4-stage pipeline + per-modality block diagrams.
- [[problem-formulation]] — windowing (W=5, 300 ms, ΔT=1.5 s), labels, loss, metrics.
- [[results]] — Table I (F1/AUC t+1..t+5) & Table II (timings) to reproduce.

## Modalities (★ = priority)
- ★ [[camera]] — ResNet-18 + LSTM(128); best modality (97.1% F1).
- ★ [[radar]] — complex `(4,256,250)`→`(8,256,64)`; 3×Conv2D + LSTM(64); best partner.
- [[gps]] — 18-D handcrafted motion features + 2-LSTM; weakest modality.
- [[lidar]] — BEV stack `(700,1200,15)` + ResNet-18 (15-ch); hurts strong configs.
- [[fusion]] — late fusion; softmax-over-validation-F1 weighted average.

## Dataset (★ make-or-break)
- [[scenario31-structure]] — dirs, CSV columns, per-modality raw facts, data quality.
- [[deepsense-hardware]] — official testbed/sensor specs (array, radar AWR2243, camera, LiDAR, FOV).
- ★ [[sequences-and-batching]] — scenes via `seq_index`; windowing & split rules.
- ★ [[blockage-label]] — ✅ RESOLVED: `label`∈{blocked,not_blocked} in the labelled CSV (2.85% pos).
- [[abnormalities]] — every ⚠️ inconsistency/risk, ranked.
- [[legacy-code-audit]] — bugs in the prior dataloader + the sequence-correct rewrite (`src/data/`).
- [[multi-scenario]] — ★ scenarios 31-34 combined; cross-scenario split (test=unseen 31).
- [[cross-scenario-investigation]] — ★ why cross-scenario fails (domain shift + radar no-signal).

## Concepts
- [[blockage-prediction]] · [[late-fusion]] · [[lstm-temporal]] · [[class-imbalance]]

## Doing the work
- [[replication-plan]] — phased roadmap (Phase 0 blockers → camera → radar → fusion).
- [[improvement-plan]] — ★ post-first-results fixes (loading/time architecture, radar norm, eval CV).
- [[open-questions]] — unresolved decisions, ranked.
- [[lessons-learned]] — hard-won insights (living).

## Raw sources (immutable, outside vault)
- `../RLC_Bockage.pdf` (paper) · `../_pdf_text.txt` · `../_fig1.png` · `../_fig2.png`
- `../scenario31_new/scenario31/` (dataset)
