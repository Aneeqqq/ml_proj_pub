---
title: Multi-Scenario Setup (31-34, cross-scenario)
tags: [dataset, multi-scenario, splits]
updated: 2026-06-07
status: verified
---

# Multi-Scenario Setup — DeepSense 31-34, cross-scenario protocol

All four paper scenarios are now present (added 2026-06-07). Same schema; the power-derived label
([[blockage-label]]) applies to all with no hand-labeling.

## Inventory (📊 measured)
| scenario | samples | sequences | derived blocked % | folder | CSV |
|---|---|---|---|---|---|
| 31 | 7012 | 52 | 6.3% | `scenario31_new/scenario31/` | `scenario31_dev_labelled.csv` |
| 32 | 3235 | 15 | 6.9% | `scenario32/` | `scenario32_dev.csv` |
| 33 | 3981 | 18 | 9.3% | `scenario33/` | `scenario33_dev.csv` |
| 34 | 4439 | 31 | 4.9% | `scenario34_new/scenario34/` | `scenario34.csv` |
| **total** | **18,667** | **116** | | | |

All ~10 Hz (native dt ~92 ms), all files present, identical 20-column schema. None ship a blockage
label (31's hand-label is replaced by the derived one).

## Structural gotchas (handled in code)
- **Inconsistent nesting/CSV names** per scenario → each listed with its own `csv`+`root` in
  `configs/data.yaml: scenarios:`.
- **`seq_index` collides across scenarios** → use `seq_uid = "<scenario>:<seq_index>"` everywhere
  (`configs/data.yaml: seq_col: seq_uid`).
- **Per-scenario path roots** → `scripts/build_dataset.py` rewrites sensor paths to be relative to
  ML_Proj_Claude/ (`scenario32/unit1/...`) and concatenates → **`data/dataset_all.csv`** (the single
  pipeline source; `data_root: .`).

## Split protocol (cross_scenario) — DECISION 2026-06-07
`configs/data.yaml: split.protocol: cross_scenario`, `test_scenarios: ["31"]`, `val_frac: 0.18`.
- **train + val = scenarios 32/33/34** (sequence-level, stratified on blockage); **test = ALL of
  unseen scenario 31** — the DeepSense-challenge / paper protocol; honestly measures generalization
  to a new location. `src/data/splits.py::cross_scenario_split` (dispatch via `split_from_config`).
- 📊 Verified split: train 52 seqs/9671 samples (6.8% pos), val 12/1984 (7.7%), test 52/7012 (6.3%);
  windows train 8368 / val 1683 / test 5640; smoke_test passes (no leakage, paths resolve, shapes ok).
- `pooled` protocol (stratified split across all 4) also available via config.

## Pipeline order
`build_dataset` → `make_splits` → `smoke_test` → `fit_radar_norm` → `train_camera`/`train_radar` →
`fuse_eval`. (build_dataset replaces the single-scenario `derive_label` step.)

## ⚠️ GPU PC needs the data copied
`scenario32/`, `scenario33/`, `scenario34_new/` are gitignored (large) — copy them via USB to the GPU
box like scenario 31. `data/dataset_all.csv` is regenerated locally by `build_dataset`.

Related: [[blockage-label]] · [[sequences-and-batching]] · [[replication-plan]] · [[deepsense-hardware]]
