---
title: Legacy Dataloader Audit + Rewrite
tags: [dataset, code, audit, dataloader]
updated: 2026-06-01
status: verified
source: repo-root train/data_setup.py, DataLabel2/dataloader.py, splits/*.csv
---

# Legacy Dataloader Audit + the Rewrite

The repo root (`G:/Projects/ML_proj`) already had a first-pass dataloader (branch
`feature/blockage-dataloader`). Audited 2026-06-01; it has exactly the failure modes the vault
warned about. New, sequence-correct code lives in **`ML_Proj_Claude/src/data/`**.

## Where things are
- **Dataset (provided, used by everything):** `ML_Proj_Claude/scenario31_new/scenario31/` incl.
  `scenario31_dev_labelled.csv`. → [[scenario31-structure]]
- **Labelling tool (how `label` was made):** repo `DataLabel2/label.py` — a Tkinter GUI. It shows
  camera frames + the mmWave power trace and lets the human mark blockage; a **reminder fires when a
  25-sample moving-average power drops ≥ 3 dB below baseline** (assist heuristic). So the label is
  **human-confirmed, power-drop-assisted**. → [[blockage-label]]
- **Legacy loaders:** repo `train/data_setup.py` (`Scenario31Dataset`), `DataLabel2/dataloader.py`
  (`ScenarioSequenceDataset`). Kept for reference; superseded by the rewrite.

## 🔴 Bugs found in the legacy loaders
1. **Cross-sequence leakage.** `Scenario31Dataset.__len__ = len(df) - seq_len` and windows slide over
   the **whole dataframe ignoring `seq_index`** → windows span two different drives. (`ScenarioSequenceDataset`
   *does* group by a scenario id, but only one id exists for scenario 31, so within-31 it still ignores
   `seq_index`.) → [[sequences-and-batching]]
2. **Wrong window length.** `seq_len=4` default; paper uses **W=5**.
3. **Single-step label, no horizon.** `label = df.iloc[idx+seq_len]["label"]` — one next-step label,
   not the **K=5 vector** `y_{t+1..t+5}` the paper predicts. → [[problem-formulation]]
4. **No sub-sampling.** `step_size=1` at native ~10 Hz; paper operates at **300 ms** (need step≈3). → [[problem-formulation]]
5. **Random row-level split.** `create_dataloaders` uses `torch.utils.data.random_split` over windows
   → scene leakage again, and no positive-class stratification.
6. **Camera only.** No radar/lidar/gps path.

## 📊 Audit of the existing `splits/` CSVs
`splits/scenario31_dev_train.csv` (4908) + `_val.csv` (2104): **seq_index overlap = 1** between train
and val (almost sequence-level, but one sequence leaks); only train/val (no test). Positives present
in both (130 train / 70 val). Replaced by the stratified split below.

## ✅ The rewrite (`ML_Proj_Claude/src/data/`)
- **`splits.py`** — `stratified_sequence_split`: whole `seq_index` → one split; positive (15) and
  negative (37) sequences allocated separately by ratio so val/test get blockage episodes. Asserts
  disjoint + complete. → [[sequences-and-batching]]
- **`radar_features.py`** — `radar_raw_to_features`: complex `(4,256,250)` → **8-channel `(8,256,64)`**
  (magnitude, phase, Doppler-FFT spectral, mean/std across RX, entropy, Doppler mean-velocity,
  spectral width); Doppler 250→64 by center-crop. → [[radar]]
- **`dataset.py`** — `BlockageWindowDataset`: builds windows **per sequence** (no crossing),
  **W=5**, **K=5 horizon label vector**, **step=3** sub-sample, camera `(W,3,256,256)` + radar
  `(W,8,256,64)`, train-time augmentation; `compute_pos_weight` for weighted BCE (×α=1.1);
  `make_loaders` builds train/val/test from the split. → [[camera]] [[radar]] [[class-imbalance]]
- **Config:** `configs/data.yaml`. **Scripts:** `scripts/make_splits.py`, `scripts/smoke_test.py`
  (asserts: no cross-seq window, disjoint splits, correct shapes, finite radar).
- **Env:** `ML_Proj_Claude/.venv` (torch CPU) — never global. `requirements.txt` pins torch 2.8.0.

Related: [[sequences-and-batching]] · [[blockage-label]] · [[replication-plan]] · [[lessons-learned]]
