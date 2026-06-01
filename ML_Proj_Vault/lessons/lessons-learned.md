---
title: Lessons Learned (hub)
tags: [lessons, hub]
updated: 2026-06-01
status: living
---

# Lessons Learned — living log

Hard-won insights for this replication. Append as we go (newest on top within a section).

## Dataset (the part that silently breaks everything)
- 💡 **`seq_index` = scene boundary. Treat it as sacred.** Data is 52 continuous drives, not iid
  frames. Window *inside* a sequence; split *by* sequence. Sample-level splitting leaks the same
  scene into train+test and inflates F1. → [[sequences-and-batching]]
- 💡 **The blockage label wasn't in `*_dev.csv`** (beam/power only) — the human supplied it as
  `scenario31_dev_labelled.csv` (`label`∈{blocked,not_blocked}). Lesson holds: always confirm the
  label exists & which file to use *before* coding. → [[blockage-label]]
- 💡 **Positives are scarce AND lumpy, not just rare.** 200 positives (2.85%) sounds like ordinary
  imbalance, but they sit in only **15/52 sequences** as **23 contiguous episodes**. With mandatory
  sequence-level splits, that means val/test can accidentally get ~zero positives → F1 noise. Must
  **stratify the split by blockage presence**. Imbalance + grouping interact. → [[sequences-and-batching]]
- 💡 **Native rate (~10 Hz) ≠ paper rate (300 ms).** A sub-sampling step (~3) is implied, not given.
  Every window depends on this; fix it once and apply it uniformly across modalities. → [[problem-formulation]]
- 💡 **Timestamps are irregular** (duplicates at Δt=0, gaps >500 ms). Don't trust a uniform clock;
  rely on `index` order within `seq_index`. → [[sequences-and-batching]]
- 💡 **Keep modalities aligned by `index`.** Camera/radar/lidar/gps for one sample share an `index`;
  build windows on that axis so a "frame t" is the same instant in every modality.

- 💡 **Cross-check the dataset's official spec page, not just the paper.** deepsense6g.net resolved
  several paper ambiguities: array is on Unit 1 (not the vehicle), radar dim-0 is RX antennas (not
  azimuth), 10 Hz sweep (confirms native rate), and scenarios 31–34 ship no native blockage label.
  The official source outranks the paper prose. → [[deepsense-hardware]]
- 💡 **deepsense6g.net is JS-rendered — WebFetch returns only the title.** Use WebSearch (which
  surfaces server-rendered summaries) or mirror pages (emergentmind, wi-lab, arXiv) instead.
- 💡 **Check NaNs early.** 440 rows (6.3%) have NaN GPS telemetry here; DeepSense documents ~200 NaN
  samples across 31–34 from sensor errors. Core camera/radar columns were clean, but always verify.

## Paper reading
- 💡 **Render the figures.** PyPDF2 text extraction mangled math symbols and missed the per-block
  pipelines; the architecture only became clear from the rendered `_fig2.png` (PyMuPDF at 4× zoom).
  For any paper: extract text AND render diagram pages. → [[architecture]]
- 💡 **Paper text vs figure vs data can disagree.** Antenna side (text) and the 64-beam owner (data)
  conflict; fusion is described for 3–4 modalities but evaluated over 15. When they clash, the
  *figure* and the *data* usually win over prose. → [[abnormalities]]
- 💡 **Reported metrics are directional targets, not exact goals** until the label + split match the
  paper's (partly undisclosed) setup. → [[results]]

## Tooling/env
- 💡 PyMuPDF (`pip install pymupdf`) renders pages reliably on this Windows box; `pdftoppm`/poppler
  is **not** available. numpy/pandas/PyPDF2/PIL present by default.
- 💡 Console is cp1252 — writing extracted PDF text to a UTF-8 file and Reading it avoids
  UnicodeEncodeError on math symbols.

## Code / pipeline
- 💡 **The prior dataloader had the exact leak the vault predicted.** `Scenario31Dataset` slid
  windows over the whole dataframe (`len(df)-seq_len`) ignoring `seq_index`, used seq_len=4, a single
  next-step label, no sub-sampling, and a random row-level split. Confirms why the
  sequence-and-batching rules matter in practice. → [[legacy-code-audit]]
- 💡 **The label is power-drop-*assisted*, human-confirmed.** `DataLabel2/label.py` flags a ≥3 dB drop
  in a 25-sample moving average vs baseline as a labelling reminder — the human then confirms. Good
  to know the positive definition's origin. → [[blockage-label]]
- 💡 **Env hygiene:** project code + `.venv` live in `ML_Proj_Claude/`; never install to global
  Python; torch is the CPU build (no CUDA on this box). The dataset is self-contained under
  `ML_Proj_Claude/scenario31_new/`.

## Modeling
- 💡 **Decimate-then-window silently destroyed 2/3 of positives.** The first loader did `frames[::3]`
  then windowed → test had only ~6 positives/horizon (F1 = noise) and radar couldn't learn. Switching
  to **dense anchoring + timestamp-based 300 ms selection** kept all positives (test 6→17/horizon,
  windows 1353/315/217 → 4032/948/660) at the same paper timing. With sparse labels, *how* you
  subsample matters as much as the model. → [[improvement-plan]]
- 💡 **Per-sample (instance) normalization can erase the signal.** Z-scoring each radar frame
  independently removed the absolute magnitude level that encodes a blocker → radar AUC 0.41. Global
  **train-set** normalization preserves it (per-channel frame-mean AUC up to 0.79). Normalize with
  dataset stats, not per-sample, when the *level* carries information. → [[radar]]
- 💡 **Don't early-stop on a metric you can't calibrate.** F1@0.5 under heavy `pos_weight` is ~0 and
  noisy → it stopped training at epoch 9 and picked a poor checkpoint. Early-stop on **AUC** (or loss);
  tune the threshold only at the end.
- 💡 **Kill orphaned training processes.** `timeout`-killed runs left ~8 python processes (one 6 GB)
  swapping the 16 GB box, making everything crawl. Clean up with `Get-Process python | Stop-Process`.
- 💡 **Track AUC, not just F1, while training.** The camera model hit **val/test AUC ≈ 0.90 after only
  3 CPU epochs** while F1@0.5 stayed low (~0.08–0.25). With heavy `pos_weight` the sigmoid outputs are
  uncalibrated, so a fixed 0.5 threshold gives poor F1 even when ranking is excellent. AUC reveals the
  model is learning; F1@0.5 alone would look like failure. → [[results]]
- 💡 **Tune the decision threshold per horizon on the validation set** before reporting F1; don't use
  0.5. The paper's F1 numbers presumably sit at sensible operating points. (TODO in the trainer.)
- 💡 **CPU is the bottleneck**, not the code. ResNet-18 × 5 frames × ~1.3k windows ≈ minutes/epoch on
  CPU; convergence (paper-scale) needs a GPU. The pipeline itself is verified end-to-end. GPU run is
  prepped: `requirements-gpu.txt` (cu128) + `SETUP_GPU.md`; code auto-uses CUDA, pins memory, and
  enables AMP when a GPU is present (auto-disabled on CPU, so the same scripts run anywhere).
- _(append more as we train radar, fusion, etc.)_

Related: [[abnormalities]] · [[open-questions]] · [[replication-plan]]
