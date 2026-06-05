# Log — ML_Proj_Vault

Append-only, chronological. Prefix entries `## [YYYY-MM-DD] <type> | <title>`.
Grep the timeline: `grep "^## \[" log.md | tail -5`.

## [2026-06-01] ingest | Paper "Multi-Modal Sensor Fusion for Proactive Blockage Prediction" (arXiv:2507.15769v1)
- Read all 5 pages (text via PyPDF2 → `../_pdf_text.txt`; figures rendered via PyMuPDF →
  `../_fig1.png` system model, `../_fig2.png` architecture).
- Captured: [[paper-summary]], [[system-model]], [[architecture]], [[problem-formulation]], [[results]].
- Per-modality pages from Fig. 2 block diagrams: [[camera]], [[radar]], [[gps]], [[lidar]], [[fusion]].
- Key facts: W=5 frames, 300 ms step, ΔT=1.5 s, k=5 horizons; weighted BCE w_pos=1.1·N0/N1;
  late fusion softmax(F1); camera+radar best (97.2% F1 @ 95.7 ms).

## [2026-06-01] ingest | Dataset DeepSense 6G — Scenario 31
- Measured structure from `scenario31_dev.csv` (7012 rows, 20 cols) + raw files.
- Captured: [[scenario31-structure]], [[sequences-and-batching]], [[blockage-label]], [[abnormalities]].
- Measurements: 52 sequences (`seq_index` 2..63, gaps), lengths 66–281, all index-contiguous;
  native ~10 Hz (mean Δt ≈100.8 ms, irregular); camera 960×540 RGB; radar (4,256,250) complex64;
  mmWave 64-D power; beam 1..64; max_pwr 0.141–0.754 (median 0.180, 77%<0.25).

## [2026-06-01] decision | Critical risks flagged
- 🔴 No blockage label in CSV → must fetch official or derive from power. → [[blockage-label]]
- 🔴 Scene leakage → sequence-level windowing & splits mandatory. → [[sequences-and-batching]]
- 🔴 300 ms (paper) vs ~100 ms (native) → sub-sampling step s≈3 to be confirmed. → [[problem-formulation]]
- All open items tracked in [[open-questions]]; insights in [[lessons-learned]].

## [2026-06-01] ingest | Labels added — scenario31_dev_labelled.csv
- Human labelled the dataset. New file `scenario31_dev_labelled.csv` = original + `label` column
  (`blocked`/`not_blocked`). 7012 rows, 21 cols; row order/index unchanged.
- Measured: 200 blocked (2.85%), N0/N1=34.06, w_pos@α1.1≈37.5. 23 contiguous episodes (len 5–21,
  med ~8), in only 15/52 sequences; 37 sequences fully clear.
- 🔴 risk #1 (missing label) RESOLVED. New risk: scarce/lumpy positives → **stratified** seq-level split.
- Updated: [[blockage-label]] (resolved), [[sequences-and-batching]], [[class-imbalance]],
  [[scenario31-structure]], [[abnormalities]], [[open-questions]], [[00_overview]], [[replication-plan]].

## [2026-06-01] ingest | Web research — DeepSense6G official specs (deepsense6g.net + dataset paper)
- New page [[deepsense-hardware]]. Confirmed: Unit1=stationary BS (16-elem 60GHz array, 64-beam
  codebook, 90° FOV, 10 Hz), Unit2=mobile car (GPS-RTK). Camera 960×540@30fps; radar TI AWR2243
  76–81GHz 4RX×1TX; LiDAR Ouster OS1-32. Radar npy (4,256,250)=(RX,range,Doppler) — paper's
  "azimuth" for dim0 is wrong (it's antennas).
- Scenarios 31–34 = Beam Prediction Challenge scenarios → **no native blockage label** (confirms
  manual labelling was required). DeepSense notes ~200 NaN samples across 31–34.
- 📊 Our file: 440 rows NaN in unit2 GPS telemetry; core modalities clean.
- Resolved antenna-side abnormality (paper prose wrong). Updated [[system-model]], [[radar]],
  [[camera]], [[gps]], [[problem-formulation]], [[blockage-label]], [[scenario31-structure]],
  [[abnormalities]], [[index]].

## [2026-06-01] decision | Scope + workspace
- Human: build on **Scenario 31 only** for now; **rewrite** the dataloader (vs patch) with **camera +
  radar**; put code + **.venv** inside `ML_Proj_Claude/`; **no global pip**; **prefer PyTorch**.

## [2026-06-01] build | Sequence-correct dataloader (camera+radar)
- Audited legacy loaders (repo `train/data_setup.py`, `DataLabel2/dataloader.py`): cross-sequence
  leakage, seq_len=4, single-step label, no subsample, random row split. Existing `splits/` had 1
  seq leaking train↔val. → [[legacy-code-audit]]
- Wrote `ML_Proj_Claude/src/data/{splits,radar_features,dataset}.py` + `configs/data.yaml` +
  `scripts/{make_splits,smoke_test}.py`. W=5, K=5, step=3; stratified seq-level split; radar
  (4,256,250)→(8,256,64); per-horizon pos_weight.
- Created `.venv` (torch 2.8.0 CPU). **Smoke test PASSED**: splits sequence-disjoint, no window
  crosses a boundary; windows train=1353/val=315/test=217; batch shapes camera (B,5,3,256,256),
  radar (B,5,8,256,64), label (B,5); radar finite; per-horizon pos_weight ≈ 30–34 (×1.1 ≈ 33–37).
- Split (stratified, seed 42): train 37 seqs/11 pos (2.99%), val 8/2 (2.86%), test 7/2 (2.04%).

## [2026-06-01] result | Camera model (ResNet-18+LSTM) — first real run (3 epochs, CPU)
- Built `src/models/camera.py`, `src/train/{metrics,engine}.py`, `scripts/train_camera.py`, `configs/camera.yaml`.
- 3-epoch CPU run (pretrained): train_loss 1.38→0.99→0.77, val_loss 0.89→0.62→0.40,
  **val AUC≈0.91–0.96**, **test AUC≈0.90**. F1@0.5 low (val~0.25, test~0.08).
- 💡 High AUC + low F1@0.5 = good ranking, bad threshold. Need (a) more epochs (CPU slow ~min/epoch),
  (b) **per-horizon threshold tuning on val** (don't use fixed 0.5 with heavy pos_weight). → [[lessons-learned]]
- TODO: train to convergence (GPU), tune threshold, then radar + camera+radar fusion.

## [2026-06-01] build | Radar model (3xConv2D + LSTM-64)
- Built `src/models/radar.py` (RadarBlockageModel), `configs/radar.yaml`, `scripts/train_radar.py`
  (reuses engine/metrics). Per-frame 3×Conv2D 8→32→64→128 → adaptive pool → LSTM(64) → 2-FC (drop 0.3).
- Smoke run OK (loop + shapes verified on CPU). Full training deferred (CPU; user paused long runs).
- Next options: per-horizon threshold tuning, GPU convergence runs, camera+radar late fusion.

## [2026-06-01] build | GPU-run prep
- Added `requirements-gpu.txt` (CUDA cu128 torch 2.8.0) + kept `requirements.txt` (CPU). Wrote
  `SETUP_GPU.md` (driver check, install, verify, recommended GPU settings, run commands).
- Code now GPU-aware: dataloader pins memory + persistent/prefetch workers when CUDA; engine + both
  train scripts support **AMP** (autocast + GradScaler), auto-disabled off-CUDA; configs gained
  `amp: true` and GPU batch/worker guidance.
- Re-verified CPU smoke (camera + radar) — no regression. Ready to train to convergence on a GPU.

## [2026-06-01] build | Threshold tuning + camera/radar late fusion
- Added per-horizon `tune_thresholds` (F1-max on val) + `evaluate(..., thresholds=)`; train scripts
  now report F1@0.5 AND F1@tuned and save thresholds in the checkpoint. Verified synthetic
  f1_mean 0.315→0.602.
- Added `src/fusion/late_fusion.py` (`predict_probs`, `softmax_f1_weights`, `fuse_probs`) +
  `scripts/fuse_eval.py` (loads both models over one aligned loader, weights = softmax(val F1),
  tunes thresholds on fused val, scores test). Generic over modality subsets.
- ✅ Verified on CPU with existing checkpoints. T=1 weights near-uniform (cam .509/rad .491) →
  flat-softmax abnormality confirmed; `--temperature`/`--per-horizon-weights` to sharpen. → [[fusion]]
- GPU-ready: same scripts produce paper-comparable F1 once models converge.

## [2026-06-02] result+analysis | First GPU run underperformed → improvement plan
- Camera test AUC 0.90 (ranks well) but F1 collapses (0.42@t+1→0@t+3+); early-stopped epoch 9.
- Radar AUC 0.41 (broken) — but trivial Doppler-FFT-mean AUC 0.725 → features fine, training bug.
- Deep timestamp study: within-seq dt median 0.092s (p95 0.184, max 0.547), 41 dup stamps, only 1
  internal gap >0.5s (seq51), seqs 16–438s apart → seq_index is a clean scene key; time handling
  coherent. BUT `::3` decimation discards 68% of positives → test only ~6 pos/horizon (noise).
- Root causes: (1) decimate-then-window destroys positives; (2) radar per-sample instance-norm strips
  magnitude signal; (3) early-stop on f1@0.5; (4) pos_weight≈37 over-predicts; (5) tiny test split.
- Wrote [[improvement-plan]] (lead: dense anchoring + timestamp-based frame selection + coherence
  guard + train-set norm; AUC early-stop; CV). Updated [[sequences-and-batching]], [[radar]], [[index]].

## [2026-06-02] build | Implemented Improvement Plan §A (loading/time) + B1 + E1
- Rewrote `src/data/dataset.py`: **dense anchoring** (every native frame a window-end) +
  **timestamp-based frame selection** (closest to anchor±j*300ms) + **coherence guard** (tol 130ms)
  → windows train 4032/val 948/test 660 (was 1353/315/217), ~3× positives, no time-gap spanning.
- Train-set radar norm: `radar_features(stats=)` + `scripts/fit_radar_norm.py`. Verified global norm
  preserves cross-frame signal (ch6 AUC 0.787, ch3 0.663) — fixes the instance-norm bug (radar 0.41).
- Train scripts now early-stop/checkpoint on **val AUC** (not f1@0.5). Configs use `step_ms/tol_ms`,
  radar `norm.path`. Updated smoke_test (time-guard assert) + fuse_eval (loads radar_norm).
- CPU smokes PASS: smoke_test (splits/windowing/time-guard/shapes), train_camera + train_radar
  (full loop, AUC early-stop, radar loads train-set norm). `--smoke` now forces num_workers=0/batch=2
  (CPU-safe); full runs keep the GPU config (batch32/8 workers). Re-run on GPU pending. → [[improvement-plan]]
- ⚠️ ENV: 8 DataLoader workers (GPU config) on this 4-core CPU spawned ~17 orphan python procs (one
  6 GB) that exhausted the 16 GB RAM → OOM/timeouts. Killed via `Get-Process python | Stop-Process`.

## [2026-06-02] result+fix | 2nd GPU run: camera improved, radar still flat (AMP overflow)
- Camera (dense loader) history: val AUC up to 0.990, val F1_mean 0.4–0.49 (was ~0.25), F1@t+5 up to
  0.58 → dense anchoring worked. Radar history: train_loss frozen ~1.407, val AUC random (0.10–0.89).
- Diagnosed radar: train-set norm of tiny-std magnitude channels → values ~1e5 → **fp16/AMP overflow
  → inf grads → GradScaler skips every step**. Fix: clip normalized radar feats to ±10 + nan_to_num.
- Also test JSONs didn't write (camera stale / radar+fusion missing) — `--smoke` skipped that block.
  Wrapped test eval in try/except + flushed prints so the next run surfaces results/errors. → [[lessons-learned]]
- Pushed fixes; user to re-run train_radar (+ train_camera for clean test json) + fuse_eval on GPU.

## [2026-06-02] change | Freeze camera ResNet-18 backbone
- During code verification: confirmed camera inputs (B,5,3,256,256), 300ms timestamp-spaced (every
  ~3rd native frame, not consecutive), ImageNet normalization, Fig.2-faithful architecture.
- User change: `freeze_backbone: true`. Implemented correctly — `requires_grad=False` + overrode
  `model.train()` to keep backbone in eval (BN stats frozen). Trainable params 11.5M → 346K (LSTM+head).
- Verified: 0 trainable backbone params, param_groups=1, backbone.training=False under .train(). → [[camera]]

## [2026-06-05] investigate+build | Power-derived blockage label (replaces hand-label)
- Thorough label verification: hand-label correlates with nothing (AUC≤0.65), is manual/subjective
  (DataLabel2 GUI), invisible in camera, doesn't track power. Relative-drop model fails.
- LOS-envelope model works: blocked = max_pwr ≥3 dB below per-seq rolling-q90 envelope for ≥3 frames
  (+ deep −4.5 dB override). Verified positives are genuine −4..−6 dB fades.
- Implemented `src/data/derive_label.py` + `scripts/derive_label.py` → `scenario31_dev_derived.csv`
  (`label_derived`). Threaded `label.column` config through splits/dataset/all scripts.
- Result: 6.3% pos / 32 seqs (vs hand 2.9% / 15 seqs); val/test now 5 pos seqs each; pos_weight ~14.
  derive→make_splits→smoke_test all pass. Hand label kept for comparison. → [[blockage-label]]

## [2026-06-01] build | Vault scaffolded
- Created schema [[CLAUDE]], [[index]], this log, [[00_overview]], concept pages, [[replication-plan]].
- Vault is the project's permanent memory; structured per the LLM-Wiki / Memex pattern.
