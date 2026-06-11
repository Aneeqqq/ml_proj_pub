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

## [2026-06-07] build | Multi-scenario (31-34) cross-scenario pipeline
- Added scenarios 32 (3235/15), 33 (3981/18), 34 (4439/31) — same schema as 31. Total 18,667/116.
  Derived label generalizes (4.9-9.3% blocked, files present, ~92ms). → [[multi-scenario]]
- `scripts/build_dataset.py`: derive label per scenario + combine → `data/dataset_all.csv` with
  `scenario`, `seq_uid` (collision-safe), and ML_Proj_Claude-relative paths. Config reworked to a
  `scenarios:` list + `seq_col: seq_uid` + `split.protocol: cross_scenario`.
- `splits.py`: `cross_scenario_split` (test=31, stratified train/val on 32/33/34) + `split_from_config`
  dispatcher. Threaded `seq_col` through dataset/build_windows/all scripts.
- DECISION (user): **cross-scenario** protocol (train 32/33/34, test unseen 31). Verified:
  train 8368 / val 1683 / test 5640 windows; smoke_test passes. → [[multi-scenario]]

## [2026-06-08] result | First cross-scenario run (t+1) — generalization FAILS
- Setup: derived label, frozen camera backbone, K=1 (t+1), train/val=32/33/34, test=unseen 31.
- Camera: val AUC 0.68 (on 32/33/34 val) but **TEST AUC 0.46 on unseen scn31 — below random**, F1 0.08.
  → camera learned scene-specific geometry; does NOT transfer to a new location (inverts on 31).
- Radar: **train_loss frozen ~1.354**, val_f1 constant 0.147 → not learning at all (separate bug,
  not AMP). Test AUC 0.54.
- Fused: AUC 0.455 (no help).
- KEY: cross-scenario generalization fails. NOTE: the paper likely **pooled** 31-34 (within-distribution),
  not held-out — so pooled protocol is the fair replication; cross-scenario is a harder honesty test.
- TODO: (1) run POOLED protocol (sanity + paper-comparable); (2) debug radar flat train_loss;
  (3) consider unfreezing camera. Results committed on GPU box (push pending interactive creds). → [[multi-scenario]]

## [2026-06-11] result | pooled_v2_r2p1d (R(2+1)D-18) — incomplete, box went down at epoch 12
- v2 fixes active (balanced sampler, brightness aug, cosine LR, Kinetics-pretrained video model).
- 12 epochs ran: best **val AUC 0.726 @ep6**; train_loss 0.44→0.08 (fast overfit), val oscillating
  0.60–0.73. NOT better than frozen-resnet pooled baseline's val 0.748 so far. No test eval (laptop
  slept/shut down mid-run; SSH dead). Checkpoint ep6 should exist on box: outputs/pooled_v2_r2p1d/.
- TODO when box returns: add --eval-only mode or rerun; also pooled_v2_resnet never ran. Consider
  stronger regularization for r2p1d (it memorizes 12k windows in ~6 epochs: more dropout/weight
  decay, freeze early stages, or fewer epochs).

## [2026-06-09] state | Experiments + tracking; SSH-driven GPU runs
- Diagnosis done (see [[cross-scenario-investigation]]): cross-scenario fails (domain shift: 31/32 day,
  33/34 night, diff intersections); radar has no signal for the 60GHz-fade label; radar model healthy.
- Pooled (within-dist) confirms: camera test AUC 0.68, radar 0.50, fused 0.68 (α=1.1, frozen).
- Infra added: per-run output folders `outputs/<run_name>/`; experiment tracking (TensorBoard +
  `outputs/experiments.jsonl` registry, `scripts/experiments.py` table); AMP re-enabled + persistent
  train workers (speed); SSH access to GPU box (key `~/.ssh/gpu_ml`, host computer house@192.168.18.86).
- Driving GPU over SSH from this PC: `git checkout -- outputs & git pull --rebase && python -m scripts.X`.
  GPU can't `git push` (no interactive creds) — read results via SSH / capture from logs.
- ⏸ PENDING (resume tomorrow): run `pooled_a037_unfrozen` (unfrozen backbone, α=0.367) to see if
  fine-tuning lifts camera above the 0.68 frozen baseline. Box idle, repo at ec9c7c2.

## [2026-06-01] build | Vault scaffolded
- Created schema [[CLAUDE]], [[index]], this log, [[00_overview]], concept pages, [[replication-plan]].
- Vault is the project's permanent memory; structured per the LLM-Wiki / Memex pattern.

## [2026-06-11] result | pooled_v2_r2p1d ep6 ckpt scored (--eval-only) + regularized rerun launched
- Scored the surviving epoch-6 checkpoint: **test AUC 0.641, F1 tuned 0.193** (val 0.726) - *below* the frozen-resnet pooled baseline (0.678 / 0.212). Confirms the unregularized r2p1d overfit.
- Per-scenario test: scn31 auc 0.588 | scn33 auc 0.554 | scn34 auc **0.915** (474 win). Scenario 34 is far easier; 31/33 drag the average. Worth a vault page if it repeats.
- New code (e29c3a8): `--eval-only` flag; R2Plus1DBlockage `freeze_stages` (frozen stem+L1+L2 kept in eval), dropout 0.4->0.5, weight_decay 1e-4->5e-4.
- Launched **pooled_v2_r2p1d_reg** (chained after eval). Epoch 1: train_loss 0.458, val AUC 0.636.

## [2026-06-11] result | s2s5_r2p1d - Scenario 2+5, NATIVE labels, R(2+1)D-18: val AUC 0.968
- New experiment per user: camera-only, scenarios 2+5 (stationary testbed, real DeepSense `labels_unit1.csv` blockage labels - first native labels in the project), train/val only (85/15, no test), fresh Kinetics weights, v2 regularization (freeze stem+L1+L2, dropout 0.5, wd 5e-4).
- Data: builder joins labels onto scenarioX.csv by image filename (labels file is power-file-string-ordered with garbage timestamps). 5274 rows, 63 seqs, ~8 Hz frames; 3899 train / 699 val windows.
- Result: **best val AUC 0.9681 @ep9** (ep1 already 0.964), early stop @19. F1@0.5 noisy (0.0-0.66) - calibration, not ranking; threshold tuning would fix.
- Read: with real labels + visible testbed blockages the model is excellent -> the 31-34 weakness is the label/visibility problem, not the architecture. Checkpoint: outputs/s2s5_r2p1d/camera_best.pt on GPU box.

## [2026-06-11] verify | s2s5 scene/timestamp loading is correct
Rechecked scene loading via timestamps (the make-or-break concern). 63 scenes, 4598 windows:
- 0 non-monotonic scenes; intra-scene dt median 0.125s (~8 Hz), p95 0.166s.
- 0 time-range overlaps between consecutive scenes (scenes genuinely distinct).
- 11 scenes have internal dropped-frame gaps (1.0-1.33s) - handled: windows can't span them.
- Window builder: 0 cross-scene, 0 gap-spanning (max adjacent gap 0.500s < step+2tol bound 0.560s);
  worst per-frame deviation from the 0.3s grid 0.120s < tol 0.13. The 37 "jumps" = benign single-drop jitter.
- Timestamp cross-checks the filename time (image_BS1_396_02_11_07 <-> ['02-11-07-0']).
Conclusion: s2s5_r2p1d val AUC 0.968 is on scene-coherent, correctly-windowed data.
