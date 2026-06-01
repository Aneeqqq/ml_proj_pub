---
title: Dataset — Scenario 31 Structure
tags: [dataset, deepsense, structure]
updated: 2026-06-01
status: verified
source: ../scenario31_new/scenario31/ (measured 2026-06-01)
---

# DeepSense 6G — Scenario 31 (the provided data)

Path: `../scenario31_new/scenario31/`. This is **one scenario** of the 31–34 the paper uses.
Everything below is 📊 **measured from the data**.

> ✅ **Use `scenario31_dev_labelled.csv`** (21 cols = the 20 below + `label` ∈ {blocked,
> not_blocked}). The original `scenario31_dev.csv` is unlabelled. See [[blockage-label]].

## Directory layout
```
scenario31/
  scenario31_dev.csv          <- master index (7012 rows + header), source of truth for samples
  resources/
    scen_31_zoom.png/.txt      <- map view + bounding box of the scene
  unit1/                       <- RSU (stationary; sensors + 60 GHz beam array)
    camera_data/   image_<i>.jpg          (960×540 RGB)   7012 files
    radar_data/    radar_data_<i>.npy      (4,256,250) complex64   7012 files
    lidar_data/    lidar_data_<i>.ply      point clouds    7012 files
    mmWave_data/   mmWave_power_<i>.txt     64-D power vector  7012 files
    GPS_data/      gps_location.txt         single fixed RSU lat/lon
  unit2/                       <- vehicle (mobile)
    GPS_data/      GPS_location_<i>.txt     per-sample lat/lon
```

## `scenario31_dev.csv` — 20 columns (📊)
| # | column | meaning |
|---|---|---|
| 1 | `index` | global sample id (range **173 → 8535**, with gaps; 7012 rows) |
| 2 | `unit1_rgb` | rel path to camera jpg → [[camera]] |
| 3 | `unit1_pwr_60ghz` | rel path to 64-D mmWave power txt (per-beam received power) |
| 4 | `unit1_lidar` | rel path to .ply → [[lidar]] |
| 5 | `unit1_radar` | rel path to .npy → [[radar]] |
| 6 | `unit1_loc` | RSU GPS (constant) |
| 7 | `unit2_loc` | vehicle GPS per sample → [[gps]] |
| 8 | `unit1_beam` | **best beam index ∈ 1..64** (argmax of the 64-D power) |
| 9 | `unit1_max_pwr` | **max received power** (0.141 → 0.754) — key for [[blockage-label]] |
| 10 | `time_stamp` | `HH:MM:SS-microseconds` |
| 11 | `seq_index` | **scene/sequence id** — CRITICAL → [[sequences-and-batching]] |
| 12–19 | unit2 telemetry | speed_kmph, num_sats, altitude, geo_sep, mode_fix_type, pdop, hdop, vdop |
| 20 | `unit2_interpolated_position` | 0/1 flag, GPS fix interpolated |
| 21 | `label` | **(labelled CSV only)** `blocked`/`not_blocked` → y∈{1,0} → [[blockage-label]] |

## Per-modality raw facts (📊)
- **Camera:** 960×540 RGB JPEG. → [[camera]]
- **Radar:** `(4, 256, 250)` complex64 = (4 antenna ch, 256 range, 250 Doppler). → [[radar]]
- **mmWave power:** 64-D float vector per sample (per-beam power); `unit1_beam`=argmax,
  `unit1_max_pwr`=max. **This is the link-quality signal blockage is defined from.** → [[blockage-label]]
- **LiDAR:** `.ply` point clouds. → [[lidar]]
- **GPS:** unit2 per-sample lat/lon (+telemetry); unit1 fixed. → [[gps]]

## mmWave / beam facts (📊)
- **64 beams** (1..64), 60 GHz (`pwr_60ghz`). Beam index **increases monotonically within a
  sequence** as the vehicle drives through (beam tracking the moving user).
- `unit1_max_pwr`: mean 0.228, median 0.180, range 0.141–0.754; right-skewed (77% < 0.25).

## Counts (📊)
- 7012 samples, **52 sequences** (`seq_index` 2..63 with gaps), sequence lengths **66–281**.
- Native sampling **~10 Hz** (within-seq mean Δt ≈ 100.8 ms). Full breakdown → [[sequences-and-batching]].

## Data quality (📊)
- **440 rows (6.3%) NaN in unit2 GPS telemetry columns**; camera/radar/lidar/beam/max_pwr/label and
  all path columns are clean; sampled files exist on disk. → [[deepsense-hardware]] / [[gps]]

## Hardware/testbed (📄 WEB — full specs)
- Unit1 = stationary BS (16-elem 60 GHz array, 64-beam codebook, 90° FOV, 10 Hz) + sensor suite;
  Unit2 = mobile car (GPS-RTK). Camera 960×540@30fps; radar TI AWR2243 (76–81 GHz, 4RX×1TX); LiDAR
  Ouster OS1-32. Two-way city street. Full page → [[deepsense-hardware]].

Related: [[sequences-and-batching]] · [[blockage-label]] · [[abnormalities]] · [[problem-formulation]]
