---
title: GPS Modality
tags: [modality, gps]
updated: 2026-06-01
status: verified
source: paper §III-A4, §III-C, Fig. 2
priority: low
---

# GPS Blockage Model

Weakest modality (gps-only ≈ 60.6% F1). Secondary for our camera+radar focus, but documented for the
full-fusion configs. → [[results]]

## Raw data (📊 DATA — Scenario 31)
- **unit2 (vehicle):** `unit2/GPS_data/GPS_location_<index>.txt` (column `unit2_loc`) — one lat/lon
  pair per sample, e.g. `33.42001…, -111.92890…`. Plus telemetry columns in the CSV:
  `unit2_spd_over_grnd_kmph`, `unit2_num_sats`, `unit2_altitude`, `unit2_geo_sep`,
  `unit2_mode_fix_type`, `unit2_pdop/hdop/vdop`, `unit2_interpolated_position` (flag).
- **unit1 (RSU):** `unit1/GPS_data/gps_location.txt` — a **single fixed** lat/lon (RSU is stationary).
  - 📊 RSU ≈ (33.42031, -111.92911). Scene bounding box (`resources/scen_31_zoom.txt`):
    (33.41948, -111.92916) → (33.42036, -111.92873).

## Preprocessing — handcrafted 18-D feature vector (📄 PAPER §III-A4)
Computed over **5 consecutive GPS readings**, using 1st/2nd-order temporal derivatives:
- **6 displacement** values, **4 instantaneous speeds**, **3 angular changes**, **3 accelerations**,
  **1 angular velocity**, **1 curvature** → **18 elements total**.
- **Min-max normalize** with a **scaler pre-fit on the training set** (apply same scaler to val/test).

## Model (📄 PAPER §III-C + Fig. 2 ②)
- Input: 18-D feature vector (sequence of GPS-derived motion features).
- **2-layer LSTM, hidden = 128** → final hidden state.
- **FC classifier: 64-unit hidden layer → ReLU → Dropout → output** blockage prob.

- Source: **GPS-RTK, sub-10 cm, 10 Hz** ([[deepsense-hardware]]).

## Notes / abnormalities
- ⚠️ **440 rows (6.3%) have NaN in the 8 unit2 GPS *telemetry* columns** (speed, num_sats, altitude,
  geo_sep, mode_fix_type, pdop, hdop, vdop). The lat/lon (`unit2_loc`) and the label are present.
  Since the 18-D features are derived from **lat/lon over 5 readings**, telemetry NaNs are mostly
  non-blocking, but impute/mask any NaN you do use. → [[deepsense-hardware]] / [[abnormalities]]
- ⚠️ `unit2_interpolated_position` flags interpolated GPS fixes — these are less reliable; consider
  whether to include. → [[abnormalities]]
- Paper finds GPS adds little/negative value (no environmental awareness). Keep, but don't expect it
  to help camera+radar.

Related: [[architecture]] · [[fusion]] · [[scenario31-structure]]
