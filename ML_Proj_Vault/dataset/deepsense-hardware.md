---
title: DeepSense 6G — Testbed & Hardware Specs (Scenario 31)
tags: [dataset, hardware, reference, deepsense]
updated: 2026-06-01
status: verified
source: deepsense6g.net + DeepSense6G paper (web research 2026-06-01)
---

# DeepSense 6G — Testbed 5 / Scenario 31 Hardware

Official specs gathered from deepsense6g.net and the DeepSense6G dataset paper (web, 2026-06-01).
These ground the raw-data shapes measured in [[scenario31-structure]].

## Environment & units (📄 WEB — official)
- **Scenario 31:** outdoor, **two-way city street**, DeepSense **Testbed 5**.
- **Unit 1 = receiving base station (STATIONARY).** Hosts the full sensor suite **and** the mmWave
  phased array. Every sample (RGB, 64×1 power vector, LiDAR, radar) is collected by **Unit 1**.
- **Unit 2 = mobile car (TRANSMITTER).** Carries GPS-RTK; drives through the scene.
- ✅ This **confirms the data** (`unit1_*` holds the beam/power; `unit2_*` is GPS only) and
  **contradicts the paper prose** that put the array on the vehicle. → [[system-model]] / [[abnormalities]]

## mmWave communication link (the thing that gets blocked)
- **60 GHz** band. **16-element phased array** at Unit 1.
- **Over-sampled codebook of 64 pre-defined beams** sweeping a **90° field of view** at **10 Hz**.
- Per sample: a **64×1 receive-power vector** from beam training → `unit1_pwr_60ghz`;
  `unit1_beam` = argmax beam (1..64), `unit1_max_pwr` = max power. → [[scenario31-structure]] / [[blockage-label]]
- **10 Hz** ✅ corroborates the measured ~100 ms native sampling. → [[problem-formulation]]

## Sensors at Unit 1
| Sensor | Model / spec | Notes |
|---|---|---|
| **Camera** | RGB **960×540**, base **30 fps** | matches measured 960×540. → [[camera]] |
| **Radar** | **TI AWR2243BOOST** FMCW, **76–81 GHz**, **750 MHz** BW, up to 20 Hz, **4 RX × 1 TX** | separate band from the 60 GHz comms link! → [[radar]] |
| **LiDAR** | **Ouster OS1-32**, 32 × 1024, 120 m range, 20 Hz | → [[lidar]] |
| **GPS (unit2)** | **GPS-RTK**, sub-10 cm, **10 Hz** | telemetry has NaNs (below). → [[gps]] |

## Radar frame format (📄 WEB — confirms our npy)
- DeepSense radar frame = **(# RX antennas) × (# samples per chirp) × (# chirps per frame)**.
- Our files: **`(4, 256, 250)` complex64** = **(4 RX, 256 range-samples, 250 Doppler-chirps)**.
- Standard DeepSense processing: raw I/Q → **range-angle (RA)** and **range-velocity (RV)** magnitude
  maps via **2D FFTs**; range resolution ≈ 0.2 m, max range ≈ 45 m. → [[radar]]
- ⚠️ The paper calls dim-0 "azimuth"; it is actually the **4 RX antennas**. Azimuth/angle is only
  obtained *after* an angle-FFT across those 4 RX. Treat dim-0 as antennas. → [[radar]] / [[abnormalities]]

## Provenance / context (📄 WEB)
- Scenarios **31–34** are the **Multi-Modal Beam Prediction Challenge 2022** scenarios. In the
  challenge, **dev = scenarios 32/33/34** and the **test set's "unseen" half = scenario 31**.
- ⚠️ These challenge scenarios ship **beam/power vectors, NOT a native blockage label** — so a
  blockage label must be **added by the user** (which is exactly what was done →
  `scenario31_dev_labelled.csv`). The paper authors must have labelled blockage themselves too. → [[blockage-label]]
- 📄 WEB: DeepSense notes **~200 samples with NaNs across scenarios 31–34** from real sensor errors.

## Data quality in OUR file (📊 measured, `scenario31_dev_labelled.csv`)
- **440 rows (6.3%)** have **NaN in the 8 unit2 GPS *telemetry* columns** (`unit2_spd_over_grnd_kmph`,
  `unit2_num_sats`, `unit2_altitude`, `unit2_geo_sep`, `unit2_mode_fix_type`, `unit2_pdop`,
  `unit2_hdop`, `unit2_vdop`).
- **Clean (0 NaN):** `unit1_beam`, `unit1_max_pwr`, `label`, and all sensor *path* columns
  (camera/radar/lidar/power/`unit2_loc`). Sampled camera & radar files all exist on disk.
- **Implication:** core camera+radar pipeline is unaffected; only the **GPS** modality must handle
  the 440 telemetry NaNs (impute/mask, or rely on `unit2_loc` lat/lon which is present). → [[gps]]

Related: [[scenario31-structure]] · [[system-model]] · [[radar]] · [[camera]] · [[problem-formulation]]
