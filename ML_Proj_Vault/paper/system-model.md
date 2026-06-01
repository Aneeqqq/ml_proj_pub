---
title: System Model (Fig. 1)
tags: [paper, system-model]
updated: 2026-06-01
status: verified
source: paper §II, Fig. 1 (../_fig1.png)
---

# System Model — §II / Fig. 1

## The scene (from Fig. 1, rendered at ../_fig1.png)
An **RSU (roadside unit)** at an intersection carries a **base station (BS)** co-located and
**time-synchronized** with a multi-sensor suite: **camera, GPS, radar, LiDAR**. The RSU beams
mmWave signals (green lobes in the figure) to **vehicles** on the road. A large vehicle (a **bus**)
sits between the RSU and a target car, **blocking the LOS** — this is the blockage event to predict.
Output is a binary decision: **"Clear Signal" vs "Blockage"** ("Proactive I2V Blockage Prediction").

## Components (📄 PAPER §II)
- **Two geo-tagged components:**
  1. A **vehicle** equipped with a **uniform linear array (ULA)** of **M antennas**.
  2. A **fixed RSU** with a **single-antenna BS**, co-located + time-synced with camera, GPS, radar, LiDAR.
- The RSU **periodically captures** sensor data and feeds it to the blockage model.
- Setting: **I2V** (infrastructure-to-vehicle), mmWave band (60 GHz — see [[scenario31-structure]]).

## ⚠️ ABNORMALITY — which unit holds the array? (RESOLVED: data & DeepSense are right; paper prose is swapped)
The paper text says the **vehicle** has the M-antenna ULA and the **RSU** is single-antenna.
But the **DeepSense official spec confirms the opposite** ([[deepsense-hardware]]): **Unit 1 =
stationary base station** with a **16-element 60 GHz phased array**, a **64-beam codebook over a
90° FOV at 10 Hz**; **Unit 2 = mobile car (transmitter)** carrying only **GPS-RTK**. The data agrees
(`unit1_pwr_60ghz`/`unit1_beam` exist; `unit2` is GPS only). So **the RSU side does the 64-beam
measurement**, and blockage is inferred from Unit 1's received-power behavior. The paper prose is
simply mis-stated; trust the data + DeepSense. Tracked in [[abnormalities]].

## How this maps to the data
- `unit1_*` = RSU (sensors + 60 GHz beam power + best beam). → [[scenario31-structure]]
- `unit2_*` = vehicle (GPS lat/lon + telemetry). → [[gps]]
- Blockage = degradation of the unit1↔unit2 mmWave link. The **label derivation** is the open
  problem — see [[blockage-label]].

Related: [[problem-formulation]] · [[architecture]] · [[blockage-prediction]]
