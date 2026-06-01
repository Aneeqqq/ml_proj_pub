"""Radar preprocessing: raw complex cube -> 8-channel feature tensor.

Native DeepSense Scenario 31 radar file: ``(4, 256, 250)`` complex64 =
    (4 RX antennas, 256 range bins / samples-per-chirp, 250 Doppler bins / chirps).
(See ML_Proj_Vault/dataset/deepsense-hardware.md — the paper mislabels dim-0 "azimuth";
it is the 4 RX antennas. Angle/azimuth would require an angle-FFT across them.)

Target (paper §III-A6, ML_Proj_Vault/modalities/radar.md): stack **8 feature maps**, each
normalised to spatial size ``(256, 64)`` -> output ``(8, 256, 64)`` float32.

The paper's exact 8-way assignment is under-specified; we implement a faithful, documented
set and flag the choices (see OPEN items in radar.md):

    ch0  magnitude            |x| of RX0                       (amplitude)
    ch1  phase                angle(x) of RX0                  (phase)
    ch2  doppler-FFT spectral  |FFT_doppler(x)| mean over RX   (spectral energy vs velocity)
    ch3  mean magnitude        mean_RX |x|                     (across antenna channels)
    ch4  std magnitude         std_RX  |x|                     (across antenna channels)
    ch5  entropy               Shannon entropy of norm. mag    (scene complexity proxy)
    ch6  doppler mean velocity 1st moment of power spectrum    (per-range, broadcast)
    ch7  doppler spectral width 2nd moment (sqrt) of spectrum  (per-range, broadcast)

Doppler axis is reduced 250 -> 64 by **center-crop** (trim); if ever < 64 we zero-pad.
1-D (per-range) descriptors are broadcast across the 64 Doppler columns.
"""

from __future__ import annotations

import numpy as np

RANGE_BINS = 256
DOPPLER_OUT = 64
N_FEATURES = 8


def _fit_doppler(arr: np.ndarray, out: int = DOPPLER_OUT) -> np.ndarray:
    """Trim (center-crop) or zero-pad the last axis to length `out`."""
    d = arr.shape[-1]
    if d == out:
        return arr
    if d > out:                                   # center-crop
        start = (d - out) // 2
        return arr[..., start:start + out]
    pad = out - d                                  # zero-pad symmetrically
    left = pad // 2
    return np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(left, pad - left)])


def _broadcast_range(vec: np.ndarray, out: int = DOPPLER_OUT) -> np.ndarray:
    """(256,) per-range descriptor -> (256, out) by repeating across Doppler cols."""
    return np.repeat(vec[:, None], out, axis=1)


def _entropy_map(mag: np.ndarray) -> np.ndarray:
    """Row-wise (per range bin) Shannon entropy of the magnitude distribution -> (256,)."""
    p = mag / (mag.sum(axis=-1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)
    return ent


def radar_raw_to_features(raw: np.ndarray, stats=None, normalize: bool = True) -> np.ndarray:
    """Convert one raw radar cube to an (8, 256, 64) float32 feature tensor.

    Parameters
    ----------
    raw : complex array of shape (4, 256, 250)  [RX, range, doppler]
    stats : optional (mean, std) per-channel arrays of shape (8,) fit on the TRAIN split.
        When given, applies **global** normalization (preferred — preserves the absolute
        magnitude level that encodes blockage; see improvement-plan.md / radar.md).
    normalize : if no `stats`, fall back to per-sample instance z-score (legacy; strips the
        magnitude level — only used when stats are unavailable).
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"expected (RX, range, doppler), got {raw.shape}")
    # tolerate minor range-bin drift; assert antenna dim
    if raw.shape[0] != 4:
        raise ValueError(f"expected 4 RX antennas, got shape {raw.shape}")

    mag = np.abs(raw)                                  # (4, 256, 250)
    # ch0/ch1: per-antenna amplitude & phase (use RX0)
    ch0 = mag[0]                                       # (256, 250)
    ch1 = np.angle(raw[0])                             # (256, 250)
    # ch2: Doppler-FFT spectral magnitude, mean over antennas
    spec = np.abs(np.fft.fft(raw, axis=-1))            # (4, 256, 250)
    spec = np.fft.fftshift(spec, axes=-1)              # zero-Doppler centered
    ch2 = spec.mean(axis=0)                            # (256, 250)
    # ch3/ch4: mean & std magnitude across antennas
    ch3 = mag.mean(axis=0)                             # (256, 250)
    ch4 = mag.std(axis=0)                              # (256, 250)
    # ch5: entropy per range bin (from mean-antenna magnitude), broadcast
    ent = _entropy_map(ch3)                            # (256,)
    ch5 = _broadcast_range(ent)                        # (256, 64)
    # ch6/ch7: Doppler moments from the power spectrum (mean over antennas)
    power = ch2 ** 2                                    # (256, 250)
    dop_bins = np.linspace(-1.0, 1.0, power.shape[-1]) # normalised velocity axis
    psum = power.sum(axis=-1) + 1e-12                  # (256,)
    mean_v = (power * dop_bins[None, :]).sum(axis=-1) / psum                 # (256,)
    var_v = (power * (dop_bins[None, :] - mean_v[:, None]) ** 2).sum(-1) / psum
    ch6 = _broadcast_range(mean_v)                     # (256, 64)
    ch7 = _broadcast_range(np.sqrt(np.maximum(var_v, 0.0)))                  # (256, 64)

    # fit the 2-D maps' Doppler axis to 64
    maps2d = [_fit_doppler(m) for m in (ch0, ch1, ch2, ch3, ch4)]
    feats = np.stack(maps2d + [ch5, ch6, ch7], axis=0).astype(np.float32)    # (8, 256, 64)

    if stats is not None:                                   # global (train-set) normalization
        mean, std = stats
        mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1) + 1e-6
        feats = (feats - mean) / std
    elif normalize:                                         # legacy per-sample instance norm
        m = feats.mean(axis=(1, 2), keepdims=True)
        s = feats.std(axis=(1, 2), keepdims=True) + 1e-6
        feats = (feats - m) / s
    return feats


def channel_stats(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel accumulators for fitting train-set normalization.

    Returns (sum_c, sumsq_c, n_pixels): sum and sum-of-squares over the 256x64 pixels per channel
    (shape (8,)) plus the pixel count, so callers can accumulate across many frames.
    """
    feats = radar_raw_to_features(raw, stats=None, normalize=False)   # (8,256,64) unnormalized
    return feats.sum(axis=(1, 2)), (feats ** 2).sum(axis=(1, 2)), feats.shape[1] * feats.shape[2]


def add_gaussian_noise(raw: np.ndarray, sigma: float = 0.0, rng: np.random.Generator | None = None):
    """Train-time augmentation: add complex Gaussian noise to the raw cube (paper §III-A1)."""
    if sigma <= 0:
        return raw
    rng = rng or np.random.default_rng()
    noise = rng.normal(scale=sigma, size=raw.shape) + 1j * rng.normal(scale=sigma, size=raw.shape)
    return raw + noise.astype(raw.dtype)
