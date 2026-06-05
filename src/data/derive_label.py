"""Derive a clean, self-consistent blockage label from received power.

Rationale (see ML_Proj_Vault/dataset/blockage-label.md): DeepSense scenarios 31-34 ship no blockage
truth, and the manual hand-label correlates with neither power nor camera. We define blockage
physically as a **fade below the line-of-sight (LOS) envelope**:

  * per sequence, smooth `unit1_max_pwr` (3-frame median) and take the LOS envelope as the rolling
    90th-percentile (~4 s window) = the clear-LOS power achievable at that range;
  * drop_db = 10*log10(power / envelope);
  * blocked = power >= `thr_db` below the envelope for >= `min_dur` consecutive frames,
    OR a deep fade <= `deep_db` (caught regardless of duration).

Applied per sequence (never across `seq_index`), so it transfers unchanged to other scenarios.
Default: thr_db=-3 dB (half LOS power), min_dur=3 (~0.3 s), deep_db=-4.5 dB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POS, NEG = "blocked", "not_blocked"


def _drop_db(df: pd.DataFrame, env_win: int, q: float, pwr_col: str) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("seq_index"):
        g = g.sort_values("index").copy()
        s = g[pwr_col].rolling(3, center=True, min_periods=1).median()
        env = s.rolling(env_win, center=True, min_periods=max(5, env_win // 3)).quantile(q)
        g["_los_env"] = env
        g["_drop_db"] = 10 * np.log10((s / env).clip(lower=1e-6))
        out.append(g)
    return pd.concat(out).sort_index()


def _apply_threshold(drop_db: np.ndarray, thr_db: float, min_dur: int, deep_db: float) -> np.ndarray:
    """Per-sequence array -> bool. min_dur run-length on the thr crossing, plus deep-fade override."""
    raw = drop_db <= thr_db
    out = np.zeros(len(raw), bool)
    j, n = 0, len(raw)
    while j < n:
        if raw[j]:
            k = j
            while k < n and raw[k]:
                k += 1
            if k - j >= min_dur:
                out[j:k] = True
            j = k
        else:
            j += 1
    out |= (drop_db <= deep_db)            # deep brief fades counted regardless of duration
    return out


def derive_blockage_label(
    df: pd.DataFrame,
    pwr_col: str = "unit1_max_pwr",
    env_win: int = 41,
    q: float = 0.90,
    thr_db: float = -3.0,
    min_dur: int = 3,
    deep_db: float = -4.5,
) -> pd.Series:
    """Return a Series of 'blocked'/'not_blocked' aligned to df.index."""
    bb = _drop_db(df, env_win, q, pwr_col)
    lab = np.zeros(len(bb), int)
    for _, g in bb.groupby("seq_index"):
        g = g.sort_values("index")
        flags = _apply_threshold(g["_drop_db"].to_numpy(), thr_db, min_dur, deep_db)
        lab[g.index.to_numpy()] = flags.astype(int)
    return pd.Series(np.where(lab == 1, POS, NEG), index=bb.sort_index().index, name="label_derived")


def add_derived_label(df: pd.DataFrame, **kw) -> pd.DataFrame:
    df = df.copy()
    df["label_derived"] = derive_blockage_label(df, **kw).values
    return df
