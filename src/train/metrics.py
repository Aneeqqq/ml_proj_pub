"""Per-horizon metrics: F1 and AUC-ROC (paper §IV KPIs) + threshold tuning.

The model outputs K logits (t+1..t+K). We compute F1 and AUC-ROC per horizon, plus means.
AUC is NaN-guarded when a horizon's labels are single-class.

Because positives are scarce and the loss uses a large `pos_weight`, the sigmoid outputs are
uncalibrated -> a fixed 0.5 threshold gives pessimistic F1 even when ranking (AUC) is strong.
So we **tune one threshold per horizon on the validation set** (maximising F1) and apply those
to the test set. See ML_Proj_Vault/lessons-learned.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def tune_thresholds(logits: np.ndarray, labels: np.ndarray, grid: int = 199) -> np.ndarray:
    """Return per-horizon thresholds that maximise F1 on (logits, labels). Shape (K,).

    Horizons with no positive labels fall back to 0.5.
    """
    probs = _sigmoid(logits)
    K = labels.shape[1]
    candidates = np.linspace(0.005, 0.995, grid)
    thr = np.full(K, 0.5, dtype=float)
    for k in range(K):
        y = labels[:, k].astype(int)
        if y.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            f1 = f1_score(y, (probs[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def per_horizon_metrics(logits: np.ndarray, labels: np.ndarray, thresholds=None) -> dict:
    """logits, labels: (N, K). `thresholds`: None->0.5, scalar, or array (K,)."""
    probs = _sigmoid(logits)
    K = labels.shape[1]
    if thresholds is None:
        thr = np.full(K, 0.5)
    elif np.isscalar(thresholds):
        thr = np.full(K, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=float)

    f1s, aucs = [], []
    for k in range(K):
        y = labels[:, k].astype(int)
        preds = (probs[:, k] >= thr[k]).astype(int)
        f1s.append(f1_score(y, preds, zero_division=0))
        aucs.append(roc_auc_score(y, probs[:, k]) if len(np.unique(y)) > 1 else float("nan"))
    return {
        "thresholds": [round(float(t), 3) for t in thr],
        "f1_per_horizon": [round(float(v), 4) for v in f1s],
        "auc_per_horizon": [round(float(v), 4) for v in aucs],
        "f1_mean": float(np.nanmean(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "f1_t5": float(f1s[-1]),
        "auc_t5": float(aucs[-1]),
    }
