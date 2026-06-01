"""Late fusion primitives: per-modality probabilities -> softmax(F1)-weighted average.

Key requirement: the per-modality probabilities must be aligned to the **same windows**. We
therefore run all models over ONE combined loader (modalities=("camera","radar",...)), so window
order is identical across modalities.
"""

from __future__ import annotations

import numpy as np
import torch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def predict_probs(models: dict, loader, device: str, max_batches: int | None = None):
    """Run each modality model over a shared loader.

    models: {"camera": model, "radar": model, ...} where the key is the batch modality key.
    Returns (probs: {modality: (N,K)}, labels: (N,K)).
    """
    for m in models.values():
        m.eval()
    logits = {k: [] for k in models}
    labels = []
    nb = device != "cpu"
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        labels.append(batch["label"].numpy())
        for key, model in models.items():
            x = batch[key].to(device, non_blocking=nb)
            logits[key].append(model(x).float().cpu().numpy())
    probs = {k: _sigmoid(np.concatenate(v, 0)) for k, v in logits.items()}
    labels = np.concatenate(labels, 0)
    return probs, labels


def softmax_f1_weights(f1_by_modality: dict, temperature: float = 1.0) -> dict:
    """w_i = softmax(F1_i / T) over modalities.

    f1_by_modality: {modality: scalar F1}  OR  {modality: array (K,)} for per-horizon weights.
    Returns weights in the same shape per modality (summing to 1 across modalities).

    NOTE (see fusion.md ABNORMALITY): raw F1 values are close, so plain softmax (T=1) is nearly
    uniform. `temperature < 1` sharpens toward the better modality; the paper uses T=1.
    """
    keys = list(f1_by_modality)
    vals = np.stack([np.asarray(f1_by_modality[k], dtype=float) for k in keys], axis=0)  # (M,) or (M,K)
    z = np.exp(vals / temperature)
    w = z / z.sum(axis=0, keepdims=True)
    return {k: w[i] for i, k in enumerate(keys)}


def fuse_probs(probs: dict, weights: dict) -> np.ndarray:
    """P_fused = sum_i w_i * P_i. weights[i] is scalar or (K,) (broadcast over N)."""
    keys = list(probs)
    fused = np.zeros_like(probs[keys[0]])
    for k in keys:
        w = np.asarray(weights[k], dtype=float)
        fused += probs[k] * (w if w.ndim == 0 else w[None, :])
    return fused
