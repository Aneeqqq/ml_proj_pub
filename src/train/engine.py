"""Train / eval steps for a single-modality blockage model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .metrics import per_horizon_metrics


def _to_logits(model, batch, modality, device):
    nb = device != "cpu"
    x = batch[modality].to(device, non_blocking=nb)
    y = batch["label"].to(device, non_blocking=nb)    # (B, K)
    return model(x), y


def train_one_epoch(model, loader, optimizer, criterion, device, modality="camera",
                    max_batches: int | None = None, scaler=None, use_amp: bool = False) -> float:
    """One training epoch. Set use_amp=True (+ a GradScaler) for mixed precision on CUDA."""
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, y = _to_logits(model, batch, modality, device)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, y = _to_logits(model, batch, modality, device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, modality="camera", criterion=None,
            max_batches: int | None = None):
    """Run the model over a loader. Returns (logits, labels, loss) as numpy / float."""
    model.eval()
    total, n = 0.0, 0
    all_logits, all_labels = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits, y = _to_logits(model, batch, modality, device)
        if criterion is not None:
            total += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.float().cpu().numpy())
    logits = np.concatenate(all_logits, 0)
    labels = np.concatenate(all_labels, 0)
    loss = (total / max(n, 1)) if criterion is not None else float("nan")
    return logits, labels, loss


def evaluate(model, loader, criterion, device, modality="camera",
             max_batches: int | None = None, thresholds=None) -> dict:
    logits, labels, loss = predict(model, loader, device, modality, criterion, max_batches)
    metrics = per_horizon_metrics(logits, labels, thresholds=thresholds)
    metrics["loss"] = loss
    return metrics
