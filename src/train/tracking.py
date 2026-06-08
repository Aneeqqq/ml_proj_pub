"""Lightweight experiment tracking: TensorBoard per-epoch curves + a JSONL run registry.

- TensorBoard: per-run scalars under outputs/<run_name>/tb/ (view with `tensorboard --logdir outputs`).
  Gracefully no-ops if the `tensorboard` package isn't installed.
- Registry: one JSON line per finished run appended to outputs/experiments.jsonl, so all runs can be
  compared in a table (see scripts/experiments.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:                       # tensorboard not installed -> skip TB, keep registry
    _HAS_TB = False


def get_writer(outdir):
    """TensorBoard writer at outdir/tb (or None if tensorboard unavailable)."""
    if not _HAS_TB:
        return None
    return SummaryWriter(log_dir=str(Path(outdir) / "tb"))


def log_scalars(writer, step: int, **scalars) -> None:
    if writer is None:
        return
    for k, v in scalars.items():
        try:
            if v is not None and v == v:    # skip NaN
                writer.add_scalar(k, float(v), step)
        except Exception:
            pass


def close_writer(writer) -> None:
    if writer is not None:
        writer.close()


def append_run(row: dict, registry="outputs/experiments.jsonl") -> None:
    """Append one finished-run record (config + metrics) to the JSONL registry."""
    p = Path(registry)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
