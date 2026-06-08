"""Print the experiment registry (outputs/experiments.jsonl) as a comparison table.

Run:  .venv/Scripts/python.exe -m scripts.experiments
TensorBoard curves:  tensorboard --logdir outputs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reg = ROOT / "outputs" / "experiments.jsonl"
    if not reg.exists():
        print("no runs yet:", reg)
        return
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    cols = ["timestamp", "run_name", "modality", "protocol", "alpha", "freeze_backbone",
            "batch_size", "K", "epochs_ran", "best_val_auc", "test_auc", "test_f1_tuned", "test_f1_0p5"]
    df = df[[c for c in cols if c in df.columns]]
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
