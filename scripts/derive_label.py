"""Write a CSV with a power-derived `label_derived` column (keeps the hand `label` intact).

Run:  .venv/Scripts/python.exe -m scripts.derive_label
Reads the CSV in configs/data.yaml, derives the LOS-envelope fade label, writes
scenario31_dev_derived.csv next to it, and prints stats + per-sequence positive counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.derive_label import add_derived_label

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    src_csv = ROOT / d["paths"]["csv_handlabel"] if "csv_handlabel" in d["paths"] else ROOT / d["paths"]["csv"]
    lc = d.get("label", {})
    params = dict(env_win=lc.get("env_win", 41), q=lc.get("env_q", 0.90),
                  thr_db=lc.get("thr_db", -3.0), min_dur=lc.get("min_dur", 3),
                  deep_db=lc.get("deep_db", -4.5))
    print("deriving label with", params)

    df = pd.read_csv(src_csv)
    df = add_derived_label(df, **params)
    out = src_csv.with_name("scenario31_dev_derived.csv")
    df.to_csv(out, index=False)

    y = (df["label_derived"] == "blocked")
    print(f"\nderived positive rate: {y.mean():.3f} ({int(y.sum())}/{len(df)})")
    if "label" in df.columns:
        h = (df["label"] == "blocked")
        both = int((y & h).sum())
        print(f"hand-label positive rate: {h.mean():.3f} | frames blocked in BOTH: {both}")
    nseq = df[y].groupby("seq_index").size()
    print(f"sequences with >=1 derived blockage: {nseq.size}/{df['seq_index'].nunique()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
