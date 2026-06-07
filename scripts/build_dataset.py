"""Build the combined multi-scenario dataset CSV.

For each scenario in configs/data.yaml:
  * derive the power-based blockage label (`label_derived`);
  * tag rows with `scenario` and a globally-unique `seq_uid` = "<scenario>:<seq_index>";
  * rewrite sensor path columns to be relative to ML_Proj_Claude/ (root + "./unit1/..").
Concatenate all scenarios -> data/dataset_all.csv (the pipeline's single source).

Run:  .venv/Scripts/python.exe -m scripts.build_dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.derive_label import derive_blockage_label

ROOT = Path(__file__).resolve().parents[1]
PATH_COLS = ["unit1_rgb", "unit1_radar", "unit1_lidar", "unit1_pwr_60ghz"]


def main() -> None:
    d = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    lc = d["label"]
    params = dict(env_win=lc["env_win"], q=lc["env_q"], thr_db=lc["thr_db"],
                  min_dur=lc["min_dur"], deep_db=lc["deep_db"])

    frames = []
    for sc in d["scenarios"]:
        name, root = sc["name"], sc["root"].rstrip("/")
        df = pd.read_csv(ROOT / sc["csv"])
        df["label_derived"] = derive_blockage_label(df, **params).values
        df["scenario"] = name
        df["seq_uid"] = name + ":" + df["seq_index"].astype(str)
        for c in PATH_COLS:
            if c in df.columns:
                df[c] = root + "/" + df[c].astype(str).str.lstrip("./")
        pos = (df.label_derived == "blocked")
        print(f"  scenario {name}: {len(df):>5} rows, {df.seq_index.nunique():>2} seqs, "
              f"{100*pos.mean():.1f}% blocked")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out = ROOT / d["paths"]["csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\ntotal: {len(combined)} rows, {combined.seq_uid.nunique()} sequences across "
          f"{combined.scenario.nunique()} scenarios -> {out}")
    print("scenario blocked rates:")
    print((combined.assign(b=combined.label_derived == "blocked")
           .groupby("scenario").b.mean().round(3)).to_string())


if __name__ == "__main__":
    main()
