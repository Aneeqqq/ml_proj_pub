"""Build the combined camera dataset for DeepSense Scenario 2 + 5 (NATIVE blockage labels).

Unlike scenarios 31-34 (no labels -> power-derived), Scenario 2/5 ship `labels_unit1.csv`
with a real `label` column (blocked / not_blocked). That file is row-ordered by the mmWave
power-file *string* number (0, 1, 10, 100, ...), NOT temporal order, and its time_stamp /
seq_index columns are garbage - so we join its `label` onto the main scenario CSV (which has
proper seq_index + time_stamp[UTC]) by the unique camera image filename.

Output: data/dataset_s2s5.csv with the standard pipeline columns
  scenario, seq_uid, seq_index, index, time_stamp (HH:MM:SS-microseconds), unit1_rgb, label

Run: .venv/Scripts/python.exe -m scripts.build_dataset_s2s5
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [("2", "Scenario2", "scenario2.csv"),
             ("5", "Scenario5", "scenario5.csv")]


def std_time(ts: str) -> str:
    """"['02-11-07-142']" (HH-MM-SS-milliseconds) -> "02:11:07-142000" (pipeline format)."""
    h, m, s, ms = ts.strip("[]'").split("-")
    return f"{h}:{m}:{s}-{int(ms) * 1000}"


def main() -> None:
    frames = []
    for name, folder, csv in SCENARIOS:
        main_df = pd.read_csv(ROOT / folder / csv)
        lab = pd.read_csv(ROOT / folder / "labels_unit1.csv")
        main_df["img"] = main_df["unit1_rgb"].str.split("/").str[-1]
        lab["img"] = lab["unit1_rgb"].str.split("/").str[-1]
        m = main_df.merge(lab[["img", "label"]], on="img", how="left", validate="one_to_one")
        assert m["label"].notna().all(), f"{folder}: unmatched labels after image-name join"
        out = pd.DataFrame({
            "scenario": name,
            "seq_uid": name + ":" + m["seq_index"].astype(str),
            "seq_index": m["seq_index"],
            "index": m["index"],
            "time_stamp": m["time_stamp[UTC]"].map(std_time),
            "unit1_rgb": "./" + folder + "/" + m["unit1_rgb"].str.replace(r"^\./", "", regex=True),
            "label": m["label"],
        })
        pos = (out["label"] == "blocked").mean()
        print(f"scenario {name}: {len(out)} rows, {out['seq_uid'].nunique()} seqs, "
              f"pos_rate {pos:.4f}")
        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)
    dest = ROOT / "data" / "dataset_s2s5.csv"
    dest.parent.mkdir(exist_ok=True)
    all_df.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(all_df)} rows, {all_df['seq_uid'].nunique()} sequences)")


if __name__ == "__main__":
    main()
