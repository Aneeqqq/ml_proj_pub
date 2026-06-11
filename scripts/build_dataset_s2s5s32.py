"""Build combined dataset: Scenario 2+5 (native labels) + Scenario 32 (derived labels).

Scenarios 2+5 ship labels_unit1.csv with real blockage (`blocked` / `not_blocked`).
Scenario 32 has no native labels -> use power-derived label (see src/data/derive_label.py).
This allows side-by-side inspection of native vs derived labels, and validation that the
derived label in scenario 32 corresponds to visible blockages.

Run: .venv/Scripts/python.exe -m scripts.build_dataset_s2s5s32
Output: data/dataset_s2s5s32.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.derive_label import add_derived_label

ROOT = Path(__file__).resolve().parents[1]


def std_time(ts: str) -> str:
    """['HH-MM-SS-ms'] or "HH:MM:SS-us" -> "HH:MM:SS-microseconds" (pipeline format)."""
    if ts.startswith("["):  # Scenario 2/5 format: ['HH-MM-SS-ms']
        h, m, s, ms = ts.strip("[]'").split("-")
        return f"{h}:{m}:{s}-{int(ms) * 1000}"
    else:  # Scenario 32 format: HH:MM:SS-us
        return ts


def main() -> None:
    frames = []

    # Scenarios 2+5: native labels
    for name, folder in [("2", "Scenario2"), ("5", "Scenario5")]:
        main_df = pd.read_csv(ROOT / folder / "scenario2.csv" if folder == "Scenario2"
                               else ROOT / folder / "scenario5.csv")
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
        print(f"scenario {name} (native): {len(out)} rows, {out['seq_uid'].nunique()} seqs, "
              f"pos_rate {pos:.4f}")
        frames.append(out)

    # Scenario 32: derive blockage labels from power
    csv32 = pd.read_csv(ROOT / "scenario32" / "scenario32_dev.csv")
    m32 = add_derived_label(csv32)
    out32 = pd.DataFrame({
        "scenario": "32",
        "seq_uid": "32:" + m32["seq_index"].astype(str),
        "seq_index": m32["seq_index"],
        "index": m32["index"],
        "time_stamp": m32["time_stamp"].map(std_time) if "time_stamp[UTC]" not in m32.columns else m32["time_stamp[UTC]"].map(std_time),
        "unit1_rgb": "./scenario32/" + m32["unit1_rgb"].str.replace(r"^\./", "", regex=True),
        "label": m32["label_derived"].values,
    })
    pos32 = (out32["label"] == "blocked").mean()
    print(f"scenario 32 (derived):  {len(out32)} rows, {out32['seq_uid'].nunique()} seqs, "
          f"pos_rate {pos32:.4f}")
    frames.append(out32)

    all_df = pd.concat(frames, ignore_index=True)
    dest = ROOT / "data" / "dataset_s2s5s32.csv"
    dest.parent.mkdir(exist_ok=True)
    all_df.to_csv(dest, index=False)
    print(f"\nwrote {dest} ({len(all_df)} rows, {all_df['seq_uid'].nunique()} sequences)")
    print(f"scenarios 2+5 (native labels) + 32 (derived labels)")


if __name__ == "__main__":
    main()
