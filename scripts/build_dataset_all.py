"""Build the FULL combined camera dataset: scenarios 2, 5, 31, 32, 33, 34.

Labels (consistent orientation: blocked = object in unit1->unit2 LOS):
  - 2, 5  : native DeepSense labels (join onto main CSV by image filename)
  - 31-34 : experiment power-fade label (fade=blocked), from scenarioNN_dev_labelled_CLAUDE.csv
            (beam-overlay verified orientation; see scripts/exp_label_scenario.py)

Caveats handled:
  * seq_index integrity: 2/5 joined onto the MAIN scenario CSV (real seq_index +
    time_stamp[UTC]); 31-34 already have real seq_index. seq_uid is scenario-prefixed
    so scene ids never collide across scenarios.
  * label-orientation: 31-34 use the corrected fade=blocked (not the inverted s32 audit).
  * per-scenario domain shift / imbalance is recorded in the printout and carried via the
    `scenario` column so splits/eval can be done per-scenario.

Output: data/dataset_all.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 30s scenarios: folder containing unit1/, and dev csv name, experiment-label csv name
SCN30 = {
    31: (ROOT / "scenario31_new" / "scenario31", "scenario31_dev_labelled_CLAUDE.csv", "scenario31_new/scenario31"),
    32: (ROOT / "scenario32", "scenario32_dev_labelled_CLAUDE.csv", "scenario32"),
    33: (ROOT / "scenario33", "scenario33_dev_labelled_CLAUDE.csv", "scenario33"),
    34: (ROOT / "scenario34_new" / "scenario34", "scenario34_dev_labelled_CLAUDE.csv", "scenario34_new/scenario34"),
}


def std_time_s25(ts: str) -> str:
    """"['02-11-07-142']" (HH-MM-SS-ms) -> "02:11:07-142000"."""
    h, m, s, ms = ts.strip("[]'").split("-")
    return f"{h}:{m}:{s}-{int(ms) * 1000}"


def proc_native(scn):
    main = pd.read_csv(ROOT / f"Scenario{scn}" / f"scenario{scn}.csv")
    lab = pd.read_csv(ROOT / f"Scenario{scn}" / "labels_unit1.csv")
    main["img"] = main["unit1_rgb"].str.split("/").str[-1]
    lab["img"] = lab["unit1_rgb"].str.split("/").str[-1]
    m = main.merge(lab[["img", "label"]], on="img", how="left", validate="one_to_one")
    assert m["label"].notna().all(), f"Scenario{scn}: unmatched labels"
    out = pd.DataFrame({
        "index": m["index"],
        "unit1_rgb": f"./Scenario{scn}/" + m["unit1_rgb"].str.replace(r"^\./", "", regex=True),
        "time_stamp": m["time_stamp[UTC]"].map(std_time_s25),
        "seq_index": m["seq_index"],
        "scenario": scn,
        "seq_uid": f"{scn}:" + m["seq_index"].astype(str),
        "label": m["label"],
    })
    return out


def proc_exp(scn):
    root, csv, prefix = SCN30[scn]
    df = pd.read_csv(root / csv)
    out = pd.DataFrame({
        "index": df["index"],
        "unit1_rgb": f"./{prefix}/" + df["unit1_rgb"].str.replace(r"^\./", "", regex=True),
        "time_stamp": df["time_stamp"].astype(str),   # already HH:MM:SS-us
        "seq_index": df["seq_index"],
        "scenario": scn,
        "seq_uid": f"{scn}:" + df["seq_index"].astype(str),
        "label": df["label_derived"],
    })
    return out


def main():
    print("Building FULL dataset: scenarios 2, 5, 31, 32, 33, 34")
    frames = []
    print("\nNative (2, 5):")
    for s in (2, 5):
        o = proc_native(s); frames.append(o)
        b = (o.label == "blocked").sum()
        print(f"  s{s}: {len(o)} rows, {o.seq_uid.nunique()} seqs, {b} blocked ({100*b/len(o):.1f}%)")
    print("\nExperiment fade=blocked (31, 32, 33, 34):")
    for s in (31, 32, 33, 34):
        o = proc_exp(s); frames.append(o)
        b = (o.label == "blocked").sum()
        print(f"  s{s}: {len(o)} rows, {o.seq_uid.nunique()} seqs, {b} blocked ({100*b/len(o):.1f}%)")

    alldf = pd.concat(frames, ignore_index=True)
    b = (alldf.label == "blocked").sum()
    print(f"\n{'-'*60}")
    print(f"TOTAL: {len(alldf)} rows, {alldf.seq_uid.nunique()} seqs, "
          f"{alldf.scenario.nunique()} scenarios, {b} blocked ({100*b/len(alldf):.1f}%)")
    dest = ROOT / "data" / "dataset_all.csv"
    dest.parent.mkdir(exist_ok=True)
    alldf.to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
