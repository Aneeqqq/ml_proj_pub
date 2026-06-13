"""Build combined dataset: scenarios 2+5 (native labels) + scenarios 32+33 (audited power-derived).

Combines:
  - Scenario 2: native DeepSense labels (visible traffic blockage)
  - Scenario 5: native DeepSense labels (visible traffic blockage)
  - Scenario 32: power-derived i2v labels, audited (10.9% blocked)
  - Scenario 33: power-derived i2v labels, audited (20.2% blocked)

Audit mapping:
  - "visible" (path clear in camera) → not_blocked
  - "not_visible" (obstruction blocks view) → blocked

Output: data/dataset_s2s5s32s33.csv with unified labels & timestamps.
"""

from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]


def std_time_s25(ts: str) -> str:
    """"['02-11-07-142']" (HH-MM-SS-milliseconds) -> "02:11:07-142000" (pipeline format)."""
    h, m, s, ms = ts.strip("[]'").split("-")
    return f"{h}:{m}:{s}-{int(ms) * 1000}"


def process_scenario_2_5():
    """Process scenarios 2 and 5 (native labels).

    labels_unit1.csv has the real `label` but its seq_index/time_stamp are garbage
    (row-ordered by power-file string number). Join `label` onto the MAIN scenario CSV
    (proper seq_index + time_stamp[UTC]) by unique camera image filename.
    """
    dfs = []
    for scn in [2, 5]:
        main_path = ROOT / f"Scenario{scn}" / f"scenario{scn}.csv"
        lab_path = ROOT / f"Scenario{scn}" / "labels_unit1.csv"

        if not main_path.exists() or not lab_path.exists():
            print(f"  WARNING: scenario {scn} CSVs not found, skipping")
            continue

        main_df = pd.read_csv(main_path)
        lab = pd.read_csv(lab_path)
        main_df["img"] = main_df["unit1_rgb"].str.split("/").str[-1]
        lab["img"] = lab["unit1_rgb"].str.split("/").str[-1]
        m = main_df.merge(lab[["img", "label"]], on="img", how="left", validate="one_to_one")
        assert m["label"].notna().all(), f"Scenario{scn}: unmatched labels after image-name join"

        out = pd.DataFrame({
            "index": m["index"],
            "unit1_rgb": f"./Scenario{scn}/" + m["unit1_rgb"].str.replace(r"^\./", "", regex=True),
            "time_stamp": m["time_stamp[UTC]"].map(std_time_s25),
            "seq_index": m["seq_index"],
            "scenario": scn,
            "seq_uid": f"{scn}:" + m["seq_index"].astype(str),
            "label": m["label"],
        })
        blocked = (out["label"] == "blocked").sum()
        print(f"  Scenario {scn}: {len(out)} rows, {out['seq_uid'].nunique()} sequences, "
              f"{blocked} blocked ({100*blocked/len(out):.1f}%)")
        dfs.append(out)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def process_scenario_32():
    """Process scenario 32 with the CORRECTED label (fade = blocked).

    The prior audit (scenario32_audit.json) used the opposite convention and inverted
    the model (test AUC 0.394). The corrected copy scenario32_dev_labelled_CLAUDE.csv
    carries label_derived = power-fade label (fade=blocked), beam-overlay verified to
    be correctly oriented. We use it directly.
    """
    csv_path = ROOT / "scenario32" / "scenario32_dev_labelled_CLAUDE.csv"

    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["label"] = df["label_derived"]   # fade=blocked, verified orientation

    df["scenario"] = 32
    # time_stamp already in HH:MM:SS-microseconds format; keep as-is
    df["seq_uid"] = "32:" + df["seq_index"].astype(str)
    # Add scenario prefix to paths
    df["unit1_rgb"] = "./scenario32/" + df["unit1_rgb"].str.replace(r"^\./", "", regex=True)

    blocked = (df["label"] == "blocked").sum()
    print(f"  Scenario 32: {len(df)} rows, {df['seq_index'].nunique()} sequences, {blocked} blocked ({100*blocked/len(df):.1f}%)")
    return df[["index", "unit1_rgb", "time_stamp", "seq_index", "scenario", "seq_uid", "label"]]


def process_scenario_33():
    """Process scenario 33 (audited power-derived labels).

    Audit mapping: visible → not_blocked, not_visible → blocked
    """
    csv_path = ROOT / "scenario33" / "scenario33_dev_labelled.csv"

    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Load audit results
    audit_path = ROOT / "outputs" / "scenario33_audit.json"
    with open(audit_path) as f:
        audit = json.load(f)

    # Mapping: visible → not_blocked, not_visible → blocked
    def get_label(row_idx):
        audit_result = audit.get(str(int(row_idx)), None)
        if audit_result == "visible":
            return "not_blocked"
        else:  # not_visible or unclear
            return "blocked"

    df["label"] = df.reset_index(drop=True).index.map(lambda i: get_label(i))

    df["scenario"] = 33
    # time_stamp already in HH:MM:SS-microseconds format; keep as-is
    df["seq_uid"] = "33:" + df["seq_index"].astype(str)
    # Add scenario prefix to paths
    df["unit1_rgb"] = "./scenario33/" + df["unit1_rgb"].str.replace(r"^\./", "", regex=True)

    blocked = (df["label"] == "blocked").sum()
    print(f"  Scenario 33: {len(df)} rows, {df['seq_index'].nunique()} sequences, {blocked} blocked ({100*blocked/len(df):.1f}%)")
    return df[["index", "unit1_rgb", "time_stamp", "seq_index", "scenario", "seq_uid", "label"]]


def main():
    print("Building combined dataset s2s5s32s33...")

    print("\nProcessing scenarios 2+5 (native labels):")
    df_s25 = process_scenario_2_5()

    print("\nProcessing scenario 32 (audited):")
    df_s32 = process_scenario_32()

    print("\nProcessing scenario 33 (audited):")
    df_s33 = process_scenario_33()

    # Combine all
    dfs = [df_s25, df_s32, df_s33]
    dfs = [d for d in dfs if not d.empty]

    if not dfs:
        print("ERROR: No data loaded!")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Statistics
    print(f"\n{'-'*60}")
    print(f"Combined dataset: {len(combined)} rows, {combined['scenario'].nunique()} scenarios")
    for scn in sorted(combined['scenario'].unique()):
        scn_data = combined[combined['scenario'] == scn]
        blocked = (scn_data['label'] == 'blocked').sum()
        print(f"  Scenario {scn}: {len(scn_data)} rows, {blocked} blocked ({100*blocked/len(scn_data):.1f}%)")

    blocked_total = (combined['label'] == 'blocked').sum()
    print(f"\nOverall blockage rate: {100*blocked_total/len(combined):.1f}% ({blocked_total}/{len(combined)})")

    # Save
    out_csv = ROOT / "data" / "dataset_s2s5s32s33.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
