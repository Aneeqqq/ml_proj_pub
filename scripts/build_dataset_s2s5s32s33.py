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


def std_time(ts: str) -> str:
    """Standardize timestamp format to HH:MM:SS-us."""
    if ts.startswith("["):  # Scenario 2/5 format: ['HH-MM-SS-ms']
        h, m, s, ms = ts.strip("[]'").split("-")
        return f"{h}:{m}:{s}-{int(ms) * 1000}"
    else:  # Scenario 32/33 format: HH:MM:SS-us
        return ts


def process_scenario_2_5():
    """Process scenarios 2 and 5 (native labels from labels_unit1.csv)."""
    dfs = []
    for scn in [2, 5]:
        csv_path = ROOT / f"Scenario{scn}" / "labels_unit1.csv"

        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        # Ensure unit1_rgb column exists
        if "unit1_rgb" not in df.columns:
            if "image_name" in df.columns:
                df["unit1_rgb"] = "./unit1/camera_data/" + df["image_name"]
            else:
                df["unit1_rgb"] = "./unit1/camera_data/image_" + df["index"].astype(str) + ".jpg"

        df["scenario"] = scn
        # time_stamp is already in format (numeric or HH:MM:SS-us), keep as-is
        df["seq_uid"] = df["seq_index"].astype(str)

        # Use native label as-is
        if "label" not in df.columns and "blocked" in df.columns:
            df["label"] = df["blocked"].map({1: "blocked", 0: "not_blocked"})

        print(f"  Scenario {scn}: {len(df)} rows, {df['seq_index'].nunique()} sequences")
        dfs.append(df[["index", "unit1_rgb", "time_stamp", "seq_index", "scenario", "seq_uid", "label"]])

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def process_scenario_32():
    """Process scenario 32 (audited power-derived labels).

    Audit mapping: visible → not_blocked, not_visible → blocked
    """
    csv_path = ROOT / "scenario32" / "scenario32_dev_labelled.csv"

    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Load audit results
    audit_path = ROOT / "outputs" / "scenario32_audit.json"
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

    df["scenario"] = 32
    df["time_stamp"] = df["time_stamp"].apply(std_time)
    df["seq_uid"] = df["seq_index"].astype(str)

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
    df["time_stamp"] = df["time_stamp"].apply(std_time)
    df["seq_uid"] = df["seq_index"].astype(str)

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
