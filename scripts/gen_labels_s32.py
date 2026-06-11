"""Generate power-derived labels for scenario 32 from audit results."""

from pathlib import Path
import pandas as pd
import json
from src.data.derive_label import add_derived_label

ROOT = Path(__file__).resolve().parents[1]
s32_csv = ROOT / "scenario32" / "scenario32_dev.csv"
audit_path = ROOT / "outputs" / "scenario32_audit.json"

# First, derive labels using LOS envelope
params = dict(env_win=41, q=0.90, thr_db=-3.0, min_dur=3, deep_db=-4.5)
print(f"Reading {s32_csv}")
df = pd.read_csv(s32_csv)
print(f"  {len(df)} rows, {df['seq_index'].nunique()} sequences")

print(f"Deriving blockage labels with {params}")
df = add_derived_label(df, **params)

# Now apply audit mapping: visible → not_blocked, not_visible → blocked
with open(audit_path) as f:
    audit = json.load(f)

def get_label(row_idx):
    audit_result = audit.get(str(int(row_idx)), None)
    if audit_result == "visible":
        return "not_blocked"
    else:
        return "blocked"

df["label_derived"] = df.reset_index(drop=True).index.map(lambda i: get_label(i))

# Save with _labelled suffix
out_csv = s32_csv.with_stem("scenario32_dev_labelled")
df.to_csv(out_csv, index=False)

# Print stats
blocked = (df["label_derived"] == "blocked").sum()
print(f"\nAudited label rate: {blocked}/{len(df)} blocked ({100*blocked/len(df):.1f}%)")
print(f"Wrote {out_csv}")
