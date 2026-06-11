"""Generate power-derived labels for scenario 33, using same LOS envelope method as s32."""

from pathlib import Path
import pandas as pd
from src.data.derive_label import add_derived_label

ROOT = Path(__file__).resolve().parents[1]
s33_csv = ROOT / "scenario33" / "scenario33_dev.csv"

# Same parameters used for scenario 32
params = dict(env_win=41, q=0.90, thr_db=-3.0, min_dur=3, deep_db=-4.5)

print(f"Reading {s33_csv}")
df = pd.read_csv(s33_csv)
print(f"  {len(df)} rows, {df['seq_index'].nunique()} sequences")

print(f"Deriving blockage labels with {params}")
df = add_derived_label(df, **params)

# Save with _labelled suffix like scenario 32
out_csv = s33_csv.with_stem("scenario33_dev_labelled")
df.to_csv(out_csv, index=False)

# Print stats
y = (df["label_derived"] == "blocked")
print(f"\nDerived positive rate: {y.mean():.3f} ({int(y.sum())}/{len(df)})")
nseq = df[y].groupby("seq_index").size()
print(f"Sequences with >=1 blockage: {nseq.size}/{df['seq_index'].nunique()}")
print(f"Wrote {out_csv}")
