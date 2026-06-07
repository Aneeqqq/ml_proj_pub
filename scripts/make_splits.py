"""Generate the stratified, sequence-level split and save it.

Run:  .venv/Scripts/python.exe -m scripts.make_splits
Outputs:
  splits/seq_assignment.csv   (seq_index, split)
  prints per-split stats (sequences, positive sequences, samples, positive rate)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.splits import split_from_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    csv = ROOT / cfg["paths"]["csv"]
    df = pd.read_csv(csv)

    res = split_from_config(df, cfg)

    print(f"=== per-split stats (protocol: {cfg['split'].get('protocol','pooled')}) ===")
    print(res.stats.to_string(index=False))

    # warn if any split has no positive sequences
    bad = res.stats[(res.stats["n_pos_seqs"] == 0)]
    if len(bad):
        print("\n[WARN] splits with ZERO positive sequences:", bad["split"].tolist())
    else:
        print("\n[OK] every split has >=1 positive sequence")

    out = ROOT / "splits" / "seq_assignment.csv"
    out.parent.mkdir(exist_ok=True)
    rows = [{"seq_uid": s, "split": sp} for s, sp in sorted(res.assignment.items())]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} sequences)")


if __name__ == "__main__":
    main()
