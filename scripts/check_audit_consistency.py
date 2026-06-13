"""Cross-reference audit decisions vs independent power-derived fade label for s32 & s33.

If the two audits used the same convention, the audit<->power agreement should be
similar across scenarios. A flipped convention shows up as anti-correlation.
"""
from pathlib import Path
import json
import pandas as pd
from src.data.derive_label import derive_blockage_label

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(env_win=41, q=0.90, thr_db=-3.0, min_dur=3, deep_db=-4.5)

for scn in [32, 33]:
    dev = pd.read_csv(ROOT / f"scenario{scn}" / f"scenario{scn}_dev.csv")
    power = derive_blockage_label(dev, **PARAMS).reset_index(drop=True)  # 'blocked'/'not_blocked'
    power_blocked = (power == "blocked").to_numpy()

    audit = json.load(open(ROOT / "outputs" / f"scenario{scn}_audit.json"))
    # audit keyed by 0..N-1 row position
    aud = pd.Series([audit.get(str(i), None) for i in range(len(dev))])
    vis = (aud == "visible").to_numpy()
    nvis = (aud == "not_visible").to_numpy()

    # Crosstab: power fade vs audit decision
    pf_vis = (power_blocked & vis).sum()       # power says fade, audit says "visible"
    pf_nvis = (power_blocked & nvis).sum()      # power says fade, audit says "not_visible"
    clear_vis = (~power_blocked & vis).sum()
    clear_nvis = (~power_blocked & nvis).sum()

    print(f"\n=== Scenario {scn} ===")
    print(f"  power-fade frames: {power_blocked.sum()}  ({100*power_blocked.mean():.1f}%)")
    print(f"  audit visible={vis.sum()}  not_visible={nvis.sum()}")
    print(f"  crosstab (rows=power, cols=audit):")
    print(f"               audit:visible   audit:not_visible")
    print(f"    power FADE:   {pf_vis:6d}          {pf_nvis:6d}")
    print(f"    power CLEAR:  {clear_vis:6d}          {clear_nvis:6d}")
    # Of the power fades, what fraction did the audit call 'not_visible' (=blocked under 33 convention)?
    if power_blocked.sum():
        frac = pf_nvis / power_blocked.sum()
        print(f"  -> of power-fades, {100*frac:.1f}% audited 'not_visible'")
