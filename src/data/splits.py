"""Stratified, sequence-level train/val/test splitting for Scenario 31.

Why this exists
---------------
The data is organised into *scenes* (`seq_index`). Splitting at the row/window level
leaks the same drive into train+test and inflates F1. So we assign **whole sequences**
to one split. Additionally, blockage positives are scarce and lumpy — only ~15 of 52
sequences contain any blockage — so a naive random split can leave val/test with zero
positives. We therefore **stratify on blockage presence**: positive and negative
sequences are allocated to splits separately, each by the requested ratios.

See ML_Proj_Vault/dataset/sequences-and-batching.md and blockage-label.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pandas as pd


@dataclass
class SplitResult:
    assignment: dict[int, str]                 # seq_index -> "train"|"val"|"test"
    seqs: dict[str, list[int]] = field(default_factory=dict)   # split -> [seq_index]
    stats: pd.DataFrame = None                  # per-split summary

    def split_of(self, seq_index: int) -> str:
        return self.assignment[int(seq_index)]


def _allocate(seq_ids: list[int], ratios: dict[str, float]) -> dict[str, list[int]]:
    """Allocate an (already-shuffled) list of sequence ids across splits by ratio.

    Uses largest-remainder rounding so counts sum exactly to len(seq_ids), and biases
    leftover sequences toward val/test first so the small/positive pool is not starved.
    """
    n = len(seq_ids)
    order = ["train", "val", "test"]
    raw = {k: ratios.get(k, 0.0) * n for k in order}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    assigned = sum(base.values())
    remainder = n - assigned
    # distribute the remaining sequences by largest fractional part, val/test first on ties
    frac_order = sorted(order, key=lambda k: (raw[k] - base[k], k in ("val", "test")), reverse=True)
    for i in range(remainder):
        base[frac_order[i % len(frac_order)]] += 1

    out: dict[str, list[int]] = {k: [] for k in order}
    cursor = 0
    for k in order:
        out[k] = seq_ids[cursor: cursor + base[k]]
        cursor += base[k]
    return out


def stratified_sequence_split(
    df: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    label_col: str = "label",
    positive: str = "blocked",
    seq_col: str = "seq_index",
) -> SplitResult:
    """Return a sequence-level, blockage-stratified split.

    Guarantees:
      * every seq_index lands in exactly one split (no leakage);
      * positive sequences are distributed across splits by the same ratios, so val/test
        receive blockage episodes whenever the pool allows.
    """
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}

    per_seq = (
        df.assign(_pos=(df[label_col] == positive).astype(int))
        .groupby(seq_col)
        .agg(n=("_pos", "size"), n_pos=("_pos", "sum"))
        .reset_index()
    )
    per_seq["has_block"] = per_seq["n_pos"] > 0

    pos_seqs = per_seq.loc[per_seq["has_block"], seq_col].tolist()
    neg_seqs = per_seq.loc[~per_seq["has_block"], seq_col].tolist()

    # deterministic shuffle of each pool
    rng = pd.Series(pos_seqs).sample(frac=1.0, random_state=seed).tolist()
    rng_neg = pd.Series(neg_seqs).sample(frac=1.0, random_state=seed + 1).tolist()

    alloc_pos = _allocate(rng, ratios)
    alloc_neg = _allocate(rng_neg, ratios)

    assignment: dict = {}
    seqs: dict[str, list] = {"train": [], "val": [], "test": []}
    for split in seqs:
        for s in alloc_pos[split] + alloc_neg[split]:
            assignment[s] = split            # seq id may be str (seq_uid) or int (seq_index)
            seqs[split].append(s)

    # ---- sanity: disjoint & complete ----
    all_assigned = [s for v in seqs.values() for s in v]
    assert len(all_assigned) == len(set(all_assigned)), "a sequence landed in >1 split"
    assert set(all_assigned) == set(per_seq[seq_col]), "missing sequences"

    # ---- per-split stats ----
    rows = []
    for split, ss in seqs.items():
        sub = per_seq[per_seq[seq_col].isin(ss)]
        rows.append(
            dict(
                split=split,
                n_seqs=len(ss),
                n_pos_seqs=int(sub["has_block"].sum()),
                n_samples=int(sub["n"].sum()),
                n_pos_samples=int(sub["n_pos"].sum()),
                pos_rate=round(sub["n_pos"].sum() / max(sub["n"].sum(), 1), 4),
            )
        )
    stats = pd.DataFrame(rows)

    return SplitResult(assignment=assignment, seqs=seqs, stats=stats)


def _stats(df, seqs, seq_col, label_col, positive):
    per = (df.assign(_pos=(df[label_col] == positive).astype(int))
           .groupby(seq_col).agg(n=("_pos", "size"), n_pos=("_pos", "sum")).reset_index())
    per["has_block"] = per["n_pos"] > 0
    rows = []
    for split, ss in seqs.items():
        sub = per[per[seq_col].isin(ss)]
        rows.append(dict(split=split, n_seqs=len(ss), n_pos_seqs=int(sub["has_block"].sum()),
                         n_samples=int(sub["n"].sum()), n_pos_samples=int(sub["n_pos"].sum()),
                         pos_rate=round(sub["n_pos"].sum() / max(sub["n"].sum(), 1), 4)))
    return pd.DataFrame(rows)


def cross_scenario_split(
    df: pd.DataFrame,
    test_scenarios: list[str],
    val_frac: float = 0.18,
    seed: int = 42,
    label_col: str = "label_derived",
    positive: str = "blocked",
    seq_col: str = "seq_uid",
    scen_col: str = "scenario",
) -> SplitResult:
    """Held-out-scenario protocol: TEST = all sequences of `test_scenarios`; the remaining
    scenarios' sequences are split into train/val (stratified on blockage presence, by val_frac).
    """
    test_scn = {str(s) for s in test_scenarios}
    df = df.copy()
    df[scen_col] = df[scen_col].astype(str)

    test_seqs = df.loc[df[scen_col].isin(test_scn), seq_col].unique().tolist()
    tv = df[~df[scen_col].isin(test_scn)]

    per = (tv.assign(_pos=(tv[label_col] == positive).astype(int))
           .groupby(seq_col).agg(n_pos=("_pos", "sum")).reset_index())
    pos_seqs = per.loc[per.n_pos > 0, seq_col].tolist()
    neg_seqs = per.loc[per.n_pos == 0, seq_col].tolist()
    rng_p = pd.Series(pos_seqs).sample(frac=1.0, random_state=seed).tolist()
    rng_n = pd.Series(neg_seqs).sample(frac=1.0, random_state=seed + 1).tolist()

    def take_val(lst):
        k = round(len(lst) * val_frac)
        return lst[:k], lst[k:]          # val, train

    vp, tp = take_val(rng_p)
    vn, tn = take_val(rng_n)
    seqs = {"train": tp + tn, "val": vp + vn, "test": [str(s) for s in test_seqs]}

    assignment = {}
    for split, ss in seqs.items():
        for s in ss:
            assignment[s] = split
    all_assigned = [s for v in seqs.values() for s in v]
    assert len(all_assigned) == len(set(all_assigned)), "a sequence landed in >1 split"

    stats = _stats(df, seqs, seq_col, label_col, positive)
    return SplitResult(assignment=assignment, seqs=seqs, stats=stats)


def split_from_config(df: pd.DataFrame, cfg: dict) -> SplitResult:
    """Dispatch to the configured split protocol (cross_scenario | pooled)."""
    sp, lc = cfg["split"], cfg["label"]
    seqc = cfg.get("seq_col", "seq_index")
    if sp.get("protocol") == "cross_scenario":
        return cross_scenario_split(df, sp["test_scenarios"], sp.get("val_frac", 0.18), sp["seed"],
                                    label_col=lc["column"], positive=lc["positive"], seq_col=seqc)
    return stratified_sequence_split(df, sp["ratios"], sp["seed"], label_col=lc["column"],
                                     positive=lc["positive"], seq_col=seqc)
