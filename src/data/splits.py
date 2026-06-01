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

    assignment: dict[int, str] = {}
    seqs: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for split in seqs:
        for s in alloc_pos[split] + alloc_neg[split]:
            assignment[int(s)] = split
            seqs[split].append(int(s))

    # ---- sanity: disjoint & complete ----
    all_assigned = [s for v in seqs.values() for s in v]
    assert len(all_assigned) == len(set(all_assigned)), "a sequence landed in >1 split"
    assert set(all_assigned) == set(int(s) for s in per_seq[seq_col]), "missing sequences"

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
