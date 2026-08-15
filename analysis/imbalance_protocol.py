"""Paired comparison of the imbalance strategies against class weighting.

Reads the 120 runs from `program/experiment_imbalance_protocol.py` and pairs each
one against the class-weighted baseline from the main grid, matched on model and
seed. The baseline is reused rather than re-run because both came from the same
training code, so the pairing is exact rather than approximate.

Two granularities come out, because they answer different questions and neither
subsumes the other.

Per model and strategy, twenty-four tests over five seeds each. This is the
convention the rest of the paper uses, the same one behind
`char_vs_word_paired.csv`, and Holm runs across all twenty-four.

Per strategy, three tests pooling the forty paired differences. Pairs are not
independent across models here, so this is the weaker of the two and is reported
as the summary line rather than the evidence.

Nothing in this file chooses a strategy. The manuscript reports the class-weighted
objective because it was fixed before any of this ran.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "tables" / "imbalance_protocol" / "imbalance_runs.csv"
BASE = ROOT / "results" / "final" / "24_grid_attention_pooling" / "multiseed_runs.csv"
OUT = ROOT / "results" / "final" / "40_imbalance_protocol"
SEEDS = [42, 7, 123, 2024, 777]
T_CRIT = 2.776          # t(0.975, df = 4)
MODELS = ["CharBiRNN", "CharBiLSTM", "CharBiGRU", "CharTransformer",
          "WordBiRNN", "WordBiLSTM", "WordBiGRU", "WordTransformer"]


def holm(p):
    """Holm step-down, returned in the order the p-values came in."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(diff):
    """Mean, sd, interval and paired t against zero, on differences in points."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t, p = stats.ttest_rel(d, np.zeros(n)) if n > 1 else (np.nan, np.nan)
    return {"n": n, "mean_pp": m, "sd_pp": sd,
            "ci95_lo_pp": m - T_CRIT * se, "ci95_hi_pp": m + T_CRIT * se,
            "t": t, "p": p, "cohens_dz": m / sd if sd else np.nan}


def main() -> int:
    runs = pd.read_csv(RUNS)
    base = (pd.read_csv(BASE)[["Model", "Seed", "F1"]]
            .rename(columns={"F1": "F1_weighted"}))
    # three strategies share one baseline row per model and seed, and the
    # validation still matters because a duplicated baseline row would silently
    # double every difference
    d = runs.merge(base, on=["Model", "Seed"], how="left", validate="many_to_one")
    if d.F1_weighted.isna().any():
        missing = d[d.F1_weighted.isna()][["Strategy", "Model", "Seed"]]
        print("no baseline for:\n", missing.to_string(index=False))
        return 1
    d["diff_pp"] = (d.F1 - d.F1_weighted) * 100
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "paired_runs.csv", index=False)

    strategies = [s for s in ["unweighted", "oversampling", "balanced"]
                  if s in set(d.Strategy)]
    expected = len(strategies) * len(MODELS) * len(SEEDS)
    print(f"{len(d)} of {expected} paired runs, {len(strategies)} strateg"
          f"{'y' if len(strategies) == 1 else 'ies'}")

    # per model and strategy, the convention the rest of the paper uses
    rows = []
    for s in strategies:
        for m in MODELS:
            g = d[(d.Strategy == s) & (d.Model == m)]
            if len(g) < 2:
                continue
            rows.append({"Strategy": s, "Model": m,
                         "F1_weighted": g.F1_weighted.mean(),
                         "F1_strategy": g.F1.mean(), **paired(g.diff_pp.values)})
    per = pd.DataFrame(rows)
    if not per.empty:
        per["holm_p"] = holm(per.p.values)
        per["favours_strategy"] = per.ci95_lo_pp > 0
        per["favours_weighting"] = per.ci95_hi_pp < 0
        per.round(6).to_csv(OUT / "per_model_paired.csv", index=False)

    # per strategy, pooling the forty differences. Weaker, and labelled so.
    rows = []
    for s in strategies:
        g = d[d.Strategy == s]
        rows.append({"Strategy": s, "models": g.Model.nunique(),
                     "worst_model_pp": g.groupby("Model").diff_pp.mean().min(),
                     "best_model_pp": g.groupby("Model").diff_pp.mean().max(),
                     **paired(g.diff_pp.values)})
    pooled = pd.DataFrame(rows)
    if not pooled.empty:
        pooled["holm_p"] = holm(pooled.p.values)
        pooled.round(6).to_csv(OUT / "per_strategy_pooled.csv", index=False)

    pd.DataFrame([{
        "runs": len(d), "strategies": len(strategies), "models": d.Model.nunique(),
        "seeds": d.Seed.nunique(),
        "any_strategy_beats_weighting": bool(per.favours_strategy.any()) if not per.empty else False,
        "cells_favouring_weighting": int(per.favours_weighting.sum()) if not per.empty else 0,
        "cells_surviving_holm": int((per.holm_p < 0.05).sum()) if not per.empty else 0,
        "largest_gain_pp": float(per.mean_pp.max()) if not per.empty else np.nan,
        "largest_loss_pp": float(per.mean_pp.min()) if not per.empty else np.nan,
    }]).round(6).to_csv(OUT / "headline.csv", index=False)

    if not pooled.empty:
        print("\npooled over the eight models")
        print(pooled[["Strategy", "n", "mean_pp", "ci95_lo_pp", "ci95_hi_pp",
                      "holm_p"]].round(4).to_string(index=False))
    if not per.empty:
        print(f"\n{int((per.holm_p < 0.05).sum())} of {len(per)} model and strategy "
              f"cells survive Holm")
        print(per[["Strategy", "Model", "mean_pp", "ci95_lo_pp", "ci95_hi_pp",
                   "holm_p"]].round(4).to_string(index=False))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
