from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "final" / "24_grid_attention_pooling" / "multiseed_runs.csv"
OUT = ROOT / "results" / "final" / "39_architecture"

def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, ps[i] * (len(ps) - rank)))
        out[i] = run
    return out

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(RUNS)
    f1 = r.pivot(index="Seed", columns="Model", values="F1")
    params = r.groupby("Model").Params.first()

    rows = []
    for level, prefix in (("character", "Char"), ("word", "Word")):
        models = sorted([m for m in f1.columns if m.startswith(prefix)],
                        key=lambda m: -f1[m].mean())
        pairs = list(itertools.combinations(models, 2))
        raw = [float(stats.ttest_rel(f1[a], f1[b]).pvalue) for a, b in pairs]
        for (a, b), p_raw, p_adj in zip(pairs, raw, holm(raw)):
            d = (f1[a] - f1[b]).values * 100
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            rows.append({
                "level": level, "better": a, "worse": b,
                "f1_better": round(float(f1[a].mean()), 4),
                "f1_worse": round(float(f1[b].mean()), 4),
                "diff_pp": round(float(d.mean()), 3),
                "ci95_lo_pp": round(float(lo), 3), "ci95_hi_pp": round(float(hi), 3),
                "cohens_dz": round(float(d.mean() / d.std(ddof=1)), 2),
                "p_raw": p_raw, "p_holm": round(p_adj, 4),
                "separates": bool(p_adj < 0.05),
                "params_better": int(params[a]), "params_worse": int(params[b])})

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "within_level_pairs.csv", index=False)

    spread = []
    for level, prefix in (("character", "Char"), ("word", "Word")):
        ms = [m for m in f1.columns if m.startswith(prefix)]
        best, worst = f1[ms].mean().max(), f1[ms].mean().min()
        sub = d[d.level == level]
        spread.append({"level": level,
                       "best_f1": round(float(best), 4), "worst_f1": round(float(worst), 4),
                       "spread_pp": round(float((best - worst) * 100), 3),
                       "comparisons": len(sub),
                       "separating_after_holm": int(sub.separates.sum()),
                       "top_model": f1[ms].mean().idxmax()})
    s = pd.DataFrame(spread)
    s.to_csv(OUT / "within_level_spread.csv", index=False)

    bench = ROOT / "results" / "final" / "11_inference_benchmark" / "tables" / "inference_benchmark.csv"
    if bench.exists():
        b = pd.read_csv(bench)
        base = b.loc[b.Params.idxmin()]
        b["params_over_smallest"] = (b.Params / base.Params).round(0).astype("Int64")
        b["disk_over_smallest"] = (b.Disk_MB / base.Disk_MB).round(0).astype("Int64")
        b["smallest_model"] = base.Model
        b[["Model", "Family", "Params", "Disk_MB", "params_over_smallest",
           "disk_over_smallest", "smallest_model"]].to_csv(
            OUT / "size_ratios.csv", index=False)
        print(b[["Model", "Params", "params_over_smallest",
                 "disk_over_smallest"]].to_string(index=False))
        print()

    print(d[["level", "better", "worse", "diff_pp", "ci95_lo_pp", "ci95_hi_pp",
             "p_holm", "separates"]].to_string(index=False))
    print()
    print(s.to_string(index=False))
    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
