"""F1 per registration year (2024 vs 2025) from the final grid, five seeds.

Used for Fig. 15(a). Predictions are read from
`results/final/24_grid_attention_pooling/val_predictions.csv` (anonymized,
ordered by `row_id`). The first registration year is not released, so the
`FIRST_YEAR` column is read from `data/splits/val_2024_2026.csv`, which, like
the training scripts, is not included in this repository. Row i of that split
is `row_id` i.

Output in results/final/43_per_year_f1_grid/ (aggregates, no names):
- per_seed.csv                F1 per model, seed, and year
- per_year_f1.csv             mean over five seeds per model and year
- delta_2025_minus_2024.csv   difference per model, mean and range across seeds
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
GRID = ROOT / "results" / "final" / "24_grid_attention_pooling"
OUT = ROOT / "results" / "final" / "43_per_year_f1_grid"
SEEDS = [42, 7, 123, 2024, 777]
MODELS = ["CharBiRNN", "CharBiLSTM", "CharBiGRU", "CharTransformer",
          "WordBiRNN", "WordBiLSTM", "WordBiGRU", "WordTransformer"]


def f1_exact(y, p) -> Fraction:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return Fraction(2 * tp, 2 * tp + fp + fn)


def dec(fr: Fraction, places: str) -> Decimal:
    return (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal(places), ROUND_HALF_UP)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(GRID / "val_predictions.csv").sort_values("row_id")
    val = pd.read_csv(ROOT / "data" / "splits" / "val_2024_2026.csv")
    assert len(val) == len(pred) and (pred.row_id.values == range(1, len(val) + 1)).all()
    assert ((val.LABEL == "P").astype(int).values == pred.label.values).all()
    y, year = pred.label.values, val.FIRST_YEAR.values

    rows = []
    for m in MODELS:
        for s in SEEDS:
            p = pred[f"{m}__seed{s}"].values
            for yr in (2024, 2025):
                k = year == yr
                rows.append({"Model": m, "Seed": s, "Year": yr, "N": int(k.sum()),
                             "F1_exact": f1_exact(y[k], p[k])})
    ps = pd.DataFrame(rows)
    out_seed = ps.assign(F1=[float(v) for v in ps.F1_exact]).drop(columns="F1_exact")
    out_seed.round(6).to_csv(OUT / "per_seed.csv", index=False)

    mean_rows, delta_rows = [], []
    for m in MODELS:
        g = ps[ps.Model == m]
        means = {}
        for yr in (2024, 2025):
            v = g[g.Year == yr].F1_exact
            means[yr] = sum(v, Fraction(0)) / len(v)
            mean_rows.append({"Year": yr, "Model": m, "N": int(g[g.Year == yr].N.iloc[0]),
                              "F1": float(means[yr]), "F1_4dp": str(dec(means[yr], "0.0001"))})
        per_seed_d = [(g[(g.Seed == s) & (g.Year == 2025)].F1_exact.iloc[0]
                       - g[(g.Seed == s) & (g.Year == 2024)].F1_exact.iloc[0]) * 100 for s in SEEDS]
        d = means[2025] - means[2024]
        delta_rows.append({"Model": m, "delta_pp": str(dec(d * 100, "0.001")),
                           "min_seed_pp": str(dec(min(per_seed_d), "0.001")),
                           "max_seed_pp": str(dec(max(per_seed_d), "0.001")),
                           "seeds_higher_2025": sum(x > 0 for x in per_seed_d)})
    pd.DataFrame(mean_rows).to_csv(OUT / "per_year_f1.csv", index=False)
    dd = pd.DataFrame(delta_rows)
    dd.to_csv(OUT / "delta_2025_minus_2024.csv", index=False)
    print(dd.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
