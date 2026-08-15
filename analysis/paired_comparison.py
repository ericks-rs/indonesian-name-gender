"""Paired character-versus-word comparison, the table the manuscript leads with.

Reviewer B asked for standard deviations, confidence intervals, effect sizes and
significance testing. Per-model means alone do not answer that for a paired
design, so this reports the paired difference itself: its mean, a 95 percent
interval on that mean, Cohen's d_z, and a Holm-adjusted p-value across the four
architecture pairs.

Seeds are matched, so every difference is computed within a seed before being
averaged. Values are in percentage points of F1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
OUT = FINAL / "00_summary"

PAIRS = [("CharBiRNN", "WordBiRNN", "BiRNN"),
         ("CharBiLSTM", "WordBiLSTM", "BiLSTM"),
         ("CharBiGRU", "WordBiGRU", "BiGRU"),
         ("CharTransformer", "WordTransformer", "Transformer")]
T_CRIT = 2.776  # t(0.975, df = 4)


def load_runs() -> pd.DataFrame:
    """The prediction-keeping grid run, and only that one.

    An earlier fallback silently reached for an archived run when the current
    file was missing, which is how two vintages ended up feeding one table.
    Failing loudly is the safer behaviour.
    """
    p = FINAL / "24_grid_attention_pooling" / "multiseed_runs.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}, run pipeline/seeds_grid_with_predictions.py first")
    print(f"source: {p.relative_to(ROOT)}")
    return pd.read_csv(p)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    out = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = min(1.0, max(running, pvals[i] * (len(pvals) - rank)))
        out[i] = running
    return out


def row(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    d = (a - b) * 100
    mean, sd = d.mean(), d.std(ddof=1)
    half = T_CRIT * sd / np.sqrt(len(d))
    t, p = stats.ttest_rel(a, b)
    return {"comparison": label, "char_f1": round(a.mean(), 4), "word_f1": round(b.mean(), 4),
            "diff_pp": round(mean, 3), "ci95_lo_pp": round(mean - half, 3),
            "ci95_hi_pp": round(mean + half, 3), "sd_pp": round(sd, 3),
            "cohens_dz": round(mean / sd, 2), "t": round(float(t), 2), "p_raw": float(p)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    per = {m: x.set_index("Seed").F1 for m, x in runs.groupby("Model")}
    seeds = sorted(set.intersection(*(set(s.index) for s in per.values())))
    print(f"matched seeds: {seeds}")

    rows = [row(per[c].loc[seeds].values, per[w].loc[seeds].values, lab) for c, w, lab in PAIRS]
    for r, adj in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = adj

    # No best-to-best row. Which character cell sits on top is separated by
    # 0.044 points with an interval spanning zero, so naming one after the fact
    # would be selection dressed as a result. The whole family is compared in
    # char_family_vs_baselines.py instead.

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "char_vs_word_paired.csv", index=False)

    show = df[["comparison", "char_f1", "word_f1", "diff_pp", "ci95_lo_pp",
               "ci95_hi_pp", "cohens_dz", "p_holm"]]
    print("\n" + show.to_string(index=False))
    print(f"\nlowest confidence bound across the four pairs: "
          f"{min(r['ci95_lo_pp'] for r in rows):+.2f} pp")
    print(f"Written to {OUT / 'char_vs_word_paired.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
