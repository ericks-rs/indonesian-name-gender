"""Comparison that does not depend on where the threshold sits.

Every number the manuscript reports, F1, accuracy, precision and recall, is taken
at a fixed cut of 0.5. Every model was also trained with a class-weighted
objective, which moves where a model puts its probabilities. Two models can
therefore differ in reported F1 while ranking the names identically, or agree on
F1 while one is far better calibrated. Nothing in the result set could tell those
cases apart, because no threshold-free metric existed anywhere.

AUC answers the ranking question and Brier answers the calibration one. Both come
straight from the stored probabilities, so this refits nothing. Reported per seed
with the same interval treatment as the rest of the paper.

If the AUC ordering matches the F1 ordering, the paper's claims stand and gain a
second line of support. If it does not, the abstract has to be written differently,
which is why this runs before the writing does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).parent.parent
GRID = ROOT / "results" / "final" / "24_grid_attention_pooling"
OUT = ROOT / "results" / "final" / "38_threshold_free"
SEEDS = [42, 7, 123, 2024, 777]
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
T_CRIT = 2.776   # t(0.975, df = 4)


def per_seed(p: pd.DataFrame, y: np.ndarray, mask=None) -> list[dict]:
    rows = []
    m = np.ones(len(y), dtype=bool) if mask is None else mask
    for name in CHAR + WORD:
        for s in SEEDS:
            col = f"{name}__seed{s}"
            if col not in p.columns:
                continue
            q = p[col].values[m]
            rows.append({"Model": name, "Seed": s,
                         "auc": roc_auc_score(y[m], q),
                         "brier": brier_score_loss(y[m], q)})
    return rows


def summarise(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby("Model")[col].agg(mean="mean", sd=lambda v: v.std(ddof=1))
    g["ci95_lo"] = g["mean"] - T_CRIT * g["sd"] / np.sqrt(len(SEEDS))
    g["ci95_hi"] = g["mean"] + T_CRIT * g["sd"] / np.sqrt(len(SEEDS))
    return g.round(6).reset_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(GRID / "val_probabilities.csv")
    y = p.label.values

    d = pd.DataFrame(per_seed(p, y))
    d.to_csv(OUT / "auc_brier_per_seed.csv", index=False)
    auc, brier = summarise(d, "auc"), summarise(d, "brier")
    auc.to_csv(OUT / "auc_summary.csv", index=False)
    brier.to_csv(OUT / "brier_summary.csv", index=False)

    # paired, matched seeds, the same treatment the F1 comparison gets
    piv_a = d.pivot(index="Seed", columns="Model", values="auc")
    piv_b = d.pivot(index="Seed", columns="Model", values="brier")
    pairs = []
    for fam in ("BiRNN", "BiGRU", "BiLSTM", "Transformer"):
        for lab, piv, sign in (("auc", piv_a, 1), ("brier", piv_b, -1)):
            diff = (piv["Char" + fam] - piv["Word" + fam]).values
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            pairs.append({"comparison": fam, "metric": lab,
                          "char": round(float(piv["Char" + fam].mean()), 6),
                          "word": round(float(piv["Word" + fam].mean()), 6),
                          "diff": round(float(diff.mean()), 6),
                          "ci95_lo": round(float(diff.mean() - T_CRIT * se), 6),
                          "ci95_hi": round(float(diff.mean() + T_CRIT * se), 6),
                          # Brier is a loss, so character is better when it is lower
                          "favours_character": bool(sign * diff.mean() > 0)})
    pr = pd.DataFrame(pairs)
    pr.to_csv(OUT / "char_vs_word_threshold_free.csv", index=False)

    f1 = pd.read_csv(GRID / "multiseed_summary.csv").set_index("Model")["F1_mean"]
    order_f1 = list(f1.sort_values(ascending=False).index)
    order_auc = list(auc.set_index("Model")["mean"].sort_values(ascending=False).index)
    same = order_f1 == order_auc
    moved = [m for m in order_f1 if order_f1.index(m) != order_auc.index(m)]

    ranks = pd.DataFrame({"by_f1": order_f1, "by_auc": order_auc})
    ranks.to_csv(OUT / "ranking_f1_against_auc.csv", index=False)

    print(auc.to_string(index=False))
    print()
    print(brier.to_string(index=False))
    print()
    print(pr.to_string(index=False))
    print()
    print(f"ranking identical between F1 and AUC: {same}")
    if not same:
        print(f"  models that move: {', '.join(moved)}")
        print("  by F1 :", " > ".join(order_f1))
        print("  by AUC:", " > ".join(order_auc))
    ch = auc[auc.Model.isin(CHAR)]["mean"]
    wd = auc[auc.Model.isin(WORD)]["mean"]
    print(f"\ncharacter AUC {ch.min():.4f} to {ch.max():.4f}, "
          f"word AUC {wd.min():.4f} to {wd.max():.4f}, "
          f"{'no overlap' if ch.min() > wd.max() else 'OVERLAP'}")
    print(f"Written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
