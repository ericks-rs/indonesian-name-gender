from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "final" / "24_grid_attention_pooling"
OUT = ROOT / "results" / "final" / "27_error_analysis"
DATA = ROOT / "data" / "splits"
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
SEEDS = [42, 7, 123, 2024, 777]
FEM = ("wati", "ati", "ani", "ika", "ita", "sih", "ningsih", "ah", "iyah", "yanti")
MAS = ("wan", "man", "din", "udin", "anto", "arto", "yanto", "ono", "adi", "aji")

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(SRC / "val_predictions.csv")
    tr = pd.read_csv(DATA / "train_1990_2021.csv")
    y = pred.label.values
    names = pred.name.astype(str).values
    low = np.array([s.lower() for s in names])

    wrong = {}
    for m in CHAR:
        w = np.zeros(len(y), dtype=int)
        for s in SEEDS:
            w += (pred[f"{m}__seed{s}"].values != y).astype(int)
        wrong[m] = w
    W = pd.DataFrame(wrong)
    total = W.sum(axis=1)

    df = pd.DataFrame({"name": names, "label": y,
                       "n_wrong_of_20": total.values,
                       "n_models_ever_wrong": (W > 0).sum(axis=1).values,
                       "n_chars": [len(s) for s in low],
                       "n_tokens": [len(s.split()) for s in low]})
    for m in CHAR:
        df[f"wrong_{m}"] = W[m].values
    df["ends_listed_suffix"] = [s.endswith(FEM) or s.endswith(MAS) for s in low]

    seen = set()
    for n in tr.NAMA.astype(str).str.lower():
        seen.update(n.split())
    df["unseen_tokens"] = [sum(t not in seen for t in s.split()) for s in low]
    df["all_tokens_unseen"] = df.unseen_tokens == df.n_tokens

    df.to_csv(OUT / "per_name_errors.csv", index=False)

    hard = df[df.n_wrong_of_20 == 20]
    easy = df[df.n_wrong_of_20 == 0]
    part = df[(df.n_wrong_of_20 > 0) & (df.n_wrong_of_20 < 20)]
    print(f"{len(df):,} names. Every character model and seed correct on "
          f"{len(easy):,} ({len(easy)/len(df)*100:.2f}%), all wrong on "
          f"{len(hard):,} ({len(hard)/len(df)*100:.2f}%), disputed on "
          f"{len(part):,} ({len(part)/len(df)*100:.2f}%)")

    rows = []
    for label, sub in [("always correct", easy), ("disputed", part), ("always wrong", hard)]:
        rows.append({"group": label, "n": len(sub),
                     "share_pct": round(len(sub) / len(df) * 100, 2),
                     "mean_chars": round(float(sub.n_chars.mean()), 2),
                     "mean_tokens": round(float(sub.n_tokens.mean()), 2),
                     "pct_female": round(float(sub.label.mean() * 100), 2),
                     "pct_listed_suffix": round(float(sub.ends_listed_suffix.mean() * 100), 2),
                     "pct_all_tokens_unseen": round(float(sub.all_tokens_unseen.mean() * 100), 2),
                     "mean_unseen_tokens": round(float(sub.unseen_tokens.mean()), 3)})
    prof = pd.DataFrame(rows)
    prof.to_csv(OUT / "error_group_profile.csv", index=False)
    print("\n" + prof.to_string(index=False))

    curve = []
    for cut in (20, 19, 18, 16, 15, 10, 5, 1):
        sub = df[df.n_wrong_of_20 >= cut]
        curve.append({"at_least_wrong_of_20": cut, "n": len(sub),
                      "pct_female": round(float(sub.label.mean() * 100), 2)})
    cv = pd.DataFrame(curve)
    cv["above_base_pp"] = (cv.pct_female - float(df.label.mean() * 100)).round(2)
    cv.to_csv(OUT / "female_skew_by_cutoff.csv", index=False)
    print("")
    print("female share as the cutoff moves, base rate "
          f"{df.label.mean()*100:.2f} percent")
    print(cv.to_string(index=False))

    by_tok = df.assign(err=df.n_wrong_of_20 / 20).groupby(
        df.n_tokens.clip(1, 6)).agg(n=("name", "size"), error_rate=("err", "mean")).round(4)
    by_tok.to_csv(OUT / "error_rate_by_token_count.csv")
    print("\nerror rate by token count\n" + by_tok.to_string())

    by_suf = df.assign(err=df.n_wrong_of_20 / 20).groupby("ends_listed_suffix").agg(
        n=("name", "size"), error_rate=("err", "mean")).round(4)
    by_suf.to_csv(OUT / "error_rate_by_suffix.csv")
    print("\nerror rate by listed suffix\n" + by_suf.to_string())

    ex = hard.sort_values(["n_tokens", "name"])[
        ["name", "label", "n_tokens", "ends_listed_suffix", "all_tokens_unseen"]]
    ex.to_csv(OUT / "always_wrong_names.csv", index=False)
    print(f"\n{len(ex):,} names defeat all twenty fits. "
          f"{ex.all_tokens_unseen.mean()*100:.1f}% have no token seen in training, "
          f"against {df.all_tokens_unseen.mean()*100:.1f}% overall.")
    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
