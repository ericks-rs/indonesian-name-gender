from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
OUT = FINAL / "19_external_unified"

PAIRS = [("CharBiRNN", "WordBiRNN", "BiRNN"),
         ("CharBiLSTM", "WordBiLSTM", "BiLSTM"),
         ("CharBiGRU", "WordBiGRU", "BiGRU"),
         ("CharTransformer", "WordTransformer", "Transformer")]
T_CRIT = 2.776

def holm(p: list[float]) -> list[float]:
    if any(not np.isfinite(v) for v in p):
        raise ValueError("non-finite p-value passed to Holm correction")
    order = np.argsort(p)
    out = [0.0] * len(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, p[i] * (len(p) - rank)))
        out[i] = run
    return out

def load() -> pd.DataFrame:
    frames = []

    n = pd.read_csv(FINAL / "24_grid_attention_pooling" / "multiseed_runs.csv")
    if "Ext_F1" not in n.columns:
        raise SystemExit("multiseed snapshot has no external column, promote the newer run first")
    frames.append(pd.DataFrame({"Model": n.Model, "Seed": n.Seed, "Family": "from scratch",
                                "internal_f1": n.F1, "external_f1": n.Ext_F1}))

    t = pd.read_csv(FINAL / "16_external_tfidf" / "tfidf_external_runs.csv")
    frames.append(pd.DataFrame({"Model": t.Model, "Seed": t.Seed, "Family": "classical",
                                "internal_f1": t.internal_f1, "external_f1": t.external_f1}))

    p = pd.read_csv(FINAL / "04_seeds_transformers" / "transformer_seed_runs.csv")
    frames.append(pd.DataFrame({"Model": p.Model, "Seed": p.Seed, "Family": "pretrained",
                                "internal_f1": p.val_f1, "external_f1": p.ext_f1}))

    return pd.concat(frames, ignore_index=True)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    runs.to_csv(OUT / "all_models_per_seed.csv", index=False)
    print(f"{runs.Model.nunique()} models, {len(runs)} model-seed rows, "
          f"seeds {sorted(runs.Seed.unique())}")

    s = runs.groupby(["Family", "Model"]).agg(
        internal=("internal_f1", "mean"), internal_sd=("internal_f1", "std"),
        external=("external_f1", "mean"), external_sd=("external_f1", "std")).round(4)
    s["drop_pp"] = ((s.internal - s.external) * 100).round(2)
    s = s.sort_values("external", ascending=False)
    s.to_csv(OUT / "external_unified_summary.csv")
    print("\n" + s.to_string())

    per = {m: g.set_index("Seed").external_f1 for m, g in runs.groupby("Model")}
    seeds = sorted(set(runs[runs.Model == "CharBiRNN"].Seed))
    rows = []
    for c, w, lab in PAIRS:
        a, b = per[c].loc[seeds].values, per[w].loc[seeds].values
        d = (a - b) * 100
        sd = d.std(ddof=1)
        half = T_CRIT * sd / np.sqrt(len(d))
        t, pv = stats.ttest_rel(a, b)
        rows.append({"comparison": lab, "char_ext_f1": round(a.mean(), 4),
                     "word_ext_f1": round(b.mean(), 4), "diff_pp": round(d.mean(), 3),
                     "ci95_lo_pp": round(d.mean() - half, 3),
                     "ci95_hi_pp": round(d.mean() + half, 3),
                     "cohens_dz": round(d.mean() / sd, 2), "p_raw": float(pv)})
    for r, adj in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = adj
    ext = pd.DataFrame(rows)
    ext.to_csv(OUT / "char_vs_word_external_paired.csv", index=False)
    print("\ncharacter versus word on the public benchmark")
    print(ext[["comparison", "char_ext_f1", "word_ext_f1", "diff_pp", "ci95_lo_pp",
               "ci95_hi_pp", "cohens_dz", "p_holm"]].to_string(index=False))
    print(f"lowest confidence bound: {min(r['ci95_lo_pp'] for r in rows):+.2f} pp")

    fam = runs.assign(drop=(runs.internal_f1 - runs.external_f1) * 100)
    fam["Level"] = np.where(fam.Model.str.startswith("Char"), "character",
                    np.where(fam.Model.str.startswith("Word"), "word", fam.Family))
    g = fam.groupby("Level")["drop"].agg(["count", "mean", "min", "max"]).round(2)
    print("\nF1 lost when moving to the public benchmark, percentage points")
    print(g.to_string())
    g.to_csv(OUT / "transfer_loss_by_level.csv")

    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
