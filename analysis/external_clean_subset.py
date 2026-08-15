from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
OUT = FINAL / "21_external_clean"

SOURCES = [
    (FINAL / "24_grid_attention_pooling" / "external_predictions.csv", "from scratch"),
    (FINAL / "16_external_tfidf" / "tfidf_external_predictions.csv", "classical"),
    (FINAL / "04_seeds_transformers" / "transformer_seed_external_predictions.csv", "pretrained"),
]
PAIRS = [("CharBiRNN", "WordBiRNN", "BiRNN"), ("CharBiLSTM", "WordBiLSTM", "BiLSTM"),
         ("CharBiGRU", "WordBiGRU", "BiGRU"),
         ("CharTransformer", "WordTransformer", "Transformer")]
T_CRIT = 2.776

def holm(p: list[float]) -> list[float]:
    if any(not np.isfinite(v) for v in p):
        raise ValueError("non-finite p-value passed to Holm correction")
    order = np.argsort(p)
    out, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, p[i] * (len(p) - rank)))
        out[i] = run
    return out

def paired(runs: pd.DataFrame, col: str, dest: Path, basis: str) -> pd.DataFrame:
    per = {m: g.set_index("Seed")[col] for m, g in runs.groupby("Model")}
    seeds = sorted(per["CharBiRNN"].index)
    out = []
    for c, w, lab in PAIRS:
        a, b = per[c].loc[seeds].values, per[w].loc[seeds].values
        d = (a - b) * 100
        sd = d.std(ddof=1)
        half = T_CRIT * sd / np.sqrt(len(d))
        _, pv = stats.ttest_rel(a, b)
        out.append({"comparison": lab, "char_f1": round(a.mean(), 4),
                    "word_f1": round(b.mean(), 4), "diff_pp": round(d.mean(), 3),
                    "ci95_lo_pp": round(d.mean() - half, 3),
                    "ci95_hi_pp": round(d.mean() + half, 3),
                    "cohens_dz": round(d.mean() / sd, 2), "p_raw": float(pv)})
    for r, adj in zip(out, holm([r["p_raw"] for r in out])):
        r["p_holm"] = adj
    df = pd.DataFrame(out)
    df.to_csv(dest, index=False)
    print()
    print(f"character versus word, {basis}")
    print(df[["comparison", "char_f1", "word_f1", "diff_pp", "ci95_lo_pp",
              "ci95_hi_pp", "cohens_dz", "p_holm"]].to_string(index=False))
    print(f"  range {df.diff_pp.min():.2f} to {df.diff_pp.max():.2f} pp, "
          f"lowest bound {df.ci95_lo_pp.min():+.2f}")
    return df

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "data" / "splits" / "train_1990_2021.csv")
    seen = set(tr.NAMA.str.strip().str.lower())

    ref = pd.read_csv(SOURCES[0][0])
    refkey = ref.name.str.strip().str.lower()
    for path, _ in SOURCES[1:]:
        other = pd.read_csv(path)
        aligned = ((other.name.str.strip().str.lower() == refkey).all()
                   and (other.label.values == ref.label.values).all())
        if not aligned:
            raise ValueError(f"{path.name} is not row aligned with the grid predictions")
    first = ~refkey.duplicated().values

    rows = []
    for path, family in SOURCES:
        p = pd.read_csv(path)
        key = p.name.str.strip().str.lower()
        clean = ~key.isin(seen).values
        dedup = clean & first
        y = p.label.values
        for col in p.columns:
            if col in ("name", "label"):
                continue
            model, _, seed = col.partition("__")
            model = model.replace("pred_", "")
            pred = p[col].values
            rows.append({"Family": family, "Model": model,
                         "Seed": int(seed.replace("seed", "")) if seed else 42,
                         "external_full": f1_score(y, pred),
                         "external_clean": f1_score(y[clean], pred[clean]),
                         "external_clean_dedup": f1_score(y[dedup], pred[dedup])})
        n_clean, n_all, n_ded = int(clean.sum()), len(p), int(dedup.sum())
        print(f"{path.parent.name:<28} rows {n_all}, clean {n_clean}, "
              f"clean and deduplicated {n_ded}")

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "external_clean_per_seed.csv", index=False)
    print(f"\n{runs.Model.nunique()} models, {len(runs)} model-seed rows")

    s = runs.groupby(["Family", "Model"]).agg(
        full=("external_full", "mean"), clean=("external_clean", "mean"),
        clean_dedup=("external_clean_dedup", "mean"),
        clean_dedup_sd=("external_clean_dedup", "std"),
        clean_sd=("external_clean", "std")).round(4)
    s["delta_pp"] = ((s.clean - s.full) * 100).round(2)
    s["delta_dedup_pp"] = ((s.clean_dedup - s.full) * 100).round(2)
    s = s.sort_values("clean_dedup", ascending=False)
    s.to_csv(OUT / "external_clean_summary.csv")
    print("\n" + s.to_string())

    for basis, col, fname in (
            ("uncontaminated rows", "external_clean", "char_vs_word_clean_paired.csv"),
            ("uncontaminated, one row per name", "external_clean_dedup",
             "char_vs_word_clean_dedup_paired.csv")):
        paired(runs, col, OUT / fname, basis)

    lvl = runs.assign(level=np.where(runs.Model.str.startswith("Char"), "character",
                              np.where(runs.Model.str.startswith("Word"), "word", runs.Family)))
    g = lvl.groupby("level")[["external_full", "external_clean",
                              "external_clean_dedup"]].mean().round(4)
    g["drop_pp"] = ((g.external_clean - g.external_full) * 100).round(2)
    g["drop_dedup_pp"] = ((g.external_clean_dedup - g.external_full) * 100).round(2)
    g.to_csv(OUT / "contamination_effect_by_level.csv")
    print("\neffect of removing the contaminated rows, by level")
    print(g.to_string())
    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
