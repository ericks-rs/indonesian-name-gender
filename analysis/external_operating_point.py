"""The operating point of the linear margin, and what happens when it is moved.

external_rank_flip.py showed that almost all of the external loss is precision,
and that TF-IDF+SVM gives up more precision than any character encoder while the
positive rate falls from 0.5084 to 0.4488. This script tests the reading that
follows from those two facts. A hard linear margin has no probability and no
threshold, so it meets a different class mix at whatever operating point the
training mix produced.

Two measurements, neither of which selects a reported result on the benchmark.

  1. Predicted positive rate per model per partition, against the true rate.
     This is the non-circular version of the claim, because it uses only the
     predictions the manuscript already reports.
  2. An oracle-threshold diagnostic. The SVM decision function is swept and the
     best achievable external F1 is recorded. The threshold is chosen on the
     benchmark labels, so the number is an upper bound and a diagnostic, never a
     reported model score. It answers one question only: is the external deficit
     recoverable by moving the operating point, or is the ranking learned?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "splits"
EXTERNAL = ROOT / "data" / "external" / "indonesian-names.csv"
FINAL = ROOT / "results" / "final"
OUT = FINAL / "42_external_rank_flip"

SEEDS = [42, 7, 123, 2024, 777]
CHAR_MODELS = ["CharBiLSTM", "CharBiGRU", "CharBiRNN", "CharTransformer"]
T_CRIT = 2.776


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA / "train_1990_2021.csv")
    te = pd.read_csv(DATA / "val_2024_2026.csv")
    ext = pd.read_csv(EXTERNAL)

    ext_tf = pd.read_csv(FINAL / "16_external_tfidf" / "tfidf_external_predictions.csv")
    ext_gr = pd.read_csv(FINAL / "24_grid_attention_pooling" / "external_predictions.csv")
    int_tf = pd.read_csv(FINAL / "03_seeds_tfidf" / "tfidf_seed_predictions.csv").rename(
        columns={"NAMA": "name", "LABEL_ENC": "label"})
    int_gr = pd.read_csv(FINAL / "24_grid_attention_pooling" / "val_predictions.csv")

    key = ext_tf.name.str.strip().str.lower()
    seen = set(tr.NAMA.str.strip().str.lower())
    dedup = (~key.isin(seen).values) & (~key.duplicated().values)
    full = np.ones(len(te), dtype=bool)

    # 1. predicted positive rate against the true positive rate
    rows = []
    for part, tfp, grp, mask in (("test 2024-2026", int_tf, int_gr, full),
                                 ("external, clean and deduplicated", ext_tf, ext_gr, dedup)):
        true_rate = float(tfp["label"].values[mask].mean())
        for model, src in [("TF-IDF+SVM", tfp)] + [(m, grp) for m in CHAR_MODELS]:
            r = np.array([src[f"{model}__seed{s}"].values[mask].mean() for s in SEEDS])
            rows.append({"partition": part, "model": model,
                         "true_positive_rate": round(true_rate, 4),
                         "predicted_positive_rate": round(float(r.mean()), 4),
                         "excess_pp": round(float(r.mean() - true_rate) * 100, 2)})
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "predicted_positive_rate.csv", index=False)
    print("predicted positive rate against the true rate")
    print(rates.to_string(index=False))

    piv = rates.set_index(["model", "partition"]).excess_pp
    drift = pd.DataFrame([{"model": m,
                           "excess_internal_pp": piv.loc[(m, "test 2024-2026")],
                           "excess_external_pp": piv.loc[(m, "external, clean and deduplicated")],
                           "drift_pp": round(piv.loc[(m, "external, clean and deduplicated")]
                                             - piv.loc[(m, "test 2024-2026")], 2)}
                          for m in ["TF-IDF+SVM"] + CHAR_MODELS])
    drift.to_csv(OUT / "positive_rate_drift.csv", index=False)
    print("\ndrift in over-prediction of the positive class, external minus internal")
    print(drift.to_string(index=False))

    # 2. oracle threshold on the SVM margin. Diagnostic only.
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000)
    Xtr = tfidf.fit_transform(tr.NAMA.str.lower())
    Xex = tfidf.transform(ext.name.str.lower())
    ytr = (tr.LABEL == "P").astype(int).values
    yex = ext.gender.map({"m": 0, "f": 1}).values
    yd = yex[dedup]

    sweep = []
    for seed in SEEDS:
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=seed, class_weight="balanced")
        clf.fit(Xtr, ytr)
        margin = clf.decision_function(Xex)[dedup]
        base = f1_score(yd, (margin > 0).astype(int))
        grid = np.quantile(margin, np.linspace(0.02, 0.98, 193))
        scores = [f1_score(yd, (margin > t).astype(int)) for t in grid]
        i = int(np.argmax(scores))
        sweep.append({"seed": seed, "f1_at_zero": base, "best_f1": scores[i],
                      "best_threshold": float(grid[i]),
                      "gain_pp": (scores[i] - base) * 100})
        print(f"  seed {seed:<5} default {base:.4f}  oracle {scores[i]:.4f} "
              f"at margin {grid[i]:+.4f}", flush=True)
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "svm_oracle_threshold.csv", index=False)
    v = sw.gain_pp.values
    half = T_CRIT * v.std(ddof=1) / np.sqrt(len(v)) if v.std(ddof=1) > 0 else 0.0
    print("\noracle threshold diagnostic for TF-IDF+SVM on the external benchmark")
    print(f"  default margin F1 {sw.f1_at_zero.mean():.4f}, "
          f"best achievable {sw.best_f1.mean():.4f}, "
          f"gain {v.mean():.3f} pp [{v.mean()-half:.3f}, {v.mean()+half:.3f}]")
    print("  the threshold was chosen on the benchmark labels, so this is an upper")
    print("  bound and is reported as a diagnostic, not as a model score")

    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
