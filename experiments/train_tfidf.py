import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUT = PROJECT_ROOT / "results" / "final" / "03_seeds_tfidf"

SEEDS = [42, 7, 123, 2024, 777]

def build_classifiers(seed):
    return [
        ("TF-IDF+SVM", LinearSVC(C=1.0, max_iter=5000, random_state=seed,
                                 class_weight="balanced")),
        ("TF-IDF+LR", LogisticRegression(C=1.0, max_iter=1000, random_state=seed,
                                         class_weight="balanced", solver="liblinear")),
        ("TF-IDF+RF", RandomForestClassifier(n_estimators=300, max_depth=30, n_jobs=-1,
                                             random_state=seed, class_weight="balanced")),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(DATA_DIR / "train_1990_2021.csv")
    va = pd.read_csv(DATA_DIR / "val_2024_2025.csv")
    tr["LABEL_ENC"] = (tr["LABEL"] == "P").astype(int)
    va["LABEL_ENC"] = (va["LABEL"] == "P").astype(int)
    print(f"Train {len(tr):,} | Val {len(va):,}", flush=True)

    t0 = time.time()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000)
    Xtr = tfidf.fit_transform(tr["NAMA"].str.lower())
    Xva = tfidf.transform(va["NAMA"].str.lower())
    ytr, yva = tr["LABEL_ENC"].values, va["LABEL_ENC"].values
    print(f"TF-IDF fitted: {Xtr.shape[1]:,} features in {time.time()-t0:.1f}s", flush=True)

    runs, preds = [], {"NAMA": va["NAMA"].values, "LABEL_ENC": yva}
    for seed in SEEDS:
        for name, clf in build_classifiers(seed):
            t0 = time.time()
            clf.fit(Xtr, ytr)
            p = clf.predict(Xva)
            preds[f"{name}__seed{seed}"] = p
            runs.append({
                "Model": name, "Seed": seed,
                "accuracy": accuracy_score(yva, p),
                "precision": precision_score(yva, p),
                "recall": recall_score(yva, p),
                "f1": f1_score(yva, p),
                "train_s": round(time.time() - t0, 1),
            })
            print(f"  {name:<12} seed {seed:<5} F1={runs[-1]['f1']:.4f}  "
                  f"({runs[-1]['train_s']}s)", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out / "tfidf_seed_runs.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "tfidf_seed_predictions.csv", index=False)

    rows = []
    for model, g in runs_df.groupby("Model", sort=False):
        row = {"Model": model, "n_seeds": len(g)}
        for m in ["accuracy", "precision", "recall", "f1"]:
            v = g[m].values
            sd = v.std(ddof=1)
            half = 2.776 * sd / np.sqrt(len(v))
            row[f"{m}_mean"] = round(v.mean(), 4)
            row[f"{m}_std"] = round(sd, 4)
            row[f"{m}_ci95_lo"] = round(v.mean() - half, 4)
            row[f"{m}_ci95_hi"] = round(v.mean() + half, 4)
        row["f1_min"], row["f1_max"] = round(g.f1.min(), 4), round(g.f1.max(), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tfidf_seed_summary.csv", index=False)

    print("\n" + summary[["Model", "n_seeds", "f1_mean", "f1_std", "f1_min", "f1_max"]].to_string(index=False))
    print(f"\nWritten to {out}", flush=True)

if __name__ == "__main__":
    main()
