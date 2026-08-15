"""Count every model fit behind the revision, from the artifacts themselves.

Reviewer A asked for evidence that the shared configuration is not doing the
work, and Reviewer B asked for variance and significance. Both answers rest on a
large number of separate fits, so the count has to be stated in the manuscript
and it has to be recomputed rather than remembered. Every number below is the
row count of a file on disk, except the Transformer sweep, which is still queued
and is reported as planned.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
LIVE = ROOT / "results" / "tables"
OUT = FINAL / "00_summary"

# label, unit of work, path, whether the fits are neural
BLOCKS = [
    ("Main grid, class weighting", "8 architectures, seed 42",
     FINAL / "01_main" / "tables" / "neural_weighted_only.csv", "neural"),
    ("Seed repetition, grid", "8 architectures x 5 seeds",
     ROOT / "_archive" / "v3g_superseded_dirs" / "02_seeds_grid" / "tables" / "multiseed" / "multiseed_runs.csv", "neural"),
    ("Seed repetition, TF-IDF", "3 classifiers x 5 seeds",
     FINAL / "03_seeds_tfidf" / "tfidf_seed_runs.csv", "classical"),
    ("Seed repetition, pretrained", "3 encoders x 5 seeds",
     FINAL / "04_seeds_transformers" / "transformer_seed_runs.csv", "pretrained"),
    ("Sensitivity, recurrent, test only", "6 architectures x 14 settings",
     LIVE / "sensitivity" / "hyperparam_sensitivity.csv", "neural"),
    ("Sensitivity, dev and test", "8 architectures x 14 settings",
     FINAL / "31_sensitivity_dev" / "sensitivity_dev_and_test.csv", "neural"),
    ("Imbalance, class balancing", "8 architectures",
     FINAL / "15_imbalance_standby" / "balanced" / "neural_balanced_only.csv", "neural"),
    ("Imbalance, oversampling", "8 architectures",
     FINAL / "15_imbalance_standby" / "oversample" / "neural_oversample_only.csv", "neural"),
    ("Imbalance, SMOTE", "8 architectures",
     FINAL / "15_imbalance_standby" / "smote" / "neural_smote_only.csv", "neural"),
    ("Imbalance, generated names", "8 architectures",
     FINAL / "15_imbalance_standby" / "llm_aug" / "neural_llm_aug_only.csv", "neural"),
    ("Temporal coverage", "3 training windows",
     FINAL / "10_temporal_drift" / "tables" / "temporal" / "cross_decade_results.csv", "neural"),
    ("Cross-dataset, TF-IDF refit", "3 classifiers x 5 seeds",
     FINAL / "16_external_tfidf" / "tfidf_external_runs.csv", "classical"),
    ("Grid rerun keeping predictions", "8 architectures x 5 seeds",
     ROOT / "_archive" / "v3g_superseded_dirs" / "20_seeds_grid_predictions" / "multiseed_runs.csv", "neural"),
    ("Grid with attention pooling", "8 architectures x 5 seeds",
     FINAL / "24_grid_attention_pooling" / "multiseed_runs.csv", "neural"),
    ("Selection and window decomposition", "CharBiGRU, 3 arms x 5 seeds",
     FINAL / "14_selection_bias" / "selection_bias_charbigru.csv", "neural"),
]

# analyses that reuse stored checkpoints or predictions and fit nothing new
REUSE = [
    ("Cross-dataset, all fourteen models", "stored predictions on the public benchmark",
     FINAL / "21_external_clean" / "external_clean_summary.csv"),
    ("Attention position", "8 models x 5 seeds over the full evaluation partition",
     FINAL / "26_attention_position" / "table_attention_position.csv"),
    ("Error analysis", "error pool over twenty character-model fits",
     FINAL / "27_error_analysis" / "error_group_profile.csv"),
    ("Inference cost", "latency and throughput per stored model",
     FINAL / "11_inference_benchmark" / "tables" / "inference_benchmark.csv"),
    ("Inference cost, repeated", "11 models x 7 independent trials x 200 calls",
     FINAL / "32_latency_repeats" / "latency_per_trial.csv"),
    ("Paired character versus word test", "matched seeds, Holm corrected",
     FINAL / "00_summary" / "char_vs_word_paired.csv"),
]

PLANNED = {"Sensitivity, dev and test": 112, "Sensitivity, recurrent, test only": 84}

# Most files hold one fit per row. The decomposition holds one seed per row with
# a column per arm, so its row count is a third of the fits behind it.
FITS_PER_ROW = {"Selection and window decomposition": 3}


def count(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, unit, path, kind in BLOCKS:
        got = count(path) * FITS_PER_ROW.get(label, 1)
        plan = PLANNED.get(label, got)
        rows.append({"Block": label, "Design": unit, "Kind": kind,
                     "Fits_done": got, "Fits_planned": plan,
                     "Status": "complete" if got >= plan else
                               ("running" if got else "queued")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "experiment_inventory.csv", index=False)

    w = max(len(r["Block"]) for r in rows)
    print(f"{'Block':<{w}}  {'Design':<32} {'done':>5} {'plan':>5}  status")
    print("-" * (w + 56))
    for r in rows:
        print(f"{r['Block']:<{w}}  {r['Design']:<32} {r['Fits_done']:>5} "
              f"{r['Fits_planned']:>5}  {r['Status']}")
    print("-" * (w + 56))
    print(f"{'TOTAL model fits':<{w}}  {'':<32} {df.Fits_done.sum():>5} "
          f"{df.Fits_planned.sum():>5}")

    print(f"\nby kind (planned): " + ", ".join(
        f"{k} {g.Fits_planned.sum()}" for k, g in df.groupby("Kind")))

    print("\nanalyses that fit nothing new")
    for label, unit, path in REUSE:
        print(f"  {label:<38} {unit:<58} {'present' if count(path) else 'MISSING'}")

    print(f"\nWritten to {OUT / 'experiment_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
