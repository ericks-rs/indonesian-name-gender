"""Why the character n-gram SVM leads internally and loses that lead off-corpus.

Reviewer A asked why TF-IDF+SVM ranks first on the 2024-2026 test partition and
third on the independent benchmark. The candidate explanation is structural. The
vectorizer freezes a vocabulary of 50,000 character n-grams selected from the
training partition, so an n-gram that the training registry never produced
carries no weight at all, whatever it spells. A character encoder reads the same
name through a 33-symbol embedding and still composes a representation for a
spelling it has not seen.

This script measures that asymmetry rather than asserting it. For every
evaluation name it counts how many of the analyzer's n-gram occurrences fall
outside the fitted vocabulary, does the same for characters against the neural
tokenizer, and then stratifies accuracy by coverage so the two model families
can be compared at matched difficulty. The interaction between coverage and the
gap between the families is bootstrapped, because a picture of two curves is not
evidence on its own.

Nothing is retrained. The vectorizer is refitted exactly as in seeds_tfidf.py
and external_tfidf.py, and every accuracy is read from the stored predictions.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "splits"
EXTERNAL = ROOT / "data" / "external" / "indonesian-names.csv"
FINAL = ROOT / "results" / "final"
TOK = ROOT / "tokenizers"
OUT = FINAL / "41_ngram_coverage"

SEEDS = [42, 7, 123, 2024, 777]
CHAR_MODELS = ["CharBiLSTM", "CharBiGRU", "CharBiRNN", "CharTransformer"]
RNG = np.random.default_rng(20260918)
N_BOOT = 10000


class CharTokenizer:
    def __init__(self): self.char2idx = {"<PAD>": 0, "<UNK>": 1}
    def encode(self, name, n):
        ids = [self.char2idx.get(c, 1) for c in name.lower()]
        return ids[:n] if len(ids) >= n else ids + [0] * (n - len(ids))
    @property
    def vocab_size(self): return len(self.char2idx)


class WordTokenizer:
    def __init__(self, min_freq=2): self.word2idx = {"<PAD>": 0, "<UNK>": 1}; self.min_freq = min_freq
    def encode(self, name, n):
        ids = [self.word2idx.get(w, 1) for w in name.lower().split()]
        return ids[:n] if len(ids) >= n else ids + [0] * (n - len(ids))
    @property
    def vocab_size(self): return len(self.word2idx)


def coverage_frame(names, analyzer, vocab, char_vocab):
    """One row per name: how much of it the frozen feature space can see."""
    rows = []
    for nm in names:
        grams = analyzer(nm)
        total = len(grams)
        in_vocab = [g for g in grams if g in vocab]
        types = set(grams)
        types_in = {g for g in types if g in vocab}
        chars = list(nm.lower())
        chars_oov = [c for c in chars if c not in char_vocab]
        rows.append({
            "name": nm,
            "n_grams": total,
            "n_grams_in_vocab": len(in_vocab),
            "ngram_oov_rate": (total - len(in_vocab)) / total if total else np.nan,
            "n_gram_types": len(types),
            "n_gram_types_in_vocab": len(types_in),
            "type_oov_rate": (len(types) - len(types_in)) / len(types) if types else np.nan,
            "n_chars": len(chars),
            "char_oov_rate": len(chars_oov) / len(chars) if chars else np.nan,
        })
    return pd.DataFrame(rows)


def pooled_correct(pred_df, model, mask):
    """Per-name mean correctness over the five seeds of one model."""
    y = pred_df["label"].values[mask]
    cols = [f"{model}__seed{s}" for s in SEEDS]
    p = pred_df[cols].values[mask]
    return (p == y[:, None]).mean(axis=1)


def boot_mean_ci(values, n=N_BOOT):
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    draws = values[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.modules["__main__"].CharTokenizer = CharTokenizer
    sys.modules["__main__"].WordTokenizer = WordTokenizer
    char_tok = pickle.load(open(TOK / "char_tokenizer.pkl", "rb"))
    char_vocab = set(char_tok.char2idx) - {"<PAD>", "<UNK>"}

    tr = pd.read_csv(DATA / "train_1990_2021.csv")
    te = pd.read_csv(DATA / "val_2024_2026.csv")
    ext = pd.read_csv(EXTERNAL)

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000)
    tfidf.fit(tr.NAMA.str.lower())
    analyzer = tfidf.build_analyzer()
    vocab = set(tfidf.vocabulary_)
    print(f"vectorizer vocabulary {len(vocab):,} n-grams, "
          f"character vocabulary {len(char_vocab)} symbols", flush=True)

    # the deduplicated uncontaminated benchmark, rebuilt exactly as in
    # external_clean_subset.py so the subset matches the reported one
    ext_pred_tfidf = pd.read_csv(FINAL / "16_external_tfidf" / "tfidf_external_predictions.csv")
    ext_pred_grid = pd.read_csv(FINAL / "24_grid_attention_pooling" / "external_predictions.csv")
    key = ext_pred_tfidf.name.str.strip().str.lower()
    if not (ext_pred_grid.name.str.strip().str.lower() == key).all():
        raise ValueError("external prediction files are not row aligned")
    seen = set(tr.NAMA.str.strip().str.lower())
    clean = ~key.isin(seen).values
    dedup = clean & (~key.duplicated().values)
    print(f"benchmark rows {len(ext):,}, uncontaminated {clean.sum():,}, "
          f"uncontaminated and deduplicated {dedup.sum():,}", flush=True)

    cov_int = coverage_frame(te.NAMA.str.lower().tolist(), analyzer, vocab, char_vocab)
    cov_int["partition"] = "test 2024-2026"
    cov_ext = coverage_frame(ext.name.str.lower().tolist(), analyzer, vocab, char_vocab)
    cov_ext["partition"] = "external benchmark"
    cov_ext_d = cov_ext[dedup].copy().reset_index(drop=True)
    cov_ext_d["partition"] = "external, clean and deduplicated"

    pd.concat([cov_int, cov_ext_d], ignore_index=True).to_csv(
        OUT / "coverage_per_name.csv", index=False)

    summary = []
    for lab, df in (("test 2024-2026", cov_int),
                    ("external benchmark", cov_ext),
                    ("external, clean and deduplicated", cov_ext_d)):
        summary.append({
            "partition": lab, "n_names": len(df),
            "mean_ngram_oov_rate": round(df.ngram_oov_rate.mean(), 4),
            "median_ngram_oov_rate": round(df.ngram_oov_rate.median(), 4),
            "mean_type_oov_rate": round(df.type_oov_rate.mean(), 4),
            "share_with_any_oov": round((df.ngram_oov_rate > 0).mean(), 4),
            "share_oov_over_10pct": round((df.ngram_oov_rate > 0.10).mean(), 4),
            "share_oov_over_25pct": round((df.ngram_oov_rate > 0.25).mean(), 4),
            "mean_char_oov_rate": round(df.char_oov_rate.mean(), 6),
            "share_with_any_char_oov": round((df.char_oov_rate > 0).mean(), 6),
            "mean_ngram_types_in_vocab": round(df.n_gram_types_in_vocab.mean(), 2),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "coverage_summary.csv", index=False)
    print("\ncoverage by partition")
    print(summ.to_string(index=False))

    a, b = cov_ext_d.ngram_oov_rate.values, cov_int.ngram_oov_rate.values
    lo_a, hi_a = boot_mean_ci(a)
    lo_b, hi_b = boot_mean_ci(b)
    print(f"\nmean n-gram OOV rate external {a.mean():.4f} [{lo_a:.4f}, {hi_a:.4f}], "
          f"test {b.mean():.4f} [{lo_b:.4f}, {hi_b:.4f}], "
          f"difference {a.mean() - b.mean():+.4f}")

    int_pred_tfidf = pd.read_csv(FINAL / "03_seeds_tfidf" / "tfidf_seed_predictions.csv")
    int_pred_grid = pd.read_csv(FINAL / "24_grid_attention_pooling" / "val_predictions.csv")

    bins = [-1e-9, 0.0, 0.05, 0.10, 0.20, 1.01]
    labels = ["0", "0 to 5", "5 to 10", "10 to 20", "over 20"]

    svm_ext = pooled_correct(ext_pred_tfidf, "TF-IDF+SVM", dedup)
    char_ext = {m: pooled_correct(ext_pred_grid, m, dedup) for m in CHAR_MODELS}
    oov_ext = cov_ext_d.ngram_oov_rate.values
    grp_ext = pd.cut(oov_ext, bins=bins, labels=labels)

    if not (int_pred_grid.name.str.lower().values == te.NAMA.str.lower().values).all():
        raise ValueError("internal grid predictions are not aligned with the test partition")
    int_tf = int_pred_tfidf.rename(columns={"NAMA": "name", "LABEL_ENC": "label"})
    if not (int_tf.name.str.lower().values == te.NAMA.str.lower().values).all():
        raise ValueError("internal TF-IDF predictions are not aligned with the test partition")
    full = np.ones(len(te), dtype=bool)
    svm_int = pooled_correct(int_tf, "TF-IDF+SVM", full)
    char_int = {m: pooled_correct(int_pred_grid, m, full) for m in CHAR_MODELS}
    oov_int = cov_int.ngram_oov_rate.values
    grp_int = pd.cut(oov_int, bins=bins, labels=labels)

    strata = []
    for part, grp, svm, chars, oov in (
            ("external, clean and deduplicated", grp_ext, svm_ext, char_ext, oov_ext),
            ("test 2024-2026", grp_int, svm_int, char_int, oov_int)):
        for lab in labels:
            m = np.asarray(grp == lab)
            if m.sum() == 0:
                continue
            row = {"partition": part, "oov_bin_pct": lab, "n_names": int(m.sum()),
                   "mean_oov_rate": round(float(oov[m].mean()), 4),
                   "TF-IDF+SVM": round(float(svm[m].mean()) * 100, 2)}
            for mod in CHAR_MODELS:
                row[mod] = round(float(chars[mod][m].mean()) * 100, 2)
            row["CharBiLSTM_minus_SVM_pp"] = round(row["CharBiLSTM"] - row["TF-IDF+SVM"], 2)
            strata.append(row)

    st = pd.DataFrame(strata)
    st.to_csv(OUT / "accuracy_by_coverage_bin.csv", index=False)
    print("\naccuracy by n-gram coverage bin, percent")
    print(st.to_string(index=False))

    # the interaction the explanation predicts: on the external benchmark the
    # character encoder should gain on the SVM as coverage falls
    interactions = []
    for mod in CHAR_MODELS:
        gap = (char_ext[mod] - svm_ext) * 100
        low = gap[oov_ext <= 0.0]
        high = gap[oov_ext > 0.10]
        if len(low) < 20 or len(high) < 20:
            continue
        obs = high.mean() - low.mean()
        draws = (RNG.choice(high, (N_BOOT, len(high))).mean(axis=1)
                 - RNG.choice(low, (N_BOOT, len(low))).mean(axis=1))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        interactions.append({"model": mod, "n_full_coverage": len(low),
                             "n_oov_over_10pct": len(high),
                             "gap_full_coverage_pp": round(float(low.mean()), 3),
                             "gap_oov_over_10pct_pp": round(float(high.mean()), 3),
                             "interaction_pp": round(float(obs), 3),
                             "ci95_lo_pp": round(float(lo), 3),
                             "ci95_hi_pp": round(float(hi), 3)})
    inter = pd.DataFrame(interactions)
    inter.to_csv(OUT / "coverage_interaction_external.csv", index=False)
    print("\ncharacter model minus TF-IDF+SVM, full coverage against coverage gaps, external")
    print(inter.to_string(index=False))

    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
