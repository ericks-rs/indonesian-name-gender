"""Accuracy split by whether a name ends in a gender-bearing pattern.

The review asked about cultural and regional bias. An earlier version of this
analysis grouped evaluation names by orthographic markers of particular naming
traditions, which put ethnic labels in a table on the strength of a word list
assembled by hand and covering only a fifth of the corpus. The finding it
produced is a special case of the one below, so nothing is lost by stating it
without the labels.

The split here is structural. A model that reads the end of a name depends on
the end of a name carrying the signal, and half the corpus does not oblige. The
patterns are found in the training partition rather than written down in
advance. Any three-character ending with at least MIN_SUPPORT training names and
at least STRONG of one gender counts as a marker, which leaves no room for a
list to be curated toward a conclusion.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
GRID = FINAL / "24_grid_attention_pooling"
OUT = FINAL / "28_suffix_coverage"
TOK = ROOT / "tokenizers"
CHAR =["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
SEEDS = [42, 7, 123, 2024, 777]
NGRAM = 3
MIN_SUPPORT = 200
STRONG = 0.80
T_CRIT = 2.776  # t(0.975, df = 4)


class WordTokenizer:  # stub so the tokenizer pickle can be loaded, only word2idx is used
    pass


sys.modules["__main__"].WordTokenizer = WordTokenizer


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "data" / "splits" / "train_1990_2021.csv")
    te = pd.read_csv(ROOT / "data" / "splits" / "val_2024_2026.csv")
    pred = pd.read_csv(GRID / "val_predictions.csv")
    y = pred.label.values

    tl = tr.NAMA.astype(str).str.lower()
    ty = (tr.LABEL == "P").astype(int).values
    end = tl.str[-NGRAM:]
    tot, fem = Counter(end), Counter(end[ty == 1])
    markers = {e: fem[e] / n for e, n in tot.items()
               if n >= MIN_SUPPORT and (fem[e] / n >= STRONG or fem[e] / n <= 1 - STRONG)}
    pd.DataFrame([{"ending": e, "n_training": tot[e], "pct_female": round(v * 100, 1),
                   "points_to": "female" if v >= 0.5 else "male"}
                  for e, v in sorted(markers.items(), key=lambda kv: -kv[1])]).to_csv(
        OUT / "marker_endings.csv", index=False)
    print(f"{len(markers)} marker endings found in training, each with at least "
          f"{MIN_SUPPORT} names and at least {STRONG*100:.0f} percent of one gender")

    lt = te.NAMA.astype(str).str.lower()
    has = lt.str[-NGRAM:].isin(markers).values
    # written out, not only printed, so the manuscript can cite counts rather than
    # a percentage somebody rounded by hand
    pd.DataFrame([{"n_eval": int(len(has)), "n_with_marker": int(has.sum()),
                   "n_without_marker": int((~has).sum()),
                   "n_marker_endings": len(markers),
                   "min_support": MIN_SUPPORT, "strong_pct": STRONG * 100,
                   "ngram": NGRAM}]).to_csv(OUT / "coverage_counts.csv", index=False)
    print(f"evaluation names ending in a marker: {int(has.sum()):,} "
          f"({has.mean()*100:.1f} percent), without: {int((~has).sum()):,} "
          f"({(~has).mean()*100:.1f} percent)\n")

    rows = []
    for fam, models in (("character", CHAR), ("word", WORD)):
        a = np.array([np.mean([(pred[f"{m}__seed{s}"].values[has] == y[has]).mean()
                               for m in models]) for s in SEEDS]) * 100
        b = np.array([np.mean([(pred[f"{m}__seed{s}"].values[~has] == y[~has]).mean()
                               for m in models]) for s in SEEDS]) * 100
        d = a - b
        half = T_CRIT * d.std(ddof=1) / np.sqrt(len(d))
        _, p = stats.ttest_1samp(d, 0.0)
        rows.append({"level": fam,
                     "acc_with_marker": round(float(a.mean()), 2),
                     "acc_without_marker": round(float(b.mean()), 2),
                     "diff_pp": round(float(d.mean()), 2),
                     "ci95_lo_pp": round(float(d.mean() - half), 2),
                     "ci95_hi_pp": round(float(d.mean() + half), 2),
                     "p_raw": float(p)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy_by_suffix_coverage.csv", index=False)
    print(df.to_string(index=False))

    # the same split, model by model, so the effect is not an artefact of averaging
    per = []
    for m in CHAR + WORD:
        a = np.array([(pred[f"{m}__seed{s}"].values[has] == y[has]).mean() for s in SEEDS]) * 100
        b = np.array([(pred[f"{m}__seed{s}"].values[~has] == y[~has]).mean() for s in SEEDS]) * 100
        per.append({"model": m, "with_marker": round(float(a.mean()), 2),
                    "without_marker": round(float(b.mean()), 2),
                    "diff_pp": round(float((a - b).mean()), 2)})
    pm = pd.DataFrame(per)
    pm.to_csv(OUT / "accuracy_by_suffix_coverage_per_model.csv", index=False)
    print("\nper model\n" + pm.to_string(index=False))
    print(f"\nevery one of the {len(pm)} models is worse on names without a marker: "
          f"{bool((pm.diff_pp > 0).all())}")

    # character versus word by number of tokens, on both the internal test and the
    # clean external benchmark (Table IV). Pooled over the 20 fits per level. The
    # external panel is the direct evidence that the out-of-distribution widening is
    # a single-token effect, so the word models are scored here, not only the char.
    ep = pd.read_csv(GRID / "external_predictions.csv")
    _k = ep.name.astype(str).str.strip().str.lower()
    # main reported external set: unseen in training AND one row per distinct name,
    # matching external_rank_flip and the deduplicated set paper_numbers reports.
    ext_dedup = (~_k.isin(set(tr.NAMA.astype(str).str.strip().str.lower()))).values & (~_k.duplicated().values)
    trows = []
    for part, names, preds, yy, base in (
            ("internal 2024-2026", te.NAMA, pred, y, np.ones(len(te), dtype=bool)),
            ("external clean dedup", ep.name, ep, ep.label.values, ext_dedup)):
        nt = names.astype(str).str.split().str.len().values
        for k in (1, 2, 3, 4, 5):
            sel = ((nt == k) if k < 5 else (nt >= 5)) & base
            if int(sel.sum()) < 10:
                continue
            cell = {"partition": part, "tokens": "5 or more" if k == 5 else str(k),
                    "n_names": int(sel.sum())}
            for fam, models in (("character", CHAR), ("word", WORD)):
                cell[f"{fam}_f1"] = round(float(np.mean(
                    [f1_score(yy[sel], preds[f"{m}__seed{s}"].values[sel])
                     for m in models for s in SEEDS])), 4)
                cell[f"{fam}_acc_pct"] = round(float(np.mean(
                    [(preds[f"{m}__seed{s}"].values[sel] == yy[sel]).mean()
                     for m in models for s in SEEDS]) * 100), 2)
            cell["gap_f1_pp"] = round((cell["character_f1"] - cell["word_f1"]) * 100, 2)
            cell["gap_acc_pp"] = round(cell["character_acc_pct"] - cell["word_acc_pct"], 2)
            trows.append(cell)
    tt = pd.DataFrame(trows)
    tt.to_csv(OUT / "accuracy_by_token_level.csv", index=False)
    print("\ncharacter versus word by token count, internal and external\n"
          + tt.to_string(index=False))

    # single-token names against the word vocabulary the grid models were trained with.
    # A token outside the vocabulary (seen fewer than twice in training) becomes <UNK>,
    # so the word model gets one shared symbol and no spelling. Counts only, no names.
    wt = pickle.load(open(TOK / "word_tokenizer.pkl", "rb"))
    vocab = wt.word2idx
    # the released vocabulary stores salted blake2s keys, so tokens are hashed the
    # same way as indonamegender.tokenizers before the lookup
    unk = vocab["<UNK>"]
    if getattr(wt, "hashed", False):
        import hashlib
        key = lambda w: hashlib.blake2s(("indonamegender-v1" + w).encode("utf-8"), digest_size=8).hexdigest()
        lookup = lambda w: vocab.get(key(w), unk)
    else:
        lookup = lambda w: vocab.get(w, unk)
    vrows = []
    for part, names, preds, yy, base in (
            ("internal 2024-2026", te.NAMA, pred, y, np.ones(len(te), dtype=bool)),
            ("external clean dedup", ep.name, ep, ep.label.values, ext_dedup)):
        toks = names.astype(str).str.lower().str.split()
        nt = toks.str.len().values
        oov = np.array([all(lookup(w) == unk for w in t) for t in toks])
        single = (nt == 1) & base
        for subset, sel in (("single-token, all", single),
                            ("single-token, outside vocabulary", single & oov),
                            ("single-token, in vocabulary", single & ~oov)):
            cell = {"partition": part, "subset": subset, "n_names": int(sel.sum()),
                    "n_single_token": int(single.sum()),
                    "pct_of_single_token": round(float(sel.sum() / single.sum() * 100), 4)}
            for fam, models in (("character", CHAR), ("word", WORD)):
                cell[f"{fam}_acc_pct"] = round(float(np.mean(
                    [(preds[f"{m}__seed{s}"].values[sel] == yy[sel]).mean()
                     for m in models for s in SEEDS]) * 100), 4)
            vrows.append(cell)
    vt = pd.DataFrame(vrows)
    vt.to_csv(OUT / "single_token_vocabulary.csv", index=False)
    print(f"\nsingle-token names against the word vocabulary ({len(vocab):,} entries)\n"
          + vt.to_string(index=False))

    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
