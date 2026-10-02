"""What actually moves when TF-IDF+SVM drops from first to third off-corpus.

The frozen-vocabulary explanation was tested first, in ngram_coverage.py, and the
data did not support it. Coverage differs by only about one point between the two
evaluation sets, no external name contains a character outside the neural
vocabulary, and the coverage interaction intervals all cross zero.

So this script asks the question a different way. It separates four candidates
that the stored predictions can settle without retraining anything.

  1. Size of the flip. Is the external difference between CharBiLSTM and
     TF-IDF+SVM separable at all, or is a 0.17 point reordering being read as a
     result when the interval covers it?
  2. Which component moved. F1 hides a precision and recall trade. The SVM runs
     a precision-heavy operating point internally, and a hard-margin linear
     decision has no threshold to recalibrate when the class mix changes.
  3. Class mix. The benchmark is not the registry, and F1 on the positive class
     responds to the positive rate.
  4. Name length. The benchmark carries far fewer n-gram types per name than the
     registry does, which is a composition difference rather than a spelling one.

Every accuracy here is read from the prediction files the manuscript already
reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "splits"
EXTERNAL = ROOT / "data" / "external" / "indonesian-names.csv"
FINAL = ROOT / "results" / "final"
OUT = FINAL / "42_external_rank_flip"

SEEDS = [42, 7, 123, 2024, 777]
CHAR_MODELS = ["CharBiLSTM", "CharBiGRU", "CharBiRNN", "CharTransformer"]
ENC_MODELS = ["IndoBERT", "mBERT", "XLM-R"]
T_CRIT = 2.776  # t(0.975, df = 4)
RNG = np.random.default_rng(20260918)
N_BOOT = 10000


def per_seed_scores(pred, model, mask):
    y = pred["label"].values[mask]
    out = []
    for s in SEEDS:
        p = pred[f"{model}__seed{s}"].values[mask]
        out.append({"seed": s, "f1": f1_score(y, p),
                    "precision": precision_score(y, p),
                    "recall": recall_score(y, p),
                    "accuracy": float((p == y).mean())})
    return pd.DataFrame(out)


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    if v.std(ddof=1) == 0:
        return v.mean(), v.mean(), v.mean()
    half = T_CRIT * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


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

    # 1 and 2. per seed precision, recall and F1 on both evaluation sets
    rows = []
    for label, tfp, grp, mask in (("test 2024-2026", int_tf, int_gr, full),
                                  ("external, clean and deduplicated", ext_tf, ext_gr, dedup)):
        for model, src in [("TF-IDF+SVM", tfp)] + [(m, grp) for m in CHAR_MODELS]:
            d = per_seed_scores(src, model, mask)
            d.insert(0, "model", model)
            d.insert(0, "partition", label)
            rows.append(d)
    per_seed = pd.concat(rows, ignore_index=True)
    per_seed.to_csv(OUT / "per_seed_scores.csv", index=False)

    agg = []
    for (part, model), g in per_seed.groupby(["partition", "model"], sort=False):
        row = {"partition": part, "model": model}
        for m in ("f1", "precision", "recall", "accuracy"):
            mu, lo, hi = mean_ci(g[m].values)
            row[m] = round(mu, 4)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
        agg.append(row)
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT / "score_components.csv", index=False)
    print("precision, recall and F1 by partition")
    print(agg[["partition", "model", "precision", "recall", "f1"]].to_string(index=False))

    # movement of each component between the two evaluation sets
    piv = agg.set_index(["model", "partition"])
    move = []
    for model in ["TF-IDF+SVM"] + CHAR_MODELS:
        a = piv.loc[(model, "test 2024-2026")]
        b = piv.loc[(model, "external, clean and deduplicated")]
        move.append({"model": model,
                     "precision_pp": round((b.precision - a.precision) * 100, 2),
                     "recall_pp": round((b.recall - a.recall) * 100, 2),
                     "f1_pp": round((b.f1 - a.f1) * 100, 2)})
    move = pd.DataFrame(move)
    move.to_csv(OUT / "component_movement.csv", index=False)
    print("\nexternal minus internal, percentage points")
    print(move.to_string(index=False))

    # 1. is the external reordering separable, matched seeds against the
    #    deterministic SVM
    svm_ext = per_seed.query("partition != 'test 2024-2026' and model == 'TF-IDF+SVM'").f1.values
    svm_int = per_seed.query("partition == 'test 2024-2026' and model == 'TF-IDF+SVM'").f1.values
    sep = []
    for model in CHAR_MODELS:
        for part, svm in (("test 2024-2026", svm_int),
                          ("external, clean and deduplicated", svm_ext)):
            ch = per_seed.query("partition == @part and model == @model").f1.values
            d = (ch - svm) * 100
            mu, lo, hi = mean_ci(d)
            sep.append({"partition": part, "model": model,
                        "char_minus_svm_pp": round(mu, 3),
                        "ci95_lo_pp": round(lo, 3), "ci95_hi_pp": round(hi, 3),
                        "interval_excludes_zero": bool(lo * hi > 0)})
    sep = pd.DataFrame(sep)
    sep.to_csv(OUT / "char_vs_svm_paired.csv", index=False)
    print("\ncharacter model minus TF-IDF+SVM, F1 percentage points over five seeds")
    print(sep.to_string(index=False))

    # 3. class mix
    # The column is named for what it holds, the share of names labelled
    # female. It was called positive_rate, which in a classification table
    # reads as the rate of positive predictions. That is a different
    # quantity, measured separately in predicted_positive_rate.csv, where it
    # runs 0.4931 to 0.5001 internally against the 0.5084 here.
    pos_int = float((te.LABEL == "P").mean())
    ext_d = ext[dedup].reset_index(drop=True)
    pos_ext = float((ext_d.gender == "f").mean())
    mix = pd.DataFrame([{"partition": "test 2024-2026", "n": len(te),
                         "female_label_rate": round(pos_int, 4)},
                        {"partition": "external, clean and deduplicated", "n": len(ext_d),
                         "female_label_rate": round(pos_ext, 4)}])
    mix.to_csv(OUT / "class_mix.csv", index=False)
    print("\nshare of names labelled female")
    print(mix.to_string(index=False))

    # 4. name length composition and accuracy by token count
    te_tok = te.NAMA.str.split().str.len().values
    ext_tok = ext_d.name.str.split().str.len().values
    comp = []
    for lab, toks in (("test 2024-2026", te_tok), ("external, clean and deduplicated", ext_tok)):
        s = pd.Series(toks)
        comp.append({"partition": lab, "n": len(s), "mean_tokens": round(s.mean(), 3),
                     "median_tokens": int(s.median()),
                     **{f"share_{k}_token": round(float((s == k).mean()), 4) for k in (1, 2, 3)},
                     "share_4_or_more": round(float((s >= 4).mean()), 4)})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "length_composition.csv", index=False)
    print("\ntoken count composition")
    print(comp.to_string(index=False))

    def pooled(pred, model, mask):
        y = pred["label"].values[mask]
        p = pred[[f"{model}__seed{s}" for s in SEEDS]].values[mask]
        return (p == y[:, None]).mean(axis=1)

    bylen = []
    for part, tfp, grp, mask, toks in (
            ("test 2024-2026", int_tf, int_gr, full, te_tok),
            ("external, clean and deduplicated", ext_tf, ext_gr, dedup, ext_tok)):
        svm = pooled(tfp, "TF-IDF+SVM", mask)
        chars = {m: pooled(grp, m, mask) for m in CHAR_MODELS}
        for k in (1, 2, 3, 4):
            sel = (toks == k) if k < 4 else (toks >= 4)
            if sel.sum() < 20:
                continue
            row = {"partition": part, "tokens": "4 or more" if k == 4 else str(k),
                   "n_names": int(sel.sum()),
                   "TF-IDF+SVM": round(float(svm[sel].mean()) * 100, 2)}
            for m in CHAR_MODELS:
                row[m] = round(float(chars[m][sel].mean()) * 100, 2)
            row["CharBiLSTM_minus_SVM_pp"] = round(row["CharBiLSTM"] - row["TF-IDF+SVM"], 2)
            bylen.append(row)
    bylen = pd.DataFrame(bylen)
    bylen.to_csv(OUT / "accuracy_by_token_count.csv", index=False)
    print("\naccuracy by token count, percent")
    print(bylen.to_string(index=False))

    # composition control. Reweight the external names so their token mix matches
    # the registry, which separates a composition shift from a transfer effect.
    svm = pooled(ext_tf, "TF-IDF+SVM", dedup)
    chars = {m: pooled(ext_gr, m, dedup) for m in CHAR_MODELS}
    target = pd.Series(te_tok).clip(upper=4).value_counts(normalize=True)
    src = pd.Series(ext_tok).clip(upper=4)
    w = src.map(target / src.value_counts(normalize=True)).values
    w = w / w.sum()
    reweighted = [{"model": "TF-IDF+SVM",
                   "external_accuracy": round(float(svm.mean()) * 100, 2),
                   "reweighted_to_registry_length_mix": round(float((svm * w).sum()) * 100, 2)}]
    for m in CHAR_MODELS:
        reweighted.append({"model": m,
                           "external_accuracy": round(float(chars[m].mean()) * 100, 2),
                           "reweighted_to_registry_length_mix": round(float((chars[m] * w).sum()) * 100, 2)})
    rw = pd.DataFrame(reweighted)
    rw["shift_pp"] = (rw.reweighted_to_registry_length_mix - rw.external_accuracy).round(2)
    rw.to_csv(OUT / "length_reweighted_external.csv", index=False)
    print("\nexternal accuracy reweighted to the registry token mix")
    print(rw.to_string(index=False))

    # 5. is the pretrained-encoder external drop composition-driven (like SVM) or
    #    encoder-specific? Same clean+dedup subset, same reweighting to the internal
    #    token mix, applied to the three encoders alongside the char models and SVM.
    int_e = pd.read_csv(FINAL / "04_seeds_transformers" / "transformer_seed_val_predictions.csv")
    ext_e = pd.read_csv(FINAL / "04_seeds_transformers" / "transformer_seed_external_predictions.csv")
    ke = ext_e.name.str.strip().str.lower()
    dedup_e = (~ke.isin(seen).values) & (~ke.duplicated().values)

    def f1_seeds(pred, model, mask, prefix):
        y = pred["label"].values[mask]
        return float(np.mean([f1_score(y, pred[f"{prefix}{model}__seed{s}"].values[mask])
                              for s in SEEDS]))

    def acc_names(pred, model, mask, prefix):  # per-name accuracy pooled over seeds
        y = pred["label"].values[mask]
        P = np.column_stack([pred[f"{prefix}{model}__seed{s}"].values for s in SEEDS])[mask]
        return (P == y[:, None]).mean(axis=1)

    def w_reweight(names, mask):  # weights mapping external token mix -> internal mix
        tok = pd.Series(names.str.split().str.len().values[mask]).clip(upper=4)
        tgt = pd.Series(te_tok).clip(upper=4).value_counts(normalize=True)
        ww = tok.map(tgt / tok.value_counts(normalize=True)).values
        return ww / ww.sum()

    comp_rows = []
    for models, intd, extd, msk, pfx, allint in (
            (ENC_MODELS, int_e, ext_e, dedup_e, "pred_", np.ones(len(int_e), bool)),
            (CHAR_MODELS, int_gr, ext_gr, dedup, "", full),
            (["TF-IDF+SVM"], int_tf, ext_tf, dedup, "", full)):
        w_e = w_reweight(extd["name"], msk)
        for m in models:
            a_ext = acc_names(extd, m, msk, pfx)
            a_int = acc_names(intd, m, allint, pfx).mean()
            a_ext_rw = float((a_ext * w_e).sum())
            comp_rows.append({
                "model": m,
                "int_F1": round(f1_seeds(intd, m, allint, pfx), 4),
                "ext_F1": round(f1_seeds(extd, m, msk, pfx), 4),
                "F1_drop_pp": round((f1_seeds(intd, m, allint, pfx) - f1_seeds(extd, m, msk, pfx)) * 100, 2),
                "int_acc": round(a_int * 100, 2),
                "ext_acc": round(a_ext.mean() * 100, 2),
                "ext_acc_reweighted": round(a_ext_rw * 100, 2),
                "reweight_recovers_pp": round((a_ext_rw - a_ext.mean()) * 100, 2),
                "residual_after_reweight_pp": round((a_int - a_ext_rw) * 100, 2)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "composition_reweighted_all.csv", index=False)
    print("\nexternal drop and composition reweighting, encoders vs char vs SVM")
    print(comp.to_string(index=False))

    # encoder accuracy by token count on the clean external subset
    etok = ext_e.name.str.split().str.len().values
    erows = []
    for k in (1, 2, 3, 4):
        sel = ((etok == k) if k < 4 else (etok >= 4)) & dedup_e
        if sel.sum() < 20:
            continue
        row = {"tokens": "4 or more" if k == 4 else str(k), "n_names": int(sel.sum())}
        for m in ENC_MODELS:
            row[m] = round(acc_names(ext_e, m, sel, "pred_").mean() * 100, 2)
        erows.append(row)
    pd.DataFrame(erows).to_csv(OUT / "encoder_accuracy_by_token_count.csv", index=False)
    print("\nencoder accuracy by token count, external clean subset, percent")
    print(pd.DataFrame(erows).to_string(index=False))

    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
