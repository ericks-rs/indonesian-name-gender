from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
GRID = FINAL / "24_grid_attention_pooling"
FIGS = ROOT / "results" / "figures"
MIRROR = FINAL / "13_figures"
DPI = 600
T_CRIT = 2.776

CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
SEEDS = [42, 7, 123, 2024, 777]
CCHAR, CWORD, CCLASS, CPRE = "#1f4e79", "#c98b3a", "#5b8c5a", "#8b1a1a"

COL = 3.4
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 6.4})

def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)
    for d in (FIGS, MIRROR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white",
                    pad_inches=0.08)

    fig.savefig((FIGS / name).with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"  {name}")

def load_all() -> pd.DataFrame:
    g = pd.read_csv(GRID / "multiseed_summary.csv")
    rows = [{"model": r.Model, "family": "character" if r.Model.startswith("Char") else "word",
             "accuracy": r.Accuracy_mean, "precision": r.Precision_mean,
             "recall": r.Recall_mean, "f1": r.F1_mean, "f1_sd": r.F1_std}
            for r in g.itertuples()]
    t = pd.read_csv(FINAL / "03_seeds_tfidf" / "tfidf_seed_summary.csv")
    rows += [{"model": r.Model, "family": "classical", "accuracy": r.accuracy_mean,
              "precision": r.precision_mean, "recall": r.recall_mean,
              "f1": r.f1_mean, "f1_sd": r.f1_std} for r in t.itertuples()]
    p = pd.read_csv(FINAL / "04_seeds_transformers" / "transformer_seed_summary.csv")
    rows += [{"model": r.Model, "family": "pretrained", "accuracy": r.val_accuracy_mean,
              "precision": r.val_precision_mean, "recall": r.val_recall_mean,
              "f1": r.val_f1_mean, "f1_sd": r.val_f1_std} for r in p.itertuples()]
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

def fig5(df):
    col = {"character": CCHAR, "word": CWORD, "classical": CCLASS, "pretrained": CPRE}
    order = df.model.tolist()
    y = np.arange(len(order))
    colours = [col[f] for f in df.family]
    for name, pair in (("fig_metrics_accuracy_precision.png", ["accuracy", "precision"]),
                       ("fig_metrics_recall_f1.png", ["recall", "f1"])):
        fig, axes = plt.subplots(2, 1, figsize=(COL, 5.6), sharex=False)
        for mi, (ax, m) in enumerate(zip(axes, pair)):
            v = df[m].values * 100
            ax.barh(y, v, 0.74, color=colours, edgecolor="white", linewidth=0.4)
            ax.set_xlim(min(v) - 3, max(v) + 2.4)
            ax.set_title(f"({chr(97 + mi)}) {m}", fontsize=7.6, loc="left")
            ax.set_yticks(y)
            ax.set_yticklabels(order, fontsize=5.8)
            ax.invert_yaxis()
            ax.set_xlabel("percent")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", color="0.9", linewidth=0.5)
            ax.set_axisbelow(True)
            for yi, vi in zip(y, v):
                ax.annotate(f"{vi:.1f}", (vi, yi), xytext=(2.4, -1.8),
                            textcoords="offset points", fontsize=5.2)
        fig.tight_layout(pad=0.5)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in col.items()]
        fig.legend(handles=handles, fontsize=5.6, frameon=False, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 1.004),
                   handlelength=1.1, handletextpad=0.4, columnspacing=1.1)
        save(fig, name)

def fig6(df):
    d = df[df.family.isin(["character", "word"])].sort_values("f1")
    y = np.arange(len(d))
    err = d.f1_sd.values * 100 * T_CRIT / np.sqrt(len(SEEDS))
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    colors = [CCHAR if f == "character" else CWORD for f in d.family]
    left = (d.f1.values * 100 - err).min() - 0.25
    ax.hlines(y, left, d.f1.values * 100, color=colors,
              linewidth=1.1, alpha=0.55)
    ax.errorbar(d.f1.values * 100, y, xerr=err, fmt="none", ecolor="0.35",
                elinewidth=0.9, capsize=2.4)
    ax.scatter(d.f1.values * 100, y, s=46, color=colors, zorder=3,
               edgecolor="white", linewidth=0.7)
    lab = d.f1.values * 100 + err
    for yi, xi, vi in zip(y, lab, d.f1.values * 100):
        ax.annotate(f"{vi:.2f}", (xi, yi), xytext=(4.5, -2.0),
                    textcoords="offset points", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(d.model, fontsize=6.2)
    ax.set_xlabel("F1 on the 2024 to 2025 partition (%), five seeds")
    ax.set_xlim(left, lab.max() + 0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    save(fig, "fig_neural_f1_lollipop.png")

def fig8(pred):
    y = pred.label.values

    for name, pair in (("fig_confusion_birnn_bigru.png", CHAR[:2]),
                       ("fig_confusion_bilstm_transformer.png", CHAR[2:])):
        fig, axes = plt.subplots(2, 1, figsize=(COL, 4.4))
        for ai, (ax, m) in enumerate(zip(axes, pair)):
            cm = np.mean([confusion_matrix(y, pred[f"{m}__seed{s}"].values)
                          for s in SEEDS], axis=0)
            pct = cm / cm.sum() * 100
            cm = np.rint(cm).astype(int)
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=55)
            for i in range(2):
                for j in range(2):
                    ax.annotate(f"{cm[i, j]:,}\n{pct[i, j]:.1f}%", (j, i), ha="center",
                                va="center", fontsize=6.8,
                                color="white" if pct[i, j] > 30 else "0.15")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["male", "female"])
            ax.set_yticklabels(["male", "female"], rotation=90, va="center")
            ax.set_title(f"({chr(97 + ai)}) {m}", fontsize=7.6, loc="left")
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
        fig.tight_layout(pad=0.5)
        save(fig, name)

    rows = []
    for m in CHAR + WORD:
        cm = np.mean([confusion_matrix(y, pred[f"{m}__seed{s}"].values)
                      for s in SEEDS], axis=0)
        tn, fp, fn, tp = cm.ravel()
        rows.append({"Model": m, "level": "character" if m in CHAR else "word",
                     "true_negative": round(float(tn), 1), "false_positive": round(float(fp), 1),
                     "false_negative": round(float(fn), 1), "true_positive": round(float(tp), 1),
                     "fn_over_fp": round(float(fn / fp), 2)})
    d = pd.DataFrame(rows)
    (MIRROR / "tables").mkdir(parents=True, exist_ok=True)
    d.to_csv(MIRROR / "tables" / "confusion_counts.csv", index=False)
    print(d.to_string(index=False))

def fig9(pred, names):
    ntok = np.array([len(str(s).lower().split()) for s in names])
    bins = np.clip(ntok, 1, 5)
    y = pred.label.values
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for models, colour, label in ((CHAR, CCHAR, "character"), (WORD, CWORD, "word")):
        acc = []
        for b in range(1, 6):
            m_ = bins == b
            vals = [(pred[f"{mm}__seed{s}"].values[m_] == y[m_]).mean()
                    for mm in models for s in SEEDS]
            acc.append(np.array(vals))
        mean = np.array([a.mean() * 100 for a in acc])
        lo = np.array([a.min() * 100 for a in acc])
        hi = np.array([a.max() * 100 for a in acc])
        x = np.arange(1, 6)
        ax.plot(x, mean, "-o", color=colour, markersize=4.4, linewidth=1.4, label=label)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
    counts = [int((bins == b).sum()) for b in range(1, 6)]
    for xi, c in zip(range(1, 6), counts):
        ax.annotate(f"n={c:,}", (xi, ax.get_ylim()[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=6.6, color="0.4")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1", "2", "3", "4", "5+"])
    ax.set_xlabel("tokens in the name")
    ax.set_ylabel("accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 1.004),
               handletextpad=0.4, columnspacing=1.4)
    save(fig, "fig_accuracy_by_token_count.png")

def fig13():
    c = pd.read_csv(FINAL / "21_external_clean" / "external_clean_summary.csv")
    g = pd.read_csv(GRID / "multiseed_summary.csv").set_index("Model")
    t = pd.read_csv(FINAL / "03_seeds_tfidf" / "tfidf_seed_summary.csv").set_index("Model")
    p = pd.read_csv(FINAL / "04_seeds_transformers" /
                    "transformer_seed_summary.csv").set_index("Model")
    internal = {}
    for m in c.Model:
        if m in g.index:
            internal[m] = g.loc[m, "F1_mean"]
        elif m in t.index:
            internal[m] = t.loc[m, "f1_mean"]
        else:
            internal[m] = p.loc[m, "val_f1_mean"]
    c["internal"] = c.Model.map(internal)
    c = c.sort_values("clean_dedup", ascending=False)
    yy = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(COL, 4.6))
    ax.barh(yy - 0.26, c.internal * 100, 0.25, color="#41618c",
            label="2024 to 2025 partition", edgecolor="white", linewidth=0.4)
    ax.barh(yy, c["full"] * 100, 0.25, color="#c98b3a", label="public benchmark",
            edgecolor="white", linewidth=0.4)
    ax.barh(yy + 0.26, c["clean_dedup"] * 100, 0.25, color="#8b6a3a",
            label="benchmark, 1,464 distinct names", edgecolor="white", linewidth=0.4)
    ax.set_yticks(yy)
    ax.set_yticklabels(c.Model, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (%)")
    ax.set_xlim(80, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 1.004), fontsize=5.6,
               handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig_external_validation.png")

def main() -> int:
    df = load_all()
    pred = pd.read_csv(GRID / "val_predictions.csv")
    print(f"{len(df)} classifiers, {len(pred):,} evaluation names")
    print("figures written")
    fig5(df)
    fig6(df)
    fig8(pred)
    fig9(pred, pred.name.values)
    fig13()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
