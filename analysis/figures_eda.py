"""Figures 1 to 4 (corpus statistics) and the descriptive-statistics tables.

Produces fig01_name_length, fig02_label_share, fig03_suffix and fig04_first_token
under results/figures (PNG and PDF at 600 dpi), and writes the suffix, first-token
and vocabulary tables the manuscript quotes into results/final/13_figures/tables.

The gender legends carry a light box, matching the other figures. Labels read
male and female rather than the registry codes. Runs on the base interpreter,
because matplotlib segfaults in the GPU environment. Reads data/splits and
nothing else.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "splits"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "final" / "13_figures" / "tables"
COL, DPI = 3.4, 600
BLUE, RED = "#4285F4", "#EA4335"
# the shared box that every legend in the manuscript now carries
LEG = dict(frameon=True, framealpha=1.0, edgecolor="0.8")
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.4})
SEX = {"L": "male", "P": "female"}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  {name}.png")


def suffix3(name):
    t = str(name).lower().split()[-1]
    return t[-3:]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA / "train_1990_2021.csv")
    dv = pd.read_csv(DATA / "dev_2022_2023.csv")
    va = pd.read_csv(DATA / "val_2024_2026.csv")
    full = pd.read_csv(DATA / "strict_clean_1990_2026.csv")
    print(f"Train {len(tr):,} | Dev {len(dv):,} | Test {len(va):,} | Corpus {len(full):,}")

    # Fig 1: name length in characters (a) and in tokens (b)
    lens = full.assign(N_CHARS=full.NAMA.str.len(), N_WORDS=full.NAMA.str.split().apply(len))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.15))
    for pi, (ax, col, xlabel) in enumerate(zip(axes, ["N_CHARS", "N_WORDS"],
                                               ["Length in characters", "Length in tokens"])):
        for label, color in [("L", BLUE), ("P", RED)]:
            s = lens[lens.LABEL == label][col]
            ax.hist(s, bins=range(1, int(s.max()) + 2), alpha=0.6, color=color,
                    label=SEX[label], density=True)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({chr(97 + pi)})", fontsize=7.6, loc="left")
    axes[0].set_ylabel("Density")
    axes[1].set_xticks(range(1, 9))
    axes[0].set_xticks([0, 10, 20, 30, 40])
    axes[1].legend(**LEG, ncol=1, loc="upper right", handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig01_name_length")

    # Fig 2: label proportions per partition
    parts = [("Train 1990–2021", tr), ("Dev 2022–2023", dv), ("Test 2024–2026", va)]
    fig, ax = plt.subplots(figsize=(COL, 0.95))
    x = range(len(parts)); w = 0.38
    for off, lab, colour in ((-w / 2, "L", BLUE), (w / 2, "P", RED)):
        vals = [df.LABEL.value_counts(normalize=True).reindex(["L", "P"])[lab] * 100 for _, df in parts]
        b = ax.bar([i + off for i in x], vals, w, color=colour, label=SEX[lab],
                   edgecolor="white", linewidth=0.5)
        ax.bar_label(b, fmt="%.1f", fontsize=5.8, padding=1.0)
    ax.set_xticks(list(x)); ax.set_xticklabels([p[0] for p in parts])
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 95); ax.set_yticks([0, 25, 50, 75])
    ax.legend(**LEG, ncol=2, loc="upper right", handlelength=1.2, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, "fig02_label_share")
    r_tr = (tr.LABEL == "L").sum() / (tr.LABEL == "P").sum()
    print(f"L:P ratio  Train {r_tr:.2f}:1")

    # Fig 3: strongest trigram suffixes per label, dot plot on a zoomed percent scale
    t3 = tr.assign(SUFFIX3=tr.NAMA.apply(suffix3))
    sg = t3.groupby(["SUFFIX3", "LABEL"]).size().unstack(fill_value=0)
    sg["TOTAL"] = sg.sum(axis=1)
    sg = sg[sg.TOTAL >= 50].copy()
    sg["P_RATIO"] = sg["P"] / sg["TOTAL"] * 100
    sg["L_RATIO"] = sg["L"] / sg["TOTAL"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.5))
    for pi, (ax, col, colour, xlabel) in enumerate([
            (axes[0], "P_RATIO", RED, "Female share (%)"),
            (axes[1], "L_RATIO", BLUE, "Male share (%)")]):
        top = sg.nlargest(10, col)[[col, "TOTAL"]]
        y = range(len(top))
        ax.hlines(list(y), 95.8, top[col], color=colour, linewidth=1.0, alpha=0.45)
        ax.plot(top[col], list(y), "o", color=colour, markersize=3.2)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"-{s}" for s in top.index], fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlim(95.8, 101.9)
        ax.set_xticks([96, 98, 100])
        for yi, (v, n) in enumerate(zip(top[col], top.TOTAL)):
            ax.text(100.25, yi, f"n={n:,}", va="center", fontsize=5.4, color="0.3")
        ax.set_xlabel(xlabel)
        ax.text(-0.30, 1.0, f"({chr(97 + pi)})", transform=ax.transAxes,
                fontsize=7.6, va="top", ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.93", linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3, w_pad=2.5)
    save(fig, "fig03_suffix")

    # the suffix shares the dataset-statistics paragraph quotes, top fifteen per label
    sgr = sg.assign(P_RATIO=sg["P"] / sg["TOTAL"], L_RATIO=sg["L"] / sg["TOTAL"])
    top_f = sgr.nlargest(15, "P_RATIO")[["P_RATIO", "TOTAL"]].reset_index()
    top_m = sgr.nlargest(15, "L_RATIO")[["L_RATIO", "TOTAL"]].reset_index()
    top_f.columns = top_m.columns = ["suffix", "share", "n_training"]
    both = pd.concat([top_f.assign(points_to="female"),
                      top_m.assign(points_to="male")], ignore_index=True)
    both["share_pct"] = (both.share * 100).round(2)
    both.drop(columns="share").to_csv(TABLES / "suffix_top15_by_gender.csv", index=False)

    # Fig 4: ten most frequent first tokens, bars split by label
    t4 = tr.assign(FIRST_WORD=tr.NAMA.str.split().str[0])
    fw = t4.groupby(["FIRST_WORD", "LABEL"]).size().unstack(fill_value=0)
    fw["TOTAL"] = fw.sum(axis=1)
    top = fw.nlargest(10, "TOTAL").copy()
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    y = range(len(top))
    ax.barh(y, top["L"], label="male", color=BLUE, height=0.82)
    ax.barh(y, top["P"], left=top["L"], label="female", color=RED, height=0.82)
    ax.set_yticks(list(y)); ax.set_yticklabels(top.index, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("Number of names in the training partition")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(**LEG, ncol=1, loc="lower right", handlelength=1.2, borderaxespad=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, "fig04_first_token")

    # first-token and vocabulary tables the manuscript quotes
    top20 = fw.nlargest(20, "TOTAL").reset_index()
    top20.columns = ["first_token", "male", "female", "total"]
    top20["share_of_training_pct"] = (top20.total / len(tr) * 100).round(3)
    top20["female_pct"] = (top20.female / top20.total * 100).round(2)
    top20.to_csv(TABLES / "first_token_top20.csv", index=False)
    toks = Counter()
    for n in tr.NAMA.astype(str):
        toks.update(n.lower().split())
    word_v = sum(1 for _, c in toks.items() if c >= 2) + 2
    char_v = len({c for n in tr.NAMA.astype(str) for c in n.lower()}) + 2
    pd.DataFrame([{
        "word_vocab_min_freq_2": word_v, "char_vocab": char_v,
        "fold_gap": round(word_v / char_v, 1),
        "word_types_total": len(toks),
        "first_token_types": int(fw.shape[0]),
        "top20_share_of_training_pct": round(float(fw.assign(share=fw.TOTAL / len(tr)).nlargest(20, "TOTAL").share.sum() * 100), 2),
    }]).to_csv(TABLES / "vocabulary_sizes.csv", index=False)
    print(f"\nfigures and tables written to {FIGS} and {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
