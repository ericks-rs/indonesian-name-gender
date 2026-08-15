from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "splits"
FIGS = ROOT / "results" / "figures"
COPY_TO = ROOT / "results" / "final" / "13_figures"

TABLES = ROOT / "results" / "final" / "13_figures" / "tables"
DPI = 600

BLUE, RED = "#4285F4", "#EA4335"

COL = 3.4
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 6.4})

def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.1)

    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
                pad_inches=0.1)
    plt.close(fig)
    return p

def suffix3(name: str) -> str:
    last = name.strip().split()[-1]
    return last[-3:].lower() if len(last) >= 3 else last.lower()

def main() -> int:
    tr = pd.read_csv(DATA / "train_1990_2021.csv")
    dv = pd.read_csv(DATA / "dev_2022_2023.csv")
    va = pd.read_csv(DATA / "val_2024_2025.csv")
    full = pd.read_csv(DATA / "strict_clean_1990_2025.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    print(f"Train {len(tr):,} | Dev {len(dv):,} | Test {len(va):,} | Corpus {len(full):,}")
    made = []

    parts = [("Train\n1990-2021", tr), ("Dev\n2022-2023", dv), ("Test\n2024-2025", va)]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    x = range(len(parts))
    w = 0.38
    for off, lab, colour in ((-w / 2, "L", BLUE), (w / 2, "P", RED)):

        vals = [df.LABEL.value_counts(normalize=True).reindex(["L", "P"])[lab] * 100
                for _, df in parts]
        b = ax.bar([i + off for i in x], vals, w, color=colour, label=lab,
                   edgecolor="white", linewidth=0.5)
        ax.bar_label(b, fmt="%.1f", fontsize=5.8, padding=1.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([p[0] for p in parts])
    ax.set_ylabel("share of the partition (%)")
    ax.set_ylim(0, 72)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    made.append(save(fig, "01_label_distribution"))
    r_tr = (tr.LABEL == "L").sum() / (tr.LABEL == "P").sum()
    r_dv = (dv.LABEL == "L").sum() / (dv.LABEL == "P").sum()
    r_va = (va.LABEL == "L").sum() / (va.LABEL == "P").sum()
    print(f"L:P ratio  Train {r_tr:.2f}:1  Dev {r_dv:.2f}:1  Test {r_va:.2f}:1")

    lens = full.assign(N_CHARS=full.NAMA.str.len(), N_WORDS=full.NAMA.str.split().apply(len))
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.9))

    for pi, (ax, col, xlabel) in enumerate(zip(axes, ["N_CHARS", "N_WORDS"],
                                               ["length in characters", "length in tokens"])):
        ax.set_title(f"({chr(97 + pi)})", fontsize=7.6, loc="left")
        for label, color in [("L", BLUE), ("P", RED)]:
            s = lens[lens.LABEL == label][col]
            ax.hist(s, bins=range(1, int(s.max()) + 2), alpha=0.6, color=color,
                    label=label, density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    made.append(save(fig, "02_name_length_distribution"))

    tr = tr.assign(SUFFIX3=tr.NAMA.apply(suffix3))
    sg = tr.groupby(["SUFFIX3", "LABEL"]).size().unstack(fill_value=0)
    sg["TOTAL"] = sg.sum(axis=1)
    sg = sg[sg.TOTAL >= 50].copy()
    sg["P_RATIO"] = sg["P"] / sg["TOTAL"]
    sg["L_RATIO"] = sg["L"] / sg["TOTAL"]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 5.4))
    for pi, (ax, col, colour, xlabel) in enumerate([
        (axes[0], "P_RATIO", RED, "P(female | suffix)"),
        (axes[1], "L_RATIO", BLUE, "P(male | suffix)"),
    ]):
        top = sg.nlargest(15, col)[[col, "TOTAL"]]
        ax.barh(top.index, top[col], color=colour, height=0.72)
        ax.set_title(f"({chr(97 + pi)})", fontsize=7.6, loc="left")
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.16)
        for i, (ratio, total) in enumerate(zip(top[col], top.TOTAL)):
            ax.text(ratio + 0.012, i, f"n={total:,}", va="center", fontsize=5.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    made.append(save(fig, "03_suffix_analysis_by_gender"))

    top_f = sg.nlargest(15, "P_RATIO")[["P_RATIO", "TOTAL"]].reset_index()
    top_m = sg.nlargest(15, "L_RATIO")[["L_RATIO", "TOTAL"]].reset_index()
    top_f.columns = top_m.columns = ["suffix", "share", "n_training"]
    both = pd.concat([top_f.assign(points_to="female"),
                      top_m.assign(points_to="male")], ignore_index=True)
    both["share_pct"] = (both.share * 100).round(2)
    both.drop(columns="share").to_csv(TABLES / "suffix_top15_by_gender.csv", index=False)

    tr = tr.assign(FIRST_WORD=tr.NAMA.str.split().str[0])
    fw = tr.groupby(["FIRST_WORD", "LABEL"]).size().unstack(fill_value=0)
    fw["TOTAL"] = fw.sum(axis=1)
    top = fw.nlargest(20, "TOTAL").copy()
    fig, ax = plt.subplots(figsize=(COL, 3.2))
    y = range(len(top))

    ax.barh(y, top["L"], label="L", color=BLUE, height=0.74)
    ax.barh(y, top["P"], left=top["L"], label="P", color=RED, height=0.74)
    ax.set_yticks(list(y))
    ax.set_yticklabels(top.index, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("training names")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    made.append(save(fig, "04_first_word_distribution"))
    fwt = top.reset_index()
    fwt.columns = ["first_token", "male", "female", "total"]
    fwt["share_of_training_pct"] = (fwt.total / len(tr) * 100).round(3)
    fwt["female_pct"] = (fwt.female / fwt.total * 100).round(2)
    fwt.to_csv(TABLES / "first_token_top20.csv", index=False)
    fwt_all = fw.assign(share=fw.TOTAL / len(tr))

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
        "top20_share_of_training_pct": round(float(fwt_all.nlargest(20, "TOTAL").share.sum() * 100), 2),
    }]).to_csv(TABLES / "vocabulary_sizes.csv", index=False)

    COPY_TO.mkdir(parents=True, exist_ok=True)
    for p in made:
        (COPY_TO / p.name).write_bytes(p.read_bytes())
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")
    print(f"\n{len(made)} figures at {DPI} dpi, copies in {COPY_TO}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
