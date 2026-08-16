from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "final" / "26_attention_position" / "attention_examples.csv"
FIGS = ROOT / "results" / "figures"
MIRROR = ROOT / "results" / "final" / "13_figures"
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
DPI = 600
CMAP = LinearSegmentedColormap.from_list(
    "attn", ["#ffffff", "#fdf1d8", "#f7c96b", "#e88a3c", "#b8431f", "#6d1a10"])

def weights_of(df, name, model):
    r = df[(df.name == name) & (df.model == model)].iloc[0]
    cols = sorted([c for c in df.columns if c.startswith("pos_")],
                  key=lambda s: int(s.split("_")[1]))
    v = r[cols].astype(float).values
    return v[~np.isnan(v)]

def strip(ax, labels, w, vmax, y, h, x0, cw, fs):
    for i, (t, v) in enumerate(zip(labels, w)):
        x = x0 + i * cw
        col = CMAP(min(v / vmax, 1.0))
        ax.add_patch(FancyBboxPatch((x + 0.07 * cw, y), cw * 0.86, h,
                                    boxstyle="round,pad=0,rounding_size=0.010",
                                    facecolor=col, edgecolor="none"))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        ax.text(x + cw / 2, y + h / 2, t.upper(), ha="center", va="center", fontsize=fs,
                color="white" if lum < 0.55 else "#23211f",
                family="DejaVu Sans Mono", weight="bold" if v >= vmax * 0.55 else "normal")

COL = 3.4

def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    examples = df[["name", "gender", "suffix"]].drop_duplicates().fillna("").values.tolist()
    maxlen = max(len(n.lower()) for n, _, _ in examples)
    vmax_c = max(weights_of(df, n, m).max() for n, _, _ in examples for m in CHAR)
    vmax_w = max(weights_of(df, n, m).max() for n, _, _ in examples for m in WORD)

    cw = 0.92 / maxlen
    S = 1 / (1 - 0.087)
    y_cap = (0.925 - 0.087) * S
    h, pad, top = 0.082 * S, 0.013 * S, (0.800 - 0.087) * S
    for fname, batch in (("fig_attention_reading_one.png", examples[:2]),
                         ("fig_attention_reading_two.png", examples[2:])):
        fig = plt.figure(figsize=(COL, 4.8))
        gs = fig.add_gridspec(2, 1, hspace=0.10, left=0.11, right=0.985,
                              top=0.955, bottom=0.10)
        for k, (name, gender, suf) in enumerate(batch):
            ax = fig.add_subplot(gs[k, 0])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            chars, toks = list(name.lower()), name.lower().split()
            n = len(chars)
            x0 = (1 - n * cw) / 2 + 0.03

            tint = "#8c2f39" if gender == "female" else "#1f4e79"
            ax.text(0.5, 0.995, name, ha="center", va="top", fontsize=8.2, weight="bold",
                    color="#23211f", family="DejaVu Sans Mono")
            cap = f"{gender}, suffix -{suf}" if suf else f"{gender}, no suffix marker"
            ax.text(0.5, y_cap, cap, ha="center", va="top", fontsize=6.2, color=tint)

            for i, m in enumerate(CHAR):
                yy = top - i * (h + pad)
                strip(ax, chars, weights_of(df, name, m), vmax_c, yy, h, x0, cw, 5.4)
                ax.text(x0 - 0.012, yy + h / 2, m.replace("Char", ""), ha="right", va="center",
                        fontsize=6.0, color="#3a3a3a")

            if suf:
                sx = x0 + (n - len(suf)) * cw
                ax.add_patch(Rectangle((sx + 0.02 * cw, top - 3 * (h + pad) - 0.014 * S),
                                       len(suf) * cw - 0.04 * cw,
                                       4 * h + 3 * pad + 0.028 * S,
                                       fill=False, edgecolor=tint, linewidth=1.3,
                                       linestyle=(0, (3.5, 2))))
                ax.text(sx + len(suf) * cw / 2, top + h + 0.016 * S, f"-{suf}", ha="center",
                        va="bottom", fontsize=6.4, color=tint, weight="bold")

            yw = top - 4 * (h + pad) - 0.048 * S
            tw = n * cw / len(toks)
            for i, m in enumerate(WORD):
                yy = yw - i * (h + pad)
                strip(ax, toks, weights_of(df, name, m), vmax_w, yy, h, x0, tw, 6.2)
                ax.text(x0 - 0.012, yy + h / 2, m.replace("Word", ""), ha="right",
                        va="center", fontsize=6.0, color="#6a6a6a")
            ax.text(x0 - 0.012, top + h + 0.02 * S, "character level", ha="right",
                    va="bottom", fontsize=5.6, color="0.45", style="italic")
            ax.text(x0 - 0.012, yw + h + 0.014 * S, "word level", ha="right", va="bottom",
                    fontsize=5.6, color="0.45", style="italic")

        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
        cax = fig.add_axes([0.28, 0.058, 0.46, 0.016])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1.0])
        cb.set_ticklabels(["none", "half", "highest"])
        cb.ax.tick_params(labelsize=5.4, length=0)
        cb.outline.set_visible(False)
        fig.text(0.5, 0.012, "attention weight, mean over five seeds, scaled separately "
                 "for the two levels", ha="center", fontsize=5.4, color="0.38")
        for d in (FIGS, MIRROR):
            fig.savefig(d / fname, dpi=DPI, bbox_inches="tight",
                        facecolor="white", pad_inches=0.08)
        fig.savefig((FIGS / fname).with_suffix(".pdf"), bbox_inches="tight",
                    facecolor="white", pad_inches=0.08)
        plt.close(fig)
        print(f"  {fname}")

    print("\nwhere the peak falls, character models")
    inside = 0
    for name, _, suf in examples:
        for m in CHAR:
            w = weights_of(df, name, m)
            k = int(np.argmax(w))
            hit = bool(suf) and k >= len(w) - len(suf)
            inside += hit
            print(f"  {name:<22} {m:<16} character {k+1:>2} of {len(w)}, {w[k]*100:5.1f}%"
                  f"{'   inside the suffix' if hit else ''}")
    marked = sum(1 for _, _, s in examples if s) * len(CHAR)
    print(f"\npeak inside the suffix in {inside} of the {marked} cases where a "
          f"suffix is marked")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
