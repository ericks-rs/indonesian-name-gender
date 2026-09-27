"""Figure 10, what each model reads when it looks at a name.

A matrix of attention weights makes a reader translate coordinates back into
letters before anything can be understood. This journal is about visualization,
so the name is drawn as the name, and each character is shaded by the weight it
received. The suffix lights up on the page rather than in a legend.

The word-level strip below each name uses the same layout at token granularity,
which puts the two reading strategies side by side. Character models pull toward
the end of the name. Word models pull toward the first token.

Weights are the mean over five seeds, computed by attention_position.py. Drawing
is separate because matplotlib exits with an access violation inside the CUDA
environment. Run this on the base interpreter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
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
                                    facecolor=col, edgecolor="none", clip_on=False))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        ax.text(x + cw / 2, y + h / 2, t.upper(), ha="center", va="center", fontsize=fs,
                color="white" if lum < 0.55 else "#23211f",
                family="DejaVu Sans Mono", weight="bold" if v >= vmax * 0.55 else "normal")


# One JOIV column is 3.49 inches.
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
    # Figure 10 carries BANOWATI LARASATI and GATOTKACA WIRAWAN in one figure.
    for fname, batch in (("fig10_attention_reading.png", [examples[0], examples[2]]),):
        panel_h = (8.2 / 72 + 6.2 / 72 + 0.035      # name, caption, gap
                   + 4 * 0.125 + 3 * 0.018          # four character strips
                   + 0.15                           # break before the word rows
                   + 4 * 0.125 + 3 * 0.018)         # four word strips
        band = 0.23        # colour bar and its tick labels, no caption
        between, above = 0.10, 0.06
        fig = plt.figure(figsize=(COL - 0.55,
                                  above + 2 * panel_h + between + band))
        fh = fig.get_size_inches()[1]
        gs = fig.add_gridspec(2, 1, hspace=between / panel_h, left=0.11, right=0.985,
                              top=1 - above / fh, bottom=band / fh)
        for k, (name, gender, suf) in enumerate(batch):
            ax = fig.add_subplot(gs[k, 0])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            chars, toks = list(name.lower()), name.lower().split()
            n = len(chars)
            x0 = (1 - n * cw) / 2 + 0.03

            ph = ax.get_position().height * fh
            title_h, cap_h, gap = 8.2 / 72, 6.2 / 72, 0.035
            strip_h, strip_pad = 0.125, 0.018
            y_cap_p = 1 - (title_h + 0.010) / ph
            top_p = 1 - (title_h + cap_h + gap + strip_h) / ph
            h_p, pad_p = strip_h / ph, strip_pad / ph
            r = h_p / h
            tint = "#8c2f39" if gender == "female" else "#1f4e79"
            ax.text(0.5, 0.995, name, ha="center", va="top", fontsize=8.2, weight="bold",
                    color="#23211f", family="DejaVu Sans Mono")
            cap = f"{gender}, suffix -{suf}" if suf else f"{gender}, no suffix marker"
            ax.text(0.5, y_cap_p, cap, ha="center", va="top", fontsize=6.2, color=tint)

            for i, m in enumerate(CHAR):
                yy = top_p - i * (h_p + pad_p)
                strip(ax, chars, weights_of(df, name, m), vmax_c, yy, h_p, x0, cw, 5.4)
                ax.text(x0 - 0.012, yy + h_p / 2, m.replace("Char", ""), ha="right", va="center",
                        fontsize=6.0, color="#3a3a3a")
            if suf:
                sx = x0 + (n - len(suf)) * cw
                ax.add_patch(Rectangle((sx + 0.02 * cw, top_p - 3 * (h_p + pad_p) - 0.014 * S * r),
                                       len(suf) * cw - 0.04 * cw,
                                       4 * h_p + 3 * pad_p + 0.028 * S * r,
                                       fill=False, edgecolor=tint, linewidth=1.3,
                                       linestyle=(0, (3.5, 2))))
                ax.text(sx + len(suf) * cw / 2, top_p + h_p + 0.016 * S * r, f"-{suf}", ha="center",
                        va="bottom", fontsize=6.4, color=tint, weight="bold")

            yw = top_p - 4 * (h_p + pad_p) - 0.095 * S * r
            tw = n * cw / len(toks)
            for i, m in enumerate(WORD):
                yy = yw - i * (h_p + pad_p)
                strip(ax, toks, weights_of(df, name, m), vmax_w, yy, h_p, x0, tw, 6.2)
                ax.text(x0 - 0.012, yy + h_p / 2, m.replace("Word", ""), ha="right",
                        va="center", fontsize=6.0, color="#6a6a6a")
            ax.text(x0 - 0.012, top_p + h_p + 0.02 * S * r, "character level", ha="right",
                    va="bottom", fontsize=5.6, color="0.45", style="italic")
            ax.text(x0 - 0.012, yw + h_p + 0.004 * S * r, "word level", ha="right", va="bottom",
                    fontsize=5.6, color="0.45", style="italic")

        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
        cax = fig.add_axes([0.28, 0.105 / fh, 0.46, 0.040 / fh])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1.0])
        cb.set_ticklabels(["none", "half", "highest"])
        cb.ax.tick_params(labelsize=5.4, length=0)
        cb.outline.set_visible(False)
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
    marked = sum(1 for _, _, s in examples if s) * len(CHAR)
    print(f"peak inside the suffix in {inside} of the {marked} marked cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
