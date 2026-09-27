"""Training and development F1 curves, Figure 7 rebuilt from the current run.

The submitted version of this figure came from a single run under the earlier
protocol, and its per-epoch history was never kept, so it could not be redrawn
against the numbers the revision reports. The grid now records history for every
fit, which means the curve can show the spread across five seeds instead of one
trajectory, and the training F1 is measured over the whole training partition
rather than estimated from a sample.

Curves are truncated at the epoch each run stopped, so a line ending early means
early stopping fired there rather than the model diverging.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "final" / "24_grid_attention_pooling"
FIGS = ROOT / "results" / "figures"
OUT = SRC
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
COL = {"CharBiRNN": "#1f4e79", "CharBiGRU": "#2e7d32", "CharBiLSTM": "#b8860b",
       "CharTransformer": "#8b1a1a", "WordBiRNN": "#1f4e79", "WordBiGRU": "#2e7d32",
       "WordBiLSTM": "#b8860b", "WordTransformer": "#8b1a1a"}


# One JOIV column is 3.403 inches on the current template. The two panels sit
# side by side rather than stacked, which keeps the figure short enough to share
# a column with running text. COL is already the colour map in this module,
# hence the separate name.
COLW = 3.403
COLH = 1.70
WPAD = 1.5
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 6.4})

def band(ax, h, models, col, style, label_suffix):
    for m in models:
        g = h[h.Model == m]
        if g.empty:
            continue
        piv = g.pivot(index="epoch", columns="Seed", values=col)
        mean = piv.mean(axis=1)
        lo, hi = piv.min(axis=1), piv.max(axis=1)
        x = mean.index.values
        ax.plot(x, mean.values, style, color=COL[m], linewidth=1.3,
                label=f"{m}{label_suffix}")
        ax.fill_between(x, lo.values, hi.values, color=COL[m], alpha=0.12, linewidth=0)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(SRC / "training_history.csv")
    tcol = "train_f1" if "train_f1" in h.columns else "train_f1_subsample"
    if tcol != "train_f1":
        print(f"warning: history carries {tcol}, not a full-partition training F1")
    print(f"{len(h):,} epoch rows | {h.Model.nunique()} models | seeds {sorted(h.Seed.unique())}")

    stop = h.groupby(["Model", "Seed"]).epoch.max().reset_index()
    s = stop.groupby("Model").epoch.agg(["mean", "min", "max"]).round(1)
    s.to_csv(OUT / "stopping_epoch_summary.csv")
    print("\nepoch at which each run stopped\n" + s.to_string())

    # bracket access, not attribute access. A column named "last" would resolve
    # to the DataFrame method of that name and silently compare against a bound
    # method, which is how this table came out empty the first time.
    gap = h.merge(stop.rename(columns={"epoch": "stop_epoch"}), on=["Model", "Seed"])
    final = gap[gap["epoch"] == gap["stop_epoch"]]
    fs = final.groupby("Model").agg(train_f1=(tcol, "mean"), dev_f1=("dev_f1", "mean")).round(4)
    fs["gap_pp"] = ((fs.train_f1 - fs.dev_f1) * 100).round(2)
    fs.to_csv(OUT / "train_dev_gap.csv")
    print("\nfit at the stopping epoch, mean over five seeds\n" + fs.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(COLW, COLH), sharey=True)
    for ax, models, title in ((axes[0], CHAR, "(a)"), (axes[1], WORD, "(b)")):
        band(ax, h, models, tcol, "-", "")
        band(ax, h, models, "dev_f1", "--", "")
        ax.set_xlabel("Epoch")
        # Panel letter inside the top-left corner rather than as a title above the
        # axes. The title reserved a strip of height on top of each panel, and
        # dropping it keeps the figure short without a separate label row. The
        # curves sit low at the first epoch, so the corner stays clear.
        ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.6)
        # Round 2: angka sumbu 6.4 pt seperti figure lain (7.5 lebih besar dari label sumbu)
        ax.tick_params(labelsize=6.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    # Round 2 (Ers): kedua panel memakai label dan angka sumbu y, supaya tiap panel
    # terbaca sendiri. sharey menyembunyikan angka di panel kanan, jadi dinyalakan lagi.
    for ax in axes:
        ax.set_ylabel("F1")
        ax.yaxis.set_tick_params(labelleft=True)

    handles = [plt.Line2D([], [], color=COL[m], linewidth=1.3,
                          label=m.replace("Char", "").replace("Word", "")) for m in CHAR]
    handles += [plt.Line2D([], [], color="0.3", linewidth=1.3, label="training"),
                plt.Line2D([], [], color="0.3", linewidth=1.3, linestyle="--",
                           label="development")]
    # Split across the panels. One legend of six entries filled most of a panel
    # once the figure became short, so the four model colours sit in (a) and the
    # two line styles in (b), each in the free corner of its own panel.
    axes[0].legend(handles=handles[:4], fontsize=6.0, frameon=True,
                   framealpha=1.0, edgecolor="0.8", ncol=1,
                   loc="lower right", handlelength=1.8, labelspacing=0.22,
                   borderpad=0.3, borderaxespad=0.2)
    axes[1].legend(handles=handles[4:], fontsize=6.0, frameon=True,
                   framealpha=1.0, edgecolor="0.8", ncol=1,
                   loc="lower right", handlelength=1.8, labelspacing=0.22,
                   borderpad=0.3, borderaxespad=0.2)
    # Round 2 (Ers): jarak antar panel dilebarkan, angka 40 di sumbu (a) tadinya
    # hampir menempel ke sumbu (b)
    fig.tight_layout(pad=0.7, w_pad=WPAD)
    # No repositioning here. That step pinned each letter to the left edge of the
    # figure, which was right while the panels were stacked and put both letters
    # on top of each other once they sat side by side. Each letter now stays at
    # the left edge of its own panel, where ax.set_title(loc="left") puts it.
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig05_training_curves.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {FIGS / 'fig05_training_curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
