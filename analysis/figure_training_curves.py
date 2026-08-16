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

COLW = 3.4
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

    gap = h.merge(stop.rename(columns={"epoch": "stop_epoch"}), on=["Model", "Seed"])
    final = gap[gap["epoch"] == gap["stop_epoch"]]
    fs = final.groupby("Model").agg(train_f1=(tcol, "mean"), dev_f1=("dev_f1", "mean")).round(4)
    fs["gap_pp"] = ((fs.train_f1 - fs.dev_f1) * 100).round(2)
    fs.to_csv(OUT / "train_dev_gap.csv")
    print("\nfit at the stopping epoch, mean over five seeds\n" + fs.to_string())

    fig, axes = plt.subplots(2, 1, figsize=(COLW, 4.6), sharey=True)
    for ax, models, title in ((axes[0], CHAR, "(a)"), (axes[1], WORD, "(b)")):
        band(ax, h, models, tcol, "-", "")
        band(ax, h, models, "dev_f1", "--", "")
        ax.set_xlabel("epoch")
        ax.set_title(title, fontsize=7.6, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    for a_ in axes:
        a_.set_ylabel("F1")

    handles = [plt.Line2D([], [], color=COL[m], linewidth=1.3,
                          label=m.replace("Char", "").replace("Word", "")) for m in CHAR]
    handles += [plt.Line2D([], [], color="0.3", linewidth=1.3, label="training"),
                plt.Line2D([], [], color="0.3", linewidth=1.3, linestyle="--",
                           label="development")]
    axes[0].legend(handles=handles, fontsize=5.6, frameon=False, ncol=2,
                   loc="lower right", columnspacing=1.0)
    fig.tight_layout(pad=0.7)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        p = ax.get_position()
        ax.set_title(lab, fontsize=7.6, loc="left", x=(0.004 - p.x0) / p.width)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_training_curves.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {FIGS / 'fig_training_curves.png'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
