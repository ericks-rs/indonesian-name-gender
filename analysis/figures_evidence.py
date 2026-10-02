"""Six figures for claims that so far rest on tables alone.

Each answers a specific reviewer comment. The sensitivity swarm answers the
question of whether a shared configuration is doing the work. The forest plot
puts the confidence intervals the review asked for on a page instead of in a
column. The efficiency frontier turns a latency table into a picture of what the
accuracy costs. The error profile shows who the models fail, which the review
asked to see with examples. The last two support the temporal and imbalance
arguments that currently have no visual at all.

Drawn on the base interpreter. Every value is read from results/final.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
GRID = FINAL / "24_grid_attention_pooling"
FIGS = ROOT / "results" / "figures"
MIRROR = FINAL / "13_figures"
DPI = 600
CCHAR, CWORD, CCLASS, CPRE = "#1f4e79", "#c98b3a", "#5b8c5a", "#8b1a1a"
# One JOIV column is 3.49 inches. Every figure is drawn to that width, panels
# stack downwards, and no figure carries more than two of them.
COL = 3.4
# Per-figure heights for the efficiency frontier. Both were tuned so the saved
# PNG aspect matches the display box set in the manuscript, so Word places the
# figure at full column width with no vertical stretch.
FH_LAT = 3.0
FH_PAR = 2.0141
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 6.4})

CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]


def save(fig, name):
    for d in (FIGS, MIRROR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white",
                    pad_inches=0.08)
    fig.savefig(FIGS / name.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"  {name}")


def place_labels(fig, ax, pts, fontsize=5.2, colour="0.3", avoid=()):
    """Label each point, stepping around it until the text clears everything.

    Every label used to sit at a fixed offset to the right of its marker, which
    is fine until two points land close together. CharBiGRU at 85k parameters
    and CharBiLSTM at 112k differ by 0.04 F1 points, so one label was printed
    through the other. The first candidate here is that same offset to the
    right, so a label only moves when it has to. Call after tight_layout, since
    the test is run in display coordinates and the axes must be at final size.

    Crossing the left or bottom spine is ranked last, since the tick labels and
    the axis title live there. Crossing the right spine is allowed, because that
    side is empty margin and the alternative is worse. mBERT sits 0.08 F1 under
    XLM-R, so on a short panel the only free position beside its own marker is
    the one that overhangs slightly, and a label pushed off its row instead
    reads as an unlabelled point. Anything passed in avoid, a legend drawn
    inside the axes for instance, counts as occupied.
    """
    cand = [(3.4, -1.2, "left", "baseline"), (-3.4, -1.2, "right", "baseline"),
            (0, 4.4, "center", "bottom"), (0, -4.8, "center", "top"),
            (3.4, 4.4, "left", "bottom"), (3.4, -4.8, "left", "top"),
            (-3.4, 4.4, "right", "bottom"), (-3.4, -4.8, "right", "top")]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    frame = ax.get_window_extent(r)
    taken = [a.get_window_extent(r) for a in avoid]
    marks = []
    for x, y, _ in pts:
        px, py = ax.transData.transform((x, y))
        marks.append(Bbox.from_bounds(px - 3.4, py - 3.4, 6.8, 6.8))
    placed = []
    for i in sorted(range(len(pts)), key=lambda k: -pts[k][1]):
        x, y, txt = pts[i]
        blocked = [m for j, m in enumerate(marks) if j != i] + placed + taken
        boxes = []
        for dx, dy, ha, va in cand:
            a = ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=colour, ha=ha, va=va)
            boxes.append(a.get_window_extent(r).expanded(1.04, 1.10))
            a.remove()
        clear = [k for k, bb in enumerate(boxes)
                 if not any(bb.overlaps(o) for o in blocked)]
        inside = [k for k in clear
                  if boxes[k].x0 >= frame.x0
                  and boxes[k].y0 >= frame.y0 and boxes[k].y1 <= frame.y1]
        k = (inside or clear or [0])[0]
        dx, dy, ha, va = cand[k]
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=colour, ha=ha, va=va)
        placed.append(boxes[k])


def fig_sensitivity():
    """Every one of the 112 configurations, scored on the development partition.

    The earlier version plotted the 2024-2026 figure, which meant the comparison
    across configurations was made on the partition the headline table reports.
    The claim now rests on development scores, so it never touches that
    partition, and the test column stays in the source file for transparency."""
    d = pd.read_csv(FINAL / "31_sensitivity_dev" / "sensitivity_dev_and_test.csv")
    d = d.rename(columns={"Dev_F1": "F1"})
    order = CHAR[:3] + ["CharTransformer"] + WORD[:3] + ["WordTransformer"]
    order = [m for m in order if m in set(d.Model)]
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(COL, 3.4))
    for i, m in enumerate(order):
        v = d[d.Model == m].F1.values * 100
        c = CCHAR if m.startswith("Char") else CWORD
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)), v,
                   s=26, color=c, alpha=0.72, edgecolor="white", linewidth=0.4, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="0.25", linewidth=1.4, zorder=4)
    ch = d[d.Level == "char"].F1.min() * 100
    wd = d[d.Level == "word"].F1.max() * 100
    ax.axhspan(wd, ch, color="#2e7d32", alpha=0.08, zorder=1)
    ax.axhline(ch, color="#2e7d32", linewidth=0.9, linestyle=":")
    ax.axhline(wd, color="#2e7d32", linewidth=0.9, linestyle=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([m.replace("Char", "").replace("Word", "") for m in order],
                       fontsize=5.8, rotation=42, ha="right")
    ax.set_ylabel("F1 on the development partition (%)")
    ax.axvline(3.5, color="0.85", linewidth=1.0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    for xc, lab in ((1.5, "character"), (5.5, "word")):
        ax.annotate(lab, xy=(xc, 1.012), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.4, color="0.3")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    save(fig, "fig14_sensitivity_sweep.png")


def fig_forest():
    """Paired character minus word, with the interval over five seeds."""
    a = pd.read_csv(FINAL / "00_summary" / "char_vs_word_paired.csv")
    # panel (b) uses the basis the text reports, one row per distinct name
    b = pd.read_csv(FINAL / "21_external_clean" / "char_vs_word_clean_dedup_paired.csv")
    # Panel (a) belonged to the representation-level subsection and panel (b) to
    # external validation. A figure whose panels are cited from two different
    # subsections is exactly what the manuscript no longer allows, so each gets
    # its own single-column figure.
    for name, d, col in (("fig_paired_forest_test.png", a, CCHAR),
                         ("fig_paired_forest_external.png", b, "#5b7c99")):
        # 3.0 inches matches the lollipop in figures_results.py, so the two
        # figures sit at one height when the manuscript places them together.
        fig, ax = plt.subplots(figsize=(COL, 3.0))
        d = d.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(d))
        ax.hlines(y, d.ci95_lo_pp, d.ci95_hi_pp, color=col, linewidth=2.0)
        ax.scatter(d.diff_pp, y, s=52, color=col, zorder=3, edgecolor="white",
                   linewidth=0.8)
        ax.axvline(0, color="#8b1a1a", linewidth=1.0, linestyle="--")
        for i, r in d.iterrows():
            ax.annotate(f"{r.diff_pp:+.2f} [{r.ci95_lo_pp:.2f}, {r.ci95_hi_pp:.2f}]",
                        (r.ci95_hi_pp, i), xytext=(4, -2.0), textcoords="offset points",
                        fontsize=5.4, color="0.25")
        ax.set_yticks(y)
        ax.set_yticklabels(d.comparison, fontsize=6.2)
        # The caption already says which direction the difference runs, so the
        # axis carries only the unit. The long form was 1.70 inches wide at 7 pt
        # and could not fit a two-inch panel without shrinking the type.
        ax.set_xlabel("F1 percentage points")
        ax.set_xlim(min(-0.4, d.ci95_lo_pp.min() - 0.6), d.ci95_hi_pp.max() * 1.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.5)
        save(fig, name)


def fig_efficiency():
    """What the accuracy costs to serve."""
    b = pd.read_csv(FINAL / "11_inference_benchmark" / "tables" /
                    "inference_benchmark.csv").set_index("Model")
    g = pd.read_csv(GRID / "multiseed_summary.csv").set_index("Model")
    p = pd.read_csv(FINAL / "04_seeds_transformers" /
                    "transformer_seed_summary.csv").set_index("Model")
    pts = []
    for m in b.index:
        if m in g.index:
            f1 = g.loc[m, "F1_mean"]
            fam = "character" if m.startswith("Char") else "word"
        elif m in p.index:
            f1, fam = p.loc[m, "val_f1_mean"], "pretrained"
        else:
            continue
        # the repeated-trial figure where one exists, the single run otherwise,
        # which today means TF-IDF only
        ms = b.loc[m, "CPU_fwd_ms_repeated"] if "CPU_fwd_ms_repeated" in b.columns \
            and pd.notna(b.loc[m, "CPU_fwd_ms_repeated"]) else b.loc[m, "CPU_fwd_ms"]
        pts.append({"model": m, "family": fam, "f1": f1 * 100,
                    "cpu_ms": ms, "params": b.loc[m, "Params"]})
    d = pd.DataFrame(pts)
    col = {"character": CCHAR, "word": CWORD, "pretrained": CPRE}
    # Same split as the forest. Panel (a) was cited from the inference-cost
    # subsection and panel (b) from parameter efficiency.
    # Height is per figure. Type is sized in points and does not shrink with the
    # panel, so a shorter latency figure costs nothing in legibility. What it
    # costs is room between the points, which place_labels absorbs.
    # Height and legend placement are per figure. Both figures now keep the
    # legend inside, in the lower right. The parameter figure used to put its
    # legend above the axes because that corner was said to hold points, which
    # the current data no longer support: the pretrained models sit at the top
    # right, at 95.78 to 96.17 F1, while the axis reaches down to 93.23. An
    # external legend costs height that the corner gives away for nothing.
    for name, xcol, xlab, fh, inside in (
            ("fig13_efficiency_latency.png", "cpu_ms",
             "CPU latency per name (ms, log scale)", FH_LAT, True),
            ("fig07_efficiency_params.png", "params",
             "parameters (log scale)", FH_PAR, True)):
        fig, ax = plt.subplots(figsize=(COL, fh))
        for fam, gg in d.groupby("family"):
            ax.scatter(gg[xcol], gg.f1, s=26, color=col[fam], label=fam,
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("F1 (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
        # Lift the word cluster off the bottom spine. WordBiRNN is the lowest
        # point on both panels, and with no room below its "BiRNN" label was
        # driven onto the x-axis. The headroom lets place_labels seat it clear.
        ax.set_ylim(bottom=d.f1.min() - 0.6)
        if xcol == "cpu_ms":
            # On the latency panel WordBiRNN is also the leftmost point, so pull
            # the cluster in from the left spine as well.
            ax.set_xlim(left=d.cpu_ms.min() * 0.62)
        fig.tight_layout(pad=0.5)
        h, lab = ax.get_legend_handles_labels()
        keep = [f for f in ("character", "word", "pretrained") if f in lab]
        hs = [h[lab.index(f)] for f in keep]
        if inside:
            lg = ax.legend(hs, keep, frameon=True, framealpha=1.0,
                           edgecolor="0.8", loc="lower right", ncol=1,
                           fontsize=5.8, title="tokenization", title_fontsize=6,
                           handletextpad=0.3, borderaxespad=0.6)
        else:
            lg = fig.legend(hs, keep, frameon=False, loc="lower center",
                            bbox_to_anchor=(0.5, 1.004), ncol=len(keep),
                            title="tokenization", title_fontsize=6,
                            handletextpad=0.3, columnspacing=1.2)
        place_labels(fig, ax, [(getattr(r, xcol), r.f1,
                                r.model.replace("Char", "").replace("Word", ""))
                               for r in d.itertuples()], avoid=(lg,))
        save(fig, name)


def fig_errors():
    """Who the models fail, over twenty character-model fits.

    Three panels side by side, one over the limit and far too wide for a column.
    The first two belong together, since both describe the same grouping of
    names by how many fits get them wrong. The third asks a different question
    and becomes its own figure."""
    e = pd.read_csv(FINAL / "27_error_analysis" / "per_name_errors.csv")
    prof = pd.read_csv(FINAL / "27_error_analysis" / "error_group_profile.csv")
    # Side by side rather than stacked. Stacked, the figure stood 3.7 inches tall
    # in the column and pushed the token-count figure onto the next page.
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85))

    ax = axes[0]
    counts = e.n_wrong_of_20.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#41618c", edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("Fits wrong, out of 20")
    ax.set_ylabel("Names (log scale)")
    ax.set_title("(a)", fontsize=7.6, loc="left")
    ax.annotate(f"{int(counts.get(0, 0)):,} always correct", (0, counts.get(0, 1)),
                xytext=(4, -3), textcoords="offset points", fontsize=5.8, color="0.3")
    ax.annotate(f"{int(counts.get(20, 0)):,} always wrong", (20, counts.get(20, 1)),
                xytext=(-4, 4), textcoords="offset points", fontsize=5.8, color="#8b1a1a",
                ha="right")

    ax = axes[1]
    g = prof.set_index("group").loc[["always correct", "disputed", "always wrong"]]
    x = np.arange(3)
    # Orange is reserved for word-level models in the other figures, so the groups
    # use light gray, medium gray, and dark red for errors.
    ax.bar(x, g.pct_female, 0.62, color=["#bdbdbd", "#8c8c8c", "#8b1a1a"],
           edgecolor="white", linewidth=0.5)
    ax.axhline(50, color="0.4", linestyle=":", linewidth=1.0)
    for xi, v in zip(x, g.pct_female):
        ax.annotate(f"{v:.2f}%", (xi, v), xytext=(0, 2.4), textcoords="offset points",
                    ha="center", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["always\ncorrect", "disputed", "always\nwrong"], fontsize=6.0)
    ax.set_ylabel("Female names (%)")
    ax.set_ylim(0, 92)
    ax.set_title("(b)", fontsize=7.6, loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save(fig, "fig12_error_profile.png")

    by = pd.read_csv(FINAL / "27_error_analysis" / "error_rate_by_token_count.csv")
    fig, ax = plt.subplots(figsize=(COL, 1.65))
    ax.bar(by.iloc[:, 0], by.error_rate * 100, 0.62, color="#5b7c99",
           edgecolor="white", linewidth=0.5)
    for xi, v, n in zip(by.iloc[:, 0], by.error_rate * 100, by.n):
        ax.annotate(f"{v:.1f}%\nn={int(n):,}", (xi, v), xytext=(0, 2.4),
                    textcoords="offset points", ha="center", fontsize=5.6)
    ax.set_xlabel("tokens in the name")
    ax.set_ylabel("error rate (%)")
    ax.set_ylim(0, max(by.error_rate * 100) * 1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_error_by_token.png")


def fig_temporal():
    """Accuracy year by year, and what training only on older records costs."""
    # mean of five seeds from the final grid (analysis/per_year_f1_grid.py). The older file in
    # 10_temporal_drift came from a single-seed checkpoint with Transformer mean pooling.
    y = pd.read_csv(FINAL / "43_per_year_f1_grid" / "per_year_f1.csv")
    c = pd.read_csv(FINAL / "10_temporal_drift" / "tables" / "temporal" /
                    "cross_decade_results.csv")
    # Side by side, so the figure does not take most of a column on its own.
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.30))
    ax = axes[0]
    # The per-year file holds one row per model per year, eight models across
    # two years. Handing all sixteen rows to a single plot call joined them into
    # one zigzag with a fractional year axis, which described nothing. The
    # measurement is a paired change per model, so it is drawn as one line per
    # model between the two years.
    ycol = [c_ for c_ in y.columns if c_.lower() == "f1"][0]
    scale = 100 if y[ycol].max() <= 1 else 1
    for m, g in y.groupby("Model"):
        g = g.sort_values("Year")
        colour = CCHAR if str(m).startswith("Char") else CWORD
        ax.plot(g.Year.values, g[ycol].values * scale, "-o", color=colour,
                markersize=3.0, linewidth=1.0, alpha=0.85)
    ax.set_xticks(sorted(y.Year.unique()))
    ax.set_xlim(min(y.Year) - 0.12, max(y.Year) + 0.12)
    # Shortened for the half-column panel. The caption carries the full wording,
    # "the year a name was first recorded".
    ax.set_xlabel("Year first recorded")
    ax.set_ylabel("F1 (%)")
    ax.set_title("(a)", fontsize=7.6, loc="left")
    handles = [plt.Line2D([], [], color=CCHAR, label="character"),
               plt.Line2D([], [], color=CWORD, label="word")]
    # The band between the character and word curves is the only empty part of
    # the panel once it is half a column wide. Lower left sat on a word curve.
    # With the final-grid means, the lowest char line passes through the middle left,
    # so the legend moved to the middle right, in the gap between the char and word groups at 2025.
    ax.legend(handles=handles, frameon=False, ncol=1, loc="center right",
              fontsize=5.6, handlelength=1.2, labelspacing=0.25, borderaxespad=0.3)
    ax = axes[1]
    lab = c.iloc[:, 0].astype(str).values
    val = c[[cc for cc in c.columns if "f1" in cc.lower()][0]].values
    val = val * (100 if np.nanmax(val) <= 1 else 1)
    ax.barh(np.arange(len(lab)), val, 0.6, color=["#9fb4c9", "#5b7c99", CCHAR])
    for i, v in enumerate(val):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(3.4, -1.8), textcoords="offset points",
                    fontsize=6)
    ax.set_yticks(np.arange(len(lab)))
    # year ranges use an en dash, as in the text
    ax.set_yticklabels([l.replace("_full", ", full").replace("-", "\u2013") for l in lab], fontsize=6)
    ax.set_xlim(min(val) - 1.2, max(val) + 1.8)
    ax.set_xlabel("Test F1 (%)")
    ax.set_title("(b)", fontsize=7.6, loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_temporal_drift.png")


def fig_imbalance():
    """Resampling against the class-weighted objective, under the main protocol.

    The earlier version of this figure read `15_imbalance_standby`, a one-seed run
    that selected checkpoints on the test partition. Its class-weighted baseline
    disagreed with the grid the manuscript reports by up to 0.49 points, which made
    the figure argue against the paper. The 120 runs behind the current file use the
    same five seeds, the same development-set selection and one final scoring of the
    test partition, so every bar is a difference between two matched fits."""
    f = FINAL / "40_imbalance_protocol" / "per_model_paired.csv"
    if not f.exists():
        print("  fig_imbalance skipped, the re-run has not produced its table yet")
        return
    d = pd.read_csv(f)
    order = CHAR + WORD
    strategies = [s for s in ("unweighted", "oversampling", "balanced")
                  if s in set(d.Strategy)]
    cols = {"unweighted": "#8c8c8c", "oversampling": "#7b5ea7", "balanced": "#2a9d8f"}
    fig, ax = plt.subplots(figsize=(COL, 3.8))
    yy = np.arange(len(order))
    w = 0.8 / max(len(strategies), 1)
    for i, s in enumerate(strategies):
        g = d[d.Strategy == s].set_index("Model").reindex(order)
        off = (i - (len(strategies) - 1) / 2) * w
        err = np.vstack([g.mean_pp - g.ci95_lo_pp, g.ci95_hi_pp - g.mean_pp])
        ax.barh(yy + off, g.mean_pp, w * 0.9, color=cols[s], label=s,
                edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(g.mean_pp, yy + off, xerr=err, fmt="none", ecolor="0.3",
                    elinewidth=0.6, capsize=1.4, zorder=3)
    ax.axvline(0, color="0.3", linewidth=1.0, zorder=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(order, fontsize=6)
    ax.invert_yaxis()
    # Short label. The long version ("F1 change against class weighting ...") sat
    # right above the caption and read like a second figure title.
    ax.set_xlabel("ΔF1 (percentage points)")
    # Upper left, inside the axes. Every bar runs left from zero and only
    # WordTransformer reaches past one point, so the top left corner is the one
    # empty region, and the legend costs no extra height there.
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=5.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_imbalance_strategies.png")


def main() -> int:
    print("figures written")
    for fn in (fig_sensitivity, fig_forest, fig_efficiency, fig_errors,
               fig_temporal, fig_imbalance):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} FAILED, {e.__class__.__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
