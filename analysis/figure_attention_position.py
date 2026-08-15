"""Table and figure for the positional attention analysis.

Replaces the entropy material. Entropy could say how evenly a model divided its
attention but not whether the mass landed anywhere the linguistics predicts, and
its ordering among character models did not survive a change of seed. Position
does both, so this builds the artefacts the manuscript needs from it.

Drawn on the base interpreter. matplotlib crashes with an access violation
inside the CUDA environment, so figures are always produced separately from the
runs that need torch.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "final" / "26_attention_position"
FIGS = ROOT / "results" / "figures"
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
WORD = ["WordBiRNN", "WordBiGRU", "WordBiLSTM", "WordTransformer"]
T_CRIT = 2.776  # t(0.975, df = 4)
STYLE = {"CharBiRNN": ("#1f4e79", "o", "-"), "CharBiGRU": ("#2e7d32", "s", "-"),
         "CharBiLSTM": ("#b8860b", "^", "-"), "CharTransformer": ("#8b1a1a", "D", "--")}


# One JOIV column is 3.49 inches. Panels stack downwards at that width.
COL = 3.4
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 6.4})

def holm(p):
    """Holm step-down, returned in the order the p-values came in.

    This was missing. `build_table` ran a t-test per model and wrote the raw
    p-value straight out, and the manuscript then printed that column under the
    heading `Holm p`. Ers caught it on 15 August. The family is the eight
    positional tests, so a model is judged against the whole grid rather than
    against its own representation level.
    """
    import numpy as _np
    p = _np.asarray(p, dtype=float)
    order = _np.argsort(p)
    adj = _np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ci(v):
    v = np.asarray(v, dtype=float)
    return T_CRIT * v.std(ddof=1) / np.sqrt(len(v))


def build_table(per_seed: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    uc = float(uni[uni.level == "char"].uniform_weight.iloc[0])
    uw = float(uni[uni.level == "word"].uniform_weight.iloc[0])
    rows = []
    for m in CHAR + WORD:
        g = per_seed[per_seed.Model == m]
        lvl = g.level.iloc[0]
        u = uc if lvl == "char" else uw
        last, first = g.last_unit_mass.values, g.first_unit_mass.values
        _, p_last = stats.ttest_1samp(last - u, 0.0)
        _, p_ends = stats.ttest_rel(last, first)
        rows.append({
            "Model": m, "Level": lvl,
            "Final unit": round(float(last.mean()), 4),
            "Final unit SD": round(float(last.std(ddof=1)), 4),
            "Final over uniform": round(float(last.mean() / u), 2),
            "Final CI lo": round(float(last.mean() - ci(last)), 4),
            "Final CI hi": round(float(last.mean() + ci(last)), 4),
            "p raw final vs uniform": float(p_last),
            "First unit": round(float(first.mean()), 4),
            "Final minus first": round(float((last - first).mean()), 4),
            "p raw final vs first": float(p_ends),
            "Centre of mass": round(float(g.centre_of_mass.mean()), 4)})
    t = pd.DataFrame(rows)
    # both families are the eight models, one test each
    t["Holm p final vs uniform"] = holm(t["p raw final vs uniform"].values)
    t["Holm p final vs first"] = holm(t["p raw final vs first"].values)
    return t


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(SRC / "profile_char_by_position.csv")
    wprof = pd.read_csv(SRC / "profile_word_by_token.csv")
    per_seed = pd.read_csv(SRC / "attention_position_per_seed.csv")
    uni = pd.read_csv(SRC / "uniform_reference.csv")
    uc = float(uni[uni.level == "char"].uniform_weight.iloc[0])
    uw = float(uni[uni.level == "word"].uniform_weight.iloc[0])

    tab = build_table(per_seed, uni)
    tab.to_csv(SRC / "table_attention_position.csv", index=False)
    show = tab[["Model", "Final unit", "Final unit SD", "Final over uniform",
                "First unit", "Final minus first", "p raw final vs uniform",
                "Holm p final vs uniform", "p raw final vs first",
                "Holm p final vs first", "Centre of mass"]]
    (SRC / "table_attention_position.md").write_text(
        show.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(show.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL, 5.0),)

    for m in CHAR:
        g = prof[prof.Model == m].sort_values("position")
        x = g.position.values
        y = g["mean"].values * 100
        band = g["std"].fillna(0).values * 100 * T_CRIT / np.sqrt(5)
        c, mk, ls = STYLE[m]
        ax1.plot(x, y, ls, color=c, marker=mk, markersize=3.6, linewidth=1.4, label=m)
        ax1.fill_between(x, y - band, y + band, color=c, alpha=0.13, linewidth=0)
    ax1.axhline(uc * 100, color="0.35", linestyle=":", linewidth=1.1)
    ax1.annotate(f"uniform attention, {uc*100:.2f}%", xy=(-6.5, uc * 100),
                 xytext=(0, 5), textcoords="offset points", fontsize=5.6, color="0.3",
                 ha="center")
    ax1.set_xlabel("character position, counting back from the end of the name")
    ax1.set_ylabel("mean attention weight (%)")
    ax1.set_xticks(range(-12, 0))
    ax1.set_xticklabels([str(i) for i in range(-12, -1)] + ["last"], fontsize=5.6)
    ax1.tick_params(axis="y", labelsize=7.5)
    ax1.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    ax1.set_title("(a)", fontsize=7.6, loc="left")
    ax1.spines[["top", "right"]].set_visible(False)

    idx = np.arange(len(WORD))
    firsts = [per_seed[per_seed.Model == m].first_unit_mass.mean() * 100 for m in WORD]
    lasts = [per_seed[per_seed.Model == m].last_unit_mass.mean() * 100 for m in WORD]
    ef = [ci(per_seed[per_seed.Model == m].first_unit_mass.values) * 100 for m in WORD]
    el = [ci(per_seed[per_seed.Model == m].last_unit_mass.values) * 100 for m in WORD]
    ax2.bar(idx - 0.19, firsts, 0.36, yerr=ef, capsize=2.5, color="#4a6fa5",
            edgecolor="white", linewidth=0.5, label="first token", error_kw={"lw": 0.8})
    ax2.bar(idx + 0.19, lasts, 0.36, yerr=el, capsize=2.5, color="#c98b3a",
            edgecolor="white", linewidth=0.5, label="last token", error_kw={"lw": 0.8})
    ax2.axhline(uw * 100, color="0.35", linestyle=":", linewidth=1.1)
    ax2.annotate(f"uniform, {uw*100:.1f}%", xy=(2.55, uw * 100), xytext=(0, 4),
                 textcoords="offset points", fontsize=5.6, color="0.3")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([m.replace("Word", "") for m in WORD], fontsize=6)
    ax2.set_ylabel("mean attention weight (%)")
    ax2.set_ylim(0, 62)
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.legend(fontsize=5.6, frameon=False, loc="upper center",
               ncol=2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
    ax2.set_title("(b)", fontsize=7.6, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_attention_position.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {FIGS / 'fig_attention_position.png'}")
    print(f"table written to {SRC / 'table_attention_position.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
