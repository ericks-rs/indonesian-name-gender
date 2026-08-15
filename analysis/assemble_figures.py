"""Collect the current figures into one numbered set for the manuscript.

Figures accumulated under three different protocols and two numbering schemes,
which left duplicate numbers, gaps, and thirty-odd files from runs that no
longer exist sitting in the same directory. Picking the wrong one is a silent
error, so the manuscript set lives in its own folder, is numbered in the order
the paper uses, and is rebuilt from scratch every time this runs.

Anything in results/figures that is not on the list below is stale by
construction and gets moved out of the way.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "figures"
OUT = SRC / "manuscript"
ATTIC = ROOT / "_archive" / "v3i_old_figures"
SOURCE = SRC / "_source"

# Manuscript number, source file, what the caption will be about.
#
# The order is the order of first citation, which is what IEEE and JOIV require
# and what the set did not have. Figures added late took whatever number was
# free, so the reader met 15 before 7 and 12 near the end. Three further rules
# apply from this revision on. Every figure fits one column, no figure carries
# more than two panels, and no figure has panels cited from two different
# subsections. The last rule is why the paired forest and the efficiency
# frontier each became two figures.
ORDER = [
    (1, "02_name_length_distribution.png", "name length in characters and in tokens"),
    (2, "01_label_distribution.png", "label proportions across the three partitions"),
    (3, "03_suffix_analysis_by_gender.png", "suffixes ranked by conditional gender probability"),
    (4, "04_first_word_distribution.png", "most frequent first tokens by gender"),
    (5, "fig_metrics_accuracy_precision.png", "accuracy and precision, fourteen classifiers"),
    (6, "fig_metrics_recall_f1.png", "recall and F1, fourteen classifiers"),
    (7, "fig_neural_f1_lollipop.png", "F1 of the eight neural models with the seed interval"),
    (8, "fig_paired_forest_test.png", "paired character minus word on the test partition"),
    (9, "fig_training_curves.png", "training and development F1 across epochs"),
    (10, "fig_confusion_birnn_bigru.png", "confusion matrices, BiRNN and BiGRU"),
    (11, "fig_confusion_bilstm_transformer.png", "confusion matrices, BiLSTM and Transformer"),
    (12, "fig_efficiency_params.png", "F1 against parameter count"),
    (13, "fig_accuracy_by_token_count.png", "accuracy against the number of tokens"),
    (14, "fig_attention_reading_one.png", "attention over two constructed names"),
    (15, "fig_attention_reading_two.png", "attention over two further constructed names"),
    (16, "fig_attention_position.png", "where attention falls, aggregated over the partition"),
    (17, "fig_paired_forest_external.png", "paired character minus word on the benchmark"),
    (18, "fig_external_validation.png", "F1 across the three versions of the benchmark"),
    (19, "fig_error_profile.png", "how often a name is missed, and by which gender"),
    (20, "fig_error_by_token.png", "error rate against the number of tokens"),
    (21, "fig_efficiency_latency.png", "F1 against single-thread CPU latency"),
    (22, "fig_sensitivity_sweep.png", "every configuration in the sensitivity sweep"),
    (23, "fig_imbalance_strategies.png", "resampling strategies against a weighted objective"),
    (24, "fig_temporal_drift.png", "F1 by registration year and by training window"),
]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ATTIC.mkdir(parents=True, exist_ok=True)

    used, missing = set(), []
    lines = ["| Figure | File | Content |", "|---|---|---|"]
    for n, src, what in ORDER:
        p = SRC / src
        if not p.exists():
            # a previous run already tidied the sources away, so look there too
            p = SOURCE / re.sub(r"^(?:\d+_|fig_)", "", src)
        if not p.exists():
            missing.append(src)
            continue
        used.add(src)
        stem = f"fig{n:02d}_" + src.split("_", 1)[1].replace(".png", "")
        for ext in (".png", ".pdf"):
            q = p.with_suffix(ext)
            if q.exists():
                shutil.copy2(q, OUT / f"{stem}{ext}")
        lines.append(f"| {n} | `{stem}.png` | {what} |")
        print(f"  {n:>2}  {stem}.png   <- {src}")

    moved = 0
    for p in sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.pdf")):
        if p.name in used or p.with_suffix(".png").name in used:
            continue
        shutil.move(str(p), str(ATTIC / p.name))
        moved += 1

    # The producing scripts number their output on their own, and those numbers
    # are not the manuscript's. Two of them collide outright, 14_prior_comparison
    # and 14_sensitivity_sweep, which become figures 18 and 12. Leaving a second
    # numbered set one directory above the real one invites picking the wrong
    # file, so the sources move into their own folder with the number stripped.
    SOURCE.mkdir(parents=True, exist_ok=True)
    for p in sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.pdf")):
        stem = re.sub(r"^(?:\d+_|fig_)", "", p.name)
        shutil.move(str(p), str(SOURCE / stem))
    # The mirror under results/final/13_figures carries the producing scripts'
    # own numbering, where 14_ means two different manuscript figures. Same hazard
    # as the source folder, same treatment. Its CSVs are left alone, since an
    # analysis reads them.
    mirror = ROOT / "results" / "final" / "13_figures"
    renamed = 0
    if mirror.exists():
        for q in sorted(mirror.glob("*.png")) + sorted(mirror.glob("*.pdf")):
            stem = re.sub(r"^(?:\d+_|fig_)", "", q.name)
            if stem != q.name:
                q.replace(mirror / stem)
                renamed += 1
        if renamed:
            print(f"  {renamed} mirrored figure(s) renamed without the old number")

    (SOURCE / "README.md").write_text(
        "Output of the figure scripts, kept unnumbered on purpose. The numbering "
        "the manuscript uses lives in `../manuscript/` and nowhere else. Rerunning "
        "`pipeline/chain_overnight.py` refills this folder.\n", encoding="utf-8")

    (OUT / "INDEX.md").write_text(
        "# Manuscript figures\n\nRebuilt by `pipeline/assemble_figures.py`. "
        "Every panel carries only its letter, since the caption supplies the "
        "description. All files are 600 dpi.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    print(f"\n{len(used)} figures in {OUT}")
    print(f"{moved} stale file(s) moved to {ATTIC}")
    if missing:
        print(f"MISSING: {', '.join(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
