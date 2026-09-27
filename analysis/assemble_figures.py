from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "figures"
OUT = SRC / "manuscript"
ATTIC = ROOT / "_archive" / "v3i_old_figures"
SOURCE = SRC / "_source"

# The manuscript ships fourteen figures, already numbered in results/figures/.
# Each source name below is the shipped file; this list is the numbering authority
# and matches the figure order in the paper.
ORDER = [
    (1, "fig01_name_length.png", "name length in characters and in tokens"),
    (2, "fig02_label_share.png", "label proportions across the three partitions"),
    (3, "fig03_suffix.png", "suffixes ranked by conditional gender probability"),
    (4, "fig04_first_token.png", "most frequent first tokens by gender"),
    (5, "fig05_training_curves.png", "training and development F1 across epochs"),
    (6, "fig06_confusion.png", "confusion matrices, CharBiGRU and CharBiLSTM"),
    (7, "fig07_efficiency_params.png", "F1 against parameter count"),
    (8, "fig08_attention_position_char.png", "character-level attention by position"),
    (9, "fig09_attention_position_word.png", "word-level attention, first token against last"),
    (10, "fig10_attention_reading.png", "attention over two constructed names"),
    (11, "fig11_external_validation.png", "F1 across the three versions of the benchmark"),
    (12, "fig12_error_profile.png", "how often a name is missed, and by which gender"),
    (13, "fig13_efficiency_latency.png", "F1 against single-thread CPU latency"),
    (14, "fig14_sensitivity_sweep.png", "every configuration in the sensitivity sweep"),
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

    SOURCE.mkdir(parents=True, exist_ok=True)
    for p in sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.pdf")):
        stem = re.sub(r"^(?:\d+_|fig_)", "", p.name)
        shutil.move(str(p), str(SOURCE / stem))

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
