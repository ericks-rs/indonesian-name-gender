from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
FINAL = ROOT / "results" / "final"
GRID = FINAL / "24_grid_attention_pooling"
OUT = FINAL / "27_error_analysis"
CHAR = ["CharBiRNN", "CharBiGRU", "CharBiLSTM", "CharTransformer"]
SEEDS = [42, 7, 123, 2024, 777]

NGRAM_MAX, NGRAM_MIN = 4, 2
MIN_SUPPORT = 30
STRONG = 0.75

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "data" / "splits" / "train_1990_2021.csv")
    pred = pd.read_csv(GRID / "val_predictions.csv")
    err = pd.read_csv(OUT / "per_name_errors.csv")

    tl = tr.NAMA.astype(str).str.lower()
    ty = (tr.LABEL == "P").astype(int).values
    tot, fem = {}, {}
    for k in range(NGRAM_MIN, NGRAM_MAX + 1):
        e = tl.str[-k:]
        tot[k] = Counter(e)
        fem[k] = Counter(e[ty == 1])

    def pick_ending(name: str):
        for k in range(NGRAM_MAX, NGRAM_MIN - 1, -1):
            e = name[-k:]
            if tot[k].get(e, 0) >= MIN_SUPPORT:
                return e, tot[k][e], fem[k][e] / tot[k][e]
        e = name[-NGRAM_MAX:]
        n = tot[NGRAM_MAX].get(e, 0)
        return e, n, (fem[NGRAM_MAX][e] / n if n else float("nan"))
    seen_tokens = set()
    first_tot, first_fem = Counter(), Counter()
    for n, y in zip(tl, ty):
        parts = n.split()
        seen_tokens.update(parts)
        if parts:
            first_tot[parts[0]] += 1
            first_fem[parts[0]] += y

    hard = err[err.n_wrong_of_20 == 20].copy()
    cols = [f"{m}__seed{s}" for m in CHAR for s in SEEDS]
    pm = pred.set_index("name").loc[hard.name.values]
    hard["predicted"] = pm[cols[0]].values
    hard["low"] = hard.name.str.lower()
    picked = hard.low.map(pick_ending)
    hard["ending"] = [p[0] for p in picked]
    hard["ending_len"] = hard.ending.str.len()
    hard["ending_n"] = [p[1] for p in picked]
    hard["ending_female_share"] = [p[2] for p in picked]
    hard["unseen_tokens"] = hard.low.map(lambda s: sum(t not in seen_tokens for t in s.split()))
    hard["first_token"] = hard.low.str.split().str[0]
    hard["first_n"] = hard.first_token.map(lambda t: first_tot.get(t, 0))
    hard["first_female_share"] = hard.apply(
        lambda r: (first_fem.get(r.first_token, 0) / r.first_n) if r.first_n else np.nan, axis=1)
    hard["ends_in_initial"] = hard.low.str.split().str[-1].str.len() == 1
    hard["type"] = np.where((hard.predicted == 1) & (hard.label == 0), "false positive",
                            "false negative")

    def reason(r):
        share, fshare = r.ending_female_share, r.first_female_share
        if r.ends_in_initial:
            return "name ends in a single-letter initial, so the final characters carry no suffix"
        if r.ending_n == 0:
            return "ending never seen in training"
        if r.ending_n < MIN_SUPPORT:
            return f"ending seen only {int(r.ending_n)} times in training"

        if r.label == 1 and share <= 1 - STRONG:
            return (f"ending is {(1 - share) * 100:.0f} percent male across "
                    f"{int(r.ending_n)} training names")
        if r.label == 0 and share >= STRONG:
            return (f"ending is {share * 100:.0f} percent female across "
                    f"{int(r.ending_n)} training names")

        if r.first_n >= MIN_SUPPORT and r.label == 1 and fshare <= 1 - STRONG:
            return (f"first token {r.first_token} is {(1 - fshare) * 100:.0f} percent male "
                    f"across {int(r.first_n)} training names, against the ending")
        if r.first_n >= MIN_SUPPORT and r.label == 0 and fshare >= STRONG:
            return (f"first token {r.first_token} is {fshare * 100:.0f} percent female "
                    f"across {int(r.first_n)} training names, against the ending")
        if r.unseen_tokens == r.n_tokens:
            return "no token appears in training"
        if r.n_tokens == 1:
            return "single token, no surrounding context"
        if 0.4 <= share <= 0.6:
            return (f"ending is balanced at {share * 100:.0f} percent female across "
                    f"{int(r.ending_n)} training names")
        pts = "female" if share > 0.5 else "male"
        pct = share * 100 if share > 0.5 else (1 - share) * 100
        return (f"ending is {pct:.0f} percent {pts} across {int(r.ending_n)} training names, "
                f"which agrees with the true label, so the error comes from elsewhere")

    hard["reason"] = hard.apply(reason, axis=1)

    def category(r):
        share, fshare = r.ending_female_share, r.first_female_share
        if r.ends_in_initial:
            return "ends in an initial"
        if r.ending_n == 0:
            return "ending never seen"
        if r.ending_n < MIN_SUPPORT:
            return "ending rare"
        if (r.label == 1 and share <= 1 - STRONG) or (r.label == 0 and share >= STRONG):
            return "misleading ending"
        if r.first_n >= MIN_SUPPORT and (
                (r.label == 1 and fshare <= 1 - STRONG) or
                (r.label == 0 and fshare >= STRONG)):
            return "first token overrides the ending"
        if r.unseen_tokens == r.n_tokens:
            return "no token seen"
        if r.n_tokens == 1:
            return "single token"
        if 0.4 <= share <= 0.6:
            return "ending balanced"
        return "ending agrees, cause elsewhere"

    hard["category"] = hard.apply(category, axis=1)
    keep = ["name", "type", "category", "label", "predicted", "n_tokens", "ending", "ending_len",
            "ending_n", "ending_female_share", "first_token", "first_n",
            "first_female_share", "unseen_tokens", "reason"]
    hard[keep].sort_values(["type", "ending_n"], ascending=[True, False]).to_csv(
        OUT / "always_wrong_with_reason.csv", index=False)

    print(f"{len(hard)} names missed by all twenty fits, "
          f"{int((hard.type == 'false positive').sum())} false positives and "
          f"{int((hard.type == 'false negative').sum())} false negatives\n")
    print("ending length chosen, longest still backed by "
          f"{MIN_SUPPORT} training names")
    for k, v in hard.ending_len.value_counts().sort_index().items():
        print(f"  {int(k)} characters  {v:>4} names")
    print()
    print("misleading-ending count as the marker threshold moves")
    for st in (0.65, 0.70, 0.75, 0.80, 0.85):
        m = (((hard.label == 1) & (hard.ending_female_share <= 1 - st)) |
             ((hard.label == 0) & (hard.ending_female_share >= st)))
        mark = "  <- reported" if abs(st - STRONG) < 1e-9 else ""
        print(f"  threshold {st:.2f}  {int(m.sum()):>3} of {len(hard)}{mark}")
    print()
    print("categories")
    vc = hard.category.value_counts()
    for k, v in vc.items():
        print(f"  {v:>4}  {k}")
    if int(vc.sum()) != len(hard):
        raise SystemExit("categories do not partition the set")

    cats = {"misleading ending": r"percent (?:male|female) across \d+ training names$",
            "first token overrides the ending": r"^first token",
            "ending rare in training": r"only \d+ times",
            "ending never seen": r"never seen",
            "ending agrees, cause is elsewhere": r"comes from elsewhere",
            "ending balanced": r"balanced at",
            "ends in an initial": r"single-letter initial",
            "no token seen": r"no token appears",
            "single token": r"single token, no"}
    lines = ["# Every name missed by all twenty character-level fits", "",
             f"{len(hard)} names, {int((hard.type == 'false positive').sum())} false "
             f"positives and {int((hard.type == 'false negative').sum())} false negatives. "
             "A false negative is a female name called male.", "",
             "Pick the rows you want and put the names, one per line, in "
             "`error_examples_selected.txt` next to this file. Any name listed there is "
             "used verbatim and in that order. Delete the file to fall back on the "
             "automatic choice.", ""]
    for t in ("false negative", "false positive"):
        lines += [f"## {t}", ""]
        sub = hard[hard.type == t]
        for cat, pat in cats.items():
            g = sub[sub.reason.str.contains(pat, regex=True)]
            if g.empty:
                continue
            lines += [f"### {cat}, {len(g)} names", "",
                      "| name | tokens | ending | ending n | pct female | reason |",
                      "|---|---:|---|---:|---:|---|"]
            for r in g.sort_values("ending_n", ascending=False).itertuples():
                pf = "" if r.ending_female_share != r.ending_female_share                     else f"{r.ending_female_share * 100:.1f}"
                lines.append(f"| {r.name} | {int(r.n_tokens)} | `{r.ending}` | "
                             f"{int(r.ending_n)} | {pf} | {r.reason} |")
            lines.append("")
    (OUT / "error_candidates.md").write_text(chr(10).join(lines), encoding="utf-8")
    print("")
    print(f"candidate listing at {OUT / 'error_candidates.md'}")

    picked = Path(__file__).parent / "error_examples_selected.txt"
    if picked.exists():
        want = [l.strip() for l in picked.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
        idx = hard.set_index(hard.name.str.upper())
        missing = [w for w in want if w.upper() not in idx.index]
        if missing:
            print("")
            print(f"not among the 261 names: {', '.join(missing)}")
        chosen = [idx.loc[w.upper()] for w in want if w.upper() in idx.index]
        table = pd.DataFrame(chosen)
        print("")
        print(f"using the {len(table)} names listed in {picked.name}")
    else:

        rows, per_type = [], 7
        for t in ("false negative", "false positive"):
            sub = hard[(hard.type == t) & ~hard.ends_in_initial]
            order = (sub.category.map(sub.category.value_counts())
                     .sort_values(ascending=False).index)
            sub = sub.loc[order]
            taken, endings = [], set()
            for _ in range(per_type):
                for cat in sub.category.unique():
                    pool = sub[(sub.category == cat) & ~sub.ending.isin(endings)
                               & ~sub.index.isin(taken)]
                    if pool.empty:
                        continue
                    pick = pool.sort_values("ending_n", ascending=False).index[0]
                    taken.append(pick)
                    endings.add(sub.loc[pick, "ending"])
                    if len(taken) == per_type:
                        break
                if len(taken) == per_type:
                    break
            rows.append(sub.loc[taken])
        table = pd.concat(rows)

    table = table.reset_index(drop=True)
    tag = {"false negative": "FN", "false positive": "FP"}
    seen = {"FN": 0, "FP": 0}
    ids = []
    for t in table.type:
        seen[tag[t]] += 1
        ids.append(f"{tag[t]}-{seen[tag[t]]}")
    pub = pd.DataFrame({
        "case": ids,
        "type": table.type.values,
        "category": table.category.values,
        "tokens": table.n_tokens.values,
        "ending": table.ending.values,
        "ending_n_training": table.ending_n.values,
        "ending_pct_female": (table.ending_female_share * 100).round(1).values,
        "true_label": np.where(table.label.values == 1, "female", "male"),
        "predicted": np.where(table.predicted.values == 1, "female", "male"),

        "share_of_all_261_pct": table.category.map(
            (hard.category.value_counts() / len(hard) * 100).round(1)).values,

        "reason": table.reason.str.replace(r"first token \S+ is", "first token is",
                                           regex=True).values})
    pub.to_csv(OUT / "error_examples_table.csv", index=False)
    (OUT / "error_examples_table.md").write_text(
        "# Error examples for the manuscript\n\n"
        "Names are withheld, because a name printed beside a year and a "
        "misclassification is identifying. The key from case identifier to name "
        "stays in "
        "`error_examples_key.csv`, which is local only.\n\n"
        + pub.to_markdown(index=False) + chr(10), encoding="utf-8")

    pd.DataFrame({"case": ids, "name": table.name.values}).to_csv(
        OUT / "error_examples_key.csv", index=False)
    table = pub
    print(f"\nexample table, {len(table)} rows\n")
    print(table.to_string(index=False))
    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
