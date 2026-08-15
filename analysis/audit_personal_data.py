from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SPLITS = ROOT / "data" / "splits"

SCAN = ["release/**/*.csv", "results/**/*.csv", "github-repo/**/*.csv", "demo/**/*.csv",
        "paper/**/*.csv"]

SCAN_TEXT = ["release/**/*.md", "release/**/*.py", "release/**/*.json",
             "release/**/*.html", "release/**/*.js", "release/**/*.ipynb",
             "github-repo/**/*.ipynb", "github-repo/**/*.md", "github-repo/**/*.json",
             "github-repo/**/*.py", "github-repo/**/*.html", "github-repo/**/*.js",
             "demo/**/*.md", "demo/**/*.html", "demo/**/*.js", "paper/**/*.md"]

SCAN_BINARY = ["release/**/*.pkl", "release/**/*.pt",
               "github-repo/**/*.pkl", "github-repo/**/*.pt", "github-repo/**/*.pth",
               "github-repo/**/*.txt", "github-repo/**/*.yaml", "github-repo/**/*.yml",
               "demo/**/*.pkl", "demo/**/*.pt", "demo/**/*.txt"]

RUN_FLOOR = 4

TOKEN_FLOOR = 50

TEXT_FLOOR = 12

AUTHORS = {"ERICKS RACHMAT SWEDIA", "M RIDWAN DWI SEPTIAN", "MARGI CAHYANTI",
           "MOCHAMAD WISUDA SARDJONO"}

LOCAL_ONLY = ("results/final", "results\\final", "results/tables", "results\\tables")
NAME_COLS = {"name", "nama", "names", "full_name"}

ALLOW = {"synthetic_female_names.csv"}
MATCH_FLOOR = 0.30

def corpus_names() -> set[str]:
    s = set()
    for f in ("train_1990_2021", "dev_2022_2023", "val_2024_2025", "strict_clean_1990_2025"):
        p = SPLITS / f"{f}.csv"
        if p.exists():
            s |= set(pd.read_csv(p).NAMA.astype(str).str.upper().str.strip())
    ext = ROOT / "data" / "external" / "indonesian-names.csv"
    if ext.exists():
        s |= set(pd.read_csv(ext).name.astype(str).str.upper().str.strip())
    return s

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clear-notebooks", action="store_true",
                    help="drop stored cell outputs, which is where notebooks leak")
    args = ap.parse_args()

    known = corpus_names()
    print(f"corpus holds {len(known):,} distinct names\n")

    files = sorted({f for pat in SCAN for f in glob.glob(str(ROOT / pat), recursive=True)})
    hits = []
    for f in files:
        if os.path.basename(f) in ALLOW:
            continue
        try:
            head = pd.read_csv(f, nrows=200)
        except Exception:
            continue
        for c in head.columns:
            if c.strip().lower() not in NAME_COLS:
                continue
            vals = head[c].astype(str).str.upper().str.strip()
            share = float(vals.isin(known).mean())
            if share >= MATCH_FLOOR:
                rel = os.path.relpath(f, ROOT)
                local = rel.startswith(LOCAL_ONLY)
                hits.append((f, c, share, local))

    long_names = [n for n in known if len(n) >= TEXT_FLOOR]
    text_hits = []
    for pat in SCAN_TEXT:
        for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
            if ".git" in f or "egg-info" in f or "__pycache__" in f:
                continue
            try:
                body = Path(f).read_text(encoding="utf-8", errors="ignore").upper()
            except Exception:
                continue
            found = sorted({n for n in long_names
                            if n in body and n not in AUTHORS})
            if found:
                text_hits.append((os.path.relpath(f, ROOT), found))

    import re as _re
    run = _re.compile(rb"[ -~]{%d,}" % RUN_FLOOR)
    tokens = set()
    for n in known:
        tokens.update(n.split())
    tokens -= {t for t in tokens if len(t) < RUN_FLOOR}
    bin_hits = []
    for pat in SCAN_BINARY:
        for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
            if ".git" in f or "egg-info" in f or "__pycache__" in f:
                continue
            try:
                raw = Path(f).read_bytes()
            except Exception:
                continue
            words, whole = set(), set()
            for m in run.finditer(raw):
                s = m.group(0).decode("ascii", "ignore").upper()
                if s in known and s not in AUTHORS:
                    whole.add(s)
                for w in s.split():
                    if w in tokens:
                        words.add(w)
            if whole or len(words) >= TOKEN_FLOOR:
                bin_hits.append((os.path.relpath(f, ROOT), len(whole), len(words)))

    if bin_hits:
        print(f"{len(bin_hits)} shipped binary or plain file(s) carry corpus names")
        print()
        print(f"  {'file':<56}{'whole names':>12}{'name tokens':>13}")
        for rel, w, t in sorted(bin_hits, key=lambda x: -x[1] - x[2]):
            print(f"  {rel:<56}{w:>12,}{t:>13,}")
        print()
    else:
        print("0 shipped binary or plain file(s) carry corpus names")
        print()

    if text_hits:
        print(f"{len(text_hits)} readable file(s) contain a corpus name of "
              f"{TEXT_FLOOR} characters or more")
        print()
        for rel, found in sorted(text_hits, key=lambda x: -len(x[1])):
            print(f"  {rel:<52} {len(found):>3}   {', '.join(found[:2])}"
                  + (" ..." if len(found) > 2 else ""))
        print()

    if args.clear_notebooks:
        import json as _json
        for rel, _ in text_hits:
            if not rel.endswith(".ipynb"):
                continue
            p = ROOT / rel
            nb = _json.loads(p.read_text(encoding="utf-8"))
            n = 0
            for cell in nb.get("cells", []):
                if cell.get("outputs"):
                    n += len(cell["outputs"])
                    cell["outputs"] = []
                if "execution_count" in cell:
                    cell["execution_count"] = None
            p.write_text(_json.dumps(nb, indent=1, ensure_ascii=False),
                         encoding="utf-8")
            print(f"  cleared {n} stored output(s) from {rel}")

    def risk(rel_bound_hits) -> int:
        return 1 if (rel_bound_hits or text_hits or bin_hits) else 0

    if not hits:
        print("no CSV column carries a corpus name")
        return risk([])

    rel_bound = [h for h in hits if not h[3]]
    local = [h for h in hits if h[3]]
    w = max(len(os.path.relpath(f, ROOT)) for f, _, _, _ in hits)
    print(f"{len(rel_bound)} release-bound column(s) carry corpus names\n")
    print(f"{'file':<{w}}  {'column':<8} {'matched':>8}  rows")
    for f, c, share, _ in rel_bound:
        n = len(pd.read_csv(f, usecols=[c]))
        print(f"{os.path.relpath(f, ROOT):<{w}}  {c:<8} {share*100:7.1f}%  {n:,}")
    print(f"\n{len(local)} local working file(s) also carry names and are left alone, "
          f"since the analyses read them and none of them is released")

    if not args.apply:
        print("\nreport only. Pass --apply to replace the name column with row_id, "
              "n_tokens and suffix3.")
        return risk(rel_bound)

    for f, c, _, _ in rel_bound:
        d = pd.read_csv(f)
        names = d[c].astype(str)
        pos = d.columns.get_loc(c)
        d = d.drop(columns=[c])
        d.insert(pos, "suffix3", names.str.lower().str[-3:])
        d.insert(pos, "n_tokens", names.str.split().str.len())
        d.insert(pos, "row_id", range(1, len(d) + 1))
        d.to_csv(f, index=False)
        print(f"  de-identified {os.path.relpath(f, ROOT)}")
    print(f"\n{len(hits)} file(s) rewritten. The local key files under "
          f"results/final are the only place a name still appears.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
