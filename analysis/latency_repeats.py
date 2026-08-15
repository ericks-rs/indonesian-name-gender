"""Single-thread CPU latency, measured over independent trials rather than one.

The benchmark in program/benchmark_inference.py takes the median of 200 calls in
a single process, and the numbers it produced put CharBiGRU at 1.34 ms against
CharBiLSTM at 0.39, three and a half times slower on 25 percent fewer parameters.
An ordering like that is not an architectural fact. At this size the call is
dominated by framework overhead, and one process gives one draw from whatever
state that process happened to be in.

So the same measurement is repeated across TRIALS fresh model instances, and the
spread is reported next to the point estimate. The manuscript can then quote a
range that survives being run again, instead of the fastest cell of one table.

Runs on CPU only, one thread, batch of one, which is the per-request cost in a
concurrent serving setting.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import inference as inf  # noqa: E402

OUT = ROOT / "results" / "final" / "32_latency_repeats"
PROBE = "BANOWATI LARASATI"
TRIALS, WARMUP, RUNS = 7, 30, 200
HF = {"IndoBERT": "indobenchmark/indobert-base-p1",
      "mBERT": "bert-base-multilingual-cased",
      "XLM-R": "xlm-roberta-base"}

torch.set_num_threads(1)


@torch.no_grad()
def one_trial(fn) -> float:
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # the eight grid models, rebuilt from scratch on every trial so no trial
    # inherits the allocator state of the one before it
    for t in range(TRIALS):
        p = inf.Predictor(ROOT / "results", model_suffix="")
        for name, model in p.models.items():
            model.to("cpu").eval()
            is_char = name.startswith("Char")
            tok = p.char_tok if is_char else p.word_tok
            x = torch.tensor([tok.encode(PROBE, 50 if is_char else 8)],
                             dtype=torch.long)
            rows.append({"Model": name, "trial": t,
                         "cpu_ms": one_trial(lambda m=model, x=x: m(x))})
        del p
        print(f"  trial {t + 1}/{TRIALS} done", flush=True)

    # the pretrained encoders, for the ratio the manuscript states
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for label, repo in HF.items():
            # padded to a fixed 32, matching program/benchmark_inference.py. Every
            # model in this comparison runs at a fixed serving shape, 50 for the
            # character models and 8 for the word models. Timing the encoders at
            # their natural token length instead makes them look 1.7 times faster
            # and puts them on a different protocol from everything else.
            tk = AutoTokenizer.from_pretrained(repo)
            enc = tk(PROBE, return_tensors="pt", truncation=True, max_length=32,
                     padding="max_length")
            for t in range(TRIALS):
                m = AutoModelForSequenceClassification.from_pretrained(
                    repo, num_labels=2).eval()
                rows.append({"Model": label, "trial": t,
                             "cpu_ms": one_trial(lambda m=m, e=enc: m(**e))})
                del m
            print(f"  {label} done", flush=True)
    except Exception as e:
        print(f"  pretrained encoders skipped, {e.__class__.__name__}: {str(e)[:90]}")

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "latency_per_trial.csv", index=False)

    # "median" is a DataFrame method, so the column is named mid and read through
    # brackets. Attribute access on a column that shadows a method silently hands
    # back the method, which is how two earlier tables in this project came out
    # empty without raising anything.
    g = d.groupby("Model")["cpu_ms"].agg(
        mid="median", lo="min", hi="max",
        iqr=lambda s: float(s.quantile(.75) - s.quantile(.25))).round(4)
    g["spread_pct"] = ((g["hi"] - g["lo"]) / g["mid"] * 100).round(1)
    g = g.reset_index()
    g["level"] = np.where(g["Model"].str.startswith("Char"), "character",
                          np.where(g["Model"].str.startswith("Word"), "word",
                                   "pretrained"))
    g.to_csv(OUT / "latency_summary.csv", index=False)
    print("\n" + g.to_string(index=False))

    ch = g[g["level"] == "character"]
    pre = g[g["level"] == "pretrained"]
    if not pre.empty:
        slowest = float(pre["mid"].max())
        ratio = pd.DataFrame([
            {"character_model": r["Model"], "character_ms": r["mid"],
             "slowest_pretrained_ms": slowest,
             "ratio": round(slowest / r["mid"], 1)}
            for _, r in ch.iterrows()])
        ratio.to_csv(OUT / "ratio_against_pretrained.csv", index=False)
        print("\n" + ratio.to_string(index=False))
        print(f"\nrange across the four character models: "
              f"{ratio['ratio'].min():.0f} to {ratio['ratio'].max():.0f} times")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
