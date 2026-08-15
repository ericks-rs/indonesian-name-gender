# Gender classification from Indonesian personal names

Code, trained checkpoints and result tables for *A Cross-Architecture Attention
Analysis of Character-Level and Word-Level Models for Gender Classification from
Indonesian Names*.

The study compares two input representations across four sequence encoders under
one fixed configuration, and asks where each model places its attention. It does
not propose an architecture. Every model here is a standard encoder used as
published.

## What the study found

Character-level representation beat word-level representation for all four
architectures, by 1.89 to 2.59 F1 points over five matched seeds, with every
confidence interval clear of zero. Encoder choice decided far less. Within the
character-level group no pairwise difference survived Holm correction, so the
paper names no best encoder.

| | test F1 | external F1 | parameters | CPU latency |
|---|---|---|---|---|
| CharBiGRU | 0.9593 | 0.9345 | 86,065 | 1.3433 ms |
| CharBiLSTM | 0.9589 | **0.9375** | 114,097 | **0.3952 ms** |
| CharBiRNN | 0.9574 | 0.9297 | **30,001** | 0.5238 ms |
| CharTransformer | 0.9554 | 0.9368 | 605,953 | 1.1456 ms |
| IndoBERT | 0.9578 | 0.9211 | 124,442,882 | 58.1366 ms |
| XLM-R | 0.9617 | 0.9257 | 278,045,186 | 59.3377 ms |

The character-level models match the fine-tuned encoders on the test partition
and lead on transfer, at 44.6 to 151.4 times lower single-thread CPU latency.
Full results are in `results/final/`.

## Install

```bash
pip install -e .
```

Or with the analysis extras, needed only to regenerate tables and figures.

```bash
pip install -e ".[analysis]"
```

## Predict

```python
from indonamegender import GenderPredictor

p = GenderPredictor()
p.predict("GATOTKACA WIRAWAN")
# {'gender': 'Male', 'confidence': 0.9959, 'model': 'CharBiGRU', ...}
```

The bundled model is the seed-42 `CharBiGRU`, and it reproduces the stored
prediction on all 15,923 evaluation names exactly. The other seven checkpoints
are under `models/`.

## Layout

```
src/indonamegender/   inference package, the one thing on PyPI
experiments/          the two training entry points
analysis/             statistics, figures and audits over the artifacts
models/               eight seed-42 checkpoints, one per architecture
tokenizers/           character vocabulary, and the hashed word vocabulary
results/final/        every reported number, as CSV
results/figures/      the 24 manuscript figures, PNG and PDF at 600 dpi
configs/              the machine and the training settings the run used
demo/                 local web demo
docs/                 data access, protocol, reproducibility
```

## Reproducing

Training needs the corpus, which is not redistributable. `docs/DATA.md` says what
the inputs are and what a substitute has to satisfy.

```bash
python experiments/train_grid.py          # eight architectures, five seeds
python experiments/train_imbalance.py     # three imbalance strategies, 120 runs
```

Eleven of the twenty-two analysis scripts run against a fresh clone, because
they read `results/final/` and nothing else. Each was executed from a clean
checkout rather than assumed to work.

```bash
python analysis/paired_comparison.py        # character against word, paired
python analysis/threshold_free_metrics.py   # AUC and Brier
python analysis/imbalance_protocol.py       # the 120-run robustness check
python analysis/architecture_comparison.py  # within-level encoder comparisons
python analysis/external_paired.py          # the external benchmark, paired
python analysis/audit_personal_data.py      # the privacy scan, over this tree
```

The remaining eleven stop at their first read. Nine need the corpus, and two
need the training layout rather than the released one. They ship because they
are the code that produced the reported numbers, not because they can be rerun
here, and `docs/REPRODUCIBILITY.md` says which is which.

`docs/PROTOCOL.md` gives the split and the training configuration.
`docs/REPRODUCIBILITY.md` reports what is exact and what is not, measured rather
than claimed.

## Privacy

The corpus is a university admissions registry. No file here contains a
registration.

The word vocabulary ships hashed. Every entry except the two reserved indices is
`blake2s(salt + token)`, which keeps the embedding index intact while making the
24,947-entry vocabulary unreadable, so
a released checkpoint stays usable without carrying the names it was fitted on.
Eighteen per-name tables under `results/final/` have their name column replaced
by a row identifier, the token count and the three-character ending. `analysis/audit_personal_data.py`
scans the tree, and the build fails rather than warns. Illustrative names in the
figures, the demo and the tests are drawn from Javanese shadow theatre and appear
nowhere in the corpus.

## Citation

See `CITATION.cff`. Licensed MIT.
