# Gender classification from Indonesian personal names

[![PyPI](https://img.shields.io/pypi/v/indonamegender)](https://pypi.org/project/indonamegender/)
[![Downloads](https://img.shields.io/pypi/dm/indonamegender)](https://pypistats.org/packages/indonamegender)
[![GitHub Release](https://img.shields.io/github/v/release/ericks-rs/indonesian-name-gender)](https://github.com/ericks-rs/indonesian-name-gender/releases)
[![Python](https://img.shields.io/pypi/pyversions/indonamegender)](https://pypi.org/project/indonamegender/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository provides code, trained checkpoints, and result tables for
*A Cross-Architecture Empirical Study of Character-Level and Word-Level
Representations for Gender Classification from Indonesian Names*.

The study evaluates character-level and word-level inputs across four encoder
families under a common training and evaluation protocol. Character n-gram
classifiers and pretrained encoders provide additional baselines. The analyses
examine predictive performance, transfer to an external data source, attention
patterns, and inference cost.

## What the study found

Character-level inputs improved temporal-test F1 over word-level inputs by
2.081–2.756 percentage points across the four matched encoder pairs. All four
differences were significant after Holm correction across five matched seeds.
Differences among architectures within either representation level were
smaller, at most 0.438 points. Among the character-level encoders, CharBiLSTM
and CharBiGRU significantly outperformed CharTransformer after Holm correction.

| model | level | precision | recall | F1 |
|---|---|---|---|---|
| TF-IDF+SVM | character | 0.9800 | 0.9611 | **0.9705** |
| XLM-R | subword | 0.9773 | 0.9619 | 0.9696 |
| mBERT | subword | 0.9768 | 0.9599 | 0.9682 |
| CharBiLSTM | character | 0.9732 | 0.9619 | 0.9675 |
| TF-IDF+LR | character | 0.9779 | 0.9561 | 0.9669 |
| CharBiGRU | character | 0.9726 | 0.9613 | 0.9669 |
| CharBiRNN | character | 0.9718 | 0.9585 | 0.9651 |
| IndoBERT | subword | 0.9738 | 0.9556 | 0.9646 |
| CharTransformer | character | 0.9625 | 0.9637 | 0.9631 |
| WordTransformer | word | 0.9398 | 0.9448 | 0.9423 |
| WordBiLSTM | word | 0.9385 | 0.9417 | 0.9401 |
| WordBiGRU | word | 0.9393 | 0.9394 | 0.9393 |
| WordBiRNN | word | 0.9408 | 0.9364 | 0.9386 |
| TF-IDF+RF | character | 0.9561 | 0.9005 | 0.9275 |

The table reports mean temporal-test scores across five matched seeds for all
14 classifiers. TF-IDF+SVM achieved the highest F1. XLM-R significantly
outperformed CharBiRNN, CharBiLSTM, and CharTransformer after Holm correction,
while the difference from CharBiGRU was not significant. On the external
benchmark, all four character-level neural models scored above the three
pretrained encoders. The character-level neural models also achieved more than
40-fold faster single-thread CPU forward-pass inference, based on median
per-name latencies across seven trials. Detailed performance and efficiency
results are available in `results/final/`.

## Install

```bash
pip install indonamegender
```

From a clone, add the analysis extras to regenerate tables and figures.

```bash
pip install -e ".[analysis]"
```

## Predict

```python
from indonamegender import GenderPredictor

p = GenderPredictor()
p.predict("GATOTKACA WIRAWAN")
# {'gender': 'Male', 'confidence': 0.999, 'model': 'CharBiLSTM', ...}
```

The package includes the seed-42 `CharBiLSTM` checkpoint. CharBiLSTM combines
a small model file (0.46 MB) with the lowest measured CPU forward-pass latency
among the four character-level neural models (0.3912 ms per name). Mean F1
across five seeds was 0.9675 on the temporal test and 0.9367 on the external
benchmark. These scores summarize the five training runs rather than the
bundled checkpoint alone.

The other seven are under `models/` in a clone. From a pip install they are
fetched once from the `v1.1.0` release and cached under
`~/.cache/indonamegender/`, so any architecture in the grid can be asked for by
name.

```python
GenderPredictor("CharBiGRU")        # bundled? no, downloaded once
GenderPredictor("WordTransformer")  # same
```

### Choosing a model

Every model here answers on CPU, so none of them needs a GPU to serve.
Latency is the median of seven trials of 200 single-thread calls at a fixed
serving shape. Internal F1 is measured on the 2024 to 2026 partition and
averaged over five seeds. External F1 is measured on the 1,464 names from the
public benchmark that do not appear in the training data. Use this column when
evaluating performance on unseen names.

| model | parameters | size | CPU ms per name | internal F1 | external F1 |
|---|---|---|---|---|---|
| `CharBiLSTM` (bundled) | 114,097 | 0.46 MB | 0.3912 | 0.9675 | 0.9393 |
| `CharTransformer` | 605,953 | 2.44 MB | 1.1338 | 0.9631 | 0.9389 |
| `CharBiGRU` | 86,065 | 0.35 MB | 1.3316 | 0.9669 | 0.9385 |
| `CharBiRNN` | 30,001 | 0.12 MB | 0.5237 | 0.9651 | 0.9313 |
| `WordTransformer` | 6,126,721 | 24.52 MB | 0.7538 | 0.9423 | 0.8338 |
| `WordBiRNN` | 2,432,545 | 9.73 MB | 0.1819 | 0.9386 | 0.8328 |
| `WordBiGRU` | 2,507,041 | 10.03 MB | 0.3351 | 0.9393 | 0.8325 |
| `WordBiLSTM` | 2,544,289 | 10.18 MB | 0.2478 | 0.9401 | 0.8328 |

Every character-level model exceeds every word-level model by at least 9.75
external F1 points, which translates the paper's main finding into a practical
serving decision. Internal F1 separates the two representation levels by only
about 2 points, so selecting a model from the internal results alone understates
the generalization gap. `CharBiRNN` is the smallest at 0.12 MB and 30,001
parameters, and costs 0.80 points of external F1 against the bundled model.

Parameter count does not predict latency here, so read the two columns
separately. Names are padded to 50 character positions and 8 word positions,
and a recurrent layer walks those positions one at a time, so every character
model pays 50 sequential steps whatever the name. That is why `WordBiRNN`
answers in a third of the time of `CharBiRNN` while carrying 81 times the
parameters. Inside the character group the three recurrent cells share every
dimension and differ only in gate count. At a hidden size of 192, kernel-launch
and sequential execution overhead can dominate the relatively small matrix
operations. `CharBiLSTM` runs four gates per step and still beats the
single-gate `CharBiRNN`, which is why the smallest model in the table is not
the quickest one.

## Layout

```
src/indonamegender/   inference package, published on PyPI as indonamegender
experiments/          the two training entry points
analysis/             statistics, figures and audits over the artifacts
models/               eight seed-42 checkpoints, one per architecture
tokenizers/           character vocabulary, and the hashed word vocabulary
results/final/        every reported number, as CSV
results/figures/      the 14 manuscript figures, PNG and PDF at 600 dpi
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
python analysis/audit_personal_data.py      # privacy scan (structural, see docs)
```

The remaining eleven stop at their first read. Nine need the corpus, and two
need the training layout rather than the released one. They ship because they
are the code that produced the reported numbers, not because they can be rerun
here, and `docs/REPRODUCIBILITY.md` says which is which.

`docs/PROTOCOL.md` gives the split and the training configuration.
`docs/REPRODUCIBILITY.md` reports what is exact and what is not, measured rather
than claimed.

## Privacy

No file in this repository contains a record from the training data.

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

The manuscript associated with this repository is currently under revision.
Final publication details and DOI will be added after publication.

See `CITATION.cff` for the current citation metadata.

Licensed MIT.
