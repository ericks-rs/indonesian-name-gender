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

To use another model, pass its name to `GenderPredictor`. The package downloads
the checkpoint from the v1.1.0 GitHub release on first use and stores it in
`~/.cache/indonamegender/`. Subsequent calls reuse the cached checkpoint. All
eight model names are listed in the table below.

```python
gru = GenderPredictor("CharBiGRU")
transformer = GenderPredictor("WordTransformer")
```

### Choosing a model

All eight models support CPU inference. The table compares model size, measured
latency, and predictive performance.

CPU latency measures the forward pass with preprocessed inputs, one thread, and
a batch size of one. Each value is the median of seven trial medians, with 200
timed calls per trial. Tokenization is excluded.

Temporal-test F1 is measured on 18,882 names first registered from 2024 to
2026. External F1 is measured on 1,464 distinct names from a separate public
source, excluding names present in training. Both scores are means across five
matched seeds.

| Model | Parameters | Model file size | CPU latency (ms/name) | Temporal-test F1 | External F1 |
|---|---:|---:|---:|---:|---:|
| `CharBiLSTM` (bundled) | 114,097 | 0.46 MB | 0.3912 | 0.9675 | 0.9367 |
| `CharTransformer` | 605,953 | 2.44 MB | 1.1338 | 0.9631 | 0.9372 |
| `CharBiGRU` | 86,065 | 0.35 MB | 1.3316 | 0.9669 | 0.9361 |
| `CharBiRNN` | 30,001 | 0.12 MB | 0.5237 | 0.9651 | 0.9300 |
| `WordTransformer` | 6,126,721 | 24.52 MB | 0.7538 | 0.9423 | 0.8359 |
| `WordBiRNN` | 2,432,545 | 9.73 MB | 0.1819 | 0.9386 | 0.8347 |
| `WordBiGRU` | 2,507,041 | 10.03 MB | 0.3351 | 0.9393 | 0.8344 |
| `WordBiLSTM` | 2,544,289 | 10.18 MB | 0.2478 | 0.9401 | 0.8346 |

Character-level models achieved higher F1 than word-level models on both
datasets. On the external benchmark, the advantage within matched encoder pairs
ranged from 9.53 to 10.21 percentage points.

CharBiLSTM combines a small model file with the lowest measured latency among
the character-level models. CharBiRNN has the smallest file and parameter count,
with lower mean F1. WordBiRNN has the lowest latency among the eight models, but
lower F1 than all four character-level models.

Parameter count alone does not determine inference speed. Character-level and
word-level inputs use different padded sequence lengths, and the encoders
perform different computations. Use measured latency alongside predictive
performance and storage requirements when selecting a model. The reported
timings describe the benchmark environment; performance on other hardware may
differ.

## Repository layout

```
src/indonamegender/   Python package for inference
experiments/          Training scripts for neural models and classical baselines
analysis/             Statistical analyses, figures, and artifact checks
models/               Eight seed-42 checkpoints for the character–word grid
tokenizers/           Fitted character and word tokenizers
results/final/        Stored predictions, metrics, and statistical results
results/figures/      Manuscript figures in PNG and PDF formats
configs/              Experiment settings and environment records
demo/                 Local web demo
docs/                 Data requirements, protocol, and reproduction instructions
tests/                Package tests
```

## Reproducing the analyses

The repository includes stored predictions and result tables that support
several analyses without access to the institutional corpus. Install the
analysis dependencies and run the following commands from the repository root:

```bash
pip install -e ".[analysis]"

python analysis/paired_comparison.py
python analysis/architecture_comparison.py
python analysis/threshold_free_metrics.py
```

These scripts calculate the paired character–word comparisons, comparisons
among architectures within each representation level, and AUC and Brier scores.
They read the included artifacts and write their outputs to `results/final/`.

### Training requirements

The institutional corpus is not distributed with the repository. Rerunning the
training experiments requires access to the original data and the input files
described in `docs/DATA.md`.

The training scripts cover the eight character-level and word-level models
(`experiments/train_grid.py`), the character n-gram classifiers
(`experiments/train_tfidf.py`), and the pretrained encoders
(`experiments/train_pretrained.py`). The data partitions and training settings
are described in `docs/PROTOCOL.md`.

The repository also retains `experiments/train_imbalance.py` and the associated
analysis for additional class-imbalance experiments. These experiments are not
reported in the current manuscript.

### Reproduction scope

Some analyses require private data or paths from the original experiment
environment. See `docs/REPRODUCIBILITY.md` for script requirements, environment
details, and the distinction between inference with released checkpoints and
retraining. Training on a substitute corpus evaluates the method on different
data and should not be expected to reproduce the reported scores.

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

If you use this repository in your research, please cite the associated study
using the metadata in [CITATION.cff](CITATION.cff).

The manuscript is under revision. Publication details and the DOI will be added
when available.

## License

The code is released under the [MIT License](LICENSE).
