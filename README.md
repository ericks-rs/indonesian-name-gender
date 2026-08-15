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

| model | level | precision | recall | F1 |
|---|---|---|---|---|
| TF-IDF+SVM | character | 0.9777 | 0.9484 | **0.9628** |
| XLM-R | subword | 0.9730 | 0.9506 | 0.9617 |
| mBERT | subword | 0.9737 | 0.9483 | 0.9608 |
| TF-IDF+LR | character | 0.9755 | 0.9453 | 0.9602 |
| CharBiGRU | character | 0.9698 | 0.9491 | 0.9593 |
| CharBiLSTM | character | 0.9688 | 0.9493 | 0.9589 |
| IndoBERT | subword | 0.9694 | 0.9466 | 0.9578 |
| CharBiRNN | character | 0.9677 | 0.9473 | 0.9574 |
| CharTransformer | character | 0.9633 | 0.9476 | 0.9554 |
| WordTransformer | word | 0.9361 | 0.9369 | 0.9365 |
| WordBiLSTM | word | 0.9349 | 0.9338 | 0.9344 |
| WordBiGRU | word | 0.9345 | 0.9324 | 0.9335 |
| WordBiRNN | word | 0.9374 | 0.9274 | 0.9323 |
| TF-IDF+RF | character | 0.9545 | 0.8871 | 0.9196 |

All fourteen classifiers are shown. The character-level neural models match the
fine-tuned encoders on the test partition and lead on the external benchmark, at
44.6 to 151.4 times lower single-thread CPU latency. Precision, recall, external
F1, parameter counts and latency for every model are in `results/final/`.

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

The bundled model is the seed-42 `CharBiLSTM`, which is the one the paper
recommends for names beyond the training data. It has the highest
external F1 at 0.9375 and the lowest CPU latency at 0.3952 milliseconds, and it
reproduces the stored prediction on all 15,923 evaluation names exactly. It
trails CharBiGRU internally by 0.04 points, a difference the paper reports as
not separable.

The other seven are under `models/` in a clone. From a pip install they are
fetched once from the `v1.0.0` release and cached under
`~/.cache/indonamegender/`, so any architecture in the grid can be asked for by
name.

```python
GenderPredictor("CharBiGRU")        # bundled? no, downloaded once
GenderPredictor("WordTransformer")  # same
```

## Layout

```
src/indonamegender/   inference package, published on PyPI as indonamegender
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

See `CITATION.cff`. Licensed MIT.
