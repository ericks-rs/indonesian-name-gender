# Protocol

The configuration below was fixed before any model was trained, shared by all
eight architectures, and never revised. No grid search, random search or Bayesian
optimisation was used at any point.

## Training

| setting | value |
|---|---|
| optimizer | Adam |
| initial learning rate | 0.001 |
| batch size | 512 |
| maximum epochs | 50 |
| early stopping | 6 epochs without development F1 improvement |
| scheduler | ReduceLROnPlateau on development loss, patience 2, factor 0.5 |
| loss | BCEWithLogitsLoss with pos_weight 1.5637 |
| checkpoint selection | highest development F1 |
| seeds | 42, 7, 123, 2024, 777 |

Female is the positive class. Development F1 is the only selection signal, and
the test partition is scored once, after the checkpoint is chosen.

## The eight architectures

Two representation levels crossed with four encoders. Character sequences are
padded to 50, word sequences to 8, and neither limit truncates any name. All
eight pool through the same additive attention module, so tokenization and
encoder are the only things that differ between a character model and its
word-level counterpart.

## Statistics

Comparisons are paired within seed rather than made between independent means.
Each reports the mean difference, a 95 percent confidence interval, Cohen's d_z
and a Holm-adjusted p-value. Holm is applied within each research question, and a
pooled correction is carried alongside as a robustness column. McNemar's exact
test is run per seed rather than once.

## Why the sensitivity sweep is not a search

After the experiments were complete, the fixed configuration was perturbed 112
ways and every variant scored on the development partition. Nothing reported was
selected from it. The evidence is that the reported setting is not the best
variant of a single one of the eight architectures. It trails the best variant by
0.09 to 0.62 points, and which variant is best differs from one architecture to
the next.

Each configuration was trained once, at seed 42, so a difference between two
variants of one architecture is not separable from seed noise. The sweep supports
a statement about the character and word groups, which are separated by far more
than that, and it cannot rank variants within an architecture.

## Class imbalance

The class-weighted objective was compared against unweighted training, random
oversampling and class-balanced sampling, across all eight models under the same
five seeds, which is 120 additional runs. Checkpoints were selected on the
development partition and the test partition scored once, so the check runs under
the protocol it is checking rather than a looser one.

No alternative produced a statistically significant improvement in any of the 24
combinations. `experiments/train_imbalance.py` reproduces it.
