# Reproducibility

What is exact, what is not, and by how much. Each claim below was measured rather
than assumed.

## Loading a checkpoint is exact

The bundled `CharBiLSTM` reproduces the stored seed-42 prediction on all 15,923
evaluation names, with no disagreement. Anything that only loads and predicts is
deterministic.

## Retraining is not bit-reproducible

The training script does not call `torch.use_deterministic_algorithms` and does
not disable the cuDNN autotuner, so two runs of the same seed produce weights
that differ by up to 1.6e-03.

The score does not move with them. All four character models return the same
development F1 to four decimals across repeated runs, and the weight drift sits
two orders of magnitude below the 0.068 to 0.184 point spread across seeds that
the paper already reports. Reproducing the reported numbers therefore does not
depend on reproducing the weights.

## Which scripts run against a clone

Eleven of the twenty-two do, and each was executed from a clean checkout rather
than assumed to work.

- `architecture_comparison.py`
- `audit_personal_data.py` (structural column scan only, see note)
- `environment_record.py`
- `experiment_inventory.py`
- `external_paired.py`
- `figure_attention_position.py`
- `figure_training_curves.py`
- `figures_evidence.py`
- `imbalance_protocol.py`
- `paired_comparison.py`
- `threshold_free_metrics.py`

The other eleven stop at their first read. Nine want the corpus. Two,
`latency_repeats.py` and `assemble_figures.py`, want the training layout, where
tokenizers and figures sit at paths this tree does not use.

The privacy audit deserves one caveat. `audit_personal_data.py` runs from a clone
and scans every shipped table for a column whose values look like names, which is
how it flags a disclosure. Its second half compares those values against the
corpus itself to catch a real name from the training data, and that comparison needs the private
corpus, which is not shipped. From a clone the corpus list is empty, so the
membership check is inert and only the structural scan runs. That is enough to
confirm no shipped table still carries a name column, which is what a reader wants
to verify.

## What you cannot reproduce from this repository

The corpus is not redistributed, so training cannot be rerun as published. What
can be rerun is everything downstream, since the analysis scripts read
`results/final/` rather than the data.

The corpus files themselves have no producing script in this repository. They
were built before this work began, and
`results/final/37_corpus_provenance/` records twelve invariants they satisfy,
each tested rather than remembered. Anyone rebuilding from a comparable corpus should
expect to reproduce the counts, not the procedure.

## Environment

The reported run used Python 3.10.9 and torch 2.8.0+cu128 on an NVIDIA GeForce
RTX 5080 Laptop GPU with 17.1 GB, under CUDA 12.8, with numpy 1.23.5, pandas
1.5.3, scikit-learn 1.2.1 and scipy 1.10.0. `configs/environment.csv` carries the
full record.

Latency figures are single-thread CPU at a fixed serving shape, repeated over
seven independent trials, and no model varied by more than 5.3 percent.
