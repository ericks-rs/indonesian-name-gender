# Training scripts

This directory contains training scripts for the classifiers evaluated in the study, together with an additional class-imbalance experiment.

| Script | Models | Scope |
|---|---|---|
| `train_grid.py` | Eight character-level and word-level neural models | Main experiment |
| `train_tfidf.py` | TF-IDF with SVM, logistic regression, and random forest | Classical baselines |
| `train_pretrained.py` | IndoBERT, mBERT, and XLM-R | Pretrained baselines |
| `train_imbalance.py` | Alternative imbalance treatments for the eight grid models | Additional experiment, not reported in the current manuscript |

## Data and dependencies

Training requires the institutional data under `data/splits/`, which are not distributed with the repository. The grid scripts also load the external benchmark and fitted tokenizers.

See [DATA.md](../docs/DATA.md) for input requirements. Tokenizers must use a vocabulary fitted to the training partition, and their encoding procedure must match the stored vocabulary keys.

The pretrained models require the Transformers library and access to the base checkpoints on Hugging Face.

## Main evaluation protocol

The main experiments train on names first registered through 2021, use the 2022–2023 development partition for neural checkpoint selection, and evaluate on the 2024–2026 temporal test.

The classifiers use five matched seeds. Optimization settings differ between the grid models and pretrained encoders, as specified in [PROTOCOL.md](../docs/PROTOCOL.md).

## Outputs and reproduction

`train_grid.py` and `train_imbalance.py` write their outputs under `results/tables/`. `train_tfidf.py` and `train_pretrained.py` write to `results/final/03_seeds_tfidf/` and `results/final/04_seeds_transformers/` by default, with an `--out` option for another location. The released results used by the analysis scripts are available under `results/final/`.

See [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) for the recorded environment, supported analyses, and requirements for retraining.
