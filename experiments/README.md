# Training scripts

These are the scripts that produced the reported results, one per model family.

| script | models | writes |
|---|---|---|
| `train_grid.py` | the eight character-level and word-level models | `results/tables/multiseed/` |
| `train_tfidf.py` | TF-IDF with SVM, logistic regression, random forest | `results/tables/` |
| `train_pretrained.py` | IndoBERT, mBERT, XLM-R, fully fine-tuned | `results/tables/transformer_baselines/` |
| `train_imbalance.py` | the three imbalance strategies over the eight models | `results/tables/imbalance_protocol/` |

All four read the training corpus from `data/splits/`, which is not included in
this repository. They therefore stop at the first `read_csv` from a fresh clone. `docs/DATA.md` states the layout a substitute
corpus must follow. The tokenizers under `tokenizers/` are shipped and are the
ones fitted on the reported training partition.

Every script uses the 1990 to 2021 training window, selects checkpoints on the
2022 to 2023 development partition, and scores the 2024 to 2025 test partition
once. The pretrained encoders download their base weights from Hugging Face on
first run.

The two Transformers in `train_grid.py` and `train_imbalance.py` pool through the same
attention module as the recurrent models, which is the architecture behind every
reported Transformer number. An earlier copy of both scripts pooled the Transformers by
mean. The thirty Transformer runs in `results/tables/imbalance_protocol/imbalance_runs.csv`
were therefore re-run with attention pooling, and the six recurrent models, identical in
both copies, were kept. The re-run also repeated the class-weighted baseline for the two
Transformers as a check. Its F1 differs from the reported grid by at most 0.132
percentage points per seed, the run-to-run variation of the GPU, see
`results/final/41_imbalance_transformer_attnpool/`.
