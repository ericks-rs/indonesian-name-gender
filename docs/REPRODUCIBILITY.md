# Reproducibility

This repository supports inference with released checkpoints and analysis of stored experimental results. Retraining requires the institutional corpus, which is not distributed.

## Inference with released checkpoints

The package includes the seed-42 CharBiLSTM checkpoint. The other seven character-level and word-level checkpoints are downloaded on first use.

```python
from indonamegender import GenderPredictor

predictor = GenderPredictor("CharBiLSTM", device="cpu")
result = predictor.predict("BANOWATI LARASATI")
print(result)
```

Inference uses the saved weights and fitted tokenizers without retraining. Numerical outputs may vary with hardware and library versions.

The scores reported in the manuscript summarize five matched training seeds. They should not be interpreted as the scores of the distributed seed-42 checkpoints alone.

## Analyses using included results

Several analyses read stored predictions and metrics from `results/final/` without requiring the institutional corpus.

Run these commands from the repository root:

```bash
pip install -e ".[analysis]"

python analysis/paired_comparison.py
python analysis/architecture_comparison.py
python analysis/threshold_free_metrics.py
```

These scripts recompute:

- F1 differences between matched character-level and word-level models.
- Pairwise differences among architectures within each representation level.
- AUC and Brier scores from stored prediction probabilities.

The scripts write their outputs to `results/final/`. Other analyses may require private data, additional dependencies, or paths from the original experiment environment.

## Retraining

The main training scripts are:

| Script | Models |
|---|---|
| `experiments/train_grid.py` | Eight character-level and word-level neural models |
| `experiments/train_tfidf.py` | TF-IDF with SVM, logistic regression, and random forest |
| `experiments/train_pretrained.py` | IndoBERT, mBERT, and XLM-R |

See [DATA.md](DATA.md) for input requirements and [PROTOCOL.md](PROTOCOL.md) for the data partitions and training settings. The grid training script also reads the external benchmark and fitted tokenizers.

The experiments use seeds 42, 7, 123, 2024, and 777. Fixed seeds do not guarantee identical weights or scores across repeated runs or computing environments. The grid training script sets random seeds but does not enforce deterministic PyTorch algorithms.

When using another corpus, fit the tokenizers on its training partition and ensure that training and inference use the same token-to-index mapping. Results from a substitute corpus evaluate the method on different data and are not expected to reproduce the published scores or dataset counts.

The repository also retains class-imbalance experiments in `experiments/train_imbalance.py`. These additional experiments are not reported in the current manuscript.

## Inference timing

Reported CPU forward-pass latencies use preprocessed inputs, one CPU thread, and a batch size of one. The repeated measurements summarize seven trials, each containing 200 timed calls.

These measurements exclude tokenization. They describe the benchmark environment and may differ from timings on other hardware or in applications that include preprocessing and additional processing steps.

## Privacy checks

`analysis/audit_personal_data.py` compares selected repository files against names loaded from the source data. These checks depend on access to the comparison data.

Running the script without those data does not establish that the release contains no corpus names. A successful exit should be interpreted in relation to the available inputs and the files covered by the scan.

## Recorded environment

The recorded experiment environment includes:

| Component | Version or device |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| Transformers | 5.14.1 |
| NumPy | 2.4.5 |
| pandas | 3.0.3 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.1 |

The full environment record is available in [configs/environment.csv](../configs/environment.csv). These versions describe the recorded environment rather than the package's minimum installation requirements.
