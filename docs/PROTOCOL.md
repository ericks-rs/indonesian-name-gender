# Experimental protocol

The main experiment evaluates character-level and word-level representations across four encoder families. Character n-gram classifiers and pretrained encoders provide additional baselines, giving 14 classifiers in total.

The main configurations were fixed before the sensitivity analyses. Those additional analyses did not determine the settings used for the main results.

## Data partitions

The institutional corpus is divided by the earliest registration year of each normalized name:

| Partition | Registration years | Names |
|---|---|---:|
| Training | 1990–2021 | 169,329 |
| Development | 2022–2023 | 16,882 |
| Temporal test | 2024–2026 | 18,882 |

No normalized name appears in more than one partition. The primary external benchmark contains 1,464 distinct names from a separate public source, excluding names present in training.

See [DATA.md](DATA.md) for preprocessing and input requirements.

## Character-level and word-level models

The neural grid combines two input levels with BiRNN, BiLSTM, BiGRU, and Transformer encoders.

Names are converted to lowercase during tokenization. Character sequences are padded to 50 positions, and whitespace-separated word sequences are padded to 8 positions. These limits accommodate all names in the institutional corpus.

The character vocabulary contains 33 entries, including padding and unknown symbols. The word vocabulary contains 24,947 entries and retains tokens occurring at least twice in training. Other tokens map to the shared unknown symbol.

All eight models use the same attention-pooling design. Within each encoder family, depth and dropout remain unchanged across input levels, while sequence length and embedding or model dimensions differ.

The recurrent models use one bidirectional layer with 96 hidden units per direction. Character and word embedding dimensions are 48 and 96.

The Transformers use three encoder layers, eight attention heads, and model dimensions of 128 for characters and 192 for words. Feedforward dimensions are four times the model dimension. Dropout is 0.3.

## Grid training settings

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 512 |
| Maximum epochs | 50 |
| Early stopping | Six consecutive epochs without improvement in development F1 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler input | Development loss |
| Scheduler patience | Two epochs |
| Learning-rate reduction factor | 0.5 |
| Loss | Class-weighted binary cross-entropy with logits |
| Checkpoint selection | Highest development F1 |
| Seeds | 42, 7, 123, 2024, 777 |

Female is the positive class. Its loss weight is calculated from the male-to-female ratio in the training partition, approximately 1.5596.

Development F1 controls checkpoint selection and early stopping. Development loss controls learning-rate scheduling. The selected checkpoint is then evaluated on the temporal test and external benchmark.

## Baselines

The classical baselines combine character n-gram TF-IDF features with a linear SVM, logistic regression, or random forest. Features use n-grams of length 2–5 within word boundaries, with a maximum vocabulary of 50,000 features. The classifiers use balanced class weights.

The pretrained baselines are IndoBERT, mBERT, and XLM-R. Each retains its original subword tokenizer. Names are converted to title case, and inputs are limited to 32 subword tokens.

Fine-tuning updates all parameters using AdamW, a learning rate of 0.00002, a batch size of 32, and class-weighted cross-entropy. Training runs for at most 10 epochs, with early stopping after two epochs without improvement in development F1. The checkpoint with the highest development F1 is retained.

The pretrained settings were fixed without hyperparameter search. The results describe these configurations rather than the best attainable performance of each encoder.

## Evaluation and statistical testing

All 14 classifiers use the same five seeds. Accuracy, precision, recall, and F1 are averaged across runs, with female as the positive class. AUC and Brier scores are also calculated for the eight grid models.

F1 comparisons pair models within the same seed. Each comparison reports the mean difference in percentage points, an unadjusted 95% confidence interval, Cohen’s dz, and a two-sided paired t-test with Holm correction.

Holm correction is applied separately to these comparison families:

| Comparison family | Tests |
|---|---:|
| Character versus word within matched encoder families | 4 |
| Architectures within the character level | 6 |
| Architectures within the word level | 6 |
| Character-level neural models versus pretrained encoders | 12 |
| Character-level neural models versus classical baselines | 12 |
| Every character-level neural model versus every word-level model | 16 |

A further correction across the combined 12, 12, and 16 comparisons provides a sensitivity check.

Confidence intervals use the five paired differences and a t-distribution with four degrees of freedom. They describe variation across seeds on a fixed evaluation set.

Exact McNemar tests compare prediction correctness within each seed. Results record the number of seeds with an unadjusted p-value below 0.05; predictions are not pooled across seeds.

## Additional analyses

The hyperparameter sensitivity analysis evaluates 112 configurations using seed 42 and development F1. It examines whether the representation-level difference persists across alternative settings. The main configurations were not selected from this analysis.

A separate CharBiGRU diagnostic examines training-period length and test-based monitoring. Test-based monitoring is confined to that diagnostic. The main experiment uses training data through 2021 and development-based monitoring.

Performance is also evaluated separately for each registration year in the temporal test.

The main neural experiments use class-weighted training. Additional class-imbalance experiments remain in the repository but are not reported in the current manuscript.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for artifact availability and requirements for rerunning the analyses.
