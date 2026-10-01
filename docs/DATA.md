# Data

## Institutional corpus

The institutional corpus is not distributed with this repository. The manuscript describes its source, preprocessing, and use in the experiments.

The processed corpus contains 205,093 distinct Indonesian names first registered from 1990 to 2026. Each name has a recorded male or female label. Duplicate names were consolidated, and names associated with both labels were removed.

## Temporal partitions

Each normalized name is assigned to a partition using its earliest registration year. Later registrations of the same name do not create additional examples or move the name to another partition.

| Partition | Registration years | Distinct names |
|---|---|---:|
| Training | 1990–2021 | 169,329 |
| Development | 2022–2023 | 16,882 |
| Test | 2024–2026 | 18,882 |

No normalized name appears in more than one partition. This design evaluates models on names that first entered the registry after the training period. Registration year indicates entry into the registry rather than the year a name was given.

The partition sizes are approximately 82.6%, 8.2%, and 9.2% of the corpus.

## External benchmark

The external benchmark comes from the public [Indonesian name gender dataset](https://www.kaggle.com/datasets/dionisiusdh/indonesian-names/versions/2).

The published collection contains 1,960 records representing 1,795 distinct names. The primary external evaluation uses 1,464 distinct names after removing names present in the institutional training partition and consolidating repeated names.

The collection provides names and gender labels but no registration years. It is used for external evaluation, not model training. The manuscript also reports results for alternative versions of this collection; these are different treatments of the same source.

## Input files for training

The grid training script reads the following institutional data files:

```text
data/splits/train_1990_2021.csv
data/splits/dev_2022_2023.csv
data/splits/val_2024_2026.csv
```

Each file must contain:

| Column | Contents |
|---|---|
| `NAMA` | Name stored in uppercase, with tokens separated by whitespace |
| `LABEL` | `L` for male or `P` for female |

The script also reads:

```text
data/external/indonesian-names.csv
```

This file uses the columns `name` and `gender`, with `m` and `f` as the gender labels.

Fitted tokenizer files are loaded from `tokenizers/`. The training scripts expect these files to be available before training begins.

## Using another corpus

To evaluate the method on another corpus:

- Apply consistent name normalization across all partitions.
- Keep each normalized name in only one partition.
- Use later registration periods for development and testing when following the temporal protocol.
- Fit the tokenizers using only the new training partition.
- Keep the token-to-index mapping consistent between training and inference.

The released tokenizers belong to the original experiments. Their vocabulary should not be treated as a vocabulary fitted to a replacement corpus.

Experiments on another corpus may produce different partition sizes, vocabulary coverage, and performance.

## Released result tables

Released per-name result tables use row identifiers and derived fields, including token counts and three-character endings, in place of full names.

Row identifiers refer to records within the corresponding table. Before combining tables, check that they describe the same data partition and use the same row ordering.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the analyses supported by the released artifacts and the data required for additional checks.
