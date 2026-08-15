# Data

## The training corpus is not included

The corpus used for training is not distributed with this repository. Its
composition and source are described in the paper. This document only states
what the code expects, so that the scripts can be run against a suitable corpus.

The corpus holds 202,134 unique names with a binary label after deduplication to
one row per distinct name and removal of names that carry both labels. No file in
this repository contains a name from it.

## The partitions

The split is temporal and operates at the name level. Each name is assigned to
the partition of the earliest year it appears, so a name first seen in 1998 and
seen again in 2022 stays in training. No name is shared between partitions.

| partition | years | names |
|---|---|---|
| training | 1990 to 2021 | 169,329 |
| development | 2022 to 2023 | 16,882 |
| test | 2024 to 2025 | 15,923 |

The training partition runs 1.56 male to female. Development and test are close
to balanced. No resampling was applied to produce that. This is a novel-name
temporal split rather than a random split, which is stricter, and it is why every
reported figure is conservative.

## The external benchmark

A public collection of Indonesian names, 1,960 rows over 1,795 distinct names,
used only for evaluation. The manuscript reports the 1,464 distinct names that do
not appear in training. It carries no year, so it cannot support the temporal
protocol and is never trained on.

Available at
`https://www.kaggle.com/datasets/dionisiusdh/indonesian-names/versions/2`.

## Substituting your own corpus

`experiments/train_grid.py` expects three CSV files under `data/splits/` with two
columns, `NAMA` and `LABEL`, where `L` is male and `P` is female.

```
data/splits/train_1990_2021.csv
data/splits/dev_2022_2023.csv
data/splits/val_2024_2025.csv
```

Four properties matter for the comparison to mean what it means here. Names must
be uppercase and whitespace separated. No name may appear in more than one
partition. The evaluation partitions must be later in time than training, not a
random sample of it. And the tokenizers must be refitted on the new training
partition, because the ones shipped here are fitted on a corpus you will not have.

## What the released tables contain instead of names

Eighteen of the tables under `results/final/` are per-name. Their name column is
replaced at build time by a row identifier, the token count and the
three-character ending, which is what the analyses read. The row order is
preserved, so a table can still be joined against another by position.
