# indonamegender

Predict gender labels from Indonesian personal names using character-level or word-level neural models.

The package includes a compact CharBiLSTM checkpoint and supports seven additional models downloaded on first use. It provides an inference interface for the models evaluated in *A Cross-Architecture Empirical Study of Character-Level and Word-Level Representations for Gender Classification from Indonesian Names*.

## Install

```bash
pip install indonamegender
```

## Use

```python
from indonamegender import GenderPredictor

predictor = GenderPredictor()
result = predictor.predict("BANOWATI LARASATI")
print(result)
```

The prediction includes the gender label, confidence score, and model name.

The package includes the seed-42 CharBiLSTM checkpoint, so the default model requires no additional download. Its model file is 0.46 MB. In the reported benchmark, CharBiLSTM achieved a median CPU forward-pass latency of 0.3912 ms per name across seven trials using one thread. This measurement excludes tokenization.

On the external benchmark of 1,464 names, CharBiLSTM achieved a mean F1 of 0.9367 across five training seeds. This score summarizes the five runs rather than the bundled checkpoint alone.

## Choose a model

Pass a model name to `GenderPredictor` to use another checkpoint.

```python
gru = GenderPredictor("CharBiGRU")
transformer = GenderPredictor("WordTransformer")
```

The package downloads additional checkpoints from the `v1.1.0` GitHub release on first use and stores them in `~/.cache/indonamegender/`. Subsequent calls reuse the cached files.

All eight models support CPU inference. To select the CPU explicitly:

```python
predictor = GenderPredictor("CharBiLSTM", device="cpu")
```

| Model | Input level | Temporal-test F1 | Parameters |
|---|---|---:|---:|
| CharBiLSTM | Character | 0.9675 | 114,097 |
| CharBiGRU | Character | 0.9669 | 86,065 |
| CharBiRNN | Character | 0.9651 | 30,001 |
| CharTransformer | Character | 0.9631 | 605,953 |
| WordTransformer | Word | 0.9423 | 6,126,721 |
| WordBiLSTM | Word | 0.9401 | 2,544,289 |
| WordBiGRU | Word | 0.9393 | 2,507,041 |
| WordBiRNN | Word | 0.9386 | 2,432,545 |

F1 values are means across five matched training seeds on 18,882 names first registered from 2024 to 2026. The distributed checkpoints use seed 42.

The package supports the eight neural models trained from scratch. Results for the character n-gram classifiers and pretrained encoders evaluated in the paper are available in the repository.

## Intended use

The models estimate the male or female labels recorded in the source dataset. They are intended to support aggregate analysis when gender labels are unavailable. Predictions may be incorrect and should not replace self-reported gender information or determine decisions about individuals.

Performance may vary across institutions and naming conventions. Evaluate the model on representative data before using predictions in an analysis.

## Research and source code

The repository provides training and analysis code, result tables, and documentation for the associated study.

- [GitHub repository](https://github.com/ericks-rs/indonesian-name-gender)
- [Releases and model checkpoints](https://github.com/ericks-rs/indonesian-name-gender/releases)

## License

Released under the MIT license.
