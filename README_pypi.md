# indonamegender

Character-level gender classification for Indonesian personal names.

A compact recurrent model that reads a name character by character. It matches
fully fine-tuned multilingual encoders at a fraction of the serving cost, and it
is the analysis code behind the paper *A Cross-Architecture Attention Analysis of
Character-Level and Word-Level Models for Gender Classification from Indonesian
Names*.

## Install

```bash
pip install indonamegender
```

## Use

```python
from indonamegender import GenderPredictor

p = GenderPredictor()
p.predict("SITI AMINAH")
# {'name': 'SITI AMINAH', 'gender': 'Female', 'confidence': 0.9997, 'model': 'CharBiLSTM'}
```

The bundled model is `CharBiLSTM`, the one the paper recommends for names from
outside the training source. It reaches 0.9375 F1 on an independent benchmark and
answers in 0.3952 ms on a single CPU thread. No download is needed for it.

## Other models

Seven further checkpoints are fetched once from the GitHub release and cached
locally, so any of the eight architectures in the grid can be asked for by name.

```python
GenderPredictor("CharBiGRU")
GenderPredictor("WordTransformer")
```

| model | level | test F1 | parameters |
|---|---|---|---|
| CharBiGRU | character | 0.9593 | 86,065 |
| CharBiLSTM | character | 0.9589 | 114,097 |
| CharBiRNN | character | 0.9574 | 30,001 |
| CharTransformer | character | 0.9554 | 605,953 |
| WordTransformer | word | 0.9365 | 6,126,721 |
| WordBiLSTM | word | 0.9344 | 2,544,289 |
| WordBiGRU | word | 0.9335 | 2,507,041 |
| WordBiRNN | word | 0.9323 | 2,432,545 |

Test F1 is the mean over five seeds on the 2024 to 2025 partition. `CharBiLSTM`
is bundled, the other seven download on first use. The three pretrained encoders
from the paper, IndoBERT, mBERT and XLM-R, are not distributed through this
package because their base weights come from Hugging Face. The scripts that
fine-tune them are in the repository under `experiments/`.

## Notes

The label is a binary administrative field, male or female. It reflects a
registration record, not how a person identifies, and the model estimates that
field rather than a person. Intended for completing missing fields in existing
records for aggregate analysis, not for decisions about individuals.

## Links

- Source, results, and paper: https://github.com/ericks-rs/indonesian-name-gender
- License: MIT
