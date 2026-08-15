from .predictor import GenderPredictor

__version__ = "1.0.2"
__all__ = ["GenderPredictor"]

AVAILABLE_MODELS = [
    "CharBiGRU",
    "CharBiLSTM",
    "CharBiRNN",
    "CharTransformer",
    "WordTransformer",
    "WordBiLSTM",
    "WordBiGRU",
    "WordBiRNN",
]
