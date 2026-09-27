from .predictor import GenderPredictor

__version__ = "1.1.0"
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
