import pickle
from pathlib import Path
import torch

from .models import BiRNNAttn, TransformerClf
from . import tokenizers as _tokenizers_module
from .tokenizers import CharTokenizer, WordTokenizer
from .download import download_model

class _TokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("CharTokenizer", "WordTokenizer"):
            return getattr(_tokenizers_module, name)
        return super().find_class(module, name)

def _load_tokenizer(path):
    with open(path, "rb") as f:
        return _TokenizerUnpickler(f).load()

CFG = {
    "CHAR_MAX_LEN": 50, "WORD_MAX_LEN": 8,
    "CHAR_EMB_DIM": 48, "WORD_EMB_DIM": 96,
    "HIDDEN_DIM": 192, "DROPOUT": 0.3,
    "TRF_CHAR_DIM": 128, "TRF_WORD_DIM": 192,
    "N_HEADS": 8, "N_LAYERS": 3, "FF_MULTIPLIER": 4,
}

MODEL_CONFIGS = {
    "CharBiRNN":       ("rnn",   "char", CFG["CHAR_EMB_DIM"]),
    "CharBiLSTM":      ("lstm",  "char", CFG["CHAR_EMB_DIM"]),
    "CharBiGRU":       ("gru",   "char", CFG["CHAR_EMB_DIM"]),
    "CharTransformer": ("trf",   "char", CFG["TRF_CHAR_DIM"]),
    "WordBiRNN":       ("rnn",   "word", CFG["WORD_EMB_DIM"]),
    "WordBiGRU":       ("gru",   "word", CFG["WORD_EMB_DIM"]),
    "WordBiLSTM":      ("lstm",  "word", CFG["WORD_EMB_DIM"]),
    "WordTransformer": ("trf",   "word", CFG["TRF_WORD_DIM"]),
}

LABEL_MAP = {0: "Male", 1: "Female"}

PACKAGE_DATA = Path(__file__).parent / "data"

class GenderPredictor:

    def __init__(self, model: str = "CharBiLSTM", device: str = None):
        if model not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(MODEL_CONFIGS)}"
            )
        self.model_name = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.char_tok = _load_tokenizer(PACKAGE_DATA / "char_tokenizer.pkl")
        self.word_tok = _load_tokenizer(PACKAGE_DATA / "word_tokenizer.pkl")

        kind, embed_level, dim = MODEL_CONFIGS[model]
        tok = self.char_tok if embed_level == "char" else self.word_tok

        if model == "CharBiLSTM":
            pt_path = PACKAGE_DATA / "CharBiLSTM.pt"
            if not pt_path.exists():
                pt_path = download_model(model)
        else:
            pt_path = download_model(model)

        if kind == "trf":
            max_len = CFG["CHAR_MAX_LEN"] if embed_level == "char" else CFG["WORD_MAX_LEN"]
            self.model = TransformerClf(
                vocab_size=tok.vocab_size, d_model=dim,
                n_heads=CFG["N_HEADS"], n_layers=CFG["N_LAYERS"],
                ff_dim=dim * CFG["FF_MULTIPLIER"], max_len=max_len,
                dropout=CFG["DROPOUT"],
            )
        else:
            self.model = BiRNNAttn(
                vocab_size=tok.vocab_size, emb_dim=dim,
                hidden_dim=CFG["HIDDEN_DIM"], dropout=CFG["DROPOUT"],
                rnn_type=kind,
            )

        state = torch.load(pt_path, map_location="cpu")
        try:
            self.model.load_state_dict(state)
        except RuntimeError as e:

            raise RuntimeError(
                f"{pt_path.name} does not match the architecture in this version of "
                f"indonamegender. Weights cached from an earlier release pool their "
                f"Transformer by mean and carry no attention projection. Delete the "
                f"cached file and let the package fetch it again. Original error: {e}"
            ) from None
        self.model.eval().to(self.device)

    def _encode(self, name: str) -> torch.Tensor:
        _, embed_level, _ = MODEL_CONFIGS[self.model_name]
        tok = self.char_tok if embed_level == "char" else self.word_tok
        max_len = CFG["CHAR_MAX_LEN"] if embed_level == "char" else CFG["WORD_MAX_LEN"]
        ids = tok.encode(name, max_len)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict(self, name: str) -> dict:
        x = self._encode(name)
        logit = self.model(x).item()
        prob_p = float(torch.sigmoid(torch.tensor(logit)))
        pred = 1 if prob_p >= 0.5 else 0
        return {
            "name": name,
            "gender": LABEL_MAP[pred],
            "confidence": prob_p if pred == 1 else (1 - prob_p),
            "prob_female": prob_p,
            "prob_male": 1 - prob_p,
            "model": self.model_name,
        }

    def predict_batch(self, names: list) -> list:
        return [self.predict(n) for n in names]

    @torch.no_grad()
    def get_attention(self, name: str) -> dict:
        x = self._encode(name)
        logit, weights = self.model(x, return_attention=True)
        prob_p = float(torch.sigmoid(logit).item())
        pred = 1 if prob_p >= 0.5 else 0

        _, embed_level, _ = MODEL_CONFIGS[self.model_name]
        max_len = CFG["CHAR_MAX_LEN"] if embed_level == "char" else CFG["WORD_MAX_LEN"]

        if embed_level == "char":
            tokens = list(name.lower())[:max_len]
        else:
            tokens = name.lower().split()[:max_len]

        w = weights[0].cpu().tolist()[:len(tokens)]
        total = sum(w) if sum(w) > 0 else 1.0
        w_norm = [v / total for v in w]

        return {
            "name": name,
            "gender": LABEL_MAP[pred],
            "confidence": prob_p if pred == 1 else (1 - prob_p),
            "tokens": tokens,
            "attention": w_norm,
            "model": self.model_name,
            "level": embed_level,
        }

def compare_models(name: str, device: str = None) -> dict:
    results = {}
    for m in MODEL_CONFIGS.keys():
        gp = GenderPredictor(model=m, device=device)
        results[m] = gp.predict(name)
    return results
