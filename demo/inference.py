from __future__ import annotations
import hashlib

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, glob
if os.name == "nt":
    env_root = sys.prefix
    for p in [
        os.path.join(env_root, "Library", "bin"),
        os.path.join(env_root, "Lib", "site-packages", "torch", "lib"),
    ] + glob.glob(os.path.join(env_root, "Lib", "site-packages", "nvidia", "*", "bin")):
        if os.path.isdir(p):
            os.add_dll_directory(p)

import pickle
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharTokenizer:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}

    def encode(self, name, max_len):
        ids = [self.char2idx.get(ch, 1) for ch in name.lower()]
        return ids[:max_len] if len(ids) >= max_len else ids + [0] * (max_len - len(ids))

    @property
    def vocab_size(self):
        return len(self.char2idx)

VOCAB_SALT = "indonamegender-v1"

def vocab_key(word):
    if word.startswith("<") and word.endswith(">"):
        return word
    return hashlib.blake2s((VOCAB_SALT + word).encode("utf-8"),
                           digest_size=8).hexdigest()

class WordTokenizer:
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.min_freq = min_freq
        self.hashed = True

    def _key(self, word):

        return vocab_key(word) if getattr(self, "hashed", False) else word

    def encode(self, name, max_len):
        tokens = name.lower().split()
        ids = [self.word2idx.get(self._key(w), 1) for w in tokens]
        return ids[:max_len] if len(ids) >= max_len else ids + [0] * (max_len - len(ids))

    @property
    def vocab_size(self):
        return len(self.word2idx)

import __main__
__main__.CharTokenizer = CharTokenizer
__main__.WordTokenizer = WordTokenizer

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_out, mask):
        scores = self.attn(rnn_out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        return (rnn_out * weights.unsqueeze(-1)).sum(dim=1), weights

class BiRNNAttn(nn.Module):
    RNN_CLASSES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = self.RNN_CLASSES[rnn_type](
            emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attention=False):
        mask = (x != 0).float()
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        attn_out, weights = self.attention(rnn_out, mask)
        attn_out = self.dropout(attn_out)
        logits = self.fc(attn_out).squeeze(-1)
        if return_attention:
            return logits, weights
        return logits

class TransformerClf(nn.Module):

    def __init__(self, vocab_size, d_model, n_heads, n_layers, ff_dim, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.attention = Attention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.n_heads = n_heads

    def forward(self, x, return_attention=False):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad_mask = (x == 0)
        out = self.encoder(self.tok_emb(x) + self.pos_emb(positions),
                           src_key_padding_mask=pad_mask)
        pooled, weights = self.attention(self.norm(out), (~pad_mask).float())
        logits = self.fc(self.dropout(pooled)).squeeze(-1)
        if return_attention:
            return logits, weights
        return logits

CFG = {
    "CHAR_MAX_LEN": 50, "WORD_MAX_LEN": 8,
    "CHAR_EMB_DIM": 48, "WORD_EMB_DIM": 96,
    "HIDDEN_DIM": 192, "DROPOUT": 0.3,
    "TRF_CHAR_DIM": 128, "TRF_WORD_DIM": 192,
    "N_HEADS": 8, "N_LAYERS": 3, "FF_MULTIPLIER": 4,
}

class Predictor:
    LABEL_MAP = {0: "L", 1: "P"}
    LABEL_DESC = {"L": "Laki-laki", "P": "Perempuan"}

    def __init__(self, results_dir: str | Path, model_suffix: str = "",
                 models_dir: str | Path | None = None):
        self.results_dir = Path(results_dir)
        self.model_suffix = model_suffix
        self.models_dir = Path(models_dir) if models_dir else self.results_dir / "models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.results_dir / "tokenizers" / "char_tokenizer.pkl", "rb") as f:
            self.char_tok: CharTokenizer = pickle.load(f)
        with open(self.results_dir / "tokenizers" / "word_tokenizer.pkl", "rb") as f:
            self.word_tok: WordTokenizer = pickle.load(f)

        self.model_configs = {
            "CharBiRNN":       ("rnn",   self.char_tok, CFG["CHAR_EMB_DIM"]),
            "CharBiLSTM":      ("lstm",  self.char_tok, CFG["CHAR_EMB_DIM"]),
            "CharBiGRU":       ("gru",   self.char_tok, CFG["CHAR_EMB_DIM"]),
            "WordBiRNN":       ("rnn",   self.word_tok, CFG["WORD_EMB_DIM"]),
            "WordBiLSTM":      ("lstm",  self.word_tok, CFG["WORD_EMB_DIM"]),
            "WordBiGRU":       ("gru",   self.word_tok, CFG["WORD_EMB_DIM"]),
            "CharTransformer": ("trf",   self.char_tok, CFG["TRF_CHAR_DIM"]),
            "WordTransformer": ("trf",   self.word_tok, CFG["TRF_WORD_DIM"]),
        }

        self.models: dict[str, nn.Module] = {}
        self.missing: list[str] = []
        for name, (kind, tok, dim) in self.model_configs.items():
            pt_path = self.models_dir / f"{name}{model_suffix}.pt"
            if not pt_path.exists():
                self.missing.append(name)
                continue
            if kind == "trf":
                is_char = "Char" in name
                max_len = CFG["CHAR_MAX_LEN"] if is_char else CFG["WORD_MAX_LEN"]
                model = TransformerClf(
                    vocab_size=tok.vocab_size, d_model=dim,
                    n_heads=CFG["N_HEADS"], n_layers=CFG["N_LAYERS"],
                    ff_dim=dim * CFG["FF_MULTIPLIER"], max_len=max_len,
                    dropout=CFG["DROPOUT"],
                )
            else:
                model = BiRNNAttn(
                    vocab_size=tok.vocab_size, emb_dim=dim,
                    hidden_dim=CFG["HIDDEN_DIM"], dropout=CFG["DROPOUT"],
                    rnn_type=kind,
                )
            state = torch.load(pt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.models[name] = model

        suffix_lbl = f" (suffix={model_suffix!r})" if model_suffix else ""
        print(f"[Predictor] Loaded {len(self.models)} models{suffix_lbl} on {self.device}")
        if self.missing:
            print(f"[Predictor] WARNING, {len(self.missing)} checkpoint(s) not found in "
                  f"{self.models_dir}: {', '.join(self.missing)}")

    def _encode(self, name: str, model_name: str):
        is_char = "Char" in model_name
        tok = self.char_tok if is_char else self.word_tok
        max_len = CFG["CHAR_MAX_LEN"] if is_char else CFG["WORD_MAX_LEN"]
        ids = tok.encode(name, max_len)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict_single(self, name: str, model_name: str = "CharBiLSTM"):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models)}")
        x = self._encode(name, model_name)
        logit = self.models[model_name](x).item()
        prob_p = float(torch.sigmoid(torch.tensor(logit)))
        pred = 1 if prob_p >= 0.5 else 0
        return {
            "model": model_name,
            "name": name,
            "label": self.LABEL_MAP[pred],
            "label_desc": self.LABEL_DESC[self.LABEL_MAP[pred]],
            "confidence": prob_p if pred == 1 else (1 - prob_p),
            "prob_female": prob_p,
            "prob_male": 1 - prob_p,
        }

    @torch.no_grad()
    def predict_all(self, name: str):
        return [self.predict_single(name, m) for m in self.models]

    @torch.no_grad()
    def predict_with_attention(self, name: str, model_name: str = "CharBiLSTM"):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        x = self._encode(name, model_name)
        logits, weights = self.models[model_name](x, return_attention=True)
        prob_p = float(torch.sigmoid(logits).item())
        pred = 1 if prob_p >= 0.5 else 0

        is_char = "Char" in model_name
        max_len = CFG["CHAR_MAX_LEN"] if is_char else CFG["WORD_MAX_LEN"]

        if is_char:
            tokens = list(name.lower())[:max_len]
        else:
            tokens = name.lower().split()[:max_len]

        w = weights[0].cpu().tolist()[:len(tokens)]

        total = sum(w) if sum(w) > 0 else 1.0
        w_norm = [v / total for v in w]

        return {
            "model": model_name,
            "name": name,
            "label": self.LABEL_MAP[pred],
            "label_desc": self.LABEL_DESC[self.LABEL_MAP[pred]],
            "confidence": prob_p if pred == 1 else (1 - prob_p),
            "prob_female": prob_p,
            "prob_male": 1 - prob_p,
            "tokens": tokens,
            "attention": w_norm,
            "level": "char" if is_char else "word",
        }

    @property
    def available_models(self):
        return list(self.models.keys())

if __name__ == "__main__":

    pred = Predictor(Path(__file__).parent.parent / "results")
    print("\n=== Single prediction ===")
    print(pred.predict_single("BANOWATI LARASATI"))
    print("\n=== All models ===")
    for r in pred.predict_all("ANTAREJA NURUDIN"):
        print(f"  {r['model']:<18} {r['label']} ({r['confidence']*100:.1f}%)")
    print("\n=== With attention ===")
    r = pred.predict_with_attention("ERICKS RAMA", "CharBiLSTM")
    print(f"  Predicted: {r['label']} ({r['confidence']*100:.1f}%)")
    for tok, attn in zip(r["tokens"], r["attention"]):
        bar = "#" * int(attn * 80)
        print(f"  '{tok}'  {attn:.3f}  {bar}")
