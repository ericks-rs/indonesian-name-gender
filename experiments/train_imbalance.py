from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, glob, time, pickle
if os.name == "nt":
    env_root = sys.prefix
    for p in [
        os.path.join(env_root, "Library", "bin"),
        os.path.join(env_root, "Lib", "site-packages", "torch", "lib"),
    ] + glob.glob(os.path.join(env_root, "Lib", "site-packages", "nvidia", "*", "bin")):
        if os.path.isdir(p):
            os.add_dll_directory(p)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
TOKENIZERS_DIR = PROJECT_ROOT / "tokenizers"
OUT_DIR = RESULTS_DIR / "tables" / "imbalance_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 123, 2024, 777]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | Device: {DEVICE} | seeds={SEEDS}", flush=True)

CFG = dict(CHAR_MAX_LEN=50, WORD_MAX_LEN=8, CHAR_EMB_DIM=48, WORD_EMB_DIM=96,
           HIDDEN_DIM=192, DROPOUT=0.3, TRF_CHAR_DIM=128, TRF_WORD_DIM=192,
           N_HEADS=8, N_LAYERS=3, FF_MULTIPLIER=4,
           LR=1e-3, BATCH_SIZE=512, EPOCHS=50, PATIENCE=6)

df_train = pd.read_csv(DATA_DIR / "train_1990_2021.csv")
df_dev = pd.read_csv(DATA_DIR / "dev_2022_2023.csv")
df_val = pd.read_csv(DATA_DIR / "val_2024_2025.csv")

df_ext = pd.read_csv(PROJECT_ROOT / "data" / "external" / "indonesian-names.csv")
df_ext = df_ext.rename(columns={"name": "NAMA"})
df_ext["LABEL_ENC"] = df_ext["gender"].map({"m": 0, "f": 1})
df_train["LABEL_ENC"] = (df_train["LABEL"] == "P").astype(int)
df_val["LABEL_ENC"] = (df_val["LABEL"] == "P").astype(int)
df_dev["LABEL_ENC"] = (df_dev["LABEL"] == "P").astype(int)
n_neg = int((df_train["LABEL_ENC"] == 0).sum())
n_pos = int((df_train["LABEL_ENC"] == 1).sum())
pos_weight_val = n_neg / n_pos
print(f"Train {len(df_train):,} (M={n_neg:,} F={n_pos:,}, pos_weight={pos_weight_val:.4f}) | Val {len(df_val):,}", flush=True)

class CharTokenizer:
    def __init__(self): self.char2idx = {"<PAD>": 0, "<UNK>": 1}
    def encode(self, name, max_len):
        ids = [self.char2idx.get(ch, 1) for ch in name.lower()]
        return ids[:max_len] if len(ids) >= max_len else ids + [0]*(max_len-len(ids))
    @property
    def vocab_size(self): return len(self.char2idx)

class WordTokenizer:
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}; self.min_freq = min_freq
    def encode(self, name, max_len):
        ids = [self.word2idx.get(w, 1) for w in name.lower().split()]
        return ids[:max_len] if len(ids) >= max_len else ids + [0]*(max_len-len(ids))
    @property
    def vocab_size(self): return len(self.word2idx)

sys.modules["__main__"].CharTokenizer = CharTokenizer
sys.modules["__main__"].WordTokenizer = WordTokenizer
with open(TOKENIZERS_DIR / "char_tokenizer.pkl", "rb") as f: char_tok = pickle.load(f)
with open(TOKENIZERS_DIR / "word_tokenizer.pkl", "rb") as f: word_tok = pickle.load(f)
print(f"Tokenizers: char={char_tok.vocab_size} word={word_tok.vocab_size}", flush=True)

class NameDataset(Dataset):
    def __init__(self, df, ctok, wtok, cfg):
        self.names = df["NAMA"].values; self.labels = df["LABEL_ENC"].values
        self.ctok, self.wtok, self.cfg = ctok, wtok, cfg
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        n = self.names[i]
        return {"char_ids": torch.tensor(self.ctok.encode(n, self.cfg["CHAR_MAX_LEN"]), dtype=torch.long),
                "word_ids": torch.tensor(self.wtok.encode(n, self.cfg["WORD_MAX_LEN"]), dtype=torch.long),
                "label": torch.tensor(self.labels[i], dtype=torch.float)}

train_ds = NameDataset(df_train, char_tok, word_tok, CFG)
val_ds = NameDataset(df_val, char_tok, word_tok, CFG)
val_dl = DataLoader(val_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0, pin_memory=True)

dev_ds = NameDataset(df_dev, char_tok, word_tok, CFG)
dev_dl = DataLoader(dev_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0, pin_memory=True)
ext_ds = NameDataset(df_ext, char_tok, word_tok, CFG)
ext_dl = DataLoader(ext_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0, pin_memory=True)

class Attention(nn.Module):
    def __init__(self, hd):
        super().__init__(); self.attn = nn.Linear(hd, 1, bias=False)
    def forward(self, out, mask):
        s = self.attn(out).squeeze(-1).masked_fill(mask == 0, -1e9)
        return (out * F.softmax(s, dim=1).unsqueeze(-1)).sum(dim=1)

class BiRNNAttn(nn.Module):
    RNN = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
    def __init__(self, vocab, emb, hidden, dropout, rnn_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = self.RNN[rnn_type](emb, hidden//2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden); self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        mask = (x != 0).float()
        out, _ = self.rnn(self.embedding(x))
        return self.fc(self.dropout(self.attention(out, mask))).squeeze(1)

class TransformerClf(nn.Module):
    def __init__(self, vocab, d, heads, layers, ff, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d); self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(d, 1)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad = (x == 0)
        out = self.encoder(self.tok_emb(x) + self.pos_emb(pos), src_key_padding_mask=pad)
        nz = (~pad).float().unsqueeze(-1)
        pooled = (out * nz).sum(1) / nz.sum(1).clamp(min=1)
        return self.fc(self.dropout(self.norm(pooled))).squeeze(1)

SPECS_RNN = {
    "CharBiRNN":  (char_tok.vocab_size, CFG["CHAR_EMB_DIM"], "rnn",  "char_ids"),
    "CharBiLSTM": (char_tok.vocab_size, CFG["CHAR_EMB_DIM"], "lstm", "char_ids"),
    "CharBiGRU":  (char_tok.vocab_size, CFG["CHAR_EMB_DIM"], "gru",  "char_ids"),
    "WordBiRNN":  (word_tok.vocab_size, CFG["WORD_EMB_DIM"], "rnn",  "word_ids"),
    "WordBiLSTM": (word_tok.vocab_size, CFG["WORD_EMB_DIM"], "lstm", "word_ids"),
    "WordBiGRU":  (word_tok.vocab_size, CFG["WORD_EMB_DIM"], "gru",  "word_ids"),
}
SPECS_TRF = {
    "CharTransformer": (char_tok.vocab_size, CFG["TRF_CHAR_DIM"], CFG["CHAR_MAX_LEN"], "char_ids"),
    "WordTransformer": (word_tok.vocab_size, CFG["TRF_WORD_DIM"], CFG["WORD_MAX_LEN"], "word_ids"),
}

def make_model(name):
    if name in SPECS_RNN:
        v, e, rt, key = SPECS_RNN[name]
        return BiRNNAttn(v, e, CFG["HIDDEN_DIM"], CFG["DROPOUT"], rt), key
    v, d, ml, key = SPECS_TRF[name]
    return TransformerClf(v, d, CFG["N_HEADS"], CFG["N_LAYERS"],
                          d*CFG["FF_MULTIPLIER"], ml, CFG["DROPOUT"]), key

@torch.no_grad()
def evaluate(model, dl, crit, key):
    model.eval(); tl, P, L = 0.0, [], []
    for b in dl:
        x, y = b[key].to(DEVICE), b["label"].to(DEVICE)
        lg = model(x)
        tl += crit(lg, y).item() * len(y)
        P.extend((torch.sigmoid(lg) > 0.5).float().cpu().numpy()); L.extend(y.cpu().numpy())
    p, l = np.array(P), np.array(L)
    return {"loss": tl/len(l), "accuracy": accuracy_score(l, p),
            "precision": precision_score(l, p, zero_division=0),
            "recall": recall_score(l, p, zero_division=0),
            "f1": f1_score(l, p, zero_division=0)}

def make_loader(strategy, seed, g):
    if strategy == "oversampling":
        rng = np.random.RandomState(seed)
        idx = np.arange(len(df_train))
        pos, neg = idx[df_train.LABEL_ENC.values == 1], idx[df_train.LABEL_ENC.values == 0]
        extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
        d = df_train.iloc[np.concatenate([idx, extra])].reset_index(drop=True)
        ds = NameDataset(d, char_tok, word_tok, CFG)
        return DataLoader(ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                          num_workers=0, pin_memory=True, generator=g)
    if strategy == "balanced":
        y = df_train.LABEL_ENC.values
        w = np.where(y == 1, 1.0 / (y == 1).sum(), 1.0 / (y == 0).sum())
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(y), replacement=True, generator=g)
        return DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], sampler=sampler,
                          num_workers=0, pin_memory=True)
    return DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                      num_workers=0, pin_memory=True, generator=g)

def run(name, seed, strategy):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_dl = make_loader(strategy, seed, g)
    model, key = make_model(name)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    pw = pos_weight_val if strategy == "weighted" else 1.0
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    best_f1, patience, best_state = 0.0, 0, None
    best_ep = 0
    t_start = time.time()
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for b in train_dl:
            x, y = b[key].to(DEVICE), b["label"].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        dm = evaluate(model, dev_dl, crit, key)
        sch.step(dm["loss"])
        if dm["f1"] > best_f1:
            best_f1 = dm["f1"]; best_ep = ep + 1; patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= CFG["PATIENCE"]:
                break
    model.load_state_dict(best_state)
    best = evaluate(model, val_dl, crit, key)
    dur = time.time() - t_start
    del model, opt
    torch.cuda.empty_cache()
    return best, best_ep, dur

MODELS = ["CharBiRNN", "CharBiLSTM", "CharBiGRU", "CharTransformer",
          "WordBiRNN", "WordBiLSTM", "WordBiGRU", "WordTransformer"]
STRATEGIES = ["unweighted", "oversampling", "balanced"]

rows = []
runs_csv = OUT_DIR / "imbalance_runs.csv"
total = len(MODELS) * len(SEEDS) * len(STRATEGIES)
i = 0
for strategy in STRATEGIES:
    for name in MODELS:
        for seed in SEEDS:
            i += 1
            m, ep, dur = run(name, seed, strategy)
            rows.append({"Strategy": strategy, "Model": name, "Seed": seed,
                         "Accuracy": round(m["accuracy"], 6),
                         "Precision": round(m["precision"], 6),
                         "Recall": round(m["recall"], 6), "F1": round(m["f1"], 6),
                         "Best_epoch": ep, "Train_s": round(dur, 1)})
            pd.DataFrame(rows).to_csv(runs_csv, index=False)
            print(f"[{i:>3}/{total}] {strategy:<13} {name:<16} seed={seed:<5} "
                  f"F1={m['f1']:.4f} (best ep {ep}, {dur:.0f}s)", flush=True)

print("")
print("wrote " + str(runs_csv), flush=True)
