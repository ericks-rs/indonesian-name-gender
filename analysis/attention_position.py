from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).parent.parent
if os.name == "nt":
    for p in [os.path.join(sys.prefix, "Library", "bin"),
              os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")] +\
             glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")):
        if os.path.isdir(p):
            os.add_dll_directory(p)

SRC = ROOT / "results" / "final" / "24_grid_attention_pooling"
CKPT = SRC / "checkpoints"
OUT = ROOT / "results" / "final" / "26_attention_position"
DATA = ROOT / "data" / "splits"
TOK = ROOT / "results" / "tokenizers"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 7, 123, 2024, 777]
T_CRIT = 2.776
CFG = dict(CHAR_MAX_LEN=50, WORD_MAX_LEN=8, CHAR_EMB_DIM=48, WORD_EMB_DIM=96,
           HIDDEN_DIM=192, DROPOUT=0.3, TRF_CHAR_DIM=128, TRF_WORD_DIM=192,
           N_HEADS=8, N_LAYERS=3, FF_MULTIPLIER=4)
MODELS = ["CharBiRNN", "CharBiLSTM", "CharBiGRU", "CharTransformer",
          "WordBiRNN", "WordBiLSTM", "WordBiGRU", "WordTransformer"]

FEM = ("wati", "ati", "ani", "ika", "ita", "sih", "ningsih", "ah", "iyah", "yanti")
MAS = ("wan", "man", "din", "udin", "anto", "arto", "yanto", "ono", "adi", "aji")

class CharTokenizer:
    def __init__(self): self.char2idx = {"<PAD>": 0, "<UNK>": 1}
    def encode(self, name, n):
        ids = [self.char2idx.get(c, 1) for c in name.lower()]
        return ids[:n] if len(ids) >= n else ids + [0] * (n - len(ids))
    @property
    def vocab_size(self): return len(self.char2idx)

class WordTokenizer:
    def __init__(self, min_freq=2): self.word2idx = {"<PAD>": 0, "<UNK>": 1}; self.min_freq = min_freq
    def encode(self, name, n):
        ids = [self.word2idx.get(w, 1) for w in name.lower().split()]
        return ids[:n] if len(ids) >= n else ids + [0] * (n - len(ids))
    @property
    def vocab_size(self): return len(self.word2idx)

sys.modules["__main__"].CharTokenizer = CharTokenizer
sys.modules["__main__"].WordTokenizer = WordTokenizer
char_tok = pickle.load(open(TOK / "char_tokenizer.pkl", "rb"))
word_tok = pickle.load(open(TOK / "word_tokenizer.pkl", "rb"))

class Attention(nn.Module):
    def __init__(self, hd):
        super().__init__(); self.attn = nn.Linear(hd, 1, bias=False)
    def forward(self, out, mask):
        s = self.attn(out).squeeze(-1).masked_fill(mask == 0, -1e9)
        w = torch.softmax(s, dim=1)
        return (out * w.unsqueeze(-1)).sum(dim=1), w

class BiRNNAttn(nn.Module):
    RNN = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
    def __init__(self, vocab, emb, hidden, dropout, rnn_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = self.RNN[rnn_type](emb, hidden // 2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden); self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(self.embedding(x))
        _, w = self.attention(out, (x != 0).float())
        return w

class TransformerAttn(nn.Module):
    def __init__(self, vocab, d, heads, layers, ff, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.attention = Attention(d)
        self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(d, 1)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad = (x == 0)
        out = self.encoder(self.tok_emb(x) + self.pos_emb(pos), src_key_padding_mask=pad)
        _, w = self.attention(self.norm(out), (~pad).float())
        return w

def make(name):
    if name.endswith("Transformer"):
        if name.startswith("Char"):
            return TransformerAttn(char_tok.vocab_size, CFG["TRF_CHAR_DIM"], CFG["N_HEADS"],
                                   CFG["N_LAYERS"], CFG["TRF_CHAR_DIM"] * CFG["FF_MULTIPLIER"],
                                   CFG["CHAR_MAX_LEN"], CFG["DROPOUT"]), "char"
        return TransformerAttn(word_tok.vocab_size, CFG["TRF_WORD_DIM"], CFG["N_HEADS"],
                               CFG["N_LAYERS"], CFG["TRF_WORD_DIM"] * CFG["FF_MULTIPLIER"],
                               CFG["WORD_MAX_LEN"], CFG["DROPOUT"]), "word"
    cell = {"BiRNN": "rnn", "BiLSTM": "lstm", "BiGRU": "gru"}[
        name.replace("Char", "").replace("Word", "")]
    if name.startswith("Char"):
        return BiRNNAttn(char_tok.vocab_size, CFG["CHAR_EMB_DIM"], CFG["HIDDEN_DIM"],
                         CFG["DROPOUT"], cell), "char"
    return BiRNNAttn(word_tok.vocab_size, CFG["WORD_EMB_DIM"], CFG["HIDDEN_DIM"],
                     CFG["DROPOUT"], cell), "word"

@torch.no_grad()
def weights_for(model, ids, batch=1024):
    out = []
    for i in range(0, len(ids), batch):
        xb = torch.tensor(ids[i:i + batch], dtype=torch.long, device=DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)

def summarise(W, lengths, k=4):
    n = len(W)
    tail = np.zeros(n); head = np.zeros(n)
    last = np.zeros(n); first = np.zeros(n); com = np.zeros(n)
    for i in range(n):
        L = int(lengths[i])
        w = W[i][:L]
        s = w.sum()
        w = w / s if s > 0 else w
        tail[i] = w[-min(k, L):].sum()
        head[i] = w[:min(k, L)].sum()
        last[i] = w[-1]
        first[i] = w[0]

        com[i] = float((w * np.arange(L)).sum() / (L - 1)) if L > 1 else np.nan
    return tail, head, last, first, com

def holm(p):
    order = np.argsort(p)
    out, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, p[i] * (len(p) - rank)))
        out[i] = run
    return out

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    te = pd.read_csv(DATA / "val_2024_2025.csv")
    names = te.NAMA.astype(str).tolist()
    y = (te.LABEL == "P").astype(int).values
    clen = np.array([min(len(s.lower()), CFG["CHAR_MAX_LEN"]) for s in names])
    wlen = np.array([min(len(s.lower().split()), CFG["WORD_MAX_LEN"]) for s in names])
    cids = np.stack([np.array(char_tok.encode(s, CFG["CHAR_MAX_LEN"])) for s in names])
    wids = np.stack([np.array(word_tok.encode(s, CFG["WORD_MAX_LEN"])) for s in names])
    low = [s.lower() for s in names]
    has_suffix = np.array([s.endswith(FEM) or s.endswith(MAS) for s in low])
    print(f"{len(names):,} names | {int(has_suffix.sum()):,} end in a listed suffix "
          f"| mean character length {clen.mean():.1f}", flush=True)

    rows, curves = [], {}
    for m in MODELS:
        lvl = "char" if m.startswith("Char") else "word"
        ids, lens = (cids, clen) if lvl == "char" else (wids, wlen)
        for seed in SEEDS:
            model, _ = make(m)
            model.load_state_dict(torch.load(CKPT / f"{m}__seed{seed}.pt", map_location="cpu"),
                                  strict=False)
            model = model.to(DEVICE).eval()
            W = weights_for(model, ids)
            tail, head, last, first, com = summarise(W, lens)

            unif_tail = np.minimum(4, lens) / lens
            rows.append({"Model": m, "Seed": seed, "level": lvl,
                         "tail4_mass": round(float(tail.mean()), 5),
                         "tail4_uniform": round(float(unif_tail.mean()), 5),
                         "tail4_over_uniform": round(float((tail - unif_tail).mean()), 5),
                         "head4_mass": round(float(head.mean()), 5),
                         "last_unit_mass": round(float(last.mean()), 5),
                         "first_unit_mass": round(float(first.mean()), 5),
                         "centre_of_mass": round(float(np.nanmean(com)), 5),
                         "tail4_with_suffix": round(float(tail[has_suffix].mean()), 5),
                         "tail4_without_suffix": round(float(tail[~has_suffix].mean()), 5)})
            if seed == 42 and lvl == "char":

                acc = np.zeros(12); cnt = np.zeros(12)
                for i in range(len(W)):
                    L = int(lens[i]); w = W[i][:L]
                    s = w.sum(); w = w / s if s > 0 else w
                    k = min(12, L)
                    acc[-k:] += w[-k:]; cnt[-k:] += 1
                curves[m] = np.round(acc / np.maximum(cnt, 1), 6)
            del model
            torch.cuda.empty_cache()
        print(f"  {m:<16} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "attention_position_per_seed.csv", index=False)
    if curves:
        pd.DataFrame(curves, index=[f"pos_-{i}" for i in range(12, 0, -1)]).to_csv(
            OUT / "attention_profile_last12_seed42.csv")

    s = df.groupby(["level", "Model"]).agg(
        tail4=("tail4_mass", "mean"), tail4_sd=("tail4_mass", "std"),
        uniform=("tail4_uniform", "mean"), over_uniform=("tail4_over_uniform", "mean"),
        head4=("head4_mass", "mean"), last_unit=("last_unit_mass", "mean"),
        first_unit=("first_unit_mass", "mean"),
        centre=("centre_of_mass", "mean")).round(4)
    s.to_csv(OUT / "attention_position_summary.csv")
    print("\n" + s.to_string())

    print("\nis the tail mass above what a uniform model would give, five seeds")
    tests = []
    for m in MODELS:
        g = df[df.Model == m]
        d = g.tail4_mass.values - g.tail4_uniform.values
        sd = d.std(ddof=1)
        half = T_CRIT * sd / np.sqrt(len(d))
        _, p = stats.ttest_1samp(d, 0.0)
        tests.append({"Model": m, "level": g.level.iloc[0], "excess": round(float(d.mean()), 4),
                      "ci95_lo": round(float(d.mean() - half), 4),
                      "ci95_hi": round(float(d.mean() + half), 4), "p_raw": float(p)})
    for r, adj in zip(tests, holm([t["p_raw"] for t in tests])):
        r["p_holm"] = adj
        r["above_uniform"] = bool(adj < 0.05 and r["excess"] > 0)
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "tail_mass_vs_uniform.csv", index=False)
    print(t[["Model", "level", "excess", "ci95_lo", "ci95_hi", "p_holm",
             "above_uniform"]].to_string(index=False))

    print("\nnames ending in a listed suffix against the rest, character models")
    for m in [x for x in MODELS if x.startswith("Char")]:
        g = df[df.Model == m]
        d = g.tail4_with_suffix.values - g.tail4_without_suffix.values
        sd = d.std(ddof=1)
        half = T_CRIT * sd / np.sqrt(len(d))
        _, p = stats.ttest_1samp(d, 0.0)
        print(f"  {m:<16} {d.mean():+.4f}  CI [{d.mean()-half:+.4f}, {d.mean()+half:+.4f}]  "
              f"p={p:.5f}")

    print(f"\nWritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
