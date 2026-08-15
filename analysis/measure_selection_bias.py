"""How much the earlier figures owed to selecting on the evaluation set.

The scripts behind the submitted version monitored the 2024-2025 partition to
pick the best epoch, to trigger early stopping and to drive ReduceLROnPlateau,
and then reported metrics on that same partition. The revised grid does not.
seeds_grid_complete.py selects on the 2022-2023 development partition and scores
2024-2025 once, so the question this answers is historical. It is worth answering
anyway, because a reviewer looking at F1 falling from 0.9643 to 0.9593 deserves
to know how much of the drop is the rebuilt split and how much is the change of
selection signal.

CharBiGRU is trained twice under otherwise identical settings. Arm A selects on
2024-2025, the way the earlier work did. Arm B selects on 2022-2023 and touches
2024-2025 once at the end, matching the reported grid. The gap between the two
2024-2025 scores is the optimism the earlier protocol carried.

The architecture and the loop are copied from seeds_grid_complete.py, down to the
attention projection having no bias and the scheduler stepping on development
loss. A near-copy would answer a question about a different model.

Five seeds per arm, so the gap can be read against the seed spread.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
# nothing here imports from program/ any more. The architecture and the loop
# are written out in this file so the arms answer for the reported model.
DATA = ROOT / "data" / "splits"
OUT = ROOT / "results" / "final" / "14_selection_bias"

SEEDS = [42, 7, 123, 2024, 777]
DEV_FROM_YEAR = 2022          # 2022 and 2023 become the development split in arm B
CFG = dict(CHAR_MAX_LEN=50, CHAR_EMB_DIM=48, HIDDEN_DIM=192, DROPOUT=0.3,
           LR=1e-3, BATCH_SIZE=512, EPOCHS=50, PATIENCE=6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharTok:
    def __init__(self, names):
        chars = sorted({c for n in names for c in str(n).lower()})
        self.stoi = {c: i + 2 for i, c in enumerate(chars)}
        self.size = len(self.stoi) + 2

    def encode(self, name, n):
        ids = [self.stoi.get(c, 1) for c in str(name).lower()[:n]]
        return ids + [0] * (n - len(ids))


class CharBiGRU(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, CFG["CHAR_EMB_DIM"], padding_idx=0)
        self.rnn = nn.GRU(CFG["CHAR_EMB_DIM"], CFG["HIDDEN_DIM"] // 2,
                          batch_first=True, bidirectional=True)
        # no bias, matching the Attention module in the grid
        self.attn = nn.Linear(CFG["HIDDEN_DIM"], 1, bias=False)
        self.drop = nn.Dropout(CFG["DROPOUT"])
        self.out = nn.Linear(CFG["HIDDEN_DIM"], 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        mask = (x != 0).unsqueeze(-1)
        a = self.attn(h).masked_fill(~mask, -1e9).softmax(1)
        return self.out(self.drop((h * a).sum(1))).squeeze(-1)


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loader(df, tok, seed=None, shuffle=False):
    X = torch.tensor([tok.encode(n, CFG["CHAR_MAX_LEN"]) for n in df.NAMA], dtype=torch.long)
    y = torch.tensor((df.LABEL == "P").astype(float).values, dtype=torch.float)
    g = None
    if shuffle:
        g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=CFG["BATCH_SIZE"],
                      shuffle=shuffle, generator=g)


@torch.no_grad()
def score(model, dl):
    model.eval()
    p, t = [], []
    for xb, yb in dl:
        p.append((torch.sigmoid(model(xb.to(DEVICE))) > 0.5).float().cpu().numpy())
        t.append(yb.numpy())
    return f1_score(np.concatenate(t), np.concatenate(p))


def train(train_df, select_df, test_dl, tok, seed, pos_w):
    seed_all(seed)
    tr_dl = make_loader(train_df, tok, seed, shuffle=True)
    sel_dl = make_loader(select_df, tok)
    model = CharBiGRU(tok.size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(DEVICE))
    best, wait, best_state, best_ep = 0.0, 0, None, 0
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); opt.step()
        f = score(model, sel_dl)
        sch.step(1 - f)
        if f > best:
            best, best_ep, wait = f, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["PATIENCE"]:
                break
    model.load_state_dict(best_state)
    return score(model, test_dl), best_ep


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tr21 = pd.read_csv(DATA / "train_1990_2021.csv")
    dev = pd.read_csv(DATA / "dev_2022_2023.csv")
    tr23 = pd.read_csv(DATA / "train_1990_2023.csv")
    te = pd.read_csv(DATA / "val_2024_2025.csv")

    # built on training text only, like the tokenizer the grid uses. Fitting it
    # over the evaluation names as well would be a second, smaller leak inside a
    # script whose subject is leakage.
    tok = CharTok(tr23.NAMA)
    test_dl = make_loader(te, tok)

    pw = lambda d: (d.LABEL == "L").sum() / (d.LABEL == "P").sum()

    # Arm A is what the submitted version did. B changes the training window only.
    # C changes the selection signal only. Comparing A to B isolates the data, B to
    # C isolates the selection, and A to C is the whole distance between the two
    # protocols. An earlier version of this script compared A against C directly and
    # called the difference optimism, which folded 16,882 training names into a
    # number meant to be about leakage.
    ARMS = [("A", "train 1990-2023, select on 2024-2025", tr23, te),
            ("B", "train 1990-2021, select on 2024-2025", tr21, te),
            ("C", "train 1990-2021, select on 2022-2023", tr21, dev)]
    for tag, what, trd, seld in ARMS:
        print(f"arm {tag}: {what}, {len(trd):,} training names, "
              f"{len(seld):,} for selection")
    print(flush=True)

    rows = []
    for seed in SEEDS:
        rec = {"seed": seed}
        for tag, _, trd, seld in ARMS:
            t0 = time.time()
            f1, ep = train(trd, seld, test_dl, tok, seed, pw(trd))
            rec[f"arm{tag}_test_f1"], rec[f"arm{tag}_best_epoch"] = f1, ep
            print(f"  seed {seed:<5} arm {tag}  F1 {f1:.4f}  ep {ep:<3} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["data_effect_pp"] = (df.armA_test_f1 - df.armB_test_f1) * 100
    df["selection_effect_pp"] = (df.armB_test_f1 - df.armC_test_f1) * 100
    df["total_pp"] = (df.armA_test_f1 - df.armC_test_f1) * 100
    df.to_csv(OUT / "selection_bias_charbigru.csv", index=False)

    tc = 2.776   # t(0.975, df = 4)
    summary = []
    for col, label in (("data_effect_pp", "training window, 1990-2023 against 1990-2021"),
                       ("selection_effect_pp", "selection signal, 2024-2025 against 2022-2023"),
                       ("total_pp", "both changes together")):
        v = df[col]
        se = v.std(ddof=1) / np.sqrt(len(v))
        summary.append({"effect": label, "mean_pp": round(v.mean(), 3),
                        "sd_pp": round(v.std(ddof=1), 3),
                        "ci95_lo_pp": round(v.mean() - tc * se, 3),
                        "ci95_hi_pp": round(v.mean() + tc * se, 3)})
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "selection_bias_decomposition.csv", index=False)

    print()
    for tag in "ABC":
        c = df[f"arm{tag}_test_f1"]
        print(f"arm {tag} mean {c.mean():.4f} (sd {c.std(ddof=1):.4f})")
    print()
    print(sm.to_string(index=False))
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
