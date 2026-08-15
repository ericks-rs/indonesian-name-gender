"""Pretrained baselines across five random seeds.

Reviewer B2 asked for repeated runs behind the reported improvements. The
parity claim against IndoBERT, mBERT and XLM-R was added later, so it was still
resting on a single run while the eight grid models already had five seeds.
This script closes that gap and makes a paired test across seeds possible.

Only the seed-42 checkpoint is kept. Storing all fifteen would cost roughly
7 GB and nothing downstream reads the other twelve.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
EXTERNAL_CSV = PROJECT_ROOT / "data" / "external" / "indonesian-names.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "final" / "04_seeds_transformers"
CKPT_DIR = PROJECT_ROOT / "results" / "models" / "transformer_baselines"

MODELS = {
    "IndoBERT": "indobenchmark/indobert-base-p1",
    "mBERT": "bert-base-multilingual-cased",
    "XLM-R": "xlm-roberta-base",
}
CFG = dict(MAX_LEN=32, BATCH_SIZE=32, LR=2e-5, EPOCHS=10, PATIENCE=2)
SEEDS = [42, 7, 123, 2024, 777]
KEEP_CKPT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    """Seed before the model is built, otherwise initialisation is not covered."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NameDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = list(texts), list(labels)
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    preds, labels = [], []
    for b in dl:
        logits = model(input_ids=b["input_ids"].to(DEVICE),
                       attention_mask=b["attention_mask"].to(DEVICE)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(b["label"].numpy())
    p, y = np.concatenate(preds), np.concatenate(labels)
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p),
            "recall": recall_score(y, p), "f1": f1_score(y, p)}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(DATA_DIR / "train_1990_2021.csv")
    dv = pd.read_csv(DATA_DIR / "dev_2022_2023.csv")
    va = pd.read_csv(DATA_DIR / "val_2024_2025.csv")
    for d in (tr, dv, va):
        d["label"] = (d["LABEL"] == "P").astype(int)
        d["text"] = d["NAMA"].str.title()
    ext = pd.read_csv(EXTERNAL_CSV)
    ext["label"] = ext["gender"].map({"m": 0, "f": 1})
    ext["text"] = ext["name"].str.title()

    n_neg, n_pos = int((tr.label == 0).sum()), int((tr.label == 1).sum())
    cw = torch.tensor([(n_neg + n_pos) / (2.0 * n_neg), (n_neg + n_pos) / (2.0 * n_pos)],
                      dtype=torch.float).to(DEVICE)
    print(f"Train {len(tr):,} | Dev {len(dv):,} | Test {len(va):,} | External {len(ext):,} | device {DEVICE}", flush=True)

    runs = []
    val_preds = {"NAMA": va["NAMA"].values, "label": va["label"].values}
    ext_preds = {"name": ext["name"].values, "label": ext["label"].values}

    for mname, hf_id in MODELS.items():
        for seed in SEEDS:
            tag = f"{mname}__seed{seed}"
            print(f"\n{'='*66}\n{tag}\n{'='*66}", flush=True)
            seed_everything(seed)

            tok = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())

            g = torch.Generator()
            g.manual_seed(seed)
            train_dl = DataLoader(NameDS(tr.text, tr.label, tok, CFG["MAX_LEN"]),
                                  batch_size=CFG["BATCH_SIZE"], shuffle=True,
                                  generator=g, num_workers=0, pin_memory=True)
            val_dl = DataLoader(NameDS(va.text, va.label, tok, CFG["MAX_LEN"]),
                                batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
                                num_workers=0, pin_memory=True)
            dev_dl = DataLoader(NameDS(dv.text, dv.label, tok, CFG["MAX_LEN"]),
                                batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
                                num_workers=0, pin_memory=True)
            ext_dl = DataLoader(NameDS(ext.text, ext.label, tok, CFG["MAX_LEN"]),
                                batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
                                num_workers=0, pin_memory=True)

            opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"])
            crit = nn.CrossEntropyLoss(weight=cw)

            best_f1, patience, best_state, best_ep = 0.0, 0, None, 0
            t_start = time.time()
            for ep in range(CFG["EPOCHS"]):
                model.train()
                t0, tl, seen = time.time(), 0.0, 0
                for b in train_dl:
                    y = b["label"].to(DEVICE)
                    opt.zero_grad()
                    logits = model(input_ids=b["input_ids"].to(DEVICE),
                                   attention_mask=b["attention_mask"].to(DEVICE)).logits
                    loss = crit(logits, y)
                    loss.backward()
                    opt.step()
                    tl += loss.item() * len(y)
                    seen += len(y)
                vm, _ = evaluate(model, dev_dl)
                mark = ""
                if vm["f1"] > best_f1:
                    best_f1, best_ep, patience = vm["f1"], ep + 1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    mark = " *"
                else:
                    patience += 1
                print(f"  Ep {ep+1} | tl={tl/seen:.4f} | val_f1={vm['f1']:.4f} | "
                      f"{time.time()-t0:.0f}s{mark}", flush=True)
                if patience >= CFG["PATIENCE"]:
                    print(f"  Early stop at ep {ep+1}", flush=True)
                    break

            model.load_state_dict(best_state)
            vmet, vp = evaluate(model, val_dl)
            emet, epd = evaluate(model, ext_dl)
            val_preds[f"pred_{tag}"] = vp
            ext_preds[f"pred_{tag}"] = epd

            if seed == KEEP_CKPT_SEED:
                torch.save({"state_dict": best_state, "hf_id": hf_id, "n_params": n_params},
                           CKPT_DIR / f"{mname}.pt")

            runs.append({"Model": mname, "Seed": seed, "Params": n_params,
                         "best_epoch": best_ep, "train_s": round(time.time() - t_start, 1),
                         **{f"val_{k}": v for k, v in vmet.items()},
                         **{f"ext_{k}": v for k, v in emet.items()}})
            print(f"  val F1={vmet['f1']:.4f} | external F1={emet['f1']:.4f} | "
                  f"{runs[-1]['train_s']}s", flush=True)

            del model, opt, best_state
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            pd.DataFrame(runs).to_csv(out / "transformer_seed_runs.csv", index=False)

    runs_df = pd.DataFrame(runs)
    pd.DataFrame(val_preds).to_csv(out / "transformer_seed_val_predictions.csv", index=False)
    pd.DataFrame(ext_preds).to_csv(out / "transformer_seed_external_predictions.csv", index=False)

    rows = []
    for model, g in runs_df.groupby("Model", sort=False):
        row = {"Model": model, "Params": int(g.Params.iloc[0]), "n_seeds": len(g)}
        for m in ["val_accuracy", "val_precision", "val_recall", "val_f1", "ext_f1"]:
            v = g[m].values
            sd = v.std(ddof=1)
            half = 2.776 * sd / np.sqrt(len(v))
            row[f"{m}_mean"] = round(v.mean(), 4)
            row[f"{m}_std"] = round(sd, 4)
            row[f"{m}_ci95_lo"] = round(v.mean() - half, 4)
            row[f"{m}_ci95_hi"] = round(v.mean() + half, 4)
        row["val_f1_min"], row["val_f1_max"] = round(g.val_f1.min(), 4), round(g.val_f1.max(), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "transformer_seed_summary.csv", index=False)

    print("\n" + summary[["Model", "n_seeds", "val_f1_mean", "val_f1_std",
                          "val_f1_min", "val_f1_max", "ext_f1_mean"]].to_string(index=False))
    print(f"\nTotal wall time: {runs_df.train_s.sum()/60:.1f} min")
    print(f"Written to {out}", flush=True)


if __name__ == "__main__":
    main()
