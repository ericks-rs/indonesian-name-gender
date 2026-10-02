"""Export the eight grid models to ONNX and verify them against PyTorch.

Run from anywhere:  python onnx/export_onnx.py

Each model takes `ids` (int64, shape [batch, max_len]) and returns
  prob_female  float32 [batch]          sigmoid of the logit, class P
  attention    float32 [batch, max_len] attention-pooling weights (zero on padding)
max_len is 50 for the Char models and 8 for the Word models.

Also writes the two tokenizers as JSON (so a non-Python client can encode names)
and `parity_cases.json`, a few synthetic names with the ids and probabilities that
PyTorch produces, for testing a client implementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(ROOT / "demo"))
import inference as inf  # noqa: E402

OPSET = 17
PARITY_NAMES = ["BANOWATI LARASATI", "MUHAMMAD ALI", "SRIKANDI PALUPI", "BUDI", "WULAN",
                "SETIA", "WAHYU", "DIAN", "SITI AMINAH", "A"]


class Export(nn.Module):
    """ids -> (prob_female, attention), eval mode, no dropout."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ids):
        logits, weights = self.model(ids, return_attention=True)
        return torch.sigmoid(logits), weights


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)  # keep the exported graph on the plain path
    pred = inf.Predictor(ROOT, models_dir=ROOT / "models", tokenizers_dir=ROOT / "tokenizers")

    # tokenizers for non-Python clients
    (OUT / "char_vocab.json").write_text(
        json.dumps({"max_len": inf.CFG["CHAR_MAX_LEN"], "pad": 0, "unk": 1,
                    "char2idx": pred.char_tok.char2idx}, ensure_ascii=False), encoding="utf-8")
    (OUT / "word_vocab.json").write_text(
        json.dumps({"max_len": inf.CFG["WORD_MAX_LEN"], "pad": 0, "unk": 1,
                    "hash": "blake2s", "digest_size": 8, "salt": inf.VOCAB_SALT,
                    "hashed": bool(getattr(pred.word_tok, "hashed", False)),
                    "word2idx": pred.word_tok.word2idx}, ensure_ascii=False), encoding="utf-8")

    # verification sample: synthetic cases plus a slice of the local validation split
    names = list(PARITY_NAMES)
    val = ROOT / "data" / "splits" / "val_2024_2026.csv"
    if val.exists():
        names += pd.read_csv(val).NAMA.astype(str).sample(2000, random_state=0).tolist()
    else:
        print("val split not found, verifying on the synthetic names only")

    parity = {n: {} for n in PARITY_NAMES}
    worst = {}
    for name, model in pred.models.items():
        is_char = "Char" in name
        max_len = inf.CFG["CHAR_MAX_LEN"] if is_char else inf.CFG["WORD_MAX_LEN"]
        wrapper = Export(model).eval().cpu()
        dummy = torch.zeros(2, max_len, dtype=torch.long)
        path = OUT / f"{name}.onnx"
        torch.onnx.export(
            wrapper, (dummy,), str(path), input_names=["ids"],
            output_names=["prob_female", "attention"],
            dynamic_axes={"ids": {0: "batch"}, "prob_female": {0: "batch"},
                          "attention": {0: "batch"}},
            opset_version=OPSET, dynamo=False)

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        tok = pred.char_tok if is_char else pred.word_tok
        ids = np.array([tok.encode(n, max_len) for n in names], dtype=np.int64)
        with torch.no_grad():
            ref_p, ref_a = wrapper(torch.from_numpy(ids))
        got_p, got_a = sess.run(None, {"ids": ids})
        dp = float(np.abs(got_p - ref_p.numpy()).max())
        da = float(np.abs(got_a - ref_a.numpy()).max())
        flips = int(((got_p >= 0.5) != (ref_p.numpy() >= 0.5)).sum())
        worst[name] = (dp, da, flips)
        for i, n in enumerate(PARITY_NAMES):
            parity[n][name] = {"ids": ids[i].tolist(), "prob_female": float(ref_p[i])}
        print(f"{name:<16} {path.stat().st_size/1e6:6.2f} MB  max|dp|={dp:.2e}  "
              f"max|da|={da:.2e}  label flips={flips}/{len(names)}")

    (OUT / "parity_cases.json").write_text(json.dumps(parity, indent=1), encoding="utf-8")

    # reference hashes so a client can test its BLAKE2s, including inputs longer than one
    # 64-byte block and non-ASCII text (synthetic words only)
    import random
    rnd = random.Random(0)
    words = ["a", "budi", "wulandari", "x" * 40, "y" * 47, "z" * 48, "q" * 64, "w" * 130,
             "café", "张伟", "ab cd"]
    words += ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(1, 30)))
              for _ in range(40)]
    (OUT / "hash_cases.json").write_text(
        json.dumps([{"word": w, "key": inf.vocab_key(w)} for w in words],
                   ensure_ascii=True, indent=1), encoding="utf-8")
    bad = [k for k, (dp, da, fl) in worst.items() if dp > 1e-4 or da > 1e-4 or fl]
    print("\nALL MODELS MATCH PYTORCH" if not bad else f"\nMISMATCH: {bad}")


if __name__ == "__main__":
    main()
