from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "final" / "34_environment"
PACKAGES = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
            "seaborn", "transformers", "fastapi", "uvicorn", "pydantic",
            "imbalanced-learn", "requests"]

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"item": "python", "value": sys.version.split()[0]},
            {"item": "platform", "value": f"{platform.system()} {platform.release()}"},
            {"item": "machine", "value": platform.machine()},
            {"item": "processor", "value": platform.processor() or "unreported"}]

    gpu = cuda = "none"
    try:
        import torch
        cuda = torch.version.cuda or "cpu build"
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            p = torch.cuda.get_device_properties(0)
            rows.append({"item": "gpu memory", "value": f"{p.total_memory / 1e9:.1f} GB"})
    except Exception:
        pass
    rows += [{"item": "gpu", "value": gpu}, {"item": "cuda", "value": cuda}]

    for p in PACKAGES:
        try:
            rows.append({"item": p, "value": md.version(p)})
        except Exception:
            rows.append({"item": p, "value": "not installed"})

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "environment.csv", index=False)

    train = [
        ("optimizer", "Adam"), ("initial_learning_rate", "0.001"),
        ("batch_size", "512"), ("max_epochs", "50"),
        ("early_stopping_patience_epochs", "6"),
        ("scheduler", "ReduceLROnPlateau"), ("scheduler_patience_epochs", "2"),
        ("scheduler_factor", "0.5"), ("scheduler_monitors", "development loss"),
        ("selection_metric", "development F1"),
        ("loss", "BCEWithLogitsLoss with pos_weight"),
        ("seeds", "42, 7, 123, 2024, 777"), ("n_seeds", "5"),
        ("mixed_precision", "not used"),
    ]
    pd.DataFrame(train, columns=["setting", "value"]).to_csv(
        OUT / "training_config.csv", index=False)

    get = dict(zip(d.item, d.value))
    note = (
        "The recorded benchmark environment used an "
        f"{get.get('cpu', 'unrecorded CPU')} CPU and an {get.get('gpu', 'no GPU')}, "
        f"with Python {get.get('python')}, PyTorch {get.get('torch')}, and "
        f"CUDA {get.get('cuda')} on {get.get('platform')}.\n\n"
        "CPU forward-pass latency was measured with torch.set_num_threads(1), a "
        "batch size of one, and preprocessed inputs. Character-level inputs were "
        "padded to 50 positions, word-level inputs to 8 positions, and pretrained "
        "subword inputs to 32 positions. These lengths reflect the input "
        "configurations used by each model family.\n\n"
        "Each benchmark measured 200 calls after 30 warm-up calls and reported the "
        "median latency. The repeated benchmark in results/final/32_latency_repeats "
        "summarized seven trials using the median of the seven trial medians, with "
        "models reinitialized for each trial.\n\n"
        "Forward-pass measurements exclude tokenization and other preprocessing. "
        "The reported timings describe this hardware and software environment; "
        "application latency also depends on preprocessing, batching, and "
        "deployment conditions.\n")
    (OUT / "environment_note.txt").write_text(note, encoding="utf-8")

    print(d.to_string(index=False))
    print("\n" + note)
    print(f"Written to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
