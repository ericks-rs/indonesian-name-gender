"""Download the ONNX models from the GitHub Release into onnx/models/.

    python onnx/download_models.py                 # all eight
    python onnx/download_models.py CharBiLSTM      # one model
"""
import sys
import urllib.request
from pathlib import Path

RELEASE_TAG = "v1.1.0"
BASE = f"https://github.com/ericks-rs/indonesian-name-gender/releases/download/{RELEASE_TAG}/"
MODELS = ["CharBiRNN", "CharBiLSTM", "CharBiGRU", "CharTransformer",
          "WordBiRNN", "WordBiLSTM", "WordBiGRU", "WordTransformer"]
OUT = Path(__file__).resolve().parent / "models"


def main(names: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    for name in names or MODELS:
        if name not in MODELS:
            raise SystemExit(f"Unknown model {name}. Choose from {', '.join(MODELS)}")
        dest = OUT / f"{name}.onnx"
        if dest.exists():
            print(f"{dest.name} already present")
            continue
        url = f"{BASE}{name}.onnx"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main(sys.argv[1:])
