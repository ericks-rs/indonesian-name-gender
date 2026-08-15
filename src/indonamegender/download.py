import os
import urllib.request
from pathlib import Path

WEIGHTS_TAG = "v1.0.0"
GITHUB_RELEASE_BASE = (
    f"https://github.com/ericks-rs/indonesian-name-gender/releases/download/{WEIGHTS_TAG}/"
)

DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "indonamegender"

def get_cache_dir():
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_model(model_name, cache_dir=None):
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cache_file = cache_dir / f"{model_name}.pt"
    if cache_file.exists():
        return cache_file

    url = f"{GITHUB_RELEASE_BASE}{model_name}.pt"
    print(f"[indonamegender] Downloading {model_name}.pt from GitHub Releases...")
    print(f"  URL: {url}")
    print(f"  Cache: {cache_file}")

    try:
        urllib.request.urlretrieve(url, cache_file)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {model_name}.pt from {url}. "
            f"Original error: {e}"
        )

    return cache_file
