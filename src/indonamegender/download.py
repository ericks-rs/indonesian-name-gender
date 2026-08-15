"""Auto-download model weights dari GitHub Releases."""
import os
import urllib.request
from pathlib import Path

# The seven models the package does not bundle are fetched from this tag. The
# assets published under v0.1.0 come from the run that preceded the revision and
# pool their Transformers by mean, so they no longer match the architecture in
# models.py. Bump this together with the version in pyproject.toml, and only once
# the matching assets have been uploaded to the new tag.
WEIGHTS_TAG = "v0.1.0"
GITHUB_RELEASE_BASE = (
    f"https://github.com/ericks-rs/indonesian-name-gender/releases/download/{WEIGHTS_TAG}/"
)

# Default cache dir: ~/.cache/indonamegender/
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "indonamegender"


def get_cache_dir():
    """Return cache directory, create kalau belum ada."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(model_name, cache_dir=None):
    """Download model weights dari GitHub Releases ke local cache.

    Returns: Path ke .pt file di local cache.
    """
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
