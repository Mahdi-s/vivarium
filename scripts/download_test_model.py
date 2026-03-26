#!/usr/bin/env python3
"""
Download a small chat-capable model for integration testing.

Downloads Qwen2.5-0.5B-Instruct in two formats:
  1. HuggingFace safetensors  → models/huggingface_cache/Qwen_Qwen2.5-0.5B-Instruct/
  2. Pre-quantised GGUF (Q5_K_M) → models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf

Usage:
    python scripts/download_test_model.py              # both formats
    python scripts/download_test_model.py --hf-only    # HF safetensors only
    python scripts/download_test_model.py --gguf-only  # GGUF only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILENAME = "qwen2.5-0.5b-instruct-q5_k_m.gguf"

# ---------------------------------------------------------------------------
# Path resolution (mirrors vivarium.settings without importing it)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def _models_dir() -> Path:
    env = os.environ.get("VIVARIUM_MODEL_DIR") or os.environ.get("AAM_MODEL_DIR")
    if env:
        return Path(env)
    return _repo_root() / "models"


def _hf_cache_dir() -> Path:
    env = os.environ.get("VIVARIUM_HF_CACHE") or os.environ.get("AAM_HF_CACHE")
    if env:
        return Path(env)
    return _models_dir() / "huggingface_cache"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_hf_safetensors(cache_dir: Path) -> Path:
    """Download HF safetensors via huggingface-cli or transformers."""
    local_dir = cache_dir / HF_MODEL_ID.replace("/", "_")
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"  ✓ HF weights already present: {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {HF_MODEL_ID} → {local_dir} ...")

    # Prefer huggingface-cli (faster, supports resume)
    try:
        subprocess.check_call(
            ["huggingface-cli", "download", HF_MODEL_ID,
             "--local-dir", str(local_dir)],
            stdout=sys.stdout, stderr=sys.stderr,
        )
        return local_dir
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: transformers snapshot_download
    from huggingface_hub import snapshot_download  # type: ignore
    snapshot_download(HF_MODEL_ID, local_dir=str(local_dir))
    return local_dir


def download_gguf(models_dir: Path) -> Path:
    """Download pre-quantised GGUF from the official Qwen repo."""
    dest = models_dir / GGUF_FILENAME
    if dest.exists():
        print(f"  ✓ GGUF already present: {dest}")
        return dest

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {GGUF_REPO}/{GGUF_FILENAME} → {dest} ...")

    try:
        subprocess.check_call(
            ["huggingface-cli", "download", GGUF_REPO, GGUF_FILENAME,
             "--local-dir", str(models_dir)],
            stdout=sys.stdout, stderr=sys.stderr,
        )
        return dest
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    from huggingface_hub import hf_hub_download  # type: ignore
    hf_hub_download(GGUF_REPO, GGUF_FILENAME, local_dir=str(models_dir))
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Download test model for integration tests.")
    ap.add_argument("--hf-only", action="store_true", help="Download HF safetensors only")
    ap.add_argument("--gguf-only", action="store_true", help="Download GGUF only")
    args = ap.parse_args()

    models_dir = _models_dir()
    hf_cache = _hf_cache_dir()

    print(f"Models dir : {models_dir}")
    print(f"HF cache   : {hf_cache}")
    print()

    do_hf = not args.gguf_only
    do_gguf = not args.hf_only

    if do_hf:
        print("[1/2] HuggingFace safetensors")
        hf_path = download_hf_safetensors(hf_cache)
        print(f"  → {hf_path}\n")

    if do_gguf:
        print("[2/2] GGUF (Q5_K_M)")
        gguf_path = download_gguf(models_dir)
        print(f"  → {gguf_path}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
