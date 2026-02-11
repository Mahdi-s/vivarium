#!/usr/bin/env python3
"""
Download Olmo models from HuggingFace and optionally convert them to GGUF format.

This script:
1. Downloads models from HuggingFace
2. Optionally converts them to GGUF format using llama.cpp's conversion script
3. Saves them in the configured models directory

Usage:
    # Download and convert to GGUF (default)
    python download_and_convert_olmo_models.py
    
    # Download only (skip GGUF conversion, for PyTorch-only workflows)
    python download_and_convert_olmo_models.py --torch-only
    
Environment Variables:
    AAM_MODEL_DIR: Override model storage directory
    AAM_LLAMA_CPP_ROOT: Override llama.cpp installation path
    AAM_HF_CACHE: Override HuggingFace cache directory
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import settings; fallback to local defaults if not installed
try:
    from aam.settings import settings
    USE_SETTINGS = True
except ImportError:
    USE_SETTINGS = False


# Models to download and convert
MODELS = [
    ("allenai/Olmo-3-1025-7B", "olmo-3-1025-7b-base.gguf"),
    ("allenai/Olmo-3-1125-32B", "olmo-3-1125-32b-base.gguf"),
    ("allenai/Olmo-3-7B-RL-Zero-Math", "olmo-3-7b-rl-zero-math.gguf"),
    ("allenai/Olmo-3-7B-Think", "olmo-3-7b-think.gguf"),
    ("allenai/Olmo-3-7B-Think-SFT", "olmo-3-7b-think-sft.gguf"),
    ("allenai/Olmo-3-7B-Think-DPO", "olmo-3-7b-think-dpo.gguf"),
    ("allenai/Olmo-3-7B-Instruct", "olmo-3-7b-instruct.gguf"),
    ("allenai/Olmo-3-7B-Instruct-SFT", "olmo-3-7b-instruct-sft.gguf"),
    ("allenai/Olmo-3-7B-Instruct-DPO", "olmo-3-7b-instruct-dpo.gguf"),
]


def get_repo_root() -> Path:
    """Find the repository root (fallback when settings not available)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_models_dir() -> Path:
    """Get the models directory from settings or environment."""
    if USE_SETTINGS:
        return settings.MODEL_DIR
    
    env_val = os.environ.get("AAM_MODEL_DIR")
    if env_val:
        return Path(env_val)
    return get_repo_root() / "models"


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory from settings or environment."""
    if USE_SETTINGS:
        return settings.HF_CACHE
    
    env_val = os.environ.get("AAM_HF_CACHE")
    if env_val:
        return Path(env_val)
    return get_models_dir() / "huggingface_cache"


def get_llama_cpp_path() -> Path:
    """
    Get path to llama.cpp repository.
    
    Resolution order:
    1. AAM settings (if installed)
    2. AAM_LLAMA_CPP_ROOT environment variable
    3. Default: repo_root/third_party/llama.cpp
    
    Raises:
        RuntimeError: If llama.cpp is not found at the resolved path
    """
    if USE_SETTINGS:
        llama_cpp = settings.LLAMA_CPP_ROOT
    else:
        env_val = os.environ.get("AAM_LLAMA_CPP_ROOT")
        if env_val:
            llama_cpp = Path(env_val)
        else:
            llama_cpp = get_repo_root() / "third_party" / "llama.cpp"
    
    if not llama_cpp.exists():
        raise RuntimeError(
            f"llama.cpp not found at {llama_cpp}.\n"
            "Options:\n"
            "  1. Clone it: git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp\n"
            "  2. Set AAM_LLAMA_CPP_ROOT environment variable to your llama.cpp installation\n"
            "  3. Use --torch-only flag to skip GGUF conversion"
        )
    return llama_cpp


def get_convert_script() -> Path:
    """Get path to convert_hf_to_gguf.py script."""
    llama_cpp = get_llama_cpp_path()
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            "Please ensure llama.cpp is fully cloned."
        )
    return convert_script


def download_model(model_id: str, cache_dir: Path) -> Path:
    """
    Download model from HuggingFace using huggingface-cli or transformers.
    
    Returns path to downloaded model directory.
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"{'='*60}")
    
    model_path = cache_dir / model_id.replace("/", "_")
    
    # Check if already downloaded
    if model_path.exists() and (model_path / "config.json").exists():
        print(f"✓ Model already downloaded at: {model_path}")
        return model_path
    
    # Try using huggingface-cli first (faster, direct download)
    try:
        print("  Attempting download via huggingface-cli...")
        result = subprocess.run(
            ["huggingface-cli", "download", model_id, "--local-dir", str(model_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode == 0:
            if model_path.exists() and (model_path / "config.json").exists():
                print(f"✓ Downloaded via huggingface-cli to: {model_path}")
                return model_path
            else:
                print("  Warning: Download completed but files not found, trying transformers...")
    except FileNotFoundError:
        print("  huggingface-cli not found, using transformers...")
    except subprocess.TimeoutExpired:
        print("  Download timed out, trying transformers...")
    except Exception as e:
        print(f"  huggingface-cli failed: {e}, trying transformers...")
    
    # Fallback: use transformers to download
    print("  Using transformers library to download...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        tokenizer.save_pretrained(str(model_path))
        
        print("  Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            torch_dtype="auto",
        )
        # Fix generation_config so save_pretrained() does not fail: either do_sample=True
        # or unset temperature/top_p. Prefer making do_sample True so config is preserved.
        if getattr(model, "generation_config", None) is not None:
            gc = model.generation_config
            if getattr(gc, "temperature", None) is not None or getattr(gc, "top_p", None) is not None:
                gc.do_sample = True
        model.save_pretrained(str(model_path))
        
        print(f"✓ Downloaded via transformers to: {model_path}")
        return model_path
        
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("  Install it with: pip install transformers")
        raise
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        raise


def verify_safetensors(model_path: Path) -> bool:
    """
    Verify that safetensors files exist in the model directory.
    
    Returns True if safetensors files are present.
    """
    safetensor_files = list(model_path.glob("*.safetensors"))
    if safetensor_files:
        total_size = sum(f.stat().st_size for f in safetensor_files)
        print(f"✓ Found {len(safetensor_files)} safetensors file(s) ({total_size / (1024**3):.2f} GB)")
        return True
    
    # Check for bin files as fallback
    bin_files = list(model_path.glob("*.bin"))
    if bin_files:
        total_size = sum(f.stat().st_size for f in bin_files)
        print(f"✓ Found {len(bin_files)} bin file(s) ({total_size / (1024**3):.2f} GB)")
        return True
    
    print("✗ No model weight files found")
    return False


def convert_to_gguf(model_path: Path, output_path: Path, model_id: str) -> bool:
    """
    Convert HuggingFace model to GGUF format.
    
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Converting to GGUF: {model_id}")
    print(f"{'='*60}")
    
    convert_script = get_convert_script()
    llama_cpp = get_llama_cpp_path()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run conversion script
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",  # 16-bit floating point
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    print(f"  This may take 10-30 minutes depending on model size...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(llama_cpp),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode == 0:
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ Conversion successful!")
                print(f"  Output: {output_path}")
                print(f"  Size: {size_mb:.1f} MB")
                return True
            else:
                print(f"✗ Conversion completed but output file not found!")
                return False
        else:
            print(f"✗ Conversion failed!")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Conversion timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Conversion failed with error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Olmo models and optionally convert to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  AAM_MODEL_DIR       Override model storage directory
  AAM_LLAMA_CPP_ROOT  Override llama.cpp installation path  
  AAM_HF_CACHE        Override HuggingFace cache directory

Examples:
  # Download and convert all models to GGUF
  python download_and_convert_olmo_models.py
  
  # Download only (for PyTorch-only workflows)
  python download_and_convert_olmo_models.py --torch-only
  
  # Download specific models
  python download_and_convert_olmo_models.py --models allenai/Olmo-3-7B-Instruct
        """
    )
    
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="Skip GGUF conversion, only download HuggingFace models. "
             "Use this for PyTorch-only interpretability workflows."
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model IDs to download (default: all models)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download new models"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main function to download and optionally convert models."""
    args = parse_args()
    
    models_dir = get_models_dir()
    cache_dir = get_hf_cache_dir()
    
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Olmo Model Download Script")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Mode: {'Download only (--torch-only)' if args.torch_only else 'Download + GGUF conversion'}")
    
    if not args.torch_only:
        try:
            llama_cpp = get_llama_cpp_path()
            print(f"llama.cpp: {llama_cpp}")
        except RuntimeError as e:
            print(f"\n⚠ Warning: {e}")
            print("Continuing in --torch-only mode...")
            args.torch_only = True
    
    # Filter models if specific ones requested
    if args.models:
        models_to_process = [
            (mid, out) for mid, out in MODELS 
            if any(m in mid for m in args.models)
        ]
        if not models_to_process:
            print(f"\n✗ No matching models found for: {args.models}")
            print("Available models:")
            for mid, _ in MODELS:
                print(f"  - {mid}")
            return 1
    else:
        models_to_process = MODELS
    
    print(f"Models to process: {len(models_to_process)}")
    print("=" * 60)
    
    results: List[Tuple[str, str, bool, Optional[str]]] = []
    
    for model_id, output_name in models_to_process:
        output_path = models_dir / output_name
        model_cache_path = cache_dir / model_id.replace("/", "_")
        
        # Check existing state
        gguf_exists = output_path.exists() and not args.torch_only
        hf_exists = model_cache_path.exists() and (model_cache_path / "config.json").exists()
        
        if args.verify_only:
            if args.torch_only:
                status = "✓ EXISTS" if hf_exists else "✗ MISSING"
                results.append((model_id, output_name, hf_exists, "HF cache"))
            else:
                status = "✓ EXISTS" if gguf_exists else "✗ MISSING"
                results.append((model_id, output_name, gguf_exists, "GGUF"))
            print(f"{status}: {model_id}")
            continue
        
        # Skip GGUF if already exists (and not torch-only mode)
        if not args.torch_only and gguf_exists:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n✓ Skipping {model_id} - GGUF already exists ({size_mb:.1f} MB)")
            results.append((model_id, output_name, True, "GGUF exists"))
            continue
        
        try:
            # Download model
            model_path = download_model(model_id, cache_dir)
            
            # Verify safetensors
            if not verify_safetensors(model_path):
                print(f"✗ Model weights verification failed for {model_id}")
                results.append((model_id, output_name, False, "Verification failed"))
                continue
            
            if args.torch_only:
                # In torch-only mode, we're done after download
                results.append((model_id, output_name, True, "Downloaded (torch-only)"))
            else:
                # Convert to GGUF
                success = convert_to_gguf(model_path, output_path, model_id)
                results.append((model_id, output_name, success, "Converted" if success else "Conversion failed"))
            
        except Exception as e:
            print(f"\n✗ Failed to process {model_id}: {e}")
            results.append((model_id, output_name, False, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    successful = sum(1 for _, _, success, _ in results if success)
    failed = len(results) - successful
    
    for model_id, output_name, success, note in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        note_str = f" ({note})" if note else ""
        print(f"{status}: {model_id}{note_str}")
    
    print(f"\nTotal: {len(results)} models")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠ {failed} model(s) failed. Check errors above.")
        return 1
    
    print(f"\n✓ All models processed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
