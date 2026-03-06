"""
Utility functions for working with Olmo-3 models.

This module provides helpers for:
- Detecting Olmo model variants
- Handling Think variant special tokens
- Model-specific configuration
- Downloading models from HuggingFace
- Setting up models for Ollama
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def detect_olmo_variant(model_id: str) -> str:
    """
    Detect the Olmo-3 variant from model ID.
    
    Returns: "base", "instruct", "think", "rl_zero", or "unknown"
    """
    model_id_lower = model_id.lower()
    
    if "think" in model_id_lower:
        return "think"
    elif "instruct" in model_id_lower:
        return "instruct"
    elif "rl-zero" in model_id_lower or "rlzero" in model_id_lower:
        return "rl_zero"
    elif "olmo-3" in model_id_lower or "olmo3" in model_id_lower:
        if "base" in model_id_lower or ("7b" in model_id_lower and "instruct" not in model_id_lower and "think" not in model_id_lower):
            return "base"
    
    return "unknown"


def extract_think_tokens(text: str) -> Tuple[Optional[str], str]:
    """
    Extract <think>...</think> tokens from text.
    
    Returns: (think_content, remaining_text)
    """
    # Look for <think>...</think> blocks
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the first think block and the text after it
        think_content = matches[0].strip()
        # Remove all think blocks from text
        remaining = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return think_content, remaining
    
    return None, text


def is_think_variant(model_id: str) -> bool:
    """Check if model ID indicates a Think variant."""
    return detect_olmo_variant(model_id) == "think"


def get_olmo_model_config(model_id: str) -> dict:
    """
    Get model-specific configuration for Olmo models.
    
    Returns configuration dict with variant-specific settings.
    """
    variant = detect_olmo_variant(model_id)
    
    config = {
        "variant": variant,
        "model_id": model_id,
        "has_think_tokens": variant == "think",
        "max_new_tokens": 256 if variant == "think" else 128,  # Think models need more tokens
    }
    
    return config


def normalize_olmo_response(text: str, variant: str) -> str:
    """
    Normalize Olmo model response based on variant.
    
    For Think variants, extracts the final answer after <think> blocks.
    For other variants, returns text as-is.
    """
    if variant == "think":
        _, answer = extract_think_tokens(text)
        return answer.strip()
    return text.strip()


def ensure_olmo_model_downloaded(
    model_id: str,
    models_dir: Optional[str] = None,
    import_to_ollama: bool = True,
) -> Tuple[str, bool]:
    """
    Ensure an Olmo model is downloaded from HuggingFace and optionally imported into Ollama.
    
    Args:
        model_id: HuggingFace model ID (e.g., "allenai/Olmo-3-7B-Instruct")
        models_dir: Directory to store models (default: ./models in repo root)
        import_to_ollama: If True, import the model into Ollama after downloading
    
    Returns:
        Tuple of (ollama_model_name, was_downloaded)
        - ollama_model_name: Name to use with Ollama API (e.g., "olmo-3-1025-7b-instruct")
        - was_downloaded: True if model was just downloaded, False if already existed
    """
    # Determine models directory
    if models_dir is None:
        # Try to find repo root
        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                repo_root = parent
                break
        if repo_root is None:
            repo_root = Path.cwd()
        models_dir = str(repo_root / "models")
    
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Convert model_id to Ollama format
    olmo_model_name = model_id.replace("allenai/", "").lower()
    
    # Check if model is already in Ollama
    if import_to_ollama:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if olmo_model_name in result.stdout:
                print(f"✓ Model {olmo_model_name} already available in Ollama")
                return olmo_model_name, False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass  # Ollama not available or error, continue with download
    
    # Check if model is already cached locally
    # HuggingFace cache uses underscores instead of slashes
    cache_dir = models_path / "huggingface_cache"
    model_cache_name = model_id.replace("/", "_")
    model_cache_path = cache_dir / model_cache_name
    
    # Check for key files that indicate the model is cached
    # Check both config.json and at least one model weight file
    config_exists = (model_cache_path / "config.json").exists()
    tokenizer_exists = (model_cache_path / "tokenizer.json").exists()
    has_model_weights = any(
        (model_cache_path / f).exists() 
        for f in ["model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors", "model-00001-of-00003.safetensors"]
    )
    
    model_cached = config_exists and (tokenizer_exists or has_model_weights)
    
    if model_cached:
        print(f"✓ Model {model_id} already cached locally")
        print(f"  Cache location: {model_cache_path}")
        if not tokenizer_exists:
            print(f"  Note: Tokenizer will be downloaded on first use if needed")
        return olmo_model_name, False
    
    # Model not cached - verify without loading
    print(f"Verifying model {model_id} availability...")
    
    # Just check if the directory exists (even if incomplete) - don't load anything
    if model_cache_path.exists():
        print(f"  Found partial cache at: {model_cache_path}")
        print(f"  Model will be used from cache or downloaded on first use by LiteLLM")
        return olmo_model_name, False
    
    # No cache found - but don't download here, let LiteLLM handle it
    print(f"  Model not found in cache: {model_cache_path}")
    print(f"  Model will be downloaded on first use by LiteLLM")
    print(f"  This may take several minutes and requires ~14GB of disk space for 7B models.")
    return olmo_model_name, False


def get_ollama_model_name(hf_model_id: str) -> str:
    """
    Convert HuggingFace model ID to Ollama model name format.
    
    Args:
        hf_model_id: HuggingFace model ID (e.g., "allenai/Olmo-3-1025-7B-Instruct")
    
    Returns:
        Ollama model name (e.g., "olmo-3-1025-7b-instruct")
    """
    # Remove org prefix and convert to lowercase
    name = hf_model_id.replace("allenai/", "").lower()
    return name
