"""
Centralized settings module for Vivarium.

This module provides a singleton Settings instance that resolves paths from
environment variables with fallbacks to sensible defaults.

Environment Variables:
    AAM_MODEL_DIR: Directory for storing large model weights (GGUF, Safetensors)
    AAM_LLAMA_CPP_ROOT: Path to llama.cpp repository
    AAM_ARTIFACTS_DIR: Root directory for simulation outputs (.db, activations)
    AAM_HF_CACHE: HuggingFace downloader cache location

Usage:
    from aam.settings import settings
    
    models_dir = settings.MODEL_DIR
    llama_cpp_root = settings.LLAMA_CPP_ROOT
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


@dataclass
class AAMSettings:
    """
    Centralized settings for Vivarium.
    
    Resolves paths from environment variables with fallbacks to default locations
    relative to the project root.
    
    Attributes:
        PROJECT_ROOT: Root directory of the Vivarium project
        MODEL_DIR: Directory for model weights (GGUF, safetensors)
        LLAMA_CPP_ROOT: Path to llama.cpp installation
        ARTIFACTS_DIR: Root directory for run artifacts
        HF_CACHE: HuggingFace cache directory
    """
    
    PROJECT_ROOT: Path = field(default_factory=_find_project_root)
    
    def __post_init__(self) -> None:
        # Ensure PROJECT_ROOT is a Path
        if not isinstance(self.PROJECT_ROOT, Path):
            self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
    
    @property
    def MODEL_DIR(self) -> Path:
        """Directory for model weights. Override with AAM_MODEL_DIR env var."""
        env_val = os.environ.get("AAM_MODEL_DIR")
        if env_val:
            return Path(env_val)
        return self.PROJECT_ROOT / "models"
    
    @property
    def LLAMA_CPP_ROOT(self) -> Path:
        """
        Path to llama.cpp repository. Override with AAM_LLAMA_CPP_ROOT env var.
        
        Falls back to:
        1. AAM_LLAMA_CPP_ROOT environment variable
        2. shutil.which("llama-cpp") or shutil.which("llama-server") parent
        3. PROJECT_ROOT/third_party/llama.cpp
        """
        env_val = os.environ.get("AAM_LLAMA_CPP_ROOT")
        if env_val:
            return Path(env_val)
        
        # Try to find llama-cpp or llama-server binary
        for binary_name in ("llama-cpp", "llama-server", "llama.cpp"):
            found = shutil.which(binary_name)
            if found:
                # Return the parent directory (assumed to be llama.cpp root)
                return Path(found).parent.parent
        
        # Default fallback
        return self.PROJECT_ROOT / "third_party" / "llama.cpp"
    
    @property
    def ARTIFACTS_DIR(self) -> Path:
        """Root directory for run artifacts. Override with AAM_ARTIFACTS_DIR env var."""
        env_val = os.environ.get("AAM_ARTIFACTS_DIR")
        if env_val:
            return Path(env_val)
        return self.PROJECT_ROOT / "runs"
    
    @property
    def HF_CACHE(self) -> Path:
        """HuggingFace cache directory. Override with AAM_HF_CACHE env var."""
        env_val = os.environ.get("AAM_HF_CACHE")
        if env_val:
            return Path(env_val)
        
        # Check for HF_HOME first (standard HuggingFace env var)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home)
        
        # Default to project-local cache
        return self.MODEL_DIR / "huggingface_cache"
    
    def validate_paths(self, *, require_all: bool = False) -> dict[str, bool]:
        """
        Validate that configured paths exist.
        
        Args:
            require_all: If True, raise an error if any path doesn't exist
            
        Returns:
            Dict mapping path name to existence status
            
        Raises:
            FileNotFoundError: If require_all is True and a path doesn't exist
        """
        paths = {
            "PROJECT_ROOT": self.PROJECT_ROOT,
            "MODEL_DIR": self.MODEL_DIR,
            "LLAMA_CPP_ROOT": self.LLAMA_CPP_ROOT,
            "ARTIFACTS_DIR": self.ARTIFACTS_DIR,
            "HF_CACHE": self.HF_CACHE,
        }
        
        results = {}
        missing = []
        
        for name, path in paths.items():
            exists = path.exists()
            results[name] = exists
            if not exists:
                missing.append(f"{name}: {path}")
        
        if require_all and missing:
            raise FileNotFoundError(
                f"Required paths do not exist:\n  " + "\n  ".join(missing)
            )
        
        return results
    
    def get_llama_cpp_convert_script(self) -> Path:
        """
        Get path to llama.cpp's convert_hf_to_gguf.py script.
        
        Returns:
            Path to the conversion script
            
        Raises:
            FileNotFoundError: If script is not found
        """
        script = self.LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"
        if not script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {script}. "
                f"Ensure llama.cpp is installed at {self.LLAMA_CPP_ROOT} "
                "or set AAM_LLAMA_CPP_ROOT environment variable."
            )
        return script
    
    def ensure_dirs(self) -> None:
        """Create MODEL_DIR, ARTIFACTS_DIR, and HF_CACHE if they don't exist."""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.HF_CACHE.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        return (
            f"AAMSettings(\n"
            f"  PROJECT_ROOT={self.PROJECT_ROOT},\n"
            f"  MODEL_DIR={self.MODEL_DIR},\n"
            f"  LLAMA_CPP_ROOT={self.LLAMA_CPP_ROOT},\n"
            f"  ARTIFACTS_DIR={self.ARTIFACTS_DIR},\n"
            f"  HF_CACHE={self.HF_CACHE}\n"
            f")"
        )


# Singleton instance - import this in other modules
settings = AAMSettings()

