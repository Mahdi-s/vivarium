from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _default_gpu_layers() -> int:
    """
    Get default GPU layers based on platform.
    
    On macOS with Apple Silicon, use all layers (-1) for Metal acceleration.
    On other platforms, default to 0 (CPU-only).
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon Mac - use Metal GPU acceleration by default
        return -1  # -1 means all layers
    return 0  # CPU-only by default


@dataclass(frozen=True)
class LlamaServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8081  # Default port for OpenAI-compatible API
    ctx_size: int = 4096
    n_gpu_layers: Optional[int] = None  # None = auto-detect based on platform
    extra_args: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra_args is None:
            object.__setattr__(self, "extra_args", [])
        if self.n_gpu_layers is None:
            # Set platform-specific default
            object.__setattr__(self, "n_gpu_layers", _default_gpu_layers())


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def llama_server_binary_path() -> Path:
    # Built by our vendored llama.cpp under third_party/llama.cpp/build/bin/llama-server
    root = repo_root_from_here()
    return root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"


def run_llama_server(config: LlamaServerConfig) -> subprocess.Popen:
    bin_path = llama_server_binary_path()
    if not bin_path.exists():
        raise RuntimeError(
            f"llama-server not found at {bin_path}. Build llama.cpp first (see third_party/llama.cpp)."
        )

    args = [
        str(bin_path),
        "--model",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--ctx-size",
        str(config.ctx_size),
    ]

    # GPU layers flag varies by backend.
    # On macOS with Metal, -1 means all layers, 0 means CPU-only.
    # Only add flag if explicitly set (non-zero) or if default was applied.
    if config.n_gpu_layers != 0:
        args += ["--n-gpu-layers", str(config.n_gpu_layers)]

    args += list(config.extra_args or [])

    # Start as a subprocess; caller controls lifecycle.
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


