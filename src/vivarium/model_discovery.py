from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ModelInfo:
    source: str  # "ollama" | "lmstudio" | "unknown"
    model_name: str
    gguf_path: str
    size_bytes: int
    mtime: float


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def discover_lmstudio_gguf() -> List[ModelInfo]:
    """
    Discover GGUF files that LM Studio downloaded.

    Common location (if present):
      ~/Library/Application Support/LM Studio/models/**/*.gguf

    If LM Studio has no GGUF files or uses a different storage layout, this returns [].
    """
    base = _home() / "Library" / "Application Support" / "LM Studio" / "models"
    if not base.exists():
        return []

    out: List[ModelInfo] = []
    for p in base.rglob("*.gguf"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            ModelInfo(
                source="lmstudio",
                model_name=p.stem,
                gguf_path=str(p),
                size_bytes=int(st.st_size),
                mtime=float(st.st_mtime),
            )
        )
    return out


def _ollama_models_root() -> Path:
    return _home() / ".ollama" / "models"


def discover_ollama_manifests() -> List[ModelInfo]:
    """
    Discover Ollama models by parsing manifests and resolving the model layer blob.

    Ollama stores:
      ~/.ollama/models/manifests/<registry>/<namespace>/<repo>/<tag>
      ~/.ollama/models/blobs/sha256-<hash>

    Manifest example contains:
      layers: [{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:..."}]
    """
    root = _ollama_models_root()
    manifests_root = root / "manifests"
    blobs_root = root / "blobs"
    if not manifests_root.exists() or not blobs_root.exists():
        return []

    out: List[ModelInfo] = []

    for manifest_path in manifests_root.rglob("*"):
        if manifest_path.is_dir():
            continue
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except Exception:
            continue

        layers = doc.get("layers") or []
        if not isinstance(layers, list) or not layers:
            continue

        model_digest: Optional[str] = None
        for layer in layers:
            try:
                mt = str(layer.get("mediaType", ""))
                dg = str(layer.get("digest", ""))
            except Exception:
                continue
            # Prefer the model layer
            if "ollama.image.model" in mt or mt.endswith(".model"):
                model_digest = dg
                break
        if model_digest is None:
            # Fallback: first digest that looks like sha256:...
            for layer in layers:
                dg = str((layer or {}).get("digest", ""))
                if dg.startswith("sha256:"):
                    model_digest = dg
                    break
        if not model_digest or not model_digest.startswith("sha256:"):
            continue

        blob_hash = model_digest.split("sha256:", 1)[1]
        blob_path = blobs_root / f"sha256-{blob_hash}"
        if not blob_path.exists():
            continue

        # Derive a stable model name from manifest path:
        # manifests/<registry>/<namespace>/<repo>/<tag>
        rel = manifest_path.relative_to(manifests_root)
        parts = rel.parts
        if len(parts) >= 4:
            registry, namespace, repo, tag = parts[0], parts[1], parts[2], parts[3]
            model_name = f"{namespace}/{repo}:{tag}"
        else:
            model_name = "/".join(parts)

        try:
            st = blob_path.stat()
        except OSError:
            continue

        out.append(
            ModelInfo(
                source="ollama",
                model_name=model_name,
                gguf_path=str(blob_path),
                size_bytes=int(st.st_size),
                mtime=float(st.st_mtime),
            )
        )

    # De-dup by (source, model_name, gguf_path)
    uniq: Dict[tuple[str, str, str], ModelInfo] = {}
    for m in out:
        uniq[(m.source, m.model_name, m.gguf_path)] = m
    return sorted(uniq.values(), key=lambda m: (m.source, m.model_name))


def discover_all_models() -> List[ModelInfo]:
    return sorted(discover_ollama_manifests() + discover_lmstudio_gguf(), key=lambda m: (m.source, m.model_name))


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name)


def export_models(
    *,
    models: List[ModelInfo],
    export_dir: str,
    mode: str = "symlink",
) -> List[Dict[str, Any]]:
    """
    Export discovered GGUF files into a stable directory (repo-local).

    mode:
      - \"symlink\" (default): create symlinks
      - \"copy\": copy files
    """
    out: List[Dict[str, Any]] = []
    dst_root = Path(export_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    for m in models:
        src = Path(m.gguf_path)
        # Note: Ollama blob files may have no .gguf extension, but they are the model layer.
        dst_name = f"{m.source}__{_safe_filename(m.model_name)}.gguf"
        dst = dst_root / dst_name

        if dst.exists():
            out.append({"model_name": m.model_name, "source": m.source, "path": str(dst), "status": "exists"})
            continue

        if mode == "copy":
            shutil.copy2(src, dst)
            out.append({"model_name": m.model_name, "source": m.source, "path": str(dst), "status": "copied"})
            continue

        # default symlink
        try:
            os.symlink(str(src), str(dst))
            out.append({"model_name": m.model_name, "source": m.source, "path": str(dst), "status": "symlinked"})
        except FileExistsError:
            out.append({"model_name": m.model_name, "source": m.source, "path": str(dst), "status": "exists"})
        except OSError:
            # fallback to copy if symlink fails
            shutil.copy2(src, dst)
            out.append({"model_name": m.model_name, "source": m.source, "path": str(dst), "status": "copied_fallback"})

    return out


