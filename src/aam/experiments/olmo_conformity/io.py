from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


JsonDict = Dict[str, Any]


def sha256_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str) -> List[JsonDict]:
    p = Path(path)
    out: List[JsonDict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def deterministic_prompt_hash(*, system: str, user: str, history: List[JsonDict]) -> str:
    payload = {
        "system": system,
        "user": user,
        "history": history,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    path: str


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    params: JsonDict


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    model_id: str


def load_suite_config(path: str) -> JsonDict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_paths_config(suite_config_path: str, suite_config: JsonDict) -> JsonDict:
    """
    Load paths config referenced by suite config, or return defaults.
    
    Args:
        suite_config_path: Path to the suite config file (used to resolve relative paths)
        suite_config: The loaded suite config dict
        
    Returns:
        Dict with 'models_dir' and 'runs_dir' keys (values may be None if not configured)
    """
    paths_ref = suite_config.get("paths_config")
    if paths_ref:
        config_dir = Path(suite_config_path).parent
        paths_path = config_dir / paths_ref
        if paths_path.exists():
            return json.loads(paths_path.read_text(encoding="utf-8"))
    # Return defaults if no paths config
    return {"models_dir": None, "runs_dir": None}


def clamp_items(items: List[JsonDict], limit: Optional[int]) -> List[JsonDict]:
    if limit is None:
        return items
    try:
        n = int(limit)
    except Exception:
        return items
    if n <= 0:
        return []
    return items[:n]


