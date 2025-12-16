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


