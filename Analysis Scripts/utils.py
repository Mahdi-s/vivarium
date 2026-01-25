from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class RunRef:
    run_id: str
    run_dir: str


def _extract_run_id_from_dir_name(run_dir: str) -> str:
    name = Path(run_dir).name
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return name


def find_run_dir_for_run_id(*, run_id: str, runs_dir: str) -> str:
    """
    Find the most recently modified run directory matching run_id.

    Matches both:
    - runs/<timestamp>_<run_id>
    - runs/<run_id>
    """
    base = Path(runs_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"runs_dir not found: {base}")

    matches: list[Path] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n == run_id or n.endswith(run_id) or _extract_run_id_from_dir_name(str(p)) == run_id:
            matches.append(p)

    if not matches:
        raise FileNotFoundError(f"No run directory found for run_id={run_id!r} under {str(base)!r}")

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matches[0])


def resolve_run_ref(*, run_id: Optional[str], run_dir: Optional[str], runs_dir: str) -> RunRef:
    if run_dir:
        rd = str(Path(run_dir).expanduser().resolve())
        if not Path(rd).exists():
            raise FileNotFoundError(f"run_dir not found: {rd}")
        rid = run_id or _extract_run_id_from_dir_name(rd)
        return RunRef(run_id=rid, run_dir=rd)

    if run_id:
        rd = find_run_dir_for_run_id(run_id=str(run_id), runs_dir=runs_dir)
        return RunRef(run_id=str(run_id), run_dir=rd)

    raise ValueError("Must provide either --run-id or --run-dir")


def build_arg_parser(*, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run-id", type=str, default=None, help="Run UUID (or suffix of runs/<timestamp>_<run_id>)")
    p.add_argument("--run-dir", type=str, default=None, help="Path to run directory (contains simulation.db)")
    p.add_argument("--runs-dir", type=str, default=str(REPO_ROOT / "runs"), help="Base runs/ directory for --run-id lookup")
    return p


def load_db_for_run(run_ref: RunRef):
    from aam.analytics.utils import load_simulation_db

    return load_simulation_db(run_ref.run_dir)


def ensure_artifacts_dirs(run_dir: str) -> Tuple[str, str]:
    figures_dir = os.path.join(run_dir, "artifacts", "figures")
    tables_dir = os.path.join(run_dir, "artifacts", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return figures_dir, tables_dir

