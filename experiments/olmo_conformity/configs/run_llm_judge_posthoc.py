#!/usr/bin/env python3
"""
Post-hoc LLM judge scoring for an existing OLMo conformity run.

This is a thin wrapper around:
  python -m aam.run olmo-conformity-judgeval ...

Why this wrapper exists:
- On HPC we resolve run_dir under a scratch `runs_dir` (from paths.json) using run_id.
- We also optionally load Ollama judge settings (ollama_base, ollama_judge_model) from paths.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # experiments/olmo_conformity/configs -> repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


PATHS_CONFIG_FILE = SCRIPT_DIR / "paths.json"


def _load_paths_config() -> Dict[str, str]:
    if PATHS_CONFIG_FILE.exists():
        return json.loads(PATHS_CONFIG_FILE.read_text())
    return {}


def _find_run_dir(*, runs_dir: Path, run_id: str) -> Path:
    matches = sorted(runs_dir.glob(f"*_{run_id}"), key=lambda p: p.name, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find run_dir under {runs_dir} ending with _{run_id}")
    return matches[0]


def _ollama_base_for_env(ollama_base: str) -> str:
    """
    Convert an OpenAI-compatible base (e.g. http://localhost:11434/v1) to OLLAMA_HOST
    (e.g. localhost:11434).
    """
    s = str(ollama_base).strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]
    return s


def _wait_for_ollama(*, ollama_base: str, timeout_s: float = 30.0) -> bool:
    """
    Wait until Ollama's OpenAI-compatible endpoint responds.
    """
    try:
        import httpx
    except Exception:
        return False

    url = str(ollama_base).rstrip("/") + "/models"
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _ensure_ollama_serving(*, paths: Dict[str, str], ollama_base: str, judge_model: str) -> Optional[subprocess.Popen]:
    """
    If ollama_base is unreachable and paths.json defines ollama_server_path, start
    `ollama serve` in the background and wait until it responds.

    Returns:
      Popen handle if we started a server (caller should terminate it), else None.
    """
    if _wait_for_ollama(ollama_base=ollama_base, timeout_s=2.0):
        return None

    ollama_path = paths.get("ollama_server_path")
    if not ollama_path:
        return None

    exe = Path(ollama_path)
    if not exe.exists():
        raise FileNotFoundError(f"ollama_server_path does not exist: {exe}")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = _ollama_base_for_env(ollama_base)

    print(f"Starting Ollama server: {exe} serve (OLLAMA_HOST={env['OLLAMA_HOST']}, judge_model={judge_model})")
    proc = subprocess.Popen(
        [str(exe), "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    ok = _wait_for_ollama(ollama_base=ollama_base, timeout_s=30.0)
    if not ok:
        try:
            if proc.stdout is not None:
                tail = proc.stdout.read(4000)
                if tail:
                    print("Ollama output (tail):")
                    print(tail)
        except Exception:
            pass
        proc.terminate()
        raise RuntimeError(f"Started Ollama, but {ollama_base}/models never became reachable.")

    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc Ollama LLM judge scoring for a run_id")
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--hpc", action="store_true", help="Use runs_dir/ollama settings from paths.json")
    ap.add_argument("--runs-dir", type=str, default=None, help="Override base runs directory")
    ap.add_argument("--run-dir", type=str, default=None, help="Override exact run directory")
    ap.add_argument("--db", type=str, default=None, help="Override simulation.db path")
    ap.add_argument("--judge-model", type=str, default=None, help="Override Ollama judge model name")
    ap.add_argument("--ollama-base", type=str, default=None, help="Override Ollama OpenAI-compatible base URL")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--trial-scope", type=str, default="behavioral-only", choices=["behavioral-only", "all"])
    args = ap.parse_args()

    paths = _load_paths_config() if bool(args.hpc) else {}

    runs_dir: Optional[str] = args.runs_dir
    if runs_dir is None and bool(args.hpc):
        runs_dir = paths.get("runs_dir")
    if runs_dir is None:
        runs_dir = "./runs"

    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir(runs_dir=Path(runs_dir), run_id=str(args.run_id))
    db_path = Path(args.db) if args.db else (run_dir / "simulation.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Missing simulation.db at {db_path}")

    judge_model = args.judge_model or (paths.get("ollama_judge_model") if bool(args.hpc) else None) or "gpt-oss:20b"
    ollama_base = args.ollama_base or (paths.get("ollama_base") if bool(args.hpc) else None) or "http://localhost:11434/v1"

    print("Resolved:")
    print(f"  run_id={args.run_id}")
    print(f"  run_dir={run_dir}")
    print(f"  db={db_path}")
    print(f"  judge_model={judge_model}")
    print(f"  ollama_base={ollama_base}")

    ollama_proc: Optional[subprocess.Popen] = None
    try:
        if bool(args.hpc):
            ollama_proc = _ensure_ollama_serving(paths=paths, ollama_base=str(ollama_base), judge_model=str(judge_model))

        from aam.run import main as aam_main  # imported after sys.path setup

        argv = [
            "olmo-conformity-judgeval",
            "--run-id",
            str(args.run_id),
            "--db",
            str(db_path),
            "--judge-model",
            str(judge_model),
            "--ollama-base",
            str(ollama_base),
            "--max-concurrency",
            str(int(args.max_concurrency)),
            "--trial-scope",
            str(args.trial_scope),
        ]
        if bool(args.force):
            argv.append("--force")
        if args.limit is not None:
            argv.extend(["--limit", str(int(args.limit))])

        return int(aam_main(argv))
    finally:
        if ollama_proc is not None:
            print("Stopping Ollama server...")
            try:
                ollama_proc.terminate()
                ollama_proc.wait(timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
