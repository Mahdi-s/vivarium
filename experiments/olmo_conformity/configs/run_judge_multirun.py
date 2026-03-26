#!/usr/bin/env python3
"""
Parallel LLM judge runner — processes every run directory under a given
runs folder concurrently, calling `python -m vivarium olmo-conformity-judgeval`
once per run ID.

Why one subprocess per run?
  Each run has its own isolated simulation.db, so there is no SQLite
  write-contention when processes run in parallel.  The Ollama server
  handles the actual GPU parallelism via OLLAMA_NUM_PARALLEL.

M4 Ultra tuning guide
─────────────────────
  qwen3:0.6b is tiny; with 128 GB unified memory you can easily saturate
  the 40-core GPU by letting Ollama serve many requests simultaneously.

  Recommended starting point:
    --ollama-num-parallel 8   (Ollama server-side slots)
    --max-concurrency 8       (requests per run subprocess)

  With 6 runs × 8 concurrent requests each = up to 48 in-flight requests,
  all funnelled into the Ollama server which queues extras beyond its slot
  count.  Increase --ollama-num-parallel if you see GPU utilisation below
  ~80 % (check with `sudo powermetrics --samplers gpu_power -i 1000`).

Usage
─────
  # Dry run — see what would execute
  python run_judge_multirun.py \\
      --runs-dir ../../../runs_latest/runs \\
      --judge-config judge_config_local.json \\
      --dry-run

  # Real run
  python run_judge_multirun.py \\
      --runs-dir ../../../runs_latest/runs \\
      --judge-config judge_config_local.json \\
      --max-concurrency 8 \\
      --ollama-num-parallel 8

  # Force re-score even if parsed_answer_json already present
  python run_judge_multirun.py \\
      --runs-dir ../../../runs_latest/runs \\
      --judge-config judge_config_local.json \\
      --force
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]           # configs → olmo_conformity → experiments → repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Regex that matches the standard run directory names: <timestamp>_<uuid>
_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f\-]{36}$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(runs_dir: Path) -> List[Tuple[str, Path]]:
    """Return [(run_id, db_path), ...] for every valid run under runs_dir."""
    results: List[Tuple[str, Path]] = []
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not _RUN_DIR_RE.match(entry.name):
            continue
        db = entry / "simulation.db"
        if not db.exists():
            print(f"  [skip] {entry.name} — simulation.db not found")
            continue
        # run_id is the UUID portion after the last underscore-separated timestamp
        # directory name format: YYYYMMDD_HHMMSS_<uuid>
        run_id = entry.name.split("_", 2)[-1]  # everything after first two parts
        results.append((run_id, db))

    return results


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


async def run_one(
    *,
    run_id: str,
    db_path: Path,
    judge_config: Optional[Path],
    max_concurrency: int,
    trial_scope: str,
    force: bool,
    dry_run: bool,
    env: dict,
) -> Tuple[str, int, str]:
    """
    Spawn `python -m vivarium olmo-conformity-judgeval` for a single run and
    wait for it to finish.

    Returns (run_id, returncode, combined_output).
    """
    cmd: List[str] = [
        sys.executable, "-m", "vivarium", "olmo-conformity-judgeval",
        "--run-id", run_id,
        "--db", str(db_path),
        "--max-concurrency", str(max_concurrency),
        "--trial-scope", trial_scope,
    ]
    if judge_config is not None:
        cmd += ["--judge-config", str(judge_config)]
    if force:
        cmd.append("--force")

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return (run_id, 0, "[dry-run]")

    print(f"  [start] run_id={run_id[:8]}…  db={db_path}")
    t0 = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    stdout_bytes, _ = await proc.communicate()
    output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
    elapsed = time.monotonic() - t0
    rc = proc.returncode or 0

    status = "OK" if rc == 0 else f"FAILED(rc={rc})"
    print(f"  [{status}] run_id={run_id[:8]}…  elapsed={elapsed:.1f}s")
    if rc != 0:
        # Print last 20 lines of output to help with debugging
        tail = "\n".join(output.splitlines()[-20:])
        print(f"    --- output tail ---\n{tail}\n    ---")

    return (run_id, rc, output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run olmo-conformity-judgeval in parallel across all runs in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--runs-dir",
        type=str,
        default=str(REPO_ROOT / "runs_latest" / "runs"),
        help="Directory containing run folders (default: <repo>/runs_latest/runs)",
    )
    ap.add_argument(
        "--judge-config",
        type=str,
        default=str(SCRIPT_DIR / "judge_config_local.json"),
        help="Path to judge config JSON (default: judge_config_local.json next to this script)",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent Ollama requests per run subprocess (default: 8)",
    )
    ap.add_argument(
        "--ollama-num-parallel",
        type=int,
        default=8,
        help="Set OLLAMA_NUM_PARALLEL env var — Ollama server-side GPU slots (default: 8)",
    )
    ap.add_argument(
        "--trial-scope",
        type=str,
        default="behavioral-only",
        choices=["behavioral-only", "all"],
        help="Which trials to score (default: behavioral-only)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parsed_answer_json entries",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    judge_config = Path(args.judge_config).expanduser().resolve() if args.judge_config else None

    if judge_config is not None and not judge_config.exists():
        print(f"Error: --judge-config not found: {judge_config}", file=sys.stderr)
        return 1

    print(f"Discovering runs in: {runs_dir}")
    runs = discover_runs(runs_dir)
    if not runs:
        print("No valid run directories found. Exiting.")
        return 0

    print(f"Found {len(runs)} run(s):")
    for run_id, db in runs:
        print(f"  {run_id[:8]}…  {db}")

    # Build subprocess environment: inherit everything, add PYTHONPATH and
    # OLLAMA_NUM_PARALLEL so the Ollama server serves concurrent GPU slots.
    env = os.environ.copy()
    python_path = str(SRC_DIR)
    if env.get("PYTHONPATH"):
        python_path = python_path + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path
    env["OLLAMA_NUM_PARALLEL"] = str(args.ollama_num_parallel)

    if not args.dry_run:
        print(
            f"\nStarting parallel judge run:"
            f"\n  judge_config   = {judge_config}"
            f"\n  max_concurrency (per run) = {args.max_concurrency}"
            f"\n  OLLAMA_NUM_PARALLEL       = {args.ollama_num_parallel}"
            f"\n  trial_scope    = {args.trial_scope}"
            f"\n  force          = {args.force}"
        )

    t_start = time.monotonic()

    async def run_all() -> List[Tuple[str, int, str]]:
        tasks = [
            run_one(
                run_id=run_id,
                db_path=db_path,
                judge_config=judge_config,
                max_concurrency=args.max_concurrency,
                trial_scope=args.trial_scope,
                force=args.force,
                dry_run=args.dry_run,
                env=env,
            )
            for run_id, db_path in runs
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.monotonic() - t_start
    ok = sum(1 for _, rc, _ in results if rc == 0)
    failed = len(results) - ok

    print(f"\n{'='*60}")
    print(f"Judge multirun complete  ({total_elapsed:.1f}s total)")
    print(f"  runs processed : {len(results)}")
    print(f"  succeeded      : {ok}")
    print(f"  failed         : {failed}")
    print(f"{'='*60}")

    if failed:
        print("\nFailed runs:")
        for run_id, rc, _ in results:
            if rc != 0:
                print(f"  run_id={run_id}  rc={rc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
