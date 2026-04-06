#!/usr/bin/env python3
"""
Parallel LLM judge runner for OpenRouter (and any OpenAI-compatible provider).

Discovers every ``simulation.db`` under one or more ``--runs-dir`` trees,
deduplicates by run_id (UUID), and calls ``python -m vivarium
olmo-conformity-judgeval`` once per database — up to ``--max-parallel-dbs``
databases at a time.

Default behavior
----------------
- Targets rows that were never LLM-judged (``--no-llm-judge`` flag forwarded
  to the subprocess) — this correctly handles the common case where all rows
  already have *heuristic* ``parsed_answer_json`` but lack the ``_llm_judge``
  key from a real judge model.
- Retries rows that previously produced parse errors (``--retry-parse-errors``
  forwarded automatically).
- Processes both ``runs/`` and ``runs/think/`` trees in one invocation by
  listing them in ``--runs-dir``.
- Parallelises at the database level (each DB gets its own subprocess), so
  SQLite write-contention is avoided.  In-DB concurrency is controlled
  separately via ``--max-concurrency``.

Quick-start
-----------
  # Dry run — see what would execute
  python scripts/run_judge_openrouter.py \\
      --runs-dir runs --runs-dir runs/think \\
      --judge-model qwen/qwen3-8b \\
      --api-base https://openrouter.ai/api/v1 \\
      --dry-run

  # Real run (API key from env var OPENROUTER_API_KEY)
  python scripts/run_judge_openrouter.py \\
      --runs-dir runs --runs-dir runs/think \\
      --judge-config experiments/olmo_conformity/configs/judge_config_openrouter.json \\
      --max-parallel-dbs 3 \\
      --max-concurrency 8

  # Force re-judge everything (e.g. after prompt update)
  python scripts/run_judge_openrouter.py \\
      --runs-dir runs --runs-dir runs/think \\
      --judge-config experiments/olmo_conformity/configs/judge_config_openrouter.json \\
      --force

  # Single database (useful for testing)
  python scripts/run_judge_openrouter.py \\
      --runs-dir runs/20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d \\
      --judge-model qwen/qwen3-8b \\
      --api-base https://openrouter.ai/api/v1 \\
      --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Matches the standard run directory name: YYYYMMDD_HHMMSS_<uuid>
_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f\-]{36}$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(runs_dirs: List[Path]) -> List[Tuple[str, Path]]:
    """Return deduplicated [(run_id, db_path), ...] from all supplied roots.

    Iterates each directory recursively one level deep looking for folders
    matching the YYYYMMDD_HHMMSS_<uuid> pattern that contain a simulation.db.
    When the same run_id (UUID) appears under multiple roots, the first
    occurrence wins (roots are processed in argument order).
    """
    seen: Dict[str, Path] = {}
    results: List[Tuple[str, Path]] = []

    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            print(f"  [warn] runs-dir not found, skipping: {runs_dir}")
            continue

        # The directory itself might BE a run folder (single-run mode)
        if _RUN_DIR_RE.match(runs_dir.name):
            db = runs_dir / "simulation.db"
            if db.exists():
                run_id = runs_dir.name.split("_", 2)[-1]
                if run_id not in seen:
                    seen[run_id] = db
                    results.append((run_id, db))
                    continue
                else:
                    print(f"  [dedup] run_id={run_id[:8]}… already seen, skipping {runs_dir}")
                    continue

        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not _RUN_DIR_RE.match(entry.name):
                continue
            db = entry / "simulation.db"
            if not db.exists():
                print(f"  [skip] {entry.name} — simulation.db not found")
                continue
            run_id = entry.name.split("_", 2)[-1]
            if run_id in seen:
                print(f"  [dedup] run_id={run_id[:8]}… already seen (in {seen[run_id].parent.parent.name}), skipping {entry}")
                continue
            seen[run_id] = db
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
    judge_model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
    max_concurrency: int,
    trial_scope: str,
    force: bool,
    no_llm_judge: bool,
    retry_parse_errors: bool,
    variant_filter: Optional[str],
    verbose: bool,
    dry_run: bool,
    extra_env: dict,
) -> Tuple[str, int, str]:
    """Spawn ``olmo-conformity-judgeval`` for one run and wait for it.

    Returns (run_id, returncode, combined_output).
    """
    cmd: List[str] = [
        sys.executable, "-m", "vivarium", "olmo-conformity-judgeval",
        "--run-id", run_id,
        "--db", str(db_path),
        "--max-concurrency", str(max_concurrency),
        "--trial-scope", trial_scope,
        "--parse-error-retries", "5",
    ]
    if judge_config is not None:
        cmd += ["--judge-config", str(judge_config)]
    if judge_model is not None:
        cmd += ["--judge-model", judge_model]
    if api_base is not None:
        cmd += ["--ollama-base", api_base]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    if variant_filter is not None:
        cmd += ["--variant-filter", variant_filter]
    if force:
        cmd.append("--force")
    if no_llm_judge and not force:
        cmd.append("--no-llm-judge")
    if retry_parse_errors and not force:
        cmd.append("--retry-parse-errors")
    if verbose:
        cmd.append("--verbose")

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return (run_id, 0, "[dry-run]")

    print(f"  [start] run_id={run_id[:8]}…  db={db_path}")
    t0 = time.monotonic()

    env = {**os.environ, **extra_env}
    # With --verbose, inherit stdio so judgeval prints (per-row judge output,
    # progress) stream live.  PIPE + communicate() buffers until exit and
    # we discard non-summary lines on success — so verbose appeared "stuck".
    if verbose:
        env = {**env, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=None,
            stderr=None,
            cwd=str(REPO_ROOT),
            env=env,
        )
        await proc.wait()
        output = ""
    else:
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
    if rc != 0 and not verbose:
        tail = "\n".join(output.splitlines()[-20:])
        print(f"    --- output tail ---\n{tail}\n    ---")
    elif rc != 0 and verbose:
        print("    (see judgeval output above for error details)")
    else:
        for line in output.splitlines():
            if line.startswith("judgeval_"):
                print(f"    {line}")

    return (run_id, rc, output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run olmo-conformity-judgeval in parallel across all run databases "
            "found under one or more --runs-dir paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--runs-dir",
        dest="runs_dirs",
        action="append",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory (or individual run folder) to search for simulation.db files. "
            "Can be specified multiple times: e.g. --runs-dir runs --runs-dir runs/think. "
            "Default: runs"
        ),
    )
    ap.add_argument(
        "--judge-config",
        type=str,
        default=None,
        help="Path to judge config JSON (model, api_base, temperature, etc.)",
    )
    ap.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model id, e.g. qwen/qwen3-8b or gemma-3-1b-it (overrides --judge-config)",
    )
    ap.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL, e.g. https://openrouter.ai/api/v1 (overrides --judge-config)",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Bearer API key. If omitted, falls back to OPENROUTER_API_KEY / "
            "JUDGE_API_KEY environment variables."
        ),
    )
    ap.add_argument(
        "--max-parallel-dbs",
        type=int,
        default=3,
        help="Maximum number of databases to judge concurrently (default: 3).",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent judge requests per database subprocess (default: 8).",
    )
    ap.add_argument(
        "--trial-scope",
        type=str,
        default="all",
        choices=["behavioral-only", "all"],
        help="Which trials to score (default: all).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to every subprocess — overwrites ALL existing judge labels.",
    )
    ap.add_argument(
        "--no-no-llm-judge",
        dest="no_llm_judge",
        action="store_false",
        default=True,
        help=(
            "Disable the --no-llm-judge filter. By default, rows that have heuristic "
            "parsed_answer_json but lack the _llm_judge key are targeted. Pass this "
            "flag to skip that filter (only NULL/empty rows will be targeted)."
        ),
    )
    ap.add_argument(
        "--no-retry-parse-errors",
        dest="retry_parse_errors",
        action="store_false",
        default=True,
        help="Disable automatic retry of [parse_error] rows (enabled by default).",
    )
    ap.add_argument(
        "--resume-auto",
        action="store_true",
        help=(
            "Explicitly enable resume mode: skip already-judged rows and retry parse errors. "
            "Equivalent to the default behavior (--no-llm-judge + --retry-parse-errors are on). "
            "Use interchangeably between openrouter and local runner scripts."
        ),
    )
    ap.add_argument(
        "--variant-filter",
        type=str,
        default=None,
        metavar="VARIANTS",
        help=(
            "Comma-separated list of variant names to judge "
            "(e.g. 'base,instruct,instruct_sft,instruct_dpo'). "
            "When set, only matching trials are scored. Default: all variants."
        ),
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to subprocesses (prints raw judge output per row).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    args = ap.parse_args()

    # Resolve runs dirs
    runs_dirs_raw: List[str] = args.runs_dirs or ["runs"]
    runs_dirs = [Path(d).expanduser().resolve() for d in runs_dirs_raw]

    # Resolve API key
    api_key: Optional[str] = args.api_key
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JUDGE_API_KEY") or None

    judge_config = Path(args.judge_config).expanduser().resolve() if args.judge_config else None

    print(f"Discovering runs under: {[str(d) for d in runs_dirs]}")
    runs = discover_runs(runs_dirs)
    if not runs:
        print("No valid run directories found. Exiting.")
        return 1

    judge_label = args.judge_model or (str(judge_config) if judge_config else "default")
    base_label = args.api_base or "default"
    auth_label = "yes" if api_key else "from-env-or-none"
    print(
        f"\n{len(runs)} run(s) found.\n"
        f"  judge_model    : {judge_label}\n"
        f"  api_base       : {base_label}\n"
        f"  auth           : {auth_label}\n"
        f"  parallel_dbs   : {args.max_parallel_dbs}\n"
        f"  concurrency    : {args.max_concurrency}/db\n"
        f"  trial_scope    : {args.trial_scope}\n"
        f"  variant_filter : {args.variant_filter or 'all'}\n"
        f"  no_llm_judge   : {args.no_llm_judge}\n"
        f"  retry_errors   : {args.retry_parse_errors}\n"
        f"  force          : {args.force}\n"
    )

    sem = asyncio.Semaphore(args.max_parallel_dbs)

    async def _bounded(run_id: str, db_path: Path) -> Tuple[str, int, str]:
        async with sem:
            return await run_one(
                run_id=run_id,
                db_path=db_path,
                judge_config=judge_config,
                judge_model=args.judge_model or None,
                api_base=args.api_base or None,
                api_key=api_key,
                max_concurrency=args.max_concurrency,
                trial_scope=args.trial_scope,
                force=args.force,
                no_llm_judge=args.no_llm_judge,
                retry_parse_errors=args.retry_parse_errors,
                variant_filter=args.variant_filter or None,
                verbose=args.verbose,
                dry_run=args.dry_run,
                extra_env={},
            )

    async def _run_all() -> List[Tuple[str, int, str]]:
        tasks = [asyncio.create_task(_bounded(r, d)) for r, d in runs]
        return await asyncio.gather(*tasks)

    wall_start = time.monotonic()
    all_results = asyncio.run(_run_all())
    wall_elapsed = time.monotonic() - wall_start

    ok = sum(1 for _, rc, _ in all_results if rc == 0)
    failed = [(r, rc) for r, rc, _ in all_results if rc != 0]

    print(f"\n{'='*60}")
    print(f"Done in {wall_elapsed:.1f}s.  OK={ok}  FAILED={len(failed)}")
    for run_id, rc in failed:
        print(f"  FAILED: run_id={run_id[:8]}…  rc={rc}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
