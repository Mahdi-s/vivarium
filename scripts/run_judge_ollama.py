#!/usr/bin/env python3
"""
Parallel LLM judge runner for local Ollama (GPT-OSS 20B by default).

This script mirrors scripts/run_judge_openrouter.py and adds:
1) pre-flight run inspection (model id, total rows, judged, missing, parse errors)
2) Ollama health check (GET /api/tags) before dispatching judge subprocesses

Default behavior
----------------
- Targets rows that were never LLM-judged (forwards --no-llm-judge by default).
- Retries rows that previously produced parse errors (forwards --retry-parse-errors).
- Parallelises at the database level with a conservative default of 1 DB at a time.
- Uses local Ollama by default:
    model    = gpt-oss:20b
    api_base = http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f\-]{36}$")

RunInfo = namedtuple(
    "RunInfo",
    [
        "run_id",
        "db_path",
        "model_id",
        "total_rows",
        "judged_rows",
        "missing_rows",
        "parse_error_rows",
    ],
)


def discover_runs(runs_dirs: List[Path]) -> List[Tuple[str, Path]]:
    """Return deduplicated [(run_id, db_path), ...] from supplied roots."""
    seen: Dict[str, Path] = {}
    results: List[Tuple[str, Path]] = []

    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            print(f"  [warn] runs-dir not found, skipping: {runs_dir}")
            continue

        if _RUN_DIR_RE.match(runs_dir.name):
            db = runs_dir / "simulation.db"
            if db.exists():
                run_id = runs_dir.name.split("_", 2)[-1]
                if run_id not in seen:
                    seen[run_id] = db
                    results.append((run_id, db))
                    continue
                print(f"  [dedup] run_id={run_id[:8]}... already seen, skipping {runs_dir}")
                continue

        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not _RUN_DIR_RE.match(entry.name):
                continue
            db = entry / "simulation.db"
            if not db.exists():
                print(f"  [skip] {entry.name} - simulation.db not found")
                continue
            run_id = entry.name.split("_", 2)[-1]
            if run_id in seen:
                print(f"  [dedup] run_id={run_id[:8]}... already seen, skipping {entry}")
                continue
            seen[run_id] = db
            results.append((run_id, db))

    return results


def _extract_model_id_from_config_json(config_json: Optional[str]) -> str:
    if not config_json:
        return "unknown"
    try:
        cfg = json.loads(config_json)
        models = cfg.get("suite_config", {}).get("models", [])
        if isinstance(models, list) and models:
            model_id = models[0].get("model_id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id
        return "unknown"
    except Exception:
        return "unknown"


def inspect_run_db(run_id: str, db_path: Path) -> RunInfo:
    """Inspect DB for model ID and judge coverage counters."""
    model_id = "unknown"
    total_rows = 0
    judged_rows = 0
    parse_error_rows = 0

    conn = sqlite3.connect(str(db_path))
    try:
        run_row = conn.execute("SELECT config_json FROM runs LIMIT 1").fetchone()
        model_id = _extract_model_id_from_config_json(run_row[0] if run_row else None)

        total_rows = conn.execute("SELECT COUNT(*) FROM conformity_outputs").fetchone()[0]
        rows = conn.execute("SELECT parsed_answer_json FROM conformity_outputs").fetchall()
        for row in rows:
            parsed = row[0]
            if not parsed:
                continue
            if '"_llm_judge"' in parsed:
                judged_rows += 1
            if "[parse_error]" in parsed:
                parse_error_rows += 1
    finally:
        conn.close()

    missing_rows = total_rows - judged_rows
    return RunInfo(
        run_id=run_id,
        db_path=db_path,
        model_id=model_id,
        total_rows=total_rows,
        judged_rows=judged_rows,
        missing_rows=missing_rows,
        parse_error_rows=parse_error_rows,
    )


def _print_inspection_table(infos: List[RunInfo]) -> None:
    print("\nPre-flight run inspection")
    print("Run ID       Model evaluated                           Total  Judged  Missing  ParseErr")
    print("------------ ---------------------------------------- ------ ------- -------- --------")
    for info in infos:
        print(
            f"{info.run_id[:8] + '...':<12} "
            f"{info.model_id[:40]:<40} "
            f"{info.total_rows:>6} "
            f"{info.judged_rows:>7} "
            f"{info.missing_rows:>8} "
            f"{info.parse_error_rows:>8}"
        )
    print("------------ ---------------------------------------- ------ ------- -------- --------")
    print(
        f"{'TOTAL':<12} {'':<40} "
        f"{sum(i.total_rows for i in infos):>6} "
        f"{sum(i.judged_rows for i in infos):>7} "
        f"{sum(i.missing_rows for i in infos):>8} "
        f"{sum(i.parse_error_rows for i in infos):>8}"
    )


def _is_localhost_api_base(api_base: str) -> bool:
    base = api_base.lower()
    return ("localhost" in base) or ("127.0.0.1" in base) or ("::1" in base)


def _to_ollama_tags_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/api/tags"


def check_ollama_health(api_base: str, timeout_s: float = 3.0) -> bool:
    """Ping Ollama /api/tags. Return True on healthy response."""
    url = _to_ollama_tags_url(api_base)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


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
    """Spawn olmo-conformity-judgeval for one run and wait for completion."""
    cmd: List[str] = [
        sys.executable,
        "-m",
        "vivarium",
        "olmo-conformity-judgeval",
        "--run-id",
        run_id,
        "--db",
        str(db_path),
        "--max-concurrency",
        str(max_concurrency),
        "--trial-scope",
        trial_scope,
        "--parse-error-retries",
        "5",
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

    print(f"  [start] run_id={run_id[:8]}...  db={db_path}")
    t0 = time.monotonic()
    env = {**os.environ, **extra_env}

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
    print(f"  [{status}] run_id={run_id[:8]}...  elapsed={elapsed:.1f}s")
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


def main() -> int:
    default_judge_config = (
        REPO_ROOT / "experiments" / "olmo_conformity" / "configs" / "judge_config_qwen35_9b_local.json"
    )

    ap = argparse.ArgumentParser(
        description=(
            "Run olmo-conformity-judgeval in parallel across all run databases "
            "found under one or more --runs-dir paths, targeting local Ollama."
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
            "Can be specified multiple times. Default: runs"
        ),
    )
    ap.add_argument(
        "--judge-config",
        type=str,
        default=str(default_judge_config),
        help="Path to judge config JSON (default: local gpt-oss:20b config).",
    )
    ap.add_argument(
        "--judge-model",
        type=str,
        default="qwen3.5:9b",
        help="Judge model id (default: qwen3.5:9b).",
    )
    ap.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:11434/v1",
        help="API base URL — Ollama local or any OpenAI-compatible endpoint (default: http://localhost:11434/v1).",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Bearer API key for remote providers (e.g. OpenRouter sk-or-v1-…). "
            "Falls back to OPENROUTER_API_KEY / JUDGE_API_KEY env vars. "
            "Not needed for local Ollama."
        ),
    )
    ap.add_argument(
        "--max-parallel-dbs",
        type=int,
        default=1,
        help="Maximum databases to judge concurrently (default: 1).",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent judge requests per DB subprocess (default: 4).",
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
        help="Pass --force to every subprocess (overwrite all existing judge labels).",
    )
    ap.add_argument(
        "--no-no-llm-judge",
        dest="no_llm_judge",
        action="store_false",
        default=True,
        help=(
            "Disable the --no-llm-judge filter. By default, rows missing the "
            "_llm_judge key are targeted."
        ),
    )
    ap.add_argument(
        "--no-retry-parse-errors",
        dest="retry_parse_errors",
        action="store_false",
        default=True,
        help="Disable retry of [parse_error] rows (enabled by default).",
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
        help="Pass --verbose to subprocesses.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would run without executing them.",
    )
    args = ap.parse_args()

    runs_dirs_raw: List[str] = args.runs_dirs or ["runs"]
    runs_dirs = [Path(d).expanduser().resolve() for d in runs_dirs_raw]

    # Resolve API key (only needed when pointing at a remote provider)
    api_key: Optional[str] = args.api_key
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JUDGE_API_KEY") or None

    judge_config = Path(args.judge_config).expanduser().resolve() if args.judge_config else None
    if judge_config is not None and not judge_config.exists():
        print(f"[error] judge-config not found: {judge_config}")
        return 1

    print(f"Discovering runs under: {[str(d) for d in runs_dirs]}")
    runs = discover_runs(runs_dirs)
    if not runs:
        print("No valid run directories found. Exiting.")
        return 1

    inspection: List[RunInfo] = []
    for run_id, db_path in runs:
        try:
            inspection.append(inspect_run_db(run_id, db_path))
        except Exception as exc:
            print(f"  [warn] failed to inspect run_id={run_id[:8]}... ({exc})")
            inspection.append(
                RunInfo(run_id, db_path, "unknown", 0, 0, 0, 0)
            )
    _print_inspection_table(inspection)

    if _is_localhost_api_base(args.api_base):
        if not check_ollama_health(args.api_base):
            print("\n[error] Ollama is not running. Start it with: ollama serve")
            print("        Then re-run this script.")
            return 1

    print(
        f"\n{len(runs)} run(s) found.\n"
        f"  judge_model    : {args.judge_model}\n"
        f"  api_base       : {args.api_base}\n"
        f"  parallel_dbs   : {args.max_parallel_dbs}\n"
        f"  concurrency    : {args.max_concurrency}/db\n"
        f"  trial_scope    : {args.trial_scope}\n"
        f"  variant_filter : {args.variant_filter or 'all'}\n"
        f"  no_llm_judge   : {args.no_llm_judge}\n"
        f"  retry_errors   : {args.retry_parse_errors}\n"
        f"  force          : {args.force}\n"
    )

    sem = asyncio.Semaphore(args.max_parallel_dbs)

    # Let Ollama use per-request parallelism unless caller explicitly set it.
    extra_env: Dict[str, str] = {}
    if os.environ.get("OLLAMA_NUM_PARALLEL") is None:
        extra_env["OLLAMA_NUM_PARALLEL"] = str(args.max_concurrency)

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
                extra_env=extra_env,
            )

    async def _run_all() -> List[Tuple[str, int, str]]:
        tasks = [asyncio.create_task(_bounded(r, d)) for r, d in runs]
        return await asyncio.gather(*tasks)

    wall_start = time.monotonic()
    all_results = asyncio.run(_run_all())
    wall_elapsed = time.monotonic() - wall_start

    ok = sum(1 for _, rc, _ in all_results if rc == 0)
    failed = [(r, rc) for r, rc, _ in all_results if rc != 0]

    print(f"\n{'=' * 60}")
    print(f"Done in {wall_elapsed:.1f}s.  OK={ok}  FAILED={len(failed)}")
    for run_id, rc in failed:
        print(f"  FAILED: run_id={run_id[:8]}...  rc={rc}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
