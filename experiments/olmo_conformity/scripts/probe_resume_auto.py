#!/usr/bin/env python3
"""
Read-only probe: list runs under runs_dir that match each suite (same rules as
`vivarium olmo-conformity --resume-auto`) and print trial counts per run_id /
folder. Also reports which incomplete run would be chosen for resume.

Uses the same matching and "completed trial" definition as
`_find_incomplete_run_for_suite` in runner.py. Does not run trials or modify
databases.

Examples (from repo root on the login node):

  # Scratch runs dir from experiments/olmo_conformity/configs/paths.json
  python experiments/olmo_conformity/scripts/probe_resume_auto.py --hpc \\
      experiments/olmo_conformity/configs/suite_olmo_7b_base_ablations_temp0p0.json

  # Built-in list covering ngram / Panel B / full-5k suite filenames
  python experiments/olmo_conformity/scripts/probe_resume_auto.py --hpc --preset hpc_masters

  # Explicit runs directory (overrides paths.json)
  python experiments/olmo_conformity/scripts/probe_resume_auto.py \\
      --runs-dir /scratch1/mahdisae/olmo_experiments/runs \\
      path/to/suite.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root: .../abstractAgentMachine
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CONFIGS_DIR = REPO_ROOT / "experiments" / "olmo_conformity" / "configs"
PATHS_JSON = CONFIGS_DIR / "paths.json"

# Suite JSONs used by the three master launchers (ngram, panel B, full 5k).
PRESET_HPC_MASTERS: List[str] = [
    # ngram ablations (4 variants x 2 temps)
    "suite_olmo_7b_base_ablations_temp0p0.json",
    "suite_olmo_7b_base_ablations_temp0p6.json",
    "suite_olmo_7b_instruct_sft_ablations_temp0p0.json",
    "suite_olmo_7b_instruct_sft_ablations_temp0p6.json",
    "suite_olmo_7b_instruct_dpo_ablations_temp0p0.json",
    "suite_olmo_7b_instruct_dpo_ablations_temp0p6.json",
    "suite_olmo_7b_instruct_rlvr_ablations_temp0p0.json",
    "suite_olmo_7b_instruct_rlvr_ablations_temp0p6.json",
    # Panel B extension (6 models x 2 temps)
    "suite_olmo_7b_think_sft_panelB_extension_temp0p0.json",
    "suite_olmo_7b_think_sft_panelB_extension_temp0p6.json",
    "suite_olmo_7b_think_dpo_panelB_extension_temp0p0.json",
    "suite_olmo_7b_think_dpo_panelB_extension_temp0p6.json",
    "suite_olmo_7b_think_rlvr_panelB_extension_temp0p0.json",
    "suite_olmo_7b_think_rlvr_panelB_extension_temp0p6.json",
    "suite_olmo_32b_think_sft_panelB_extension_temp0p0.json",
    "suite_olmo_32b_think_sft_panelB_extension_temp0p6.json",
    "suite_olmo_32b_think_dpo_panelB_extension_temp0p0.json",
    "suite_olmo_32b_think_dpo_panelB_extension_temp0p6.json",
    "suite_olmo_32b_think_rlvr_panelB_extension_temp0p0.json",
    "suite_olmo_32b_think_rlvr_panelB_extension_temp0p6.json",
    # 7B Think RLVR full 5k
    "suite_olmo_7b_think_rlvr_full_5k_temp0p0.json",
    "suite_olmo_7b_think_rlvr_full_5k_temp0p6.json",
]


def _load_hpc_runs_dir() -> str:
    if not PATHS_JSON.is_file():
        raise FileNotFoundError(f"Missing {PATHS_JSON} (need --runs-dir or fix paths.json)")
    with open(PATHS_JSON, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    rd = data.get("runs_dir")
    if not rd:
        raise KeyError("paths.json has no runs_dir")
    return str(rd)


def _expected_item_count(suite_config_path: Path, cfg: Dict[str, Any]) -> int:
    from vivarium.settings import settings
    from vivarium.experiments.olmo_conformity.io import clamp_items, read_jsonl

    suite_dir = suite_config_path.resolve().parent
    total = 0
    for ds in cfg.get("datasets", []):
        rel_path = str(ds["path"])
        p = Path(rel_path)
        if not p.is_absolute():
            suite_rel = (suite_dir / p).resolve()
            abs_path = str(suite_rel) if suite_rel.exists() else str((Path(str(settings.PROJECT_ROOT)) / p).resolve())
        else:
            abs_path = str(p)
        try:
            total += len(clamp_items(read_jsonl(abs_path), cfg.get("run", {}).get("max_items_per_dataset")))
        except Exception:
            pass
    return total


def _trial_stats(conn: sqlite3.Connection, run_id: str) -> Tuple[int, int, int]:
    """
    Returns (trials_with_good_output, trials_with_any_row, trials_with_error_only_output).

    "Good" matches runner._find_incomplete_run_for_suite: distinct trial_id with
    a joined output whose raw_text does not start with '<error'.
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT t.trial_id) AS n FROM conformity_trials t "
        "JOIN conformity_outputs o ON o.trial_id = t.trial_id "
        "WHERE t.run_id = ? AND o.raw_text NOT LIKE ?",
        (run_id, "<error%"),
    ).fetchone()
    good = int(row[0]) if row and row[0] is not None else 0

    row2 = conn.execute(
        "SELECT COUNT(DISTINCT trial_id) AS n FROM conformity_trials WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    any_trial = int(row2[0]) if row2 and row2[0] is not None else 0

    row3 = conn.execute(
        "SELECT COUNT(DISTINCT t.trial_id) AS n FROM conformity_trials t "
        "JOIN conformity_outputs o ON o.trial_id = t.trial_id "
        "WHERE t.run_id = ? AND o.raw_text LIKE ?",
        (run_id, "<error%"),
    ).fetchone()
    err_out = int(row3[0]) if row3 and row3[0] is not None else 0

    return good, any_trial, err_out


def _list_matching_runs_with_stats(
    runs_dir: str,
    suite_config_sha256: str,
    current_cfg: Dict[str, Any],
    model_ids: List[str],
    condition_names: List[str],
    expected_item_count: int,
) -> List[Dict[str, Any]]:
    """
    Every run folder under runs_dir whose DB matches this suite (SHA or
    normalized identity), with per-run trial counts.
    """
    from vivarium.experiments.olmo_conformity.runner import _normalize_suite_identity

    current_identity = _normalize_suite_identity(current_cfg)
    expected_cells = expected_item_count * len(condition_names) * len(model_ids)
    out: List[Dict[str, Any]] = []

    if not os.path.isdir(runs_dir):
        return out

    for name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, name)
        db_path = os.path.join(run_path, "simulation.db")
        if not os.path.isdir(run_path) or not os.path.isfile(db_path):
            continue

        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT run_id, config_json FROM runs LIMIT 1").fetchone()
            if not row:
                conn.close()
                continue
            db_run_id = str(row[0])
            raw_cfg = row[1]
            config = json.loads(raw_cfg) if raw_cfg else {}

            stored_sha = config.get("suite_config_sha256")
            matched = stored_sha is not None and stored_sha == suite_config_sha256
            if not matched:
                stored_suite_cfg = config.get("suite_config")
                if isinstance(stored_suite_cfg, dict):
                    stored_identity = _normalize_suite_identity(stored_suite_cfg)
                    matched = stored_identity == current_identity
            if not matched:
                conn.close()
                continue

            good, any_trial, err_out = _trial_stats(conn, db_run_id)
            conn.close()

            mtime = os.path.getmtime(db_path)
            complete = good >= expected_cells
            out.append(
                {
                    "folder": name,
                    "run_path": run_path,
                    "run_id": db_run_id,
                    "trials_good_output": good,
                    "trials_rows": any_trial,
                    "trials_error_output": err_out,
                    "expected_cells": expected_cells,
                    "complete": complete,
                    "db_mtime": mtime,
                }
            )
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            continue

    # Sort: incomplete first (resume interest), then most progress, then newest mtime
    out.sort(
        key=lambda r: (
            0 if not r["complete"] else 1,
            -r["trials_good_output"],
            -r["db_mtime"],
        )
    )
    return out


def _pick_resume_auto_target(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Same selection as _find_incomplete_run_for_suite: incomplete, max good count, newest mtime."""
    incomplete = [r for r in rows if not r["complete"]]
    if not incomplete:
        return None
    incomplete.sort(key=lambda r: (r["trials_good_output"], r["db_mtime"]), reverse=True)
    return incomplete[0]


def _probe_suite(
    suite_config_path: Path,
    runs_dir: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int, str]:
    from vivarium.experiments.olmo_conformity.io import load_suite_config, sha256_file

    cfg = load_suite_config(str(suite_config_path))
    suite_path = suite_config_path.resolve()
    sha = sha256_file(str(suite_path))
    model_ids = [str(m.get("model_id") or "mock") for m in cfg.get("models", [])]
    condition_names = [str(c.get("name") or "") for c in cfg.get("conditions", [])]
    n_items = _expected_item_count(suite_path, cfg)
    expected_cells = n_items * len(condition_names) * len(model_ids)

    rows = _list_matching_runs_with_stats(
        runs_dir=runs_dir,
        suite_config_sha256=sha,
        current_cfg=cfg,
        model_ids=model_ids,
        condition_names=condition_names,
        expected_item_count=n_items,
    )
    resume = _pick_resume_auto_target(rows)
    return rows, resume, expected_cells, sha[:12]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List suite-matching runs under runs_dir with trial counts; show --resume-auto target."
    )
    parser.add_argument(
        "suite_configs",
        nargs="*",
        help="Suite JSON paths (relative to repo root or absolute). If omitted, use --preset.",
    )
    parser.add_argument("--hpc", action="store_true", help=f"Use runs_dir from {PATHS_JSON.name}")
    parser.add_argument("--runs-dir", type=str, default=None, help="Override runs base directory (scratch tree)")
    parser.add_argument(
        "--preset",
        choices=["hpc_masters"],
        default=None,
        help="Use built-in suite list instead of positional suite_configs",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir or os.environ.get("AAM_RUNS_DIR") or os.environ.get("VIVARIUM_ARTIFACTS_DIR")
    if not runs_dir and args.hpc:
        runs_dir = _load_hpc_runs_dir()
    if not runs_dir:
        print(
            "ERROR: set --runs-dir, or pass --hpc with configs/paths.json, "
            "or export AAM_RUNS_DIR / VIVARIUM_ARTIFACTS_DIR.",
            file=sys.stderr,
        )
        return 2

    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        print(f"ERROR: runs_dir is not a directory: {runs_dir}", file=sys.stderr)
        return 2

    if args.preset == "hpc_masters":
        suite_paths = [CONFIGS_DIR / name for name in PRESET_HPC_MASTERS]
    elif args.suite_configs:
        suite_paths = []
        for s in args.suite_configs:
            p = Path(s)
            if not p.is_absolute():
                p = REPO_ROOT / p
            suite_paths.append(p)
    else:
        parser.error("Provide suite JSON paths or --preset hpc_masters")

    print(f"runs_dir={runs_path}")
    print(f"Suites to check: {len(suite_paths)}")
    print("Per run: trials_good = distinct trials with a non-<error output; trials_rows = distinct trials in DB.")
    print("-" * 72)

    sys.path.insert(0, str(REPO_ROOT / "src"))

    errors = 0
    for suite_path in suite_paths:
        label = suite_path.name
        if not suite_path.is_file():
            print(f"{label}\n  ERROR: file not found: {suite_path}")
            errors += 1
            continue
        try:
            rows, resume, expected_cells, sha12 = _probe_suite(suite_path, str(runs_path))
        except Exception as e:
            print(f"{label}\n  ERROR: {e}")
            errors += 1
            continue

        print(f"{label}")
        print(f"  suite_sha256_prefix={sha12}…  expected_trial_cells={expected_cells}")
        if not rows:
            print("  No matching run folders under runs_dir (suite SHA / identity).")
            print("  --resume-auto: would start a fresh UUID.")
        else:
            print(f"  Matching run folders: {len(rows)}")
            for r in rows:
                status = "COMPLETE" if r["complete"] else "incomplete"
                mark = "  <-- resume-auto picks this" if resume and r["run_id"] == resume["run_id"] else ""
                print(
                    f"    folder={r['folder']}\n"
                    f"      run_id={r['run_id']}\n"
                    f"      trials_good_output={r['trials_good_output']} / expected={r['expected_cells']}  "
                    f"trials_rows={r['trials_rows']}  trials_error_output={r['trials_error_output']}  [{status}]{mark}"
                )
            if resume:
                print(
                    f"  --resume-auto: continue run_id={resume['run_id']} "
                    f"(folder={resume['folder']}, trials_good_output={resume['trials_good_output']})"
                )
            else:
                print("  --resume-auto: no incomplete match → would start a fresh UUID.")
        print()

    print("Done (read-only; no DB writes, no model inference).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
