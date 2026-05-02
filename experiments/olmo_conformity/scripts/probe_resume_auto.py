#!/usr/bin/env python3
"""
Read-only probe: show which run --resume-auto would attach to for each suite.

Uses the same discovery logic as `vivarium olmo-conformity --resume-auto`
(`_find_incomplete_run_for_suite` in runner.py). Does not run trials or
modify databases.

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


def _probe_one(
    suite_config_path: Path,
    runs_dir: str,
) -> Tuple[Optional[str], Optional[str], int, str]:
    """
    Returns (run_id, run_path_or_none, expected_cells, sha256_prefix).
    """
    from vivarium.experiments.olmo_conformity.io import load_suite_config, sha256_file
    from vivarium.experiments.olmo_conformity.runner import _find_incomplete_run_for_suite

    cfg = load_suite_config(str(suite_config_path))
    suite_path = suite_config_path.resolve()
    sha = sha256_file(str(suite_path))
    model_ids = [str(m.get("model_id") or "mock") for m in cfg.get("models", [])]
    condition_names = [str(c.get("name") or "") for c in cfg.get("conditions", [])]
    n_items = _expected_item_count(suite_path, cfg)
    expected_cells = n_items * len(condition_names) * len(model_ids)

    result = _find_incomplete_run_for_suite(
        runs_dir=runs_dir,
        suite_config_sha256=sha,
        current_cfg=cfg,
        model_ids=model_ids,
        condition_names=condition_names,
        expected_item_count=n_items,
    )
    rid, rpath = (None, None) if result is None else (result[0], result[1])
    return rid, rpath, expected_cells, sha[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run probe for --resume-auto matching (no trials executed).")
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
            rid, rdir, expected_cells, sha12 = _probe_one(suite_path, str(runs_path))
        except Exception as e:
            print(f"{label}\n  ERROR: {e}")
            errors += 1
            continue
        if rid and rdir:
            print(f"{label}")
            print(f"  suite_sha256_prefix={sha12}…  expected_trial_cells={expected_cells}")
            print(f"  RESUME -> run_id={rid}")
            print(f"            run_dir={rdir}")
        else:
            print(f"{label}")
            print(f"  suite_sha256_prefix={sha12}…  expected_trial_cells={expected_cells}")
            print(f"  NO incomplete matching run (would start a fresh UUID)")
        print()

    print("Done (read-only; no DB writes, no model inference).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
