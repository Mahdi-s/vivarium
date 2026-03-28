#!/usr/bin/env bash
# Matrix report for runs under runs_latest/runs: conditions, temperatures, completion,
# and judge label coverage — focused on base / instruct / instruct_sft / instruct_dpo
# (same judge_valid rules as judge_report.sh).
#
# Usage:
#   ./runs_latest_report.sh
#   ./runs_latest_report.sh [runs_dir]    # optional; default runs_latest/runs under repo root
#   ./runs_latest_report.sh --aggregate
#   ./runs_latest_report.sh --all-variants
#   ./runs_latest_report.sh --variants base,instruct
#   ./runs_latest_report.sh --by-condition <folder_id> [--variant instruct] [--temperature 0.6]
#   ./runs_latest_report.sh --runs-dir /path/to/runs_parent   # must contain */simulation.db

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/scripts/runs_latest_report.py"

if [[ ! -f "$PY" ]]; then
  echo "Error: runs_latest_report.py not found at ${PY}"
  exit 1
fi

RUNS_DIR="${REPO_ROOT}/runs_latest/runs"
if [[ $# -ge 1 && "${1}" != --* ]]; then
  raw="$1"
  shift
  if [[ "$raw" = /* ]]; then
    RUNS_DIR="$raw"
  else
    RUNS_DIR="${REPO_ROOT}/${raw}"
  fi
fi

exec python3 "$PY" --runs-dir "$RUNS_DIR" "$@"
