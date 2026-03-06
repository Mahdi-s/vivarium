#!/usr/bin/env bash
# Run posthoc interpretability analysis on an existing set of trial runs.
# Requires a run-id from a completed 01_run_trials.sh execution.
#
# Usage:
#   RUN_ID=<uuid> bash 02_run_posthoc.sh [extra flags...]
#
# Extra flags are forwarded to run_interpretability_posthoc.py, e.g.:
#   RUN_ID=abc123 bash 02_run_posthoc.sh --skip-activations --skip-interventions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"

if [ -z "${RUN_ID:-}" ]; then
  echo "Usage: RUN_ID=<uuid> bash 02_run_posthoc.sh"
  exit 1
fi

echo "Run ID: ${RUN_ID}"
echo "Runs dir: ${RUNS_DIR}"

"${PYTHON_BIN}" "${CONFIGS_DIR}/run_interpretability_posthoc.py" \
  --run-id "${RUN_ID}" \
  --runs-dir "${RUNS_DIR}" \
  "$@"
