#!/usr/bin/env bash
# Run trials using the minimal suite (model-first, 1 item per dataset per condition).
# Uses run_expanded_experiments.py; metadata goes to Comparing_Experiments/runs_metadata.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SUITE="${SUITE:-${SCRIPT_DIR}/suite_expanded_localtest_by_model.json}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"

echo "Suite: ${SUITE}"
echo "Runs dir: ${RUNS_DIR}"

"${PYTHON_BIN}" "${CONFIGS_DIR}/run_expanded_experiments.py" \
  --suite "${SUITE}" \
  --runs-dir "${RUNS_DIR}" \
  --phase trials \
  "$@"
