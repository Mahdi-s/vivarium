#!/bin/bash
# ---------------------------------------------------------------------------
# OLMo-3 7B Think Variants — Local Rerun with Extended Tokens
#
# Reruns think variants with max_new_tokens=2048 (was 256) to fix truncation.
# Runs locally (no SLURM/HPC) with --no-sleep to keep Mac awake.
#
# Usage:
#   # Run both temperatures sequentially (default)
#   bash run_7b_think_rerun_local.sh
#
#   # Run a specific temperature
#   TEMPERATURE=0.0 bash run_7b_think_rerun_local.sh
#   TEMPERATURE=0.6 bash run_7b_think_rerun_local.sh
#
# Environment:
#   RUNS_DIR  — override output directory (default: repo_root/runs)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

TEMPERATURE="${TEMPERATURE:-all}"
RUNS_DIR="${RUNS_DIR:-}"

RUNS_ARG=""
if [ -n "${RUNS_DIR}" ]; then
    RUNS_ARG="--runs-dir ${RUNS_DIR}"
fi

run_suite() {
    local temp="$1"
    local suite

    if [ "${temp}" = "0.6" ]; then
        suite="suite_7b_think_rerun_temp0p6.json"
    else
        suite="suite_7b_think_rerun_temp0p0.json"
    fi

    echo ""
    echo "================================================================"
    echo "  OLMo 7B Think RERUN — T=${temp} (max_new_tokens=2048)"
    echo "  Suite: ${suite}"
    echo "  Started: $(date)"
    echo "================================================================"
    echo ""

    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${suite}" \
        --temps "${temp}" \
        --runs-only \
        --no-sleep \
        ${RUNS_ARG}
}

if [ "${TEMPERATURE}" = "all" ]; then
    run_suite "0.0"
    run_suite "0.6"
else
    run_suite "${TEMPERATURE}"
fi

echo ""
echo "================================================================"
echo "  All 7B Think reruns complete — $(date)"
echo "================================================================"
