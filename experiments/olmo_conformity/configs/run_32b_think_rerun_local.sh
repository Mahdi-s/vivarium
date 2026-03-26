#!/bin/bash
# ---------------------------------------------------------------------------
# OLMo-3 32B Think Variants — Local Rerun with Extended Tokens
#
# Reruns think variants with max_new_tokens=4096 (was 512) to fix truncation.
# Runs locally (no SLURM/HPC) with --no-sleep to keep Mac awake.
#
# NOTE: 32B models require ~64GB GPU VRAM in fp16. Ensure you have
# sufficient GPU memory (e.g., 2x A100 80GB, or 4x A40 48GB).
#
# Usage:
#   # Run both temperatures sequentially (default)
#   bash run_32b_think_rerun_local.sh
#
#   # Run a specific temperature
#   TEMPERATURE=0.0 bash run_32b_think_rerun_local.sh
#   TEMPERATURE=0.6 bash run_32b_think_rerun_local.sh
#
# Environment:
#   RUNS_DIR  — override output directory (default: repo_root/runs)
#   CUDA_VISIBLE_DEVICES — GPU selection (default: 0,1,2,3)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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
        suite="suite_32b_think_rerun_temp0p6.json"
    else
        suite="suite_32b_think_rerun_temp0p0.json"
    fi

    echo ""
    echo "================================================================"
    echo "  OLMo 32B Think RERUN — T=${temp} (max_new_tokens=4096)"
    echo "  Suite: ${suite}"
    echo "  GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  Started: $(date)"
    echo "================================================================"
    echo ""

    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${suite}" \
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
echo "  All 32B Think reruns complete — $(date)"
echo "================================================================"
