#!/bin/bash
# ---------------------------------------------------------------------------
# Master HPC launcher for the Figure 2 Panel B extension (Tier A++ in the plan;
# see .claude/plans/below-is-feedback-from-golden-sphinx.md).
#
# Submits 12 SLURM jobs: 6 OLMo-3 Think variants × 2 temperatures.
# Each job covers the same 8 conditions on the same 400-item set:
#   asch_history_5
#   asch_zhu_unbiased_diverse_plain
#   asch_zhu_unbiased_qd
#   asch_zhu_unbiased_da
#   asch_zhu_unbiased_unanimous_plain
#   asch_zhu_unbiased_unanimous_neutral
#   asch_zhu_unbiased_unanimous_uncertain
#   authority_zhu_unbiased_trust_da
#
# These are the 8 conditions Panel A of Figure 2 (Instruct family) covers
# but Panel B (Think family) currently does not. Once these runs return,
# Panel B becomes a row-for-row analog of Panel A.
#
# All jobs request 2 GPUs and 200GB of memory, matching the existing
# 32B Think SBATCH templates.
#
# Pre-flight checklist:
#   - On HPC login node, in the abstractAgentMachine repo root.
#   - aam_venv exists at /scratch1/mahdisae/aam_venv.
#   - HF model weights for all six Think variants are cached or reachable.
#
# Usage:
#   bash experiments/olmo_conformity/scripts/run_panel_b_extension_hpc.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

JOB_DIR="experiments/olmo_conformity/configs"

THINK_JOBS=(
    "job_olmo_7b_think_sft_panelB.sh"
    "job_olmo_7b_think_dpo_panelB.sh"
    "job_olmo_7b_think_rlvr_panelB.sh"
    "job_olmo_32b_think_sft_panelB.sh"
    "job_olmo_32b_think_dpo_panelB.sh"
    "job_olmo_32b_think_rlvr_panelB.sh"
)

declare -a SUBMITTED=()

for job in "${THINK_JOBS[@]}"; do
    for temp in 0.0 0.6; do
        echo "Submitting ${job} at TEMPERATURE=${temp}..."
        SLURM_ID=$(TEMPERATURE="${temp}" sbatch --parsable "${JOB_DIR}/${job}")
        echo "  -> ${SLURM_ID}"
        SUBMITTED+=("${SLURM_ID} ${job} T=${temp}")
    done
done

echo
echo "=== Submitted ${#SUBMITTED[@]} jobs ==="
for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
done
echo
echo "Track with: squeue -u \${USER} -o '%.10i %.20j %.8T %.10M %R'"
echo
echo "When all jobs complete, the new run directories under runs/ will populate"
echo "Panel B with the 8 missing conditions, ready for figure regeneration."
