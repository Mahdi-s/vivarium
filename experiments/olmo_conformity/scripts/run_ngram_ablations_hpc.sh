#!/bin/bash
# ---------------------------------------------------------------------------
# Master HPC launcher for the n-gram ablation re-runs (Phase 2 of the response
# to reviewer feedback; see .claude/plans/below-is-feedback-from-golden-sphinx.md).
#
# Submits 9 SLURM jobs:
#   - 8 GPU jobs: 4 OLMo-3 7B Instruct-pipeline checkpoints x {T=0.0, T=0.6}
#   - 1 CPU job:  matched-instruction re-run via OpenRouter on
#                 Llama-3.1-70B-Instruct and OLMo-3.1-32B-Instruct (covers
#                 both T values internally).
#
# All jobs share the same item set (50 items per dataset x 8 datasets = 400)
# and the same three conditions:
#   (i)   asch_zhu_naked_unanimous_confident
#   (ii)  ngram_sequence_baseline (original "based on the provided sequence")
#   (iii) ngram_sequence_matched_baseline ("based on your knowledge")
#
# Pre-flight checklist before running this script:
#   - You're on the HPC login node, in the abstractAgentMachine repo root.
#   - `aam_venv` virtualenv exists at /scratch1/mahdisae/aam_venv.
#   - OPENROUTER_API_KEY is exported in the environment (for the API job).
#   - The HF model weights for the four OLMo 7B variants are cached or at
#     least reachable from /scratch1/mahdisae.
#
# Usage:
#   bash experiments/olmo_conformity/scripts/run_ngram_ablations_hpc.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

JOB_DIR="experiments/olmo_conformity/configs"

OLMO_VARIANT_JOBS=(
    "job_olmo_7b_base_ablations.sh"
    "job_olmo_7b_instruct_sft_ablations.sh"
    "job_olmo_7b_instruct_dpo_ablations.sh"
    "job_olmo_7b_instruct_rlvr_ablations.sh"
)

declare -a SUBMITTED=()

for job in "${OLMO_VARIANT_JOBS[@]}"; do
    for temp in 0.0 0.6; do
        echo "Submitting ${job} at TEMPERATURE=${temp}..."
        SLURM_ID=$(TEMPERATURE="${temp}" sbatch --parsable "${JOB_DIR}/${job}")
        echo "  -> ${SLURM_ID}"
        SUBMITTED+=("${SLURM_ID} ${job} T=${temp}")
    done
done

echo "Submitting API matched-instruction re-runs (CPU, both temperatures)..."
SLURM_ID=$(sbatch --parsable "${JOB_DIR}/job_api_ablations_matched.sh")
echo "  -> ${SLURM_ID}"
SUBMITTED+=("${SLURM_ID} job_api_ablations_matched.sh T=0.0+0.6")

echo
echo "=== Submitted ${#SUBMITTED[@]} jobs ==="
for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
done
echo
echo "Track with: squeue -u \${USER} -o '%.10i %.20j %.8T %.10M %R'"
