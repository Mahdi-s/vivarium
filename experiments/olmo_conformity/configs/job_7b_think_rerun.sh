#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=l40s|a100|a40|v100
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_7B_THINK_RERUN

# ---------------------------------------------------------------------------
# OLMo-3 7B Think Variants — RERUN with Extended Tokens (2048)
#
# Fixes truncation: original 256-token limit truncated 95-99% of outputs.
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_7b_think_rerun.sh
#   TEMPERATURE=0.6 sbatch job_7b_think_rerun.sh
#
# Defaults: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE="suite_7b_think_rerun_temp0p6.json"
else
    SUITE="suite_7b_think_rerun_temp0p0.json"
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== OLMo 7B Think RERUN: suite=${SUITE}, temperature=${TEMPERATURE}, max_new_tokens=2048 ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --temps "${TEMPERATURE}" \
    --hpc --runs-only
