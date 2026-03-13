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
#SBATCH --job-name=AAM_7B_RUN

# ---------------------------------------------------------------------------
# OLMo-3 7B Expanded Conformity — Run Trials
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_7b_run.sh
#   TEMPERATURE=0.6 sbatch job_7b_run.sh
#   TEMPERATURE=0.0 SUITE=suite_7b_expanded.json sbatch job_7b_run.sh
#
# Defaults: TEMPERATURE=0.0, SUITE=suite_7b_expanded.json
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
SUITE="${SUITE:-suite_7b_expanded.json}"

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== OLMo 7B Run: suite=${SUITE}, temperature=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --temps "${TEMPERATURE}" \
    --hpc --runs-only
