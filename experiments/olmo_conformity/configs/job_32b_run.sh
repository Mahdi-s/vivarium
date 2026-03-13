#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=a100|a40
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_32B_RUN

# ---------------------------------------------------------------------------
# OLMo-3 32B Think — Run Trials
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_32b_run.sh
#   TEMPERATURE=0.6 sbatch job_32b_run.sh
#
# Defaults: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
# Pick the right suite config based on temperature
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE="suite_32b_think_temp0p6.json"
else
    SUITE="suite_32b_think_temp0p0.json"
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== OLMo 32B Think Run: suite=${SUITE}, temperature=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --hpc --runs-only
