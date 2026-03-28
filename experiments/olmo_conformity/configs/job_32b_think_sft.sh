#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=a100
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_32B_SFT

# ---------------------------------------------------------------------------
# OLMo-3 32B Think-SFT — Run T=0.0 then T=0.6 sequentially (tensor-parallel on 2 GPUs).
#
# Usage:
#   sbatch job_32b_think_sft.sh
# ---------------------------------------------------------------------------

set -euo pipefail

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

for SUITE_TEMP in "suite_32b_think_sft_temp0p0.json:0.0" "suite_32b_think_sft_temp0p6.json:0.6"; do
    SUITE="${SUITE_TEMP%%:*}"
    TEMP="${SUITE_TEMP##*:}"
    echo "=== OLMo 32B Think-SFT: suite=${SUITE}, temperature=${TEMP} ==="
    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${SUITE}" \
        --hpc --runs-only
done

echo "=== OLMo 32B Think-SFT (both temperatures) complete ==="
