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
#SBATCH --job-name=AAM_32B_DPO

# ---------------------------------------------------------------------------
# OLMo-3 32B Think-DPO — Run T=0.0 then T=0.6 sequentially (tensor-parallel on 2 GPUs).
#
# Usage:
#   sbatch job_32b_think_dpo.sh
# ---------------------------------------------------------------------------

set -euo pipefail

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

# --- Redirect all caches to scratch to avoid blowing the 100GB home quota ---
HF_SCRATCH="/scratch1/mahdisae/olmo_experiments"
export HF_HOME="${HF_SCRATCH}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_SCRATCH}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_SCRATCH}/models/huggingface_cache"
export HF_DATASETS_CACHE="${HF_SCRATCH}/hf_datasets_cache"
export TORCH_HOME="${HF_SCRATCH}/torch_home"
export XDG_CACHE_HOME="${HF_SCRATCH}/xdg_cache"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
         "${TORCH_HOME}" "${XDG_CACHE_HOME}"

for SUITE_TEMP in "suite_32b_think_dpo_temp0p0.json:0.0" "suite_32b_think_dpo_temp0p6.json:0.6"; do
    SUITE="${SUITE_TEMP%%:*}"
    TEMP="${SUITE_TEMP##*:}"
    echo "=== OLMo 32B Think-DPO: suite=${SUITE}, temperature=${TEMP} ==="
    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${SUITE}" \
        --hpc --runs-only --force-rerun
done

echo "=== OLMo 32B Think-DPO (both temperatures) complete ==="
