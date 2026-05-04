#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=AAM_7B_DPO_NGRAM

# ---------------------------------------------------------------------------
# OLMo-3 7B Instruct-DPO — n-gram ablation (3 conditions x 8 datasets x 50 items).
# Conditions: (i) naked Asch, (ii) original-wording n-gram, (iii) matched-instruction n-gram.
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_olmo_7b_instruct_dpo_ablations.sh
#   TEMPERATURE=0.6 sbatch job_olmo_7b_instruct_dpo_ablations.sh
# Defaults: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE="suite_olmo_7b_instruct_dpo_ablations_temp0p6.json"
else
    SUITE="suite_olmo_7b_instruct_dpo_ablations_temp0p0.json"
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

HF_SCRATCH="/scratch1/mahdisae/olmo_experiments"
export HF_HOME="${HF_SCRATCH}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_SCRATCH}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_SCRATCH}/models/huggingface_cache"
export HF_DATASETS_CACHE="${HF_SCRATCH}/hf_datasets_cache"
export TORCH_HOME="${HF_SCRATCH}/torch_home"
export XDG_CACHE_HOME="${HF_SCRATCH}/xdg_cache"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
         "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "=== OLMo 7B Instruct-DPO n-gram ablation: suite=${SUITE}, temperature=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --hpc --runs-only --force-rerun --resume-auto

echo "=== OLMo 7B Instruct-DPO n-gram ablation (T=${TEMPERATURE}) complete ==="
