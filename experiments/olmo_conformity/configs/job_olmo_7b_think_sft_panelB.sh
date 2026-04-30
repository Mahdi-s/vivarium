#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=a100|a40
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_7B_TS_PB

# ---------------------------------------------------------------------------
# OLMo-3 7B Think-SFT — Panel B extension: 8 conditions missing from the
# original Think suite, on the same 400-item set as Panel A.
#
# Conditions (from suite_olmo_7b_think_sft_panelB_extension_temp{0p0,0p6}.json):
#   asch_history_5
#   asch_zhu_unbiased_diverse_plain
#   asch_zhu_unbiased_qd
#   asch_zhu_unbiased_da
#   asch_zhu_unbiased_unanimous_plain
#   asch_zhu_unbiased_unanimous_neutral
#   asch_zhu_unbiased_unanimous_uncertain
#   authority_zhu_unbiased_trust_da
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_olmo_7b_think_sft_panelB.sh
#   TEMPERATURE=0.6 sbatch job_olmo_7b_think_sft_panelB.sh
# Defaults: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE="suite_olmo_7b_think_sft_panelB_extension_temp0p6.json"
else
    SUITE="suite_olmo_7b_think_sft_panelB_extension_temp0p0.json"
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
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

echo "=== 7B Think-SFT Panel B extension: suite=${SUITE}, temperature=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --hpc --runs-only --force-rerun

echo "=== 7B Think-SFT Panel B extension (T=${TEMPERATURE}) complete ==="
