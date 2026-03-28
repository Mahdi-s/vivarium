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
#SBATCH --job-name=AAM_7_THINK_SFT_DPO

# ---------------------------------------------------------------------------
# OLMo-3 7B Think-SFT + Think-DPO in parallel (one model per GPU).
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_7b_think_sft_dpo.sh
#   TEMPERATURE=0.6 sbatch job_7b_think_sft_dpo.sh
#
# Defaults: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE_SFT="suite_7b_think_sft_temp0p6.json"
    SUITE_DPO="suite_7b_think_dpo_temp0p6.json"
else
    SUITE_SFT="suite_7b_think_sft_temp0p0.json"
    SUITE_DPO="suite_7b_think_dpo_temp0p0.json"
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== OLMo 7B Think-SFT + Think-DPO (parallel): temperature=${TEMPERATURE} ==="
echo "    SFT suite=${SUITE_SFT} on CUDA device 0"
echo "    DPO suite=${SUITE_DPO} on CUDA device 1"

_run_sft() {
    export CUDA_VISIBLE_DEVICES=0
    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${SUITE_SFT}" \
        --temps "${TEMPERATURE}" \
        --hpc --runs-only
}

_run_dpo() {
    export CUDA_VISIBLE_DEVICES=1
    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${SUITE_DPO}" \
        --temps "${TEMPERATURE}" \
        --hpc --runs-only
}

_run_sft &
PID_SFT=$!
_run_dpo &
PID_DPO=$!

set +e
wait "${PID_SFT}"
RC_SFT=$?
wait "${PID_DPO}"
RC_DPO=$?
set -e

if [ "${RC_SFT}" -ne 0 ] || [ "${RC_DPO}" -ne 0 ]; then
    echo "One or both runs failed: Think-SFT exit=${RC_SFT}, Think-DPO exit=${RC_DPO}" >&2
    exit 1
fi

echo "=== Both 7B runs finished successfully ==="
