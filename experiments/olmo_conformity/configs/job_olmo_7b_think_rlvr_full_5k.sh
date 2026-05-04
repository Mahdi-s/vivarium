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
#SBATCH --job-name=AAM_7B_TR_FULL5K

# ---------------------------------------------------------------------------
# OLMo-3 7B Think (RLVR final) — comprehensive re-run at max_new_tokens=5000.
#
# Why this exists: the existing Think-RLVR DBs at runs/think/20260325_010440
# (T=0) and runs/think/20260414_092714 (T=0.6) used max_new_tokens=2048,
# which truncated 30-71% of trials mid-`<think>` block depending on
# temperature and condition. This re-run uses 5000 tokens — matching the
# budget already used for Think-SFT and Think-DPO in runs-think-hpc/ — so
# every trial has room to close its `<think>` block before the answer slot.
#
# Coverage per run: 15 conditions on the standard 400-item pool.
#   - 12 canonical conditions (control, asch_history_5, all unanimous tones,
#     diverse_peers, devil's advocate, question distillation, authoritative_bias,
#     authority_trust, authority_trust_da)
#   - 3 ablation conditions (naked Asch, original n-gram, matched-instruction
#     n-gram) — the structural-confound test the paper relies on
#
# This run subsumes:
#   - runs/think/20260325_010440_*  (T=0,   2048 tokens)  ← truncation-broken
#   - runs/think/20260414_092714_*  (T=0.6, 2048 tokens)  ← truncation-broken
#   - the Panel B extension Think-RLVR jobs:
#       job_olmo_7b_think_rlvr_panelB.sh
#     ← do NOT submit those if you submit this one; they cover a strict subset
#       of the 15 conditions handled here.
#
# Resource budget per the user's HPC limits:
#   --gpus-per-task=2          (max allowed)
#   --mem=200G                 (cap)
#   --time=48:00:00            (cap)
#   --constraint=a100|a40      (request whatever GPU class is available)
#
# Usage:
#   TEMPERATURE=0.0 sbatch job_olmo_7b_think_rlvr_full_5k.sh
#   TEMPERATURE=0.6 sbatch job_olmo_7b_think_rlvr_full_5k.sh
# Default: TEMPERATURE=0.0
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
if [ "${TEMPERATURE}" = "0.6" ]; then
    SUITE="suite_olmo_7b_think_rlvr_full_5k_temp0p6.json"
else
    SUITE="suite_olmo_7b_think_rlvr_full_5k_temp0p0.json"
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

echo "=== 7B Think (RLVR) full re-run @ 5000 tokens: suite=${SUITE}, T=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
    --suite "experiments/olmo_conformity/configs/${SUITE}" \
    --hpc --runs-only --force-rerun --resume-auto

echo "=== 7B Think (RLVR) full re-run @ 5000 tokens (T=${TEMPERATURE}) complete ==="
