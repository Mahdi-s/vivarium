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
#SBATCH --job-name=AAM_32B_SFT_RSM_T0

# ---------------------------------------------------------------------------
# Resume OLMo-3 32B Think-SFT @ temperature 0.0 for a fixed run_id.
#
# Vivarium resolves the run folder as a *direct child* of RUNS_DIR whose name
# ends with _<run-id> (e.g. 20260330_235019_81d9194a-b1ef-4261-a0fb-bb0f713e1239).
#
# paths.json "runs_dir" → /scratch1/mahdisae/olmo_experiments/runs
# If your run directories live there directly, set RUNS_DIR="${RUNS_ROOT}".
# ---------------------------------------------------------------------------

set -euo pipefail

SUITE="suite_32b_think_sft_temp0p0.json"
RUN_ID="81d9194a-b1ef-4261-a0fb-bb0f713e1239"

# From experiments/olmo_conformity/configs/paths.json ("runs_dir")
RUNS_ROOT="/scratch1/mahdisae/olmo_experiments/runs"
RUNS_DIR="${RUNS_ROOT}/runs-think-hpc/runs-32B"

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

echo "=== Resume OLMo 32B Think-SFT T=0.0 | suite=${SUITE} | run-id=${RUN_ID} | runs-dir=${RUNS_DIR} ==="

python -m vivarium olmo-conformity \
  --suite-config "experiments/olmo_conformity/configs/${SUITE}" \
  --runs-dir "${RUNS_DIR}" \
  --run-id "${RUN_ID}"

echo "=== Resume OLMo 32B Think-SFT T=0.0 (run-id=${RUN_ID}) complete ==="
