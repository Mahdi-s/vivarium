#!/bin/bash
# Shared environment for the belief-probe HPC jobs. Sourced by job_belief_*.sh.
# Conventions follow the existing job_*.sh scripts: repo in home (small quota), everything else on scratch.

REPO_ROOT="${REPO_ROOT:-/home1/mahdisae/aam/abstractAgentMachine}"
VENV="${VENV:-/scratch1/mahdisae/aam_venv}"
SCRATCH="${SCRATCH:-/scratch1/mahdisae/olmo_experiments}"

# Experiment tag = one folder per launch batch on scratch. Set TAG=... to group jobs; default = today's date.
TAG="${TAG:-$(date +%Y%m%d)_belief}"
export AAM_BELIEF_DIR="${SCRATCH}/belief_probe/${TAG}"
mkdir -p "${AAM_BELIEF_DIR}/logs"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SCRATCH}/mpl_cache/${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

# All HF / torch caches on scratch (home quota is 100 GB)
export HF_HOME="${SCRATCH}/hf_home"
export HUGGINGFACE_HUB_CACHE="${SCRATCH}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/models/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets_cache"
export TORCH_HOME="${SCRATCH}/torch_home"
export XDG_CACHE_HOME="${SCRATCH}/xdg_cache"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"
export TOKENIZERS_PARALLELISM=false

# variant -> HF model id (OLMo-3 ladders)
model_for_variant() {
  case "$1" in
    base)           echo "allenai/Olmo-3-1025-7B" ;;
    instruct_sft)   echo "allenai/Olmo-3-7B-Instruct-SFT" ;;
    instruct_dpo)   echo "allenai/Olmo-3-7B-Instruct-DPO" ;;
    instruct)       echo "allenai/Olmo-3-7B-Instruct" ;;
    rl_zero)        echo "allenai/Olmo-3-7B-RL-Zero-Math" ;;
    think_sft)      echo "allenai/Olmo-3-7B-Think-SFT" ;;
    think_dpo)      echo "allenai/Olmo-3-7B-Think-DPO" ;;
    think)          echo "allenai/Olmo-3-7B-Think" ;;
    base_32b)       echo "allenai/Olmo-3-1125-32B" ;;
    instruct_32b)   echo "allenai/Olmo-3.1-32B-Instruct" ;;
    think_sft_32b)  echo "allenai/Olmo-3-32B-Think-SFT" ;;
    think_dpo_32b)  echo "allenai/Olmo-3-32B-Think-DPO" ;;
    think_32b)      echo "allenai/Olmo-3-32B-Think" ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

run_probe() {  # run_probe VARIANT [extra args...]
  local variant="$1"; shift
  local model; model="$(model_for_variant "${variant}")" || return 1
  echo "=== belief_probe: variant=${variant} model=${model} tag=${TAG} out=${AAM_BELIEF_DIR} ($(date)) ==="
  python investigation/backstudy/tools/belief_probe.py \
      --model-id "${model}" --variant "${variant}" \
      --items-per-dataset "${ITEMS:-0}" --conditions "${CONDITIONS:-all}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-128}" --dtype "${DTYPE:-bf16}" --seed "${SEED:-42}" \
      --out-dir "${AAM_BELIEF_DIR}" --allow-download "$@" \
      2>&1 | tee -a "${AAM_BELIEF_DIR}/logs/${variant}.log"
}
