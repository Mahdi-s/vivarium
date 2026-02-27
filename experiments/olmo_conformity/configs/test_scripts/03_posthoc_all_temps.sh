#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/runs_manifest.tsv}"

LAYERS="${LAYERS:-0,1,2,3,4,5,6,7}"
LOGIT_LENS_K="${LOGIT_LENS_K:-10}"
TRIAL_SCOPE="${TRIAL_SCOPE:-behavioral-only}"

# Toggles
PARSE_THINK="${PARSE_THINK:-1}"
DO_LOGIT_LENS="${DO_LOGIT_LENS:-0}"
DO_ANSWER_LOGPROBS="${DO_ANSWER_LOGPROBS:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}"
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs"

while IFS=$'\t' read -r col1 col2 col3; do
  # Support both 3-column (temp, run_dir, db_path) and 2-column (run_dir, db_path) manifest
  if [[ -n "${col3}" ]]; then
    temp="${col1}"
    run_dir="${col2}"
    db_path="${col3}"
  else
    temp=""
    run_dir="${col1}"
    db_path="${col2}"
  fi
  [[ -z "${run_dir}" ]] && continue
  run_id="${run_dir##*/}"
  run_id="${run_id##*_}"
  log_file="${SCRIPT_DIR}/logs/posthoc_${run_id}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "temp=${temp:-n/a} run_dir=${run_dir}"

  cmd=( "${PYTHON_BIN}" -m aam olmo-conformity-posthoc
        --run-dir "${run_dir}"
        --db "${db_path}"
        --layers "${LAYERS}"
        --logit-lens-k "${LOGIT_LENS_K}"
        --trial-scope "${TRIAL_SCOPE}"
        --no-interventions
        --no-report )

  if [[ "${PARSE_THINK}" == "1" ]]; then
    cmd+=( --parse-think-tokens )
  fi
  if [[ "${DO_LOGIT_LENS}" == "1" ]]; then
    :
  else
    cmd+=( --no-logit-lens )
  fi
  if [[ "${DO_ANSWER_LOGPROBS}" == "1" ]]; then
    :
  else
    cmd+=( --no-answer-logprobs )
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"
done < "${MANIFEST}"
