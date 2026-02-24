#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_DIR="${RUNS_DIR:-${AAM_ARTIFACTS_DIR:-${AAM_RUNS_DIR:-${REPO_ROOT}/runs}}}"
API_BASE="${VVM_API_BASE:-}"
API_KEY="${VVM_API_KEY:-}"

# If you already have HF models cached under the repo's models folder, prefer that by default.
export AAM_HF_CACHE="${AAM_HF_CACHE:-${REPO_ROOT}/models/huggingface_cache}"
export AAM_MODEL_DIR="${AAM_MODEL_DIR:-${REPO_ROOT}/models}"
export AAM_ARTIFACTS_DIR="${AAM_ARTIFACTS_DIR:-${RUNS_DIR}}"

MANIFEST="${SCRIPT_DIR}/runs_manifest.tsv"
mkdir -p "${SCRIPT_DIR}/logs"

echo "runs_dir=${RUNS_DIR}"
echo "manifest=${MANIFEST}"
if [[ -n "${API_BASE}" ]]; then
  echo "api_base=${API_BASE}"
else
  echo "api_base=(unset; will try local HF inference, which may download large models)"
fi

for suite_cfg in "${SCRIPT_DIR}"/suite_expanded_localtest_temp*.json; do
  echo "============================================================"
  echo "suite_config=${suite_cfg}"

  log_file="${SCRIPT_DIR}/logs/$(basename "${suite_cfg}" .json)_$(date +%Y%m%d_%H%M%S).log"

  cmd=( "${PYTHON_BIN}" -m aam olmo-conformity
        --suite-config "${suite_cfg}"
        --runs-dir "${RUNS_DIR}" )
  if [[ -n "${API_BASE}" ]]; then
    cmd+=( --api-base "${API_BASE}" )
  fi
  if [[ -n "${API_KEY}" ]]; then
    cmd+=( --api-key "${API_KEY}" )
  fi

  # Stream output to a log file, then parse run_dir/db from the log.
  if ! "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "ERROR: run failed; log=${log_file}"
    exit 1
  fi

  run_dir="$(grep -E '^run_dir=' "${log_file}" | tail -n 1 | cut -d= -f2-)"
  db_path="$(grep -E '^db=' "${log_file}" | tail -n 1 | cut -d= -f2-)"
  temp="$(basename "${suite_cfg}" | sed -n 's/.*temp\\([0-9]\\.[0-9]\\)\\.json/\\1/p')"

  if [[ -z "${run_dir}" || -z "${db_path}" ]]; then
    echo "ERROR: failed to parse run_dir/db from output; log=${log_file}"
    exit 2
  fi

  echo -e "${temp}\t${run_dir}\t${db_path}" >> "${MANIFEST}"
  echo "OK: temp=${temp} run_dir=${run_dir} db=${db_path}"
done

echo "============================================================"
echo "DONE. Manifest written to: ${MANIFEST}"
