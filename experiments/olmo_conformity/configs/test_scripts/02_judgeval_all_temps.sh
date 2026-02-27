#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/runs_manifest.tsv}"

# Judge config (Ollama)
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:20b}"
OLLAMA_BASE="${OLLAMA_BASE:-http://localhost:11434/v1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
# Set FORCE_JUDGEVAL=1 to overwrite existing parsed_answer_json (e.g. after judge schema changes)
FORCE_JUDGEVAL="${FORCE_JUDGEVAL:-0}"

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
  run_id="${run_id##*_}"   # last segment after final _ (e.g. UUID)
  log_file="${SCRIPT_DIR}/logs/judgeval_${run_id}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "temp=${temp:-n/a} run_id=${run_id}"
  EXTRA_ARGS=()
  [[ "${FORCE_JUDGEVAL}" == "1" ]] && EXTRA_ARGS+=(--force)
  "${PYTHON_BIN}" -m aam olmo-conformity-judgeval \
    --run-id "${run_id}" \
    --db "${db_path}" \
    --judge-model "${JUDGE_MODEL}" \
    --ollama-base "${OLLAMA_BASE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${log_file}"
done < "${MANIFEST}"

