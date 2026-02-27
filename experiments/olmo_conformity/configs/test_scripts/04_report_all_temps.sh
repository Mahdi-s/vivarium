#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/runs_manifest.tsv}"

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
  log_file="${SCRIPT_DIR}/logs/report_${run_id}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "temp=${temp:-n/a} run_id=${run_id}"
  "${PYTHON_BIN}" -m aam olmo-conformity-report \
    --run-id "${run_id}" \
    --db "${db_path}" \
    --run-dir "${run_dir}" \
    2>&1 | tee "${log_file}"
done < "${MANIFEST}"

