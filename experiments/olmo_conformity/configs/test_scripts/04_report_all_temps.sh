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

while IFS=$'\t' read -r temp run_dir db_path; do
  [[ -z "${temp}" ]] && continue
  run_id="${run_dir##*_}"
  log_file="${SCRIPT_DIR}/logs/report_temp${temp}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "temp=${temp} run_id=${run_id}"
  "${PYTHON_BIN}" -m aam olmo-conformity-report \
    --run-id "${run_id}" \
    --db "${db_path}" \
    --run-dir "${run_dir}" \
    2>&1 | tee "${log_file}"
done < "${MANIFEST}"

