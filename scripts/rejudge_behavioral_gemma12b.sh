#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_PATH="${REPO_ROOT}/runs_latest/runs"
METADATA="${REPO_ROOT}/Comparing_Experiments/runs_metadata_v6.json"
JUDGE_CONFIG="${REPO_ROOT}/experiments/olmo_conformity/configs/judge_config_local.json"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

if [[ ! -f "${METADATA}" ]]; then
  echo "Missing metadata: ${METADATA}" >&2
  exit 1
fi

if [[ ! -f "${JUDGE_CONFIG}" ]]; then
  echo "Missing judge config: ${JUDGE_CONFIG}" >&2
  exit 1
fi

echo "Using metadata: ${METADATA}"
echo "Using judge config: ${JUDGE_CONFIG}"
echo "Behavioral scope: control, asch_history_5, authoritative_bias"
echo "Max concurrency: ${MAX_CONCURRENCY}"
echo

while IFS=$'\t' read -r run_id run_dir status; do
  [[ "${status}" == "completed" ]] || continue
  db="${RUNS_PATH}/${run_dir}/simulation.db"
  if [[ ! -f "${db}" ]]; then
    echo "Skip missing DB: ${db}"
    continue
  fi

  echo "=== Re-judging run ${run_id} (${run_dir}) ==="
  "${PYTHON_BIN}" -m vivarium olmo-conformity-judgeval \
    --run-id "${run_id}" \
    --db "${db}" \
    --judge-config "${JUDGE_CONFIG}" \
    --trial-scope behavioral-only \
    --force \
    --max-concurrency "${MAX_CONCURRENCY}"
  echo
done < <(
  "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("Comparing_Experiments/runs_metadata_v6.json").read_text())
for _, info in sorted(meta.get("experiments", {}).items(), key=lambda kv: float(kv[0])):
    print(f"{info.get('run_id','')}\t{info.get('run_dir','')}\t{info.get('status','')}")
PY
)

echo "=== Behavioral re-judge completed ==="
