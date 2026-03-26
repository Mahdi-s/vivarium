#!/usr/bin/env bash
set -euo pipefail

# Re-exec under caffeinate to prevent Mac from sleeping during long judge runs
if [ -z "${CAFFEINATE_CHILD:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export CAFFEINATE_CHILD=1
  exec caffeinate -i "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="${REPO_ROOT}/runs_latest/runs"
METADATA="${REPO_ROOT}/Comparing_Experiments/runs_metadata_v6.json"
JUDGE_CONFIG="${REPO_ROOT}/experiments/olmo_conformity/configs/judge_config_local.json"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

# Parse --all-runs: when set, discover every run dir under RUNS_DIR instead of using metadata.
USE_ALL_RUNS=0
for arg in "$@"; do
  if [ "${arg}" = "--all-runs" ]; then
    USE_ALL_RUNS=1
    break
  fi
done

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

if [ ! -f "${JUDGE_CONFIG}" ]; then
  echo "Missing judge config: ${JUDGE_CONFIG}" >&2
  exit 1
fi

if [ "${USE_ALL_RUNS}" = 0 ] && [ ! -f "${METADATA}" ]; then
  echo "Missing metadata: ${METADATA}" >&2
  exit 1
fi

if [ "${USE_ALL_RUNS}" = 1 ]; then
  echo "Using runs dir: ${RUNS_DIR} (all runs)"
else
  echo "Using metadata: ${METADATA}"
fi
echo "Using judge config: ${JUDGE_CONFIG}"
echo "Judge scope: missing + parse_error only (all 12 conditions; no --force)"
echo "Max concurrency: ${MAX_CONCURRENCY}"
echo

RUN_LIST="$(mktemp)"
trap 'rm -f "${RUN_LIST}"' EXIT

if [ "${USE_ALL_RUNS}" = 1 ]; then
  # Discover all run dirs: YYYYMMDD_HHMMSS_<uuid> with simulation.db; run_id = uuid part.
  "${PYTHON_BIN}" - "${RUNS_DIR}" <<'PY' > "${RUN_LIST}"
import re
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
pattern = re.compile(r"^\d{8}_\d{6}_[0-9a-f\-]{36}$")
for entry in sorted(runs_dir.iterdir()):
    if not entry.is_dir() or not pattern.match(entry.name):
        continue
    db = entry / "simulation.db"
    if not db.exists():
        continue
    run_id = entry.name.split("_", 2)[-1]
    print(f"{run_id}\t{entry.name}\tcompleted")
PY
else
  (cd "${REPO_ROOT}" && "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("Comparing_Experiments/runs_metadata_v6.json").read_text())
for _, info in sorted(meta.get("experiments", {}).items(), key=lambda kv: float(kv[0])):
    print(f"{info.get('run_id','')}\t{info.get('run_dir','')}\t{info.get('status','')}")
PY
) > "${RUN_LIST}"
fi

while IFS="$(printf '\t')" read -r run_id run_dir status; do
  [ "${status}" = "completed" ] || continue
  db="${RUNS_DIR}/${run_dir}/simulation.db"
  if [ ! -f "${db}" ]; then
    echo "Skip missing DB: ${db}"
    continue
  fi

  echo "=== Judging run ${run_id} (${run_dir}) ==="
  "${PYTHON_BIN}" -m vivarium olmo-conformity-judgeval \
    --run-id "${run_id}" \
    --db "${db}" \
    --judge-config "${JUDGE_CONFIG}" \
    --trial-scope all \
    --retry-parse-errors \
    --max-concurrency "${MAX_CONCURRENCY}"
  echo
done < "${RUN_LIST}"

echo "=== Judge (missing + parse_error only) completed ==="
