#!/usr/bin/env bash
# Backfill missing behavioral outputs for runs under runs_latest/runs.
# Runs inference only for trials that have no conformity_outputs row.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs_latest/runs}"
# Default: Ollama. Set API_BASE= to use local HuggingFace models.
API_BASE="${API_BASE:-http://localhost:11434/v1}"

echo "Runs dir: ${RUNS_DIR}"
if [[ -n "${API_BASE}" ]]; then
  echo "API base: ${API_BASE} (Ollama)"
else
  echo "API base: (local HuggingFace)"
fi
echo ""

EXTRA=()
[[ -n "${API_BASE}" ]] && EXTRA+=(--api-base "${API_BASE}")

exec python3 -m vivarium olmo-conformity-backfill-behavioral \
  --runs-dir "${RUNS_DIR}" \
  "${EXTRA[@]}" \
  "$@"
