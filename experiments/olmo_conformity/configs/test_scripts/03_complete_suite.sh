#!/usr/bin/env bash
# Complete missing trials/outputs per suite_expanded_temp{T}.json for runs in metadata.
# Creates missing trials (when HPC timed out before initializing them) and runs inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs_latest/runs}"
METADATA="${METADATA:-${REPO_ROOT}/Comparing_Experiments/runs_metadata_v5.json}"
# Default: Ollama. Set API_BASE= to use local HuggingFace models.
API_BASE="${API_BASE:-http://localhost:11434/v1}"

echo "Runs dir: ${RUNS_DIR}"
echo "Metadata: ${METADATA}"
if [[ -n "${API_BASE}" ]]; then
  echo "API base: ${API_BASE} (Ollama)"
else
  echo "API base: (local HuggingFace)"
fi
echo ""

EXTRA=()
[[ -n "${API_BASE}" ]] && EXTRA+=(--api-base "${API_BASE}")

exec python3 -m vivarium olmo-conformity-complete-suite \
  --runs-dir "${RUNS_DIR}" \
  --metadata "${METADATA}" \
  "${EXTRA[@]}" \
  "$@"
