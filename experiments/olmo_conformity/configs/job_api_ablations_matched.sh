#!/bin/bash

# ---------------------------------------------------------------------------
# Matched-instruction n-gram ablation re-runs for the OpenRouter-API models
# (Llama-3.1-70B-Instruct and OLMo-3.1-32B-Instruct). No GPU needed.
#
# Each invocation runs all four configs:
#   - suite_llama31_70b_instruct_ablations_matched_temp0p0.json
#   - suite_llama31_70b_instruct_ablations_matched_temp0p6.json
#   - suite_olmo32b_instruct_ablations_matched_temp0p0.json
#   - suite_olmo32b_instruct_ablations_matched_temp0p6.json
# ---------------------------------------------------------------------------

set -euo pipefail

# Resolve repository root from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
OPENROUTER_API_BASE="https://openrouter.ai/api/v1"

# Paste your OpenRouter API key below before running locally.
# IMPORTANT: do not commit this key to git.
OPENROUTER_API_KEY=""

if [[ "${OPENROUTER_API_KEY}" == "PASTE_OPENROUTER_API_KEY_HERE" || -z "${OPENROUTER_API_KEY}" ]]; then
  echo "Please set OPENROUTER_API_KEY in this file before running."
  exit 1
fi
export OPENROUTER_API_KEY

CONFIGS=(
  "suite_llama31_70b_instruct_ablations_matched_temp0p0.json"
  "suite_llama31_70b_instruct_ablations_matched_temp0p6.json"
  "suite_olmo32b_instruct_ablations_matched_temp0p0.json"
  "suite_olmo32b_instruct_ablations_matched_temp0p6.json"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=== API matched-instruction ablation: ${cfg} ==="
    python experiments/olmo_conformity/configs/run_expanded_experiments.py \
        --suite "experiments/olmo_conformity/configs/${cfg}" \
        --api-base "${OPENROUTER_API_BASE}" \
        --api-key "${OPENROUTER_API_KEY}" \
        --runs-only --force-rerun
    echo "=== ${cfg} complete ==="
done

echo "=== All API matched-instruction ablations complete ==="
