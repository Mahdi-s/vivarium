#!/bin/bash
set -euo pipefail

# ============================================================================
# Vivarium End-to-End Integration Smoke Test
#
# Runs a real 3-agent, 1-step simulation with an Ollama LLM, exercises the
# full pipeline: EmpiricalAgentStateSpace → persona injection → LLM call →
# WorldEngine trace → interpretability table writes.
#
# Prerequisites:
#   1. Ollama running:          ollama serve
#   2. Model pulled:            ollama pull qwen2.5:0.5b
#   3. Python deps installed:   pip install -e .[cognitive]  (or at minimum: litellm)
#
# Override model:  VVM_TEST_MODEL=llama3.2:1b ./scripts/verify_integration.sh
# Override URL:    OLLAMA_BASE_URL=http://host:11434/v1 ./scripts/verify_integration.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="${VVM_TEST_MODEL:-qwen2.5:0.5b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"

echo "=== Vivarium Integration Smoke Test ==="
echo ""

# --- Pre-flight checks ---

echo -n "Pre-flight: Ollama reachable at ${OLLAMA_URL%/v1}... "
if curl -sf "${OLLAMA_URL%/v1}/api/tags" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo ""
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_URL%/v1}"
    echo "  Start it with:  ollama serve"
    exit 1
fi

echo -n "Pre-flight: Model '${MODEL}' available... "
if curl -sf "${OLLAMA_URL%/v1}/api/tags" 2>/dev/null | python3 -c "
import sys, json
tags = json.load(sys.stdin)
names = [m['name'] for m in tags.get('models', [])]
# Match with or without :latest tag
target = '${MODEL}'
found = any(n == target or n == target + ':latest' or n.startswith(target + ':') for n in names)
sys.exit(0 if found else 1)
" 2>/dev/null; then
    echo "✓"
else
    echo "✗"
    echo ""
    echo "ERROR: Model '${MODEL}' not found in Ollama."
    echo "  Pull it with:  ollama pull ${MODEL}"
    exit 1
fi

echo -n "Pre-flight: litellm importable... "
if python3 -c "import litellm" 2>/dev/null; then
    echo "✓"
else
    echo "✗"
    echo ""
    echo "ERROR: litellm not installed."
    echo "  Install with:  pip install litellm"
    exit 1
fi

echo ""

# --- Run integration test ---
export VVM_TEST_MODEL="${MODEL}"
export OLLAMA_BASE_URL="${OLLAMA_URL}"

python3 "${SCRIPT_DIR}/integration_smoke.py"
