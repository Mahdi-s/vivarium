#!/bin/bash
set -euo pipefail

# ============================================================================
# Vivarium End-to-End Integration Smoke Test
#
# Runs a real 3-agent, 1-step simulation against one or more LLM backends,
# exercising the full pipeline: EmpiricalAgentStateSpace → persona injection
# → LLM call → WorldEngine trace → interpretability table writes.
#
# Backends:
#   --ollama       Ollama server (default if no flag given)
#   --huggingface  HuggingFace Transformers local inference
#   --llamacpp     llama.cpp llama-server (spawned automatically)
#   --all          Run all three backends sequentially
#
# Prerequisites:
#   Ollama:      ollama serve && ollama pull qwen2.5:0.5b
#   HuggingFace: python scripts/download_test_model.py --hf-only
#   llama.cpp:   python scripts/download_test_model.py --gguf-only
#                + llama-server built (third_party/llama.cpp/build/bin/llama-server)
#   All:         pip install -e .[cognitive]   (litellm, transformers, torch)
#
# Override model:  VVM_TEST_MODEL=<model> ./scripts/verify_integration.sh --ollama
# Override URL:    OLLAMA_BASE_URL=http://host:11434/v1 ./scripts/verify_integration.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
RUN_OLLAMA=0
RUN_HF=0
RUN_LLAMACPP=0

if [[ $# -eq 0 ]]; then
    # Default: ollama only (backward compat)
    RUN_OLLAMA=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_OLLAMA=1; RUN_HF=1; RUN_LLAMACPP=1; shift ;;
        --ollama)
            RUN_OLLAMA=1; shift ;;
        --huggingface|--hf)
            RUN_HF=1; shift ;;
        --llamacpp|--llama-cpp|--llama)
            RUN_LLAMACPP=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--all | --ollama | --huggingface | --llamacpp] ..."
            echo ""
            echo "Flags (combine any):"
            echo "  --ollama       Test with Ollama server (default)"
            echo "  --huggingface  Test with HuggingFace Transformers"
            echo "  --llamacpp     Test with llama.cpp llama-server"
            echo "  --all          Test all three backends"
            echo ""
            echo "Environment:"
            echo "  VVM_TEST_MODEL        Override model name (per-backend defaults apply)"
            echo "  OLLAMA_BASE_URL       Ollama API endpoint (default: http://localhost:11434/v1)"
            echo "  LLAMA_SERVER_PORT     llama-server port (default: 18081)"
            echo "  VIVARIUM_MODEL_DIR    Override models directory"
            exit 0 ;;
        *)
            echo "Unknown flag: $1 (try --help)"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Shared pre-flight
# ---------------------------------------------------------------------------

echo "=== Vivarium Integration Smoke Test ==="
echo ""

# Python interpreter — prefer the active env's python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python"
fi

echo -n "Pre-flight: Python ($PYTHON)... "
if "$PYTHON" -c "import vivarium" 2>/dev/null; then
    echo "✓"
else
    echo "✗"
    echo ""
    echo "ERROR: 'import vivarium' failed."
    echo "  Install with:  pip install -e '${REPO_DIR}'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Backend: Ollama
# ---------------------------------------------------------------------------
OVERALL_RC=0

if [[ $RUN_OLLAMA -eq 1 ]]; then
    echo ""
    echo "────────────────────────────────────────"
    echo "  Backend: Ollama"
    echo "────────────────────────────────────────"

    OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
    MODEL="${VVM_TEST_MODEL:-qwen2.5:0.5b}"

    echo -n "Pre-flight: Ollama reachable at ${OLLAMA_URL%/v1}... "
    if curl -sf "${OLLAMA_URL%/v1}/api/tags" > /dev/null 2>&1; then
        echo "✓"
    else
        echo "✗"
        echo "  ERROR: Cannot reach Ollama at ${OLLAMA_URL%/v1}"
        echo "  Start it with:  ollama serve"
        OVERALL_RC=1
        RUN_OLLAMA=0  # skip the test
    fi

    if [[ $RUN_OLLAMA -eq 1 ]]; then
        echo -n "Pre-flight: Model '${MODEL}' available... "
        if curl -sf "${OLLAMA_URL%/v1}/api/tags" 2>/dev/null | "$PYTHON" -c "
import sys, json
tags = json.load(sys.stdin)
names = [m['name'] for m in tags.get('models', [])]
target = '${MODEL}'
found = any(n == target or n == target + ':latest' or n.startswith(target + ':') for n in names)
sys.exit(0 if found else 1)
" 2>/dev/null; then
            echo "✓"
        else
            echo "✗"
            echo "  ERROR: Model '${MODEL}' not found in Ollama."
            echo "  Pull it with:  ollama pull ${MODEL}"
            OVERALL_RC=1
            RUN_OLLAMA=0
        fi
    fi

    if [[ $RUN_OLLAMA -eq 1 ]]; then
        echo -n "Pre-flight: litellm importable... "
        if "$PYTHON" -c "import litellm" 2>/dev/null; then
            echo "✓"
        else
            echo "✗"
            echo "  ERROR: litellm not installed."
            echo "  Install with:  pip install litellm"
            OVERALL_RC=1
            RUN_OLLAMA=0
        fi
    fi

    if [[ $RUN_OLLAMA -eq 1 ]]; then
        echo ""
        export VVM_TEST_MODEL="${MODEL}"
        export OLLAMA_BASE_URL="${OLLAMA_URL}"
        if ! "$PYTHON" "${SCRIPT_DIR}/integration_smoke.py" --backend ollama; then
            OVERALL_RC=1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Backend: HuggingFace
# ---------------------------------------------------------------------------

if [[ $RUN_HF -eq 1 ]]; then
    echo ""
    echo "────────────────────────────────────────"
    echo "  Backend: HuggingFace Transformers"
    echo "────────────────────────────────────────"

    HF_MODEL="${VVM_TEST_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

    echo -n "Pre-flight: transformers importable... "
    if "$PYTHON" -c "from transformers import AutoModelForCausalLM" 2>/dev/null; then
        echo "✓"
    else
        echo "✗"
        echo "  ERROR: transformers not installed."
        echo "  Install with:  pip install transformers torch"
        OVERALL_RC=1
        RUN_HF=0
    fi

    if [[ $RUN_HF -eq 1 ]]; then
        echo -n "Pre-flight: torch importable... "
        if "$PYTHON" -c "import torch" 2>/dev/null; then
            echo "✓"
        else
            echo "✗"
            echo "  ERROR: torch not installed."
            echo "  Install with:  pip install torch"
            OVERALL_RC=1
            RUN_HF=0
        fi
    fi

    # Check if model weights exist (either local cache or will download)
    if [[ $RUN_HF -eq 1 ]]; then
        echo -n "Pre-flight: HF model weights... "
        HF_CACHE="${VIVARIUM_HF_CACHE:-${REPO_DIR}/models/huggingface_cache}"
        LOCAL_DIR="${HF_CACHE}/${HF_MODEL//\//_}"
        if [[ -d "$LOCAL_DIR" ]] && ls "$LOCAL_DIR"/*.safetensors &>/dev/null 2>&1; then
            echo "✓ (cached: ${LOCAL_DIR})"
        else
            echo "⚠ not cached — will download on first run"
            echo "  Pre-download with:  python scripts/download_test_model.py --hf-only"
        fi
    fi

    if [[ $RUN_HF -eq 1 ]]; then
        echo ""
        export VVM_TEST_MODEL="${HF_MODEL}"
        if ! "$PYTHON" "${SCRIPT_DIR}/integration_smoke.py" --backend huggingface --model "${HF_MODEL}"; then
            OVERALL_RC=1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Backend: llama.cpp
# ---------------------------------------------------------------------------

if [[ $RUN_LLAMACPP -eq 1 ]]; then
    echo ""
    echo "────────────────────────────────────────"
    echo "  Backend: llama.cpp"
    echo "────────────────────────────────────────"

    GGUF_NAME="${VVM_TEST_MODEL:-qwen2.5-0.5b-instruct-q5_k_m.gguf}"
    MODELS_DIR="${VIVARIUM_MODEL_DIR:-${REPO_DIR}/models}"

    echo -n "Pre-flight: litellm importable... "
    if "$PYTHON" -c "import litellm" 2>/dev/null; then
        echo "✓"
    else
        echo "✗"
        echo "  ERROR: litellm not installed."
        echo "  Install with:  pip install litellm"
        OVERALL_RC=1
        RUN_LLAMACPP=0
    fi

    if [[ $RUN_LLAMACPP -eq 1 ]]; then
        echo -n "Pre-flight: llama-server binary... "
        LLAMA_BIN="${REPO_DIR}/third_party/llama.cpp/build/bin/llama-server"
        if [[ -x "$LLAMA_BIN" ]]; then
            echo "✓ (${LLAMA_BIN})"
        elif command -v llama-server &>/dev/null; then
            echo "✓ ($(which llama-server))"
        else
            echo "✗"
            echo "  ERROR: llama-server not found."
            echo "  Build llama.cpp:  cd third_party/llama.cpp && cmake -B build && cmake --build build"
            OVERALL_RC=1
            RUN_LLAMACPP=0
        fi
    fi

    if [[ $RUN_LLAMACPP -eq 1 ]]; then
        echo -n "Pre-flight: GGUF file... "
        if [[ -f "${MODELS_DIR}/${GGUF_NAME}" ]]; then
            echo "✓ (${MODELS_DIR}/${GGUF_NAME})"
        elif [[ -f "${GGUF_NAME}" ]]; then
            echo "✓ (${GGUF_NAME})"
        else
            echo "✗"
            echo "  ERROR: GGUF file not found: ${MODELS_DIR}/${GGUF_NAME}"
            echo "  Download with:  python scripts/download_test_model.py --gguf-only"
            OVERALL_RC=1
            RUN_LLAMACPP=0
        fi
    fi

    if [[ $RUN_LLAMACPP -eq 1 ]]; then
        echo ""
        export VVM_TEST_MODEL="${GGUF_NAME}"
        if ! "$PYTHON" "${SCRIPT_DIR}/integration_smoke.py" --backend llamacpp --model "${GGUF_NAME}"; then
            OVERALL_RC=1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
if [[ $OVERALL_RC -eq 0 ]]; then
    echo "  All selected backends passed ✓"
else
    echo "  Some backends FAILED ✗ (see above)"
fi
echo "============================================"

exit $OVERALL_RC
