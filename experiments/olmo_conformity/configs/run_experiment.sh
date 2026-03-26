#!/bin/bash
# ============================================================================
# Unified Model-Driven Experiment Launcher
#
# Generates a suite config from CLI flags and launches run_expanded_experiments.py
# with model-specific output folders, logging, and run tracking.
#
# Usage:
#   # 7B Think rerun with extended tokens
#   ./run_experiment.sh --variant think --model allenai/Olmo-3-7B-Think \
#       --max-tokens 2048 --temps "0.0,0.6" --conditions core
#
#   # 32B Think DPO
#   ./run_experiment.sh --variant think_dpo_32b --model allenai/Olmo-3-32B-Think-DPO \
#       --max-tokens 4096 --temps "0.0" --conditions core --hpc
#
#   # Full 7B instruct suite (all 12 conditions)
#   ./run_experiment.sh --variant instruct --model allenai/Olmo-3-7B-Instruct \
#       --max-tokens 128 --temps "0.0,0.2,0.4,0.6,0.8,1.0" --conditions full
#
#   # Dry run (show config, don't execute)
#   ./run_experiment.sh --variant think --model allenai/Olmo-3-7B-Think \
#       --max-tokens 2048 --temps "0.0" --conditions core --dry-run
#
# Output:
#   {runs_dir}/{variant}/{YYYYMMDD_HHMMSS}_{run_id}/
#       ├── simulation.db
#       ├── run.log
#       └── config.json
#
# Run index:
#   {runs_dir}/{variant}/runs_index.json
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
VARIANT=""
MODEL=""
MAX_TOKENS=128
TEMPS="0.0"
CONDITIONS="core"
ITEMS=50
SEED=42
RUNS_DIR="${REPO_ROOT}/runs"
DRY_RUN=0
NO_SLEEP=0
HPC=0
HAS_THINK_TOKENS="false"
PHASE="trials"
METADATA=""
RESUME_RUN_ID=""

# ── Parse flags ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)      VARIANT="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --temps)        TEMPS="$2"; shift 2 ;;
        --conditions)   CONDITIONS="$2"; shift 2 ;;
        --items)        ITEMS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --runs-dir)     RUNS_DIR="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --no-sleep)     NO_SLEEP=1; shift ;;
        --hpc)          HPC=1; shift ;;
        --think)        HAS_THINK_TOKENS="true"; shift ;;
        --phase)        PHASE="$2"; shift 2 ;;
        --metadata)     METADATA="$2"; shift 2 ;;
        --resume)       RESUME_RUN_ID="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELPEOF'
Usage: run_experiment.sh [flags]

Required:
  --variant NAME       Short variant name (think, think_sft, base, instruct, etc.)
  --model ID           HuggingFace model ID (e.g., allenai/Olmo-3-7B-Think)

Optional:
  --max-tokens N       Max new tokens (default: 128; think: 2048; 32B: 4096)
  --temps LIST         Comma-separated temperatures (default: "0.0")
  --conditions SET     "full" (12), "core" (4), or comma-separated (default: core)
  --items N            Max items per dataset (default: 50)
  --seed N             Random seed (default: 42)
  --runs-dir PATH      Output base directory (default: repo_root/runs)
  --think              Flag this model as having <think> tokens
  --phase PHASE        Pipeline phase: trials, judge, posthoc, all (default: trials)
  --metadata PATH      Custom metadata file path
  --resume RUN_ID      Resume an existing run by UUID (skips completed trials)
  --hpc                Use HPC paths from paths.json
  --no-sleep           Prevent macOS sleep (caffeinate)
  --dry-run            Show config without running

Output:
  {runs_dir}/{variant}/{YYYYMMDD_HHMMSS}_{run_id}/
      simulation.db, run.log, config.json
  {runs_dir}/{variant}/runs_index.json  (run tracking)
HELPEOF
            exit 0 ;;
        *) echo "Unknown flag: $1 (try --help)"; exit 1 ;;
    esac
done

# ── Validate required args ──────────────────────────────────────────────────
if [[ -z "$VARIANT" || -z "$MODEL" ]]; then
    echo "ERROR: --variant and --model are required. Use --help for usage."
    exit 1
fi

# Auto-detect think tokens from variant name
if [[ "$VARIANT" == think* ]]; then
    HAS_THINK_TOKENS="true"
fi

# Normalize TEMPS into:
#  - `TEMPS_JSON`: JSON array of numeric temps, e.g. [0.0,0.6]
#  - `RUN_TEMPERATURE`: first temp (used as suite fallback)
TEMPS_CLEAN="${TEMPS//[[:space:]]/}"
TEMPS_JSON="[]"
RUN_TEMPERATURE="0.0"
if [[ -n "${TEMPS_CLEAN}" ]]; then
    IFS=',' read -ra _TEMPS_ARR <<< "${TEMPS_CLEAN}"
    if (( ${#_TEMPS_ARR[@]} > 0 )); then
        RUN_TEMPERATURE="${_TEMPS_ARR[0]}"
        TEMPS_JSON="[${_TEMPS_ARR[0]}"
        for ((i=1; i<${#_TEMPS_ARR[@]}; i++)); do
            TEMPS_JSON+=",${_TEMPS_ARR[i]}"
        done
        TEMPS_JSON+="]"
    fi
fi

# ── Build conditions JSON ───────────────────────────────────────────────────
build_conditions() {
    local cond_set="$1"
    case "$cond_set" in
        core)
            cat <<'CEOF'
    { "name": "control", "params": { "type": "control" } },
    { "name": "asch_zhu_unanimous_confident", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "unanimous", "tone": "confident" } },
    { "name": "authoritative_bias", "params": { "type": "authoritative_bias", "user_claim_strength": "high" } },
    { "name": "authority_trust", "params": { "type": "authority_claim", "system_style": "control", "claim_style": "trust" } }
CEOF
            ;;
        full)
            cat <<'FEOF'
    { "name": "control", "params": { "type": "control" } },
    { "name": "asch_history_5", "params": { "type": "asch_peer_pressure", "prompt_style": "history", "confederates": 5 } },
    { "name": "asch_zhu_unbiased_unanimous_plain", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "unanimous", "tone": "plain" } },
    { "name": "asch_zhu_unbiased_unanimous_neutral", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "unanimous", "tone": "neutral" } },
    { "name": "asch_zhu_unbiased_unanimous_confident", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "unanimous", "tone": "confident" } },
    { "name": "asch_zhu_unbiased_unanimous_uncertain", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "unanimous", "tone": "uncertain" } },
    { "name": "asch_zhu_unbiased_diverse_plain", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "diverse", "tone": "plain" } },
    { "name": "asch_zhu_unbiased_da", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "da" } },
    { "name": "asch_zhu_unbiased_qd", "params": { "type": "asch_peer_pressure", "system_style": "control", "prompt_style": "conversation", "confederates": 5, "consensus": "qd" } },
    { "name": "authoritative_bias", "params": { "type": "authoritative_bias", "user_claim_strength": "high" } },
    { "name": "authority_zhu_unbiased_trust", "params": { "type": "authority_claim", "system_style": "control", "claim_style": "trust" } },
    { "name": "authority_zhu_unbiased_trust_da", "params": { "type": "authority_claim", "system_style": "control", "claim_style": "trust_da" } }
FEOF
            ;;
        *)
            # Custom: comma-separated condition names → just use core for now
            echo "WARNING: Custom conditions not yet supported, falling back to core" >&2
            build_conditions "core"
            ;;
    esac
}

# ── Generate suite config JSON ──────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUITE_JSON=$(cat <<SEOF
{
  "paths_config": "paths.json",
  "suite_name": "olmo_conformity_${VARIANT}_auto",
  "suite_version": "auto",
  "description": "Auto-generated by run_experiment.sh for ${VARIANT} (${MODEL})",
  "datasets": [
    { "name": "immutable_facts_minimal", "version": "v2", "path": "experiments/olmo_conformity/datasets/immutable_facts/minimal_items_wrong.jsonl", "category": "general" },
    { "name": "social_conventions_minimal", "version": "v2", "path": "experiments/olmo_conformity/datasets/social_conventions/minimal_items_wrong.jsonl", "category": "opinion" },
    { "name": "gsm8k", "version": "v1", "path": "experiments/olmo_conformity/datasets/math/gsm8k_items_wrong.jsonl", "category": "math" },
    { "name": "mmlu_math", "version": "v1", "path": "experiments/olmo_conformity/datasets/math/mmlu_math_items_wrong.jsonl", "category": "math" },
    { "name": "mmlu_science", "version": "v1", "path": "experiments/olmo_conformity/datasets/science/mmlu_science_items_wrong.jsonl", "category": "science" },
    { "name": "mmlu_knowledge", "version": "v1", "path": "experiments/olmo_conformity/datasets/knowledge/mmlu_knowledge_items_wrong.jsonl", "category": "knowledge" },
    { "name": "truthfulqa", "version": "v1", "path": "experiments/olmo_conformity/datasets/truthfulness/truthfulqa_items_wrong.jsonl", "category": "truthfulness" },
    { "name": "arc", "version": "v1", "path": "experiments/olmo_conformity/datasets/reasoning/arc_items_wrong.jsonl", "category": "reasoning" }
  ],
  "conditions": [
$(build_conditions "$CONDITIONS")
  ],
  "models": [
    { "variant": "${VARIANT}", "model_id": "${MODEL}", "max_new_tokens": ${MAX_TOKENS}, "has_think_tokens": ${HAS_THINK_TOKENS} }
  ],
  "default_temperatures": ${TEMPS_JSON},
  "run": {
    "seed": ${SEED},
    "temperature": ${RUN_TEMPERATURE},
    "top_k": 50,
    "top_p": 0.9,
    "max_items_per_dataset": ${ITEMS}
  }
}
SEOF
)

# ── Create model-specific output directory ──────────────────────────────────
VARIANT_DIR="${RUNS_DIR}/${VARIANT}"
mkdir -p "${VARIANT_DIR}"

# ── Write suite config to temp file ─────────────────────────────────────────
# macOS mktemp requires X's at the very end — append .json suffix after creation
_SUITE_TMP=$(mktemp "${VARIANT_DIR}/suite_${VARIANT}_XXXXXX")
SUITE_FILE="${_SUITE_TMP}.json"
mv "${_SUITE_TMP}" "${SUITE_FILE}"
echo "${SUITE_JSON}" > "${SUITE_FILE}"

# ── Display config ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Unified Experiment Launcher"
echo "================================================================"
echo "  Variant:       ${VARIANT}"
echo "  Model:         ${MODEL}"
echo "  Max Tokens:    ${MAX_TOKENS}"
echo "  Think Tokens:  ${HAS_THINK_TOKENS}"
echo "  Temperatures:  ${TEMPS}"
echo "  Conditions:    ${CONDITIONS}"
echo "  Items/dataset: ${ITEMS}"
echo "  Seed:          ${SEED}"
echo "  Phase:         ${PHASE}"
if [[ -n "$RESUME_RUN_ID" ]]; then
echo "  Resume:        ${RESUME_RUN_ID}"
fi
echo "  Output:        ${VARIANT_DIR}/"
echo "  Suite config:  ${SUITE_FILE}"
echo "  HPC:           ${HPC}"
echo "  Started:       $(date)"
echo "================================================================"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--- DRY RUN: Generated suite config ---"
    cat "${SUITE_FILE}"
    echo ""
    echo "--- Would execute ---"
    echo "python ${SCRIPT_DIR}/run_expanded_experiments.py \\"
    echo "    --suite ${SUITE_FILE} \\"
    echo "    --temps ${TEMPS} \\"
    echo "    --runs-dir ${VARIANT_DIR} \\"
    echo "    --runs-only --phase ${PHASE}"
    echo ""
    echo "Output would be saved to: ${VARIANT_DIR}/<run_dir>/"
    rm -f "${SUITE_FILE}"
    exit 0
fi

# ── Update runs_index.json (pre-run) ───────────────────────────────────────
INDEX_FILE="${VARIANT_DIR}/runs_index.json"
if [[ ! -f "$INDEX_FILE" ]]; then
    echo '{"runs":[]}' > "$INDEX_FILE"
fi

# Add a pre-run entry (status: running)
python3 -c "
import json, sys
idx = json.load(open('${INDEX_FILE}'))
idx['runs'].append({
    'variant': '${VARIANT}',
    'model_id': '${MODEL}',
    'temperatures': [float(t) for t in '${TEMPS}'.split(',')],
    'max_new_tokens': ${MAX_TOKENS},
    'conditions': '${CONDITIONS}',
    'started_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'status': 'running',
    'suite_config': '${SUITE_FILE}'
})
json.dump(idx, open('${INDEX_FILE}', 'w'), indent=2)
"

# ── Build command ───────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CMD=(
    python3 "${SCRIPT_DIR}/run_expanded_experiments.py"
    --suite "${SUITE_FILE}"
    --temps "${TEMPS}"
    --runs-dir "${VARIANT_DIR}"
    --runs-only
    --phase "${PHASE}"
)

if [[ $NO_SLEEP -eq 1 ]]; then
    CMD+=(--no-sleep)
fi
if [[ $HPC -eq 1 ]]; then
    CMD+=(--hpc)
fi
if [[ -n "$METADATA" ]]; then
    CMD+=(--metadata "${METADATA}")
fi
if [[ -n "$RESUME_RUN_ID" ]]; then
    CMD+=(--resume-run-id "${RESUME_RUN_ID}")
fi

# ── Execute with logging ───────────────────────────────────────────────────
# Log file will be created inside the run dir once we know it.
# For now, tee to the variant-level log.
VARIANT_LOG="${VARIANT_DIR}/run_${TIMESTAMP}.log"
echo "Logging to: ${VARIANT_LOG}"
echo ""

RC=0
"${CMD[@]}" 2>&1 | tee "${VARIANT_LOG}" || RC=$?

# ── Update runs_index.json (post-run) ──────────────────────────────────────
STATUS="completed"
if [[ $RC -ne 0 ]]; then
    STATUS="failed"
fi

python3 -c "
import json
idx = json.load(open('${INDEX_FILE}'))
# Update the last entry
if idx['runs']:
    idx['runs'][-1]['status'] = '${STATUS}'
    idx['runs'][-1]['log_path'] = '${VARIANT_LOG}'
json.dump(idx, open('${INDEX_FILE}', 'w'), indent=2)
"

echo ""
echo "================================================================"
echo "  Experiment ${STATUS}: ${VARIANT}"
echo "  Log: ${VARIANT_LOG}"
echo "  Index: ${INDEX_FILE}"
echo "  Finished: $(date)"
echo "================================================================"

exit $RC
