#!/usr/bin/env bash
#
# run_all.sh — Single-command end-to-end regenerator for April_analysis/.
#
# Runs the 8 7B driver scripts (which read metadata/runs_metadata.json)
# followed by the 4 cross-family + ablation driver scripts (which read
# metadata/cross_family_metadata.json). Exits non-zero on any failure.
#
# Idempotent: all outputs are written under Comparing_Experiments/April_analysis/
# and overwritten on each invocation. Existing Analysis Scripts/april_analysis/*
# scripts are reused as-is; this wrapper only orchestrates invocation order.
#
# Usage (from repo root or from this folder):
#
#     ./Comparing_Experiments/April_analysis/run_all.sh
#
# Environment variables:
#
#     PYTHON   : Python invocation to use (default: "uv run python")
#
# Post-conditions verified on success:
#
#   - Comparing_Experiments/April_analysis/validation/claim_check.md
#     shows 12 PASS / 1 FAIL (7B)
#   - Comparing_Experiments/April_analysis/validation/cross_family_claim_check.md
#     shows 9 PASS / 0 FAIL / 1 PARTIAL (cross-family + ablation)
#
set -euo pipefail

# Resolve repo root (script lives two directories deep)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-uv run python}"
ANALYSIS_DIR="Analysis Scripts/april_analysis"
OUT_DIR="Comparing_Experiments/April_analysis"

# Pretty headers so each phase is greppable in the log
hdr() {
    printf '\n============================================================\n' >&2
    printf '  %s\n' "$1" >&2
    printf '============================================================\n' >&2
}

run_step() {
    local label="$1"
    local script="$2"
    shift 2
    hdr "$label  ($script)"
    $PYTHON "$ANALYSIS_DIR/$script" "$@"
}

# ---------------------------------------------------------------------------
# 7B layer (manifest: metadata/runs_metadata.json)
# ---------------------------------------------------------------------------
# Each of these scripts uses DEFAULT_MANIFEST = runs_metadata.json and
# DEFAULT_OUT_DIR = April_analysis/. No CLI args needed.
run_step "7B/B1 behavioral_tables"       behavioral_tables.py
run_step "7B/B2 stage_decomposition"     stage_decomposition.py
run_step "7B/C2 pattern_match_gradient"  pattern_match_gradient.py
run_step "7B/C4 temperature_concentration" temperature_concentration.py
run_step "7B/C5 item_level_correlations" item_level_correlations.py
run_step "7B/C6 mitigation_taxonomy"     mitigation_taxonomy.py
run_step "7B figures.py"                 figures.py
run_step "7B validate.py"                validate.py

# ---------------------------------------------------------------------------
# Cross-family + ablation layer (manifest: metadata/cross_family_metadata.json)
# ---------------------------------------------------------------------------
# These scripts use DEFAULT_CROSS_FAMILY_MANIFEST by default. The build_
# cross_family_argparser() guard refuses to run them if --manifest is
# accidentally pointed at runs_metadata.json, so no flag is needed here.
run_step "CF/E2 cross_family_tables"     cross_family_tables.py
run_step "CF/E3 ablation_probes"         ablation_probes.py
run_step "CF/E4 cross_family_figures"    cross_family_figures.py
run_step "CF/E5 cross_family_validate"   cross_family_validate.py

# ---------------------------------------------------------------------------
# Verification summary (does not run the scorecards; just reports what the
# driver scripts wrote).
# ---------------------------------------------------------------------------
hdr "Verification summary"

SEVEN_B_CLAIM="$OUT_DIR/validation/claim_check.md"
CF_CLAIM="$OUT_DIR/validation/cross_family_claim_check.md"

if [[ ! -f "$SEVEN_B_CLAIM" ]]; then
    echo "[run_all.sh] FAIL: $SEVEN_B_CLAIM is missing" >&2
    exit 1
fi
if [[ ! -f "$CF_CLAIM" ]]; then
    echo "[run_all.sh] FAIL: $CF_CLAIM is missing" >&2
    exit 1
fi

echo "[run_all.sh] 7B scorecard: $SEVEN_B_CLAIM"
echo "[run_all.sh] Cross-family scorecard: $CF_CLAIM"
echo
echo "[run_all.sh] Done. Inspect the two scorecards above for PASS/FAIL status."
