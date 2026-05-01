#!/bin/bash
# ---------------------------------------------------------------------------
# Master HPC launcher: 7B Think (RLVR final) comprehensive re-run at 5000 tokens.
#
# Submits 2 SLURM jobs (T=0.0 and T=0.6) for the OLMo-3 7B Think RLVR final
# checkpoint (`allenai/Olmo-3-7B-Think`). Each job covers 15 conditions on
# the standard 400-item pool:
#
#   12 canonical conditions:
#     control, asch_history_5,
#     asch_zhu_unbiased_unanimous_{plain, neutral, confident, uncertain},
#     asch_zhu_unbiased_diverse_plain, asch_zhu_unbiased_da, asch_zhu_unbiased_qd,
#     authoritative_bias, authority_zhu_unbiased_trust, authority_zhu_unbiased_trust_da
#
#   3 ablation conditions (the structural-confound test):
#     asch_zhu_naked_unanimous_confident      (no system prompt)
#     ngram_sequence_baseline                 (original "based on the provided sequence")
#     ngram_sequence_matched_baseline         (matched-instruction wording)
#
# Why this re-run: the existing Think-RLVR DBs at
# runs/think/20260325_010440_* (T=0) and runs/think/20260414_092714_* (T=0.6)
# used max_new_tokens=2048, which truncated 30-71% of trials mid-`<think>`
# block. The 5000-token budget here matches the budget already used for
# Think-SFT and Think-DPO and ensures every trial has room to close.
#
# Subsumes the existing Panel B extension Think-RLVR jobs
# (`job_olmo_7b_think_rlvr_panelB.sh`) — do NOT also submit those if this is
# submitted, since the 8 conditions they cover are a strict subset of the 15
# here.
#
# Pre-flight checklist:
#   - On HPC login node, in the abstractAgentMachine repo root.
#   - aam_venv exists at /scratch1/mahdisae/aam_venv.
#   - HF model weights for allenai/Olmo-3-7B-Think are cached or reachable
#     from /scratch1/mahdisae.
#
# Usage:
#   bash experiments/olmo_conformity/scripts/run_olmo_7b_think_rlvr_full_5k_hpc.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

JOB="experiments/olmo_conformity/configs/job_olmo_7b_think_rlvr_full_5k.sh"

declare -a SUBMITTED=()

for temp in 0.0 0.6; do
    echo "Submitting ${JOB} at TEMPERATURE=${temp}..."
    SLURM_ID=$(TEMPERATURE="${temp}" sbatch --parsable "${JOB}")
    echo "  -> ${SLURM_ID}"
    SUBMITTED+=("${SLURM_ID} job_olmo_7b_think_rlvr_full_5k.sh T=${temp}")
done

echo
echo "=== Submitted ${#SUBMITTED[@]} jobs ==="
for line in "${SUBMITTED[@]}"; do
    echo "  ${line}"
done
echo
echo "Track with: squeue -u \${USER} -o '%.10i %.20j %.8T %.10M %R'"
echo
echo "When both jobs complete:"
echo "  1. Find the new run directories under runs/."
echo "  2. Add them to Comparing_Experiments/April_analysis/metadata/runs_metadata.json"
echo "     as sources_primary entries with variants: [\"think\"]."
echo "  3. Mark the old 2048-token DBs as ignore_variants: [\"think\"]:"
echo "       runs/think/20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37/simulation.db"
echo "       runs/think/20260414_092714_67aaf9da-34d7-4855-aee9-d8a6b1a6099e/simulation.db"
echo "  4. Sanity-check truncation rate (target: <5% unclosed </think>):"
echo "       sqlite3 runs/<NEW_UUID>/simulation.db \\"
echo "         \"SELECT cc.name, COUNT(*) AS n, \\"
echo "          SUM(CASE WHEN co.raw_text LIKE '%</think>%' THEN 1 ELSE 0 END) AS closed \\"
echo "          FROM conformity_trials ct \\"
echo "          JOIN conformity_conditions cc ON cc.condition_id = ct.condition_id \\"
echo "          JOIN conformity_outputs co ON co.trial_id = ct.trial_id \\"
echo "          GROUP BY cc.name;\""
echo "  5. Re-render figures: python scripts/render_stage_trajectory_with_mcnemar.py"
echo "                        python scripts/render_fig2_heatmap_variants.py"
