#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=AAM_API_NGRAM_MATCHED

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

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

# OPENROUTER_API_KEY must be exported in the calling environment (or sourced
# from a private secrets file). Do not commit the key to the repository.
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the environment.}"

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
        --hpc --runs-only --force-rerun
    echo "=== ${cfg} complete ==="
done

echo "=== All API matched-instruction ablations complete ==="
