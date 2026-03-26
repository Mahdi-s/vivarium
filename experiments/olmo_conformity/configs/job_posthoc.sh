#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=l40s|a100|a40|v100
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_POSTHOC

# ---------------------------------------------------------------------------
# Post-hoc Interpretability Analysis
#
# Requires a completed run. Provide run_id via environment or let it
# auto-resolve from Comparing_Experiments/runs_metadata.json.
#
# Usage:
#   AAM_RUN_ID=<uuid> sbatch job_posthoc.sh
#   TEMPERATURE=0.0 sbatch job_posthoc.sh        # auto-resolve from metadata
# ---------------------------------------------------------------------------

set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

# Resolve run_id if not provided
AAM_RUN_ID="${AAM_RUN_ID:-}"
if [ -z "${AAM_RUN_ID}" ]; then
  AAM_RUN_ID="$(python -c "
import json, pathlib, sys
meta = pathlib.Path('Comparing_Experiments/runs_metadata.json')
if not meta.exists(): sys.exit(0)
entry = json.loads(meta.read_text()).get('experiments',{}).get('${TEMPERATURE}',{})
if entry.get('status')=='completed': print(entry.get('run_id',''),end='')
")"
fi

if [ -z "${AAM_RUN_ID}" ]; then
  echo "ERROR: Could not resolve run_id for T=${TEMPERATURE}. Set AAM_RUN_ID or ensure runs_metadata.json is populated." >&2
  exit 1
fi

echo "=== Post-hoc Analysis: run_id=${AAM_RUN_ID}, temperature=${TEMPERATURE} ==="

python experiments/olmo_conformity/configs/run_interpretability_posthoc.py \
    --run-id "${AAM_RUN_ID}" --hpc
