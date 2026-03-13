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
#SBATCH --job-name=AAM_RESUME

# ---------------------------------------------------------------------------
# Resume / Complete an Existing Run
#
# Re-uses the same suite config and run_id to fill in missing model trials.
#
# Usage:
#   AAM_RUN_ID=<uuid> SUITE=suite_7b_expanded.json RUNS_DIR=runs-hpc-full/run_v2/runs sbatch job_resume.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SUITE="${SUITE:-suite_7b_expanded.json}"
RUNS_DIR="${RUNS_DIR:-}"
AAM_RUN_ID="${AAM_RUN_ID:-}"

if [ -z "${AAM_RUN_ID}" ]; then
  echo "ERROR: AAM_RUN_ID is required for resume. Set it to the run_id you want to complete." >&2
  exit 1
fi

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== Resume Run: suite=${SUITE}, run_id=${AAM_RUN_ID} ==="

CMD="python -m aam olmo-conformity --suite-config experiments/olmo_conformity/configs/${SUITE}"
if [ -n "${RUNS_DIR}" ]; then
  CMD="${CMD} --runs-dir ${RUNS_DIR}"
fi
CMD="${CMD} --run-id ${AAM_RUN_ID}"

eval "${CMD}"
