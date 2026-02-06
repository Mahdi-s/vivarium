#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100|v100
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=AAM_OLMO_JUDGE_T0P8

set -euo pipefail

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH}"

export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python experiments/olmo_conformity/configs/run_llm_judge_posthoc.py \
  --run-id eb777acc-3ab5-4f87-b073-249a50d25863 \
  --hpc \
  --max-concurrency 4 \
  --trial-scope behavioral-only
