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
#SBATCH --job-name=AAM_RESUME_T0P0

set -euo pipefail

# RESUME run for T=0.0
# Original run_id: 3d70c15d-9fdf-4521-9d72-0ae1feec0238
# Models to complete: instruct_dpo, think, think_sft, think_dpo, rl_zero
# Uses suite_resume config (incomplete models only) to avoid re-running base/instruct/instruct_sft.

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1

export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python -m aam olmo-conformity \
    --suite-config experiments/olmo_conformity/configs/suite_resume_temp0.0.json \
    --runs-dir /scratch1/mahdisae/olmo_experiments/runs
