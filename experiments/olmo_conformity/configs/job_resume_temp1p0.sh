#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --constraint=a100|v100
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_RESUME_T1P0

set -euo pipefail

# RESUME run for T=1.0
# Original run_id: f95d2542-172c-4249-9ca2-b35caed3fa3d
# Models to complete: think, think_sft, think_dpo, rl_zero
# (instruct_dpo already complete at this temperature)

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1

export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python -m aam olmo-conformity \
    --suite-config experiments/olmo_conformity/configs/suite_resume_temp1.0.json \
    --runs-dir /scratch1/mahdisae/olmo_experiments/runs
