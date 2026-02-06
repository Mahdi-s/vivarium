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
#SBATCH --job-name=AAM_OLMO_POSTHOC_T0P4


cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1

export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"


python experiments/olmo_conformity/configs/run_interpretability_posthoc.py --run-id 271bb5b2-572d-4ecd-8577-b07a7cd10846 --hpc
