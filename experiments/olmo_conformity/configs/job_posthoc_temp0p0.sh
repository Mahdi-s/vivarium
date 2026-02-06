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
#SBATCH --job-name=AAM_OLMO_POSTHOC_T0P0


cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1

# Ensure repo sources are importable even if not pip-installed on the cluster.
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

# Avoid matplotlib trying to write under $HOME on HPC.
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"


python experiments/olmo_conformity/configs/run_interpretability_posthoc.py --run-id 56478e99-7607-4957-9f53-a53b73a7e9d4 --hpc
