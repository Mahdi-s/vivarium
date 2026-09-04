#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100|l40s
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_THINK8K
#SBATCH --output=/scratch1/mahdisae/olmo_experiments/slurm_logs/%x_%j.out
# ---------------------------------------------------------------------------
# OPTIONAL asset repair: re-run the 7B Think ladder through the EXISTING vivarium pipeline at 8192 tokens
# (4 core conditions, T=0, one model per job). Outputs land in paths.json runs_dir on scratch, judge with job_judge.sh.
# Runtime-bound: 1,600 trials x up to 8k tokens; expect 20–40 h per model. Prefer job_belief_think_7b.sh
# (budget-forced answers) unless you specifically need judge-labelled free generations for the Think ladder.
#   MODEL=think_sft sbatch job_think_rerun8k_7b.sh    # think_sft | think_dpo | think
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p /scratch1/mahdisae/olmo_experiments/slurm_logs
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=0
MODEL="${MODEL:-think_sft}"
SUITE="experiments/olmo_conformity/configs/suite_7b_${MODEL}_rerun8k_temp0p0.json"
echo "=== Think 8k rerun: ${MODEL} suite=${SUITE} ==="
python experiments/olmo_conformity/configs/run_expanded_experiments.py --suite "${SUITE}" --temps 0.0 --hpc --runs-only --force-rerun
