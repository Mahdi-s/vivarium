#!/bin/bash
# Submit one 48 h job per checkpoint under one TAG (login node).
#   TAG=pilot_0905 ITEMS=10 ./submit_ladder.sh           # pilot first
#   TAG=full_0905 ./submit_ladder.sh                      # full 400 items
#   TAG=full_0905 ONLY="instruct_sft instruct_dpo" ./submit_ladder.sh
set -euo pipefail
cd "$(dirname "$0")"
export TAG="${TAG:-$(date +%Y%m%d)_belief}"; export ITEMS="${ITEMS:-0}"
ALL7="base instruct_sft instruct_dpo instruct rl_zero think_sft think_dpo think"
ALL32="instruct_32b base_32b think_sft_32b think_dpo_32b think_32b"
echo "TAG=${TAG} ITEMS=${ITEMS}"
for v in ${ONLY:-$ALL7 $ALL32}; do
  case "$v" in
    *32b) J=$(VARIANT=$v sbatch --parsable --export=ALL,TAG,ITEMS,VARIANT --gpus-per-task=2 --constraint=a100 --mem=200G --job-name=AAM_BELIEF_$v job_belief_one.sh) ;;
    *)    J=$(VARIANT=$v sbatch --parsable --export=ALL,TAG,ITEMS,VARIANT --gpus-per-task=1 --constraint='a100|l40s|a40' --job-name=AAM_BELIEF_$v job_belief_one.sh) ;;
  esac
  printf "%-16s -> job %s\n" "$v" "$J"
done
echo "Outputs: /scratch1/mahdisae/olmo_experiments/belief_probe/${TAG}/   (bundle/ fills in as each job finishes)"
echo "When done:  TAG=${TAG} ./collect_bundle.sh"
