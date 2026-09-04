#!/bin/bash
# This local working copy is not a git checkout (.git is empty), so sync the new files to the HPC repo by rsync.
#   ./transfer.sh mahdisae@<login-node>
# Only small source/config files are copied (no data). Run from the repo root.
set -euo pipefail
HOST="${1:?usage: transfer.sh user@host}"
DEST="${DEST:-/home1/mahdisae/aam/abstractAgentMachine}"
cd "$(dirname "$0")/../../.."
rsync -avR --progress \
  investigation/backstudy/tools/belief_probe.py \
  investigation/backstudy/analysis_belief_probe.py \
  investigation/backstudy/analysis_policy_curve.py \
  experiments/olmo_conformity/hpc/ \
  experiments/olmo_conformity/configs/suite_7b_think_rerun8k_temp0p0.json \
  experiments/olmo_conformity/configs/suite_7b_think_sft_rerun8k_temp0p0.json \
  experiments/olmo_conformity/configs/suite_7b_think_dpo_rerun8k_temp0p0.json \
  Comparing_Experiments/publication_V2/item_set.csv \
  "${HOST}:${DEST}/"
echo "Now on the HPC:  cd ${DEST}/experiments/olmo_conformity/hpc && TAG=pilot ITEMS=10 ./submit_ladder.sh"
