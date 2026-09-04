#!/bin/zsh
# Sequential local pilot: structural factorial × OLMo-3 7B Instruct-path checkpoints.
# Usage: investigation/backstudy/tools/run_pilot.sh <items_per_dataset> [conditions]
set -u
cd /Users/mahdi/repos/abstractAgentMachine
IPD=${1:-5}
CONDS=${2:-all}
OUT=investigation/backstudy/data/belief_probe
LOG=investigation/backstudy/logs
PY=.venv/bin/python
run() { # variant model_id
  echo "[$(date '+%H:%M:%S')] START $1 ($2)" >> $LOG/pilot_progress.log
  $PY investigation/backstudy/tools/belief_probe.py --model-id "$2" --variant "$1" --items-per-dataset $IPD --conditions "$CONDS" --max-new-tokens 16 --out-dir $OUT > $LOG/pilot_$1.log 2>&1
  echo "[$(date '+%H:%M:%S')] END   $1 exit=$?" >> $LOG/pilot_progress.log
}
run instruct_sft allenai/Olmo-3-7B-Instruct-SFT
run instruct     allenai/Olmo-3-7B-Instruct
run base         allenai/Olmo-3-1025-7B
# DPO: wait for download to finish (3 safetensors shards)
for i in {1..120}; do
  n=$(find ~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-DPO/snapshots -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge 3 ] && ! pgrep -f "hf download allenai/Olmo-3-7B-Instruct-DPO" > /dev/null; then break; fi
  sleep 60
done
run instruct_dpo allenai/Olmo-3-7B-Instruct-DPO
echo "[$(date '+%H:%M:%S')] PILOT COMPLETE" >> $LOG/pilot_progress.log
