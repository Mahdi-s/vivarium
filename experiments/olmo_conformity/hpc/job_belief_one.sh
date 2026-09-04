#!/bin/bash
#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --job-name=AAM_BELIEF
#SBATCH --output=/scratch1/mahdisae/olmo_experiments/slurm_logs/%x_%j.out
# ---------------------------------------------------------------------------
# ONE model per job, everything for that model inside one 48 h reservation:
#   1. belief-probe factorial (400 items x 32 conditions; batched generation; think models budget-forced)
#   2. answer-slot activation capture for probing (every 4th layer by default; CAPTURE_LAYERS=all for all)
#   3. in-job analysis: summary/contrasts + per-layer probes -> <TAG>/bundle/
# All by-products: /scratch1/mahdisae/olmo_experiments/belief_probe/<TAG>/
# Wall-clock guard: the tool stops cleanly at TIME_BUDGET_H (default 44 h) and exits 75; with AUTO_RESUBMIT=1
# (default) the job re-submits itself and resumes from the jsonl — nothing is recomputed.
#
# GPU: 7B variants need 1 GPU; 32B variants need 2 x A100 (sharded). Submit with the matching flags:
#   TAG=full_0905 VARIANT=instruct_dpo  sbatch --gpus-per-task=1 --constraint='a100|l40s|a40' job_belief_one.sh
#   TAG=full_0905 VARIANT=think_sft     sbatch --gpus-per-task=1 --constraint='a100|l40s|a40' job_belief_one.sh
#   TAG=full_0905 VARIANT=think_dpo_32b sbatch --gpus-per-task=2 --constraint=a100 --mem=200G job_belief_one.sh
# (submit_ladder.sh does this for every checkpoint.)
# Knobs: ITEMS (0=all 400), CONDITIONS, MAX_NEW_TOKENS (64), THINK_BUDGET (2048), BATCH (16), CAPTURE_LAYERS, DTYPE (bf16)
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p /scratch1/mahdisae/olmo_experiments/slurm_logs
source "$(dirname "$0")/common.sh"
VARIANT="${VARIANT:?set VARIANT=<checkpoint name>}"
MODEL="$(model_for_variant "${VARIANT}")"
case "${VARIANT}" in *32b) DEVICE="auto"; NGPU=2; RESUB_OPTS="--gpus-per-task=2 --constraint=a100 --mem=200G" ;; *) DEVICE="cuda"; NGPU=1; RESUB_OPTS="--gpus-per-task=1 --constraint=a100|l40s|a40" ;; esac
case "${VARIANT}" in think*) THINK_ARGS="--think-budget ${THINK_BUDGET:-2048}"; MAXNEW="${MAX_NEW_TOKENS:-64}" ;; *) THINK_ARGS=""; MAXNEW="${MAX_NEW_TOKENS:-64}" ;; esac
CAPTURE="${CAPTURE_LAYERS:-0,4,8,12,16,20,24,28,31,32}"
echo "=== ${VARIANT} (${MODEL}) | TAG=${TAG} | gpus=${NGPU} device=${DEVICE} | items=${ITEMS:-0} batch=${BATCH:-16} ${THINK_ARGS} | capture=${CAPTURE} | $(date) ==="

set +e
python investigation/backstudy/tools/belief_probe.py \
    --model-id "${MODEL}" --variant "${VARIANT}" \
    --items-per-dataset "${ITEMS:-0}" --conditions "${CONDITIONS:-all}" \
    --max-new-tokens "${MAXNEW}" ${THINK_ARGS} --batch-size "${BATCH:-16}" \
    --device "${DEVICE}" --dtype "${DTYPE:-bf16}" --seed "${SEED:-42}" \
    --capture-layers "${CAPTURE}" --time-budget-hours "${TIME_BUDGET_H:-44}" \
    --out-dir "${AAM_BELIEF_DIR}" --allow-download 2>&1 | tee -a "${AAM_BELIEF_DIR}/logs/${VARIANT}.log"
RC=${PIPESTATUS[0]}
set -e

if [ "${RC}" -eq 75 ]; then
  echo "=== time budget reached; rows are saved. AUTO_RESUBMIT=${AUTO_RESUBMIT:-1} ==="
  if [ "${AUTO_RESUBMIT:-1}" = "1" ]; then
    sbatch --dependency=afterany:${SLURM_JOB_ID} ${RESUB_OPTS} --job-name="${SLURM_JOB_NAME}" --export=ALL "$0"
    echo "re-submitted to resume ${VARIANT} under TAG=${TAG}"
  fi
  exit 0
elif [ "${RC}" -ne 0 ]; then
  echo "belief_probe failed with exit ${RC}" >&2; exit "${RC}"
fi

# ---- in-job analysis (CPU; the model is released) ----
B="${AAM_BELIEF_DIR}/bundle"; mkdir -p "${B}"
python investigation/backstudy/analysis_belief_probe.py --data-dir "${AAM_BELIEF_DIR}" --out-dir "${B}" || echo "summary analysis failed (non-fatal)"
python investigation/backstudy/probe_analysis.py --data-dir "${AAM_BELIEF_DIR}" --variant "${VARIANT}" --out-dir "${B}" || echo "probe analysis failed (non-fatal)"
# ---- causal validation: steer along the pressure direction (and a random-direction control) at the best probe layers
if [ "${STEER:-1}" = "1" ] && [ -f "${B}/directions_${VARIANT}.npz" ]; then
  python investigation/backstudy/tools/belief_probe.py --model-id "${MODEL}" --variant "${VARIANT}" \
      --items-per-dataset "${ITEMS:-0}" --conditions control --max-new-tokens 0 --device "${DEVICE}" --dtype "${DTYPE:-bf16}" \
      --out-dir "${AAM_BELIEF_DIR}" --allow-download --steer-from "${B}/directions_${VARIANT}.npz" \
      --steer-items "${STEER_ITEMS:-100}" --steer-alphas "${STEER_ALPHAS:--1,1}" --batch-size "${BATCH:-16}" \
      2>&1 | tee -a "${AAM_BELIEF_DIR}/logs/${VARIANT}.steer.log" || echo "steering pass failed (non-fatal)"
  python investigation/backstudy/steer_analysis.py --data-dir "${AAM_BELIEF_DIR}" --variant "${VARIANT}" --out-dir "${B}" || echo "steer analysis failed (non-fatal)"
fi
python - "${AAM_BELIEF_DIR}" "${B}" "${VARIANT}" <<'PY'
import sys, json, pandas as pd
from pathlib import Path
src, dst, v = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
df = pd.DataFrame([json.loads(l) for l in open(src / f"{v}.jsonl") if l.strip()])
d2 = df.drop(columns=[c for c in ["reasoning"] if c in df.columns])
try:
    d2.to_parquet(dst / f"{v}.parquet", index=False)
except Exception:
    d2.to_csv(dst / f"{v}.csv.gz", index=False, compression="gzip")
(dst / f"{v}.manifest.json").write_text((src / f"{v}.manifest.json").read_text())
print(f"bundle: {v}.parquet ({len(df)} rows)")
PY
echo "=== ${VARIANT} complete ($(date)) — by-products in ${AAM_BELIEF_DIR} ==="
