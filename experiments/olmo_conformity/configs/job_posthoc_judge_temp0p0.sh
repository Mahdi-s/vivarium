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
#SBATCH --job-name=AAM_OLMO_JUDGE_T0P0

set -euo pipefail

cd /home1/mahdisae/aam/abstractAgentMachine
source /scratch1/mahdisae/aam_venv/bin/activate

export PYTHONNOUSERSITE=1

# Ensure repo sources are importable even if not pip-installed on the cluster.
export PYTHONPATH="/home1/mahdisae/aam/abstractAgentMachine/src:${PYTHONPATH:-}"

# Avoid matplotlib trying to write under $HOME on HPC (some analysis modules import matplotlib).
export MPLCONFIGDIR="/scratch1/mahdisae/olmo_experiments/mpl_cache/${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

TEMP_STR="0.0"
AAM_RUN_ID="${AAM_RUN_ID:-}"
if [ -z "${AAM_RUN_ID}" ]; then
  AAM_RUN_ID="$(
    python - <<'PY'
import json
import pathlib
import sys

temp = "0.0"
meta_path = pathlib.Path("Comparing_Experiments") / "runs_metadata.json"
if not meta_path.exists():
    print("", end="")
    sys.exit(0)
data = json.loads(meta_path.read_text(encoding="utf-8"))
entry = (data.get("experiments", {}).get(temp, {}) or {})
if entry.get("status") != "completed":
    print("", end="")
    sys.exit(0)
print(str(entry.get("run_id") or ""), end="")
PY
  )"
fi
if [ -z "${AAM_RUN_ID}" ]; then
  echo "ERROR: Could not resolve run_id for temperature ${TEMP_STR}. Ensure the run is completed and Comparing_Experiments/runs_metadata.json is populated." >&2
  exit 1
fi

python experiments/olmo_conformity/configs/run_llm_judge_posthoc.py \
  --run-id "${AAM_RUN_ID}" \
  --hpc \
  --max-concurrency 4 \
  --trial-scope behavioral-only
