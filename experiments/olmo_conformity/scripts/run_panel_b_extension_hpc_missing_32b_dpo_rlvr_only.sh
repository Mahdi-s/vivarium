#!/bin/bash
# ---------------------------------------------------------------------------
# Submit ONLY 32B Panel B extension jobs that are often missing when a queue
# snapshot already has 7B Panel B + 32B SFT + full-5k:
#
#   Slurm names (#SBATCH --job-name):
#     AAM_32B_TD_PB  —  job_olmo_32b_think_dpo_panelB.sh
#     AAM_32B_TR_PB  —  job_olmo_32b_think_rlvr_panelB.sh
#
# Each job file is submitted at T=0.0 and T=0.6.
#
# If a Slurm name is already in squeue (pending/running) for this user, that
# whole variant is skipped. Override with SKIP_SQCHECK=1 (duplicates possible).
#
# Usage (repo root on HPC login node):
#   bash experiments/olmo_conformity/scripts/run_panel_b_extension_hpc_missing_32b_dpo_rlvr_only.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

JOB_DIR="experiments/olmo_conformity/configs"

declare -A SLURM_NAME=(
  ["job_olmo_32b_think_dpo_panelB.sh"]="AAM_32B_TD_PB"
  ["job_olmo_32b_think_rlvr_panelB.sh"]="AAM_32B_TR_PB"
)

MISSING_JOBS=(
  "job_olmo_32b_think_dpo_panelB.sh"
  "job_olmo_32b_think_rlvr_panelB.sh"
)

count_active_by_name() {
  local name="$1"
  squeue -u "${USER}" --name="${name}" -h 2>/dev/null | wc -l | tr -d ' '
}

declare -a SUBMITTED=()
declare -a SKIPPED=()

for job in "${MISSING_JOBS[@]}"; do
  name="${SLURM_NAME[$job]:-}"
  if [[ -z "${name}" ]]; then
    echo "ERROR: no Slurm name mapping for ${job}" >&2
    exit 1
  fi

  if [[ "${SKIP_SQCHECK:-0}" != "1" ]]; then
    n_active="$(count_active_by_name "${name}")"
    if [[ "${n_active}" -gt 0 ]]; then
      echo "SKIP ${job} (Slurm name=${name}): ${n_active} job(s) already queued/running"
      SKIPPED+=("${job} (${name} x${n_active})")
      continue
    fi
  fi

  for temp in 0.0 0.6; do
    echo "Submitting ${job} at TEMPERATURE=${temp}..."
    SLURM_ID=$(TEMPERATURE="${temp}" sbatch --parsable "${JOB_DIR}/${job}")
    echo "  -> ${SLURM_ID}"
    SUBMITTED+=("${SLURM_ID} ${job} T=${temp}")
  done
done

echo
echo "=== Submitted ${#SUBMITTED[@]} new job invocations ==="
for line in "${SUBMITTED[@]}"; do
  echo "  ${line}"
done
echo
echo "=== Skipped (already active) ${#SKIPPED[@]} variant(s) ==="
for line in "${SKIPPED[@]}"; do
  echo "  ${line}"
done
echo
echo "Track with: squeue -u \${USER} -o '%.10i %.20j %.8T %.10M %R'"
