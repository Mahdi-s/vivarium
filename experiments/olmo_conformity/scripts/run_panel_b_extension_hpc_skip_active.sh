#!/bin/bash
# ---------------------------------------------------------------------------
# Same as run_panel_b_extension_hpc.sh, but skips a variant if ANY job with
# that Slurm job name is already pending or running.
#
# Because each variant uses one fixed --job-name for BOTH temperatures (T=0.0
# and T=0.6), you will see two rows with the same name in squeue when both are
# active — that is expected.
#
# Usage:
#   bash experiments/olmo_conformity/scripts/run_panel_b_extension_hpc_skip_active.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

JOB_DIR="experiments/olmo_conformity/configs"

# job_script -> Slurm job name (must match #SBATCH --job-name in each file)
declare -A SLURM_NAME=(
  ["job_olmo_7b_think_sft_panelB.sh"]="AAM_7B_TS_PB"
  ["job_olmo_7b_think_dpo_panelB.sh"]="AAM_7B_TD_PB"
  ["job_olmo_7b_think_rlvr_panelB.sh"]="AAM_7B_TR_PB"
  ["job_olmo_32b_think_sft_panelB.sh"]="AAM_32B_TS_PB"
  ["job_olmo_32b_think_dpo_panelB.sh"]="AAM_32B_TD_PB"
  ["job_olmo_32b_think_rlvr_panelB.sh"]="AAM_32B_TR_PB"
)

THINK_JOBS=(
  "job_olmo_7b_think_sft_panelB.sh"
  "job_olmo_7b_think_dpo_panelB.sh"
  "job_olmo_7b_think_rlvr_panelB.sh"
  "job_olmo_32b_think_sft_panelB.sh"
  "job_olmo_32b_think_dpo_panelB.sh"
  "job_olmo_32b_think_rlvr_panelB.sh"
)

count_active_by_name() {
  local name="$1"
  # Count lines with this job name (pending/running/completing)
  squeue -u "${USER}" --name="${name}" -h 2>/dev/null | wc -l | tr -d ' '
}

declare -a SUBMITTED=()
declare -a SKIPPED=()

for job in "${THINK_JOBS[@]}"; do
  name="${SLURM_NAME[$job]:-}"
  if [[ -z "${name}" ]]; then
    echo "ERROR: no Slurm name mapping for ${job}" >&2
    exit 1
  fi
  n_active="$(count_active_by_name "${name}")"
  if [[ "${n_active}" -gt 0 ]]; then
    echo "SKIP ${job} (Slurm name=${name}): ${n_active} job(s) already queued/running"
    SKIPPED+=("${job} (${name} x${n_active})")
    continue
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
