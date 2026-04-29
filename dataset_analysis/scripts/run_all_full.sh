#!/usr/bin/env bash
# Full-corpus pipeline. Run AFTER scripts/download_full.py has finished.
#   cd dataset_analysis
#   export HF_TOKEN=hf_xxx
#   python scripts/download_full.py            # ~44 GB total, resume-safe
#   bash scripts/run_all_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# install the parquet reader once
python -c "import pyarrow" 2>/dev/null || pip install --break-system-packages pyarrow

echo "=== Phase 1: SFT audits (Instruct + Think) ==="
python scripts/phase1_sft_audit.py --short-name instruct-sft --tag instruct-sft
python scripts/phase1_sft_audit.py --short-name think-sft    --tag think-sft

echo "=== Phase 2: DPO audits (Instruct + Think) ==="
python scripts/phase2_dpo_audit.py --short-name instruct-dpo --tag instruct-dpo
python scripts/phase2_dpo_audit.py --short-name think-dpo    --tag think-dpo

echo "=== Phase 4: RL audits (Instruct + Think) ==="
python scripts/phase4_rl_audit.py --short-name instruct-rl
python scripts/phase4_rl_audit.py --short-name think-rl

echo "=== Phase 3: Think-buffer audit on local simulation.db ==="
python scripts/phase3_think_audit.py

echo "=== Phase 3b: dummy-buffer prompt construction (dry-run) ==="
python scripts/phase3b_dummy_buffer.py --dry-run

echo
echo "All outputs in results/:"
ls -1 results/
