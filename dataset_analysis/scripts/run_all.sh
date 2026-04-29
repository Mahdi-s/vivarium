#!/usr/bin/env bash
# End-to-end audit pipeline.
# Runs from the dataset_analysis/ directory.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/4] downloading Dolci samples ==="
python scripts/download_samples.py "$@"

echo "=== [2/4] phase 1: Dolci-Instruct-SFT ==="
python scripts/phase1_sft_audit.py

echo "=== [3/4] phase 2: Dolci-Instruct-DPO ==="
python scripts/phase2_dpo_audit.py

echo "=== [4/4] phase 3: Think buffer audit (offline, uses local sim dbs) ==="
python scripts/phase3_think_audit.py

echo "=== phase 3b dry-run (prompt construction check) ==="
python scripts/phase3b_dummy_buffer.py --dry-run

echo "Done. See dataset_analysis/results/"
ls -1 results/
