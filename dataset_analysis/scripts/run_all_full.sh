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

echo "=== Phase 5: SFT consensus + run-length audit (Pillar I, Instruct) ==="
python scripts/phase5_consensus_audit.py --short-name instruct-sft --tag instruct-sft

echo "=== Phase 6: DPO chosen-vs-rejected effect-size audit (Pillar II, Instruct) ==="
python scripts/phase6_dpo_consensus_audit.py --short-name instruct-dpo --tag instruct-dpo

echo "=== Phase 7: corpus -> BER rank correlation (Pillar III) ==="
python scripts/phase7_correlation.py

echo "=== Phase 8: TF-IDF nearest-neighbor smoking-gun case studies ==="
python scripts/phase8_case_studies.py

echo "=== Phase 9: top-1% outlier extraction for manual qualitative review ==="
python scripts/phase9_outlier_verification.py

# Phase 10 (LLM-as-judge recall/precision bound) is intentionally NOT in
# run_all_full.sh — it requires API credentials or a local vLLM/Ollama
# server. Run it manually when those are available:
#   python scripts/phase10_llm_judge.py --judge gpt-4o-mini
# (Phase 10 was deferred per the audit plan; the regex hits in phase5/6
# are pre-registered as English-only lower bounds in the meantime.)

echo
echo "All outputs in results/:"
ls -1 results/
