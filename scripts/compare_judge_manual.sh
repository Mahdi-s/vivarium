#!/usr/bin/env bash
# Compare agreement between LLM judge labels (parsed_answer_json) and manual labels
# (is_correct, refusal_flag) across all runs in runs_metadata_v6.json.
#
# Usage: ./scripts/compare_judge_manual.sh
# Or:    bash scripts/compare_judge_manual.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/compare_judge_manual_agreement.py \
  --runs-dir runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json \
  --out-dir Comparing_Experiments/judge_manual_agreement
