#!/usr/bin/env bash
# Judge agreement report for one or all runs under runs_latest/runs.
# Prints richer diagnostics per dataset/condition, plus model-level summaries.
#
# Usage:
#   ./judge_report.sh
#   ./judge_report.sh <run_id>
#   ./judge_report.sh --all [runs_dir]
#   ./judge_report.sh --config <suite_config.json> [--all [runs_dir] | <run_id> [runs_dir]] [--include-refusal] [--detailed] [--per-run]
#                     [--ready-min-cov 95] [--ready-min-match 75] [--ready-min-n 1000]
#
#   # Run inventory (scan a folder and show model/temp/trials/missing-cells table):
#   ./judge_report.sh --inventory [runs_dir]
#   ./judge_report.sh --inventory runs --show-missing
#   ./judge_report.sh --inventory runs --sort model
#   ./judge_report.sh --inventory runs --filter-model llama

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_PATH="${REPO_ROOT}/runs_latest/runs"
# Default: full 7B expanded suite (8 datasets × 12 conditions). Override with --config if your run used another suite.
SUITE_CONFIG="${REPO_ROOT}/experiments/olmo_conformity/configs/suite_7b_expanded.json"
MODE="all"
RUN_ID=""
INCLUDE_REFUSAL=0
DETAILED=0
SHOW_PER_RUN=0
READY_MIN_COV=95.0
READY_MIN_MATCH=75.0
READY_MIN_N=1000

# Inventory-mode options
INVENTORY_MODE=0
INVENTORY_DIR=""
INVENTORY_EXTRA_ARGS=()

usage() {
  echo "Usage:"
  echo "  ./judge_report.sh"
  echo "  ./judge_report.sh <run_id> [runs_dir]"
  echo "  ./judge_report.sh --all [runs_dir]"
  echo "  ./judge_report.sh --config <suite_config.json> [--all [runs_dir] | <run_id> [runs_dir]] [--include-refusal] [--detailed] [--per-run]"
  echo "                    [--ready-min-cov 95] [--ready-min-match 75] [--ready-min-n 1000]"
  echo ""
  echo "  # Run inventory — scan a folder and print a model/temp/trials/missing-cells table:"
  echo "  ./judge_report.sh --inventory [runs_dir]   (default dir: runs)"
  echo "  ./judge_report.sh --inventory runs --show-missing"
  echo "  ./judge_report.sh --inventory runs --sort {id|model|temp|trials}"
  echo "  ./judge_report.sh --inventory runs --filter-model REGEX"
}

resolve_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    echo "$raw"
  else
    echo "${REPO_ROOT}/${raw}"
  fi
}

args=("$@")
i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --config)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --config expects a path"; usage; exit 1; }
      SUITE_CONFIG="$(resolve_path "${args[$i]}")"
      ;;
    --all)
      MODE="all"
      if [[ $((i + 1)) -lt $# && "${args[$((i + 1))]}" != --* ]]; then
        i=$((i + 1))
        RUNS_PATH="$(resolve_path "${args[$i]}")"
      fi
      ;;
    --include-refusal)
      INCLUDE_REFUSAL=1
      ;;
    --detailed)
      DETAILED=1
      ;;
    --per-run)
      SHOW_PER_RUN=1
      ;;
    --ready-min-cov)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --ready-min-cov expects a number"; usage; exit 1; }
      READY_MIN_COV="${args[$i]}"
      ;;
    --ready-min-match)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --ready-min-match expects a number"; usage; exit 1; }
      READY_MIN_MATCH="${args[$i]}"
      ;;
    --ready-min-n)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --ready-min-n expects an integer"; usage; exit 1; }
      READY_MIN_N="${args[$i]}"
      ;;
    --inventory)
      INVENTORY_MODE=1
      if [[ $((i + 1)) -lt $# && "${args[$((i + 1))]}" != --* ]]; then
        i=$((i + 1))
        INVENTORY_DIR="$(resolve_path "${args[$i]}")"
      fi
      ;;
    --show-missing)
      INVENTORY_EXTRA_ARGS+=(--show-missing)
      ;;
    --sort)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --sort expects {id|model|temp|trials}"; usage; exit 1; }
      INVENTORY_EXTRA_ARGS+=(--sort "${args[$i]}")
      ;;
    --filter-model)
      i=$((i + 1))
      [[ $i -lt $# ]] || { echo "Error: --filter-model expects a regex"; usage; exit 1; }
      INVENTORY_EXTRA_ARGS+=(--filter-model "${args[$i]}")
      ;;
    -*)
      echo "Error: unknown option: $arg"
      usage
      exit 1
      ;;
    *)
      if [[ -z "$RUN_ID" ]]; then
        MODE="single"
        RUN_ID="$arg"
      elif [[ "$RUNS_PATH" == "${REPO_ROOT}/runs_latest/runs" ]]; then
        RUNS_PATH="$(resolve_path "$arg")"
      else
        echo "Error: unexpected argument: $arg"
        usage
        exit 1
      fi
      ;;
  esac
  i=$((i + 1))
done

# ── Inventory mode: delegate entirely to run_inventory.py ───────────────────
if [[ "$INVENTORY_MODE" -eq 1 ]]; then
  INVENTORY_SCRIPT="${REPO_ROOT}/scripts/run_inventory.py"
  if [[ ! -f "$INVENTORY_SCRIPT" ]]; then
    echo "Error: run_inventory.py not found at ${INVENTORY_SCRIPT}"
    exit 1
  fi
  if [[ -z "$INVENTORY_DIR" ]]; then
    INVENTORY_DIR="${REPO_ROOT}/runs"
  fi
  exec python3 "$INVENTORY_SCRIPT" --runs-dir "$INVENTORY_DIR" "${INVENTORY_EXTRA_ARGS[@]+"${INVENTORY_EXTRA_ARGS[@]}"}"
fi
# ── End inventory mode ──────────────────────────────────────────────────────

if [[ ! -f "$SUITE_CONFIG" ]]; then
  echo "Error: suite config not found: $SUITE_CONFIG"
  exit 1
fi

if [[ ! -d "$RUNS_PATH" ]]; then
  echo "Error: runs directory not found: $RUNS_PATH"
  exit 1
fi

report_one() {
  local run_id="$1"
  local db="$2"

  python - "$db" "$run_id" "$SUITE_CONFIG" "$INCLUDE_REFUSAL" "$DETAILED" "$READY_MIN_COV" "$READY_MIN_MATCH" "$READY_MIN_N" <<'PY'
import json
import os
import sqlite3
import sys

db_path, run_id, suite_config_path, include_refusal_raw, detailed_raw, ready_min_cov_raw, ready_min_match_raw, ready_min_n_raw = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
)
include_refusal = include_refusal_raw == "1"
detailed = detailed_raw == "1"
ready_min_cov = float(ready_min_cov_raw) / 100.0
ready_min_match = float(ready_min_match_raw) / 100.0
ready_min_n = int(float(ready_min_n_raw))

with open(suite_config_path, "r", encoding="utf-8") as f:
    suite = json.load(f)

dataset_names = [d["name"] for d in suite.get("datasets", [])]
condition_names = [c["name"] for c in suite.get("conditions", [])]

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

ansi = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RESET = "\033[0m" if ansi else ""
BOLD = "\033[1m" if ansi else ""
CYAN = "\033[36m" if ansi else ""
MAGENTA = "\033[35m" if ansi else ""
GREEN = "\033[32m" if ansi else ""
YELLOW = "\033[33m" if ansi else ""

def paint(text, color):
    return f"{color}{text}{RESET}" if ansi else text

def make_table(headers, rows, right_align_cols=None):
    right_align_cols = set(right_align_cols or [])
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def border(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    out = [border("-")]
    header_cells = []
    for idx, h in enumerate(headers):
        if idx in right_align_cols:
            header_cells.append(f" {h:>{widths[idx]}} ")
        else:
            header_cells.append(f" {h:<{widths[idx]}} ")
    out.append("|" + "|".join(header_cells) + "|")
    out.append(border("="))
    for row in rows:
        cells = []
        for idx, cell in enumerate(row):
            cell = str(cell)
            if idx in right_align_cols:
                cells.append(f" {cell:>{widths[idx]}} ")
            else:
                cells.append(f" {cell:<{widths[idx]}} ")
        out.append("|" + "|".join(cells) + "|")
    out.append(border("-"))
    return "\n".join(out)

def pct(num, den):
    if den <= 0:
        return "n/a"
    return f"{(100.0 * num / den):.1f}%"

def gate_status(judge_cov, match_rate, compared_n, min_cov, min_match, min_n):
    if judge_cov is None or match_rate is None:
        return "AT_RISK"
    if judge_cov < min_cov:
        return "AT_RISK"
    if match_rate < min_match:
        return "AT_RISK"
    if compared_n < min_n:
        return "AT_RISK"
    return "READY"

def gate_status(judge_cov, match_rate, compared_n, min_cov, min_match, min_n):
    if judge_cov is None or match_rate is None:
        return "AT_RISK"
    if judge_cov < min_cov:
        return "AT_RISK"
    if match_rate < min_match:
        return "AT_RISK"
    if compared_n < min_n:
        return "AT_RISK"
    return "READY"

def safe_div(num, den):
    if den <= 0:
        return None
    return num / den

def fmt_pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"

def fmt_ratio(value):
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"

def gate_status(judge_cov, match_rate, compared_n, min_cov, min_match, min_n):
    if judge_cov is None or match_rate is None:
        return "AT_RISK"
    if judge_cov < min_cov:
        return "AT_RISK"
    if match_rate < min_match:
        return "AT_RISK"
    if compared_n < min_n:
        return "AT_RISK"
    return "READY"

def confusion_metrics(tp, tn, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced_acc = None
    if recall is not None and specificity is not None:
        balanced_acc = 0.5 * (recall + specificity)
    return precision, recall, specificity, f1, balanced_acc

overall = cur.execute(
    "SELECT COUNT(*) AS n FROM conformity_trials WHERE run_id = ?",
    (run_id,),
).fetchone()["n"]

print()
print(paint("=" * 96, MAGENTA))
print(paint(f"{BOLD}Judge Agreement Report{RESET} | run_id={run_id}", CYAN))
print(paint(f"DB: {db_path}", YELLOW))
print(paint(f"Suite: {suite.get('suite_name', 'unknown')} | configured_conditions={len(condition_names)} | configured_datasets={len(dataset_names)}", YELLOW))
print(paint(f"Total trials for run: {overall}", GREEN))
print(paint("=" * 96, MAGENTA))

dataset_query = """
WITH trial_base AS (
  SELECT
    t.trial_id,
    t.model_id,
    t.variant,
    c.name AS condition_name,
    d.name AS dataset_name,
    o.output_id,
    o.parsed_answer_json,
    o.is_correct AS manual_is_correct,
    o.refusal_flag AS manual_refusal_flag
  FROM conformity_trials t
  JOIN conformity_conditions c ON c.condition_id = t.condition_id
  JOIN conformity_items i ON i.item_id = t.item_id
  JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE t.run_id = ? AND d.name = ?
)
SELECT
  condition_name,
  COUNT(*) AS total_trials,
  SUM(CASE WHEN output_id IS NULL THEN 1 ELSE 0 END) AS missing_outputs,
  SUM(CASE WHEN parsed_answer_json IS NULL OR trim(parsed_answer_json) = '' THEN 1 ELSE 0 END) AS empty_rows,
  SUM(CASE WHEN parsed_answer_json LIKE '%[parse_error]%' THEN 1 ELSE 0 END) AS parse_errors,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL THEN 1 ELSE 0 END) AS judge_valid_rows,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL THEN 1 ELSE 0 END) AS comparable_rows,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) THEN 1 ELSE 0 END) AS matches,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) != CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) THEN 1 ELSE 0 END) AS mismatches,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 1 THEN 1 ELSE 0 END) AS tp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 0 THEN 1 ELSE 0 END) AS tn,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 1 THEN 1 ELSE 0 END) AS fp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 0 THEN 1 ELSE 0 END) AS fn,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 1 THEN 1 ELSE 0 END) AS manual_pos,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 0 THEN 1 ELSE 0 END) AS manual_neg,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 1 THEN 1 ELSE 0 END) AS judge_pos,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 0 THEN 1 ELSE 0 END) AS judge_neg,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL THEN 1 ELSE 0 END) AS refusal_valid_rows,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL THEN 1 ELSE 0 END) AS refusal_comparable_rows,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) THEN 1 ELSE 0 END) AS refusal_matches,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) != CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) THEN 1 ELSE 0 END) AS refusal_mismatches,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) = 1 THEN 1 ELSE 0 END) AS refusal_tp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) = 0 THEN 1 ELSE 0 END) AS refusal_tn,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) = 1 THEN 1 ELSE 0 END) AS refusal_fp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) = 0 THEN 1 ELSE 0 END) AS refusal_fn
FROM trial_base
GROUP BY condition_name
"""

CORE_CONDITIONS = {"control", "asch_history_5", "authoritative_bias"}
overall_valid = 0
overall_compared = 0
overall_match = 0
overall_trials = 0
overall_ref_compared = 0
overall_ref_match = 0
overall_tp = 0
overall_tn = 0
overall_fp = 0
overall_fn = 0
readiness_rows = []

for dataset_name in dataset_names:
    rows = cur.execute(dataset_query, (run_id, dataset_name)).fetchall()
    by_condition = {row["condition_name"]: row for row in rows}
    coverage_rows = []
    is_rows = []
    refusal_rows = []
    compact_rows = []
    risk_rows = []
    ds_total = ds_valid = ds_compared = ds_match = ds_mismatch = ds_empty = ds_parse = ds_missing = 0
    for condition_name in condition_names:
        row = by_condition.get(condition_name)
        if row is None:
            total_trials = missing_outputs = empty_rows = parse_errors = 0
            judge_valid_rows = comparable_rows = matches = mismatches = 0
            tp = tn = fp = fn = 0
            manual_pos = manual_neg = judge_pos = judge_neg = 0
            refusal_comparable_rows = refusal_matches = refusal_mismatches = 0
            refusal_tp = refusal_tn = refusal_fp = refusal_fn = 0
        else:
            total_trials = int(row["total_trials"] or 0)
            missing_outputs = int(row["missing_outputs"] or 0)
            empty_rows = int(row["empty_rows"] or 0)
            parse_errors = int(row["parse_errors"] or 0)
            judge_valid_rows = int(row["judge_valid_rows"] or 0)
            comparable_rows = int(row["comparable_rows"] or 0)
            matches = int(row["matches"] or 0)
            mismatches = int(row["mismatches"] or 0)
            tp = int(row["tp"] or 0)
            tn = int(row["tn"] or 0)
            fp = int(row["fp"] or 0)
            fn = int(row["fn"] or 0)
            manual_pos = int(row["manual_pos"] or 0)
            manual_neg = int(row["manual_neg"] or 0)
            judge_pos = int(row["judge_pos"] or 0)
            judge_neg = int(row["judge_neg"] or 0)
            refusal_comparable_rows = int(row["refusal_comparable_rows"] or 0)
            refusal_matches = int(row["refusal_matches"] or 0)
            refusal_mismatches = int(row["refusal_mismatches"] or 0)
            refusal_tp = int(row["refusal_tp"] or 0)
            refusal_tn = int(row["refusal_tn"] or 0)
            refusal_fp = int(row["refusal_fp"] or 0)
            refusal_fn = int(row["refusal_fn"] or 0)
        ds_total += total_trials
        ds_valid += judge_valid_rows
        ds_compared += comparable_rows
        ds_match += matches
        ds_mismatch += mismatches
        ds_empty += empty_rows
        ds_parse += parse_errors
        ds_missing += missing_outputs
        overall_trials += total_trials
        overall_valid += judge_valid_rows
        overall_compared += comparable_rows
        overall_match += matches
        overall_ref_compared += refusal_comparable_rows
        overall_ref_match += refusal_matches
        overall_tp += tp
        overall_tn += tn
        overall_fp += fp
        overall_fn += fn
        p, r, s, f1, b = confusion_metrics(tp, tn, fp, fn)
        coverage_rows.append([condition_name, total_trials, judge_valid_rows, comparable_rows, pct(judge_valid_rows, total_trials), pct(comparable_rows, total_trials), matches, empty_rows, parse_errors, missing_outputs])
        is_rows.append([condition_name, comparable_rows, matches, mismatches, pct(matches, comparable_rows), tp, tn, fp, fn, manual_pos, manual_neg, judge_pos, judge_neg, fmt_pct(p), fmt_pct(r), fmt_pct(s), fmt_pct(f1), fmt_pct(b)])
        compact_row = [
            condition_name,
            total_trials,
            judge_valid_rows,
            comparable_rows,
            pct(judge_valid_rows, total_trials),
            pct(matches, comparable_rows),
            mismatches,
            empty_rows,
            parse_errors,
            missing_outputs,
        ]
        if include_refusal:
            compact_row.append(pct(refusal_matches, refusal_comparable_rows))
        compact_rows.append(compact_row)
        risk_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "compared": comparable_rows,
            "match_rate": (matches / comparable_rows) if comparable_rows > 0 else None,
            "judge_cov": (judge_valid_rows / total_trials) if total_trials > 0 else None,
            "mismatches": mismatches,
        })
        readiness_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "trials": total_trials,
            "judge_valid": judge_valid_rows,
            "compared": comparable_rows,
            "match": matches,
            "mismatch": mismatches,
            "judge_cov": (judge_valid_rows / total_trials) if total_trials > 0 else None,
            "match_rate": (matches / comparable_rows) if comparable_rows > 0 else None,
        })
        if include_refusal:
            rp, rr, rs, rf1, rb = confusion_metrics(refusal_tp, refusal_tn, refusal_fp, refusal_fn)
            refusal_rows.append([condition_name, refusal_comparable_rows, refusal_matches, refusal_mismatches, pct(refusal_matches, refusal_comparable_rows), refusal_tp, refusal_tn, refusal_fp, refusal_fn, fmt_pct(rp), fmt_pct(rr), fmt_pct(rs), fmt_pct(rf1), fmt_pct(rb)])
    print()
    print(paint(f"{BOLD}Dataset: {dataset_name}{RESET}", CYAN))
    print(paint(f"  trials={ds_total} | judge_valid={ds_valid} | compared={ds_compared} | match={ds_match} | mismatch={ds_mismatch} | empty={ds_empty} | parse_error={ds_parse} | missing_output={ds_missing}", GREEN))
    if detailed:
        print(paint("  Coverage", YELLOW))
        print(make_table(["condition", "trials", "judge_valid", "compared", "judge_cov", "compare_cov", "match", "empty", "parse_err", "missing"], coverage_rows, right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9}))
        print(paint("  is_correct agreement", YELLOW))
        print(make_table(["condition", "compared", "match", "mismatch", "match_rate", "tp", "tn", "fp", "fn", "manual_pos", "manual_neg", "judge_pos", "judge_neg", "precision", "recall", "specificity", "f1", "bal_acc"], is_rows, right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}))
        if include_refusal:
            print(paint("  refusal_flag agreement", YELLOW))
            print(make_table(["condition", "compared", "match", "mismatch", "match_rate", "tp", "tn", "fp", "fn", "precision", "recall", "specificity", "f1", "bal_acc"], refusal_rows, right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}))
    else:
        compact_headers = ["condition", "trials", "judge_valid", "compared", "judge_cov", "is_match_rate", "mismatch", "empty", "parse_err", "missing"]
        if include_refusal:
            compact_headers.append("ref_match_rate")
        print(make_table(compact_headers, compact_rows, right_align_cols=set(range(1, len(compact_headers)))))
        ranked = [r for r in risk_rows if r["compared"] > 0 and r["match_rate"] is not None]
        if ranked:
            ranked.sort(key=lambda x: (x["match_rate"], -x["mismatches"]))
            worst = ranked[0]
            print(
                paint(
                    f"  weakest_condition={worst['condition']} | compared={worst['compared']} | is_match_rate={fmt_ratio(worst['match_rate'])} | judge_cov={fmt_ratio(worst['judge_cov'])}",
                    YELLOW,
                )
            )

overall_judge_cov = safe_div(overall_valid, overall_trials)
overall_match_rate = safe_div(overall_match, overall_compared)
overall_ref_match_rate = safe_div(overall_ref_match, overall_ref_compared)
overall_gate = gate_status(
    overall_judge_cov,
    overall_match_rate,
    overall_compared,
    ready_min_cov,
    ready_min_match,
    ready_min_n,
)

print()
print(paint(f"{BOLD}Readiness Gate Scorecard{RESET}", CYAN))
scorecard_rows = [[
    "run",
    overall_trials,
    overall_valid,
    overall_compared,
    fmt_ratio(overall_judge_cov),
    fmt_ratio(safe_div(overall_compared, overall_trials)),
    fmt_ratio(overall_match_rate),
    overall_gate,
]]
if include_refusal:
    scorecard_rows[0].insert(7, fmt_ratio(overall_ref_match_rate))
scorecard_headers = [
    "scope",
    "trials",
    "judge_valid",
    "compared",
    "judge_cov",
    "compare_cov",
    "is_match_rate",
    "gate_status",
]
if include_refusal:
    scorecard_headers = [
        "scope",
        "trials",
        "judge_valid",
        "compared",
        "judge_cov",
        "compare_cov",
        "is_match_rate",
        "ref_match_rate",
        "gate_status",
    ]
print(make_table(scorecard_headers, scorecard_rows, right_align_cols=set(range(1, len(scorecard_headers) - 1))))
print(paint(f"  gates: judge_cov>={ready_min_cov*100:.1f}% | is_match_rate>={ready_min_match*100:.1f}% | compared>={ready_min_n}", YELLOW))

print()
print(paint(f"{BOLD}Dataset x Condition Readiness{RESET}", CYAN))
cell_rows = []
for r in readiness_rows:
    status = gate_status(r["judge_cov"], r["match_rate"], r["compared"], ready_min_cov, ready_min_match, max(10, ready_min_n // 12))
    cell_rows.append([
        status,
        r["dataset"],
        r["condition"],
        r["trials"],
        r["judge_valid"],
        r["compared"],
        fmt_ratio(r["judge_cov"]),
        fmt_ratio(r["match_rate"]),
        r["mismatch"],
    ])
cell_rows.sort(key=lambda x: (0 if x[0] == "AT_RISK" else 1, float(x[7].strip("%")) if x[7] != "n/a" else 999.0))
print(make_table(
    ["flag", "dataset", "condition", "trials", "judge_valid", "compared", "judge_cov", "is_match_rate", "mismatch"],
    cell_rows[:24] if not detailed else cell_rows,
    right_align_cols={3, 4, 5, 6, 7, 8},
))

print()
print(paint(f"{BOLD}LLM-judged cells (dataset × condition){RESET}", CYAN))
print(paint("  llm_judge_valid = trials with non-empty parsed_answer_json, no [parse_error], json $.is_correct present (same as judge_valid elsewhere).", YELLOW))
llm_cell_rows = []
for r in readiness_rows:
    tr = int(r["trials"] or 0)
    jv = int(r["judge_valid"] or 0)
    if tr <= 0:
        st = "EMPTY"
    elif jv >= tr:
        st = "FULL"
    elif jv > 0:
        st = "PARTIAL"
    else:
        st = "NONE"
    llm_cell_rows.append([r["dataset"], r["condition"], tr, jv, pct(jv, tr), st])
llm_cell_rows.sort(key=lambda x: (x[0], x[1]))
print(make_table(
    ["dataset", "condition", "trials", "llm_judge_valid", "judge_cov", "cell_status"],
    llm_cell_rows,
    right_align_cols={2, 3, 4},
))
full_cells = sum(1 for row in llm_cell_rows if row[5] == "FULL")
part_cells = sum(1 for row in llm_cell_rows if row[5] == "PARTIAL")
none_cells = sum(1 for row in llm_cell_rows if row[5] == "NONE")
print(
    paint(
        f"  summary: FULL={full_cells} PARTIAL={part_cells} NONE={none_cells} (of {len(llm_cell_rows)} cells; FULL => all trials in cell have valid LLM is_correct)",
        GREEN,
    )
)

print()
print(paint(f"{BOLD}Manual-vs-Judge Drift Breakdown{RESET}", CYAN))
precision, recall, _, _, _ = confusion_metrics(overall_tp, overall_tn, overall_fp, overall_fn)
drift_total = overall_tp + overall_tn + overall_fp + overall_fn
drift_rows = [[
    overall_tp,
    overall_tn,
    overall_fp,
    overall_fn,
    pct(overall_tp, drift_total),
    pct(overall_tn, drift_total),
    pct(overall_fp, drift_total),
    pct(overall_fn, drift_total),
    fmt_pct(precision),
    fmt_pct(recall),
]]
print(make_table(
    ["tp", "tn", "fp", "fn", "tp_rate", "tn_rate", "fp_rate", "fn_rate", "precision", "recall"],
    drift_rows,
    right_align_cols=set(range(10)),
))
if overall_fp > overall_fn:
    print(paint("  drift_signal=FP-heavy (judge tends to mark correct when manual marks incorrect).", YELLOW))
elif overall_fn > overall_fp:
    print(paint("  drift_signal=FN-heavy (judge tends to mark incorrect when manual marks correct).", YELLOW))
else:
    print(paint("  drift_signal=balanced FP/FN.", YELLOW))

model_query = """
WITH trial_base AS (
  SELECT
    t.model_id,
    t.variant,
    o.parsed_answer_json,
    o.is_correct AS manual_is_correct,
    o.refusal_flag AS manual_refusal_flag
  FROM conformity_trials t
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE t.run_id = ?
)
SELECT
  variant,
  model_id,
  COUNT(*) AS trials_total,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL THEN 1 ELSE 0 END) AS judge_validated,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL THEN 1 ELSE 0 END) AS is_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) THEN 1 ELSE 0 END) AS is_match,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL THEN 1 ELSE 0 END) AS ref_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) THEN 1 ELSE 0 END) AS ref_match
FROM trial_base
GROUP BY variant, model_id
ORDER BY variant, model_id
"""
model_rows = []
for row in cur.execute(model_query, (run_id,)).fetchall():
    base = [row["variant"], row["model_id"], int(row["trials_total"] or 0), int(row["judge_validated"] or 0), pct(int(row["judge_validated"] or 0), int(row["trials_total"] or 0)), int(row["is_compared"] or 0), int(row["is_match"] or 0), pct(int(row["is_match"] or 0), int(row["is_compared"] or 0))]
    if include_refusal:
        base.extend([int(row["ref_compared"] or 0), int(row["ref_match"] or 0), pct(int(row["ref_match"] or 0), int(row["ref_compared"] or 0))])
    model_rows.append(base)
print()
print(paint(f"{BOLD}Model Summary{RESET}", CYAN))
headers = ["variant", "model_id", "trials_total", "judge_validated", "judge_cov", "is_compared", "is_match", "is_match_rate"]
if include_refusal:
    headers.extend(["ref_compared", "ref_match", "ref_match_rate"])
print(make_table(headers, model_rows, right_align_cols=set(range(2, len(headers)))))

model_condition_query = """
WITH trial_base AS (
  SELECT
    t.variant,
    t.model_id,
    c.name AS condition_name,
    o.parsed_answer_json
  FROM conformity_trials t
  JOIN conformity_conditions c ON c.condition_id = t.condition_id
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE t.run_id = ?
)
SELECT
  variant,
  model_id,
  condition_name,
  COUNT(*) AS n_expected,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' THEN 1 ELSE 0 END) AS n_judge
FROM trial_base
GROUP BY variant, model_id, condition_name
ORDER BY variant, model_id, condition_name
"""
model_condition_rows = cur.execute(model_condition_query, (run_id,)).fetchall()
gates_by_model = {}
for row in model_condition_rows:
    key = (row["variant"], row["model_id"])
    if key not in gates_by_model:
        gates_by_model[key] = {
            "n_expected_3": 0,
            "n_judge_3": 0,
            "n_expected_12": 0,
            "n_judge_12": 0,
        }
    g = gates_by_model[key]
    cond = row["condition_name"]
    n_exp = int(row["n_expected"] or 0)
    n_jdg = int(row["n_judge"] or 0)
    g["n_expected_12"] += n_exp
    g["n_judge_12"] += n_jdg
    if cond in CORE_CONDITIONS:
        g["n_expected_3"] += n_exp
        g["n_judge_3"] += n_jdg

gate_rows = []
for key in sorted(gates_by_model.keys()):
    variant, model_id = key
    g = gates_by_model[key]
    cov3 = safe_div(g["n_judge_3"], g["n_expected_3"])
    cov12 = safe_div(g["n_judge_12"], g["n_expected_12"])
    gate_rows.append([
        variant,
        model_id,
        g["n_expected_3"],
        g["n_judge_3"],
        fmt_ratio(cov3),
        "READY" if (cov3 is not None and cov3 >= ready_min_cov) else "AT_RISK",
        g["n_expected_12"],
        g["n_judge_12"],
        fmt_ratio(cov12),
        "READY" if (cov12 is not None and cov12 >= ready_min_cov) else "AT_RISK",
    ])
print()
print(paint(f"{BOLD}Publication-Gating Coverage{RESET}", CYAN))
print(make_table(
    ["variant", "model_id", "n_expected_3", "n_judge_3", "cov_3cond", "gate_3cond", "n_expected_12", "n_judge_12", "cov_12cond", "gate_12cond"],
    gate_rows,
    right_align_cols={2, 3, 4, 6, 7, 8},
))

conn.close()
PY
}

report_pooled_all() {
  python - "$RUNS_PATH" "$SUITE_CONFIG" "$INCLUDE_REFUSAL" "$DETAILED" "$READY_MIN_COV" "$READY_MIN_MATCH" "$READY_MIN_N" <<'PY'
import json
import os
import sqlite3
import sys

runs_path, suite_config_path, include_refusal_raw, detailed_raw, ready_min_cov_raw, ready_min_match_raw, ready_min_n_raw = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
)
include_refusal = include_refusal_raw == "1"
detailed = detailed_raw == "1"
ready_min_cov = float(ready_min_cov_raw) / 100.0
ready_min_match = float(ready_min_match_raw) / 100.0
ready_min_n = int(float(ready_min_n_raw))

with open(suite_config_path, "r", encoding="utf-8") as f:
    suite = json.load(f)
dataset_names = [d["name"] for d in suite.get("datasets", [])]
condition_names = [c["name"] for c in suite.get("conditions", [])]

ansi = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RESET = "\033[0m" if ansi else ""
BOLD = "\033[1m" if ansi else ""
CYAN = "\033[36m" if ansi else ""
MAGENTA = "\033[35m" if ansi else ""
YELLOW = "\033[33m" if ansi else ""
GREEN = "\033[32m" if ansi else ""

def paint(text, color):
    return f"{color}{text}{RESET}" if ansi else text

def make_table(headers, rows, right_align_cols=None):
    right_align_cols = set(right_align_cols or [])
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    def border(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"
    out = [border("-")]
    out.append("|" + "|".join((f" {h:>{widths[idx]}} " if idx in right_align_cols else f" {h:<{widths[idx]}} ") for idx, h in enumerate(headers)) + "|")
    out.append(border("="))
    for row in rows:
        out.append("|" + "|".join((f" {str(cell):>{widths[idx]}} " if idx in right_align_cols else f" {str(cell):<{widths[idx]}} ") for idx, cell in enumerate(row)) + "|")
    out.append(border("-"))
    return "\n".join(out)

def pct(num, den):
    if den <= 0:
        return "n/a"
    return f"{(100.0 * num / den):.1f}%"

def gate_status(judge_cov, match_rate, compared_n, min_cov, min_match, min_n):
    if judge_cov is None or match_rate is None:
        return "AT_RISK"
    if judge_cov < min_cov:
        return "AT_RISK"
    if match_rate < min_match:
        return "AT_RISK"
    if compared_n < min_n:
        return "AT_RISK"
    return "READY"

condition_query = """
WITH trial_base AS (
  SELECT
    t.variant,
    t.model_id,
    c.name AS condition_name,
    d.name AS dataset_name,
    o.output_id,
    o.parsed_answer_json,
    o.is_correct AS manual_is_correct,
    o.refusal_flag AS manual_refusal_flag
  FROM conformity_trials t
  JOIN conformity_conditions c ON c.condition_id = t.condition_id
  JOIN conformity_items i ON i.item_id = t.item_id
  JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE t.run_id = ?
)
SELECT
  dataset_name,
  condition_name,
  COUNT(*) AS trials_total,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL THEN 1 ELSE 0 END) AS judge_validated,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL THEN 1 ELSE 0 END) AS is_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) THEN 1 ELSE 0 END) AS is_match,
  SUM(CASE WHEN output_id IS NULL THEN 1 ELSE 0 END) AS missing_outputs,
  SUM(CASE WHEN parsed_answer_json IS NULL OR trim(parsed_answer_json) = '' THEN 1 ELSE 0 END) AS empty_rows,
  SUM(CASE WHEN parsed_answer_json LIKE '%[parse_error]%' THEN 1 ELSE 0 END) AS parse_errors,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 1 THEN 1 ELSE 0 END) AS tp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 0 THEN 1 ELSE 0 END) AS tn,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 0 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 1 THEN 1 ELSE 0 END) AS fp,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = 1 AND CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) = 0 THEN 1 ELSE 0 END) AS fn,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL THEN 1 ELSE 0 END) AS ref_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) THEN 1 ELSE 0 END) AS ref_match
FROM trial_base
GROUP BY dataset_name, condition_name
"""

model_query = """
WITH trial_base AS (
  SELECT
    t.variant,
    t.model_id,
    o.parsed_answer_json,
    o.is_correct AS manual_is_correct,
    o.refusal_flag AS manual_refusal_flag
  FROM conformity_trials t
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE t.run_id = ?
)
SELECT
  variant,
  model_id,
  COUNT(*) AS trials_total,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL THEN 1 ELSE 0 END) AS judge_validated,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL THEN 1 ELSE 0 END) AS is_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL AND manual_is_correct IS NOT NULL AND CAST(manual_is_correct AS INT) = CAST(json_extract(parsed_answer_json, '$.is_correct') AS INT) THEN 1 ELSE 0 END) AS is_match,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL THEN 1 ELSE 0 END) AS ref_compared,
  SUM(CASE WHEN parsed_answer_json IS NOT NULL AND trim(parsed_answer_json) != '' AND parsed_answer_json NOT LIKE '%[parse_error]%' AND json_extract(parsed_answer_json, '$.refusal_flag') IS NOT NULL AND manual_refusal_flag IS NOT NULL AND CAST(manual_refusal_flag AS INT) = CAST(json_extract(parsed_answer_json, '$.refusal_flag') AS INT) THEN 1 ELSE 0 END) AS ref_match
FROM trial_base
GROUP BY variant, model_id
ORDER BY variant, model_id
"""

dataset_condition_totals = {}
model_totals = {}
run_dataset_metrics = {}
model_condition_totals = {}
run_count = 0

for dirname in sorted(os.listdir(runs_path)):
    run_dir = os.path.join(runs_path, dirname)
    if not os.path.isdir(run_dir):
        continue
    db_path = os.path.join(run_dir, "simulation.db")
    if not os.path.isfile(db_path):
        continue
    run_id = dirname.split("_", 2)[-1] if "_" in dirname else dirname
    if not run_id:
        continue
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    run_count += 1
    for row in cur.execute(condition_query, (run_id,)).fetchall():
        key = (row["dataset_name"], row["condition_name"])
        if key not in dataset_condition_totals:
            dataset_condition_totals[key] = {"trials_total": 0, "judge_validated": 0, "is_compared": 0, "is_match": 0, "missing_outputs": 0, "empty_rows": 0, "parse_errors": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "ref_compared": 0, "ref_match": 0}
        agg = dataset_condition_totals[key]
        for col in agg.keys():
            agg[col] += int(row[col] or 0)
        run_dataset_metrics.setdefault(run_id, {}).setdefault(row["dataset_name"], {"trials_total": 0, "judge_validated": 0, "is_compared": 0, "is_match": 0})
        ragg = run_dataset_metrics[run_id][row["dataset_name"]]
        ragg["trials_total"] += int(row["trials_total"] or 0)
        ragg["judge_validated"] += int(row["judge_validated"] or 0)
        ragg["is_compared"] += int(row["is_compared"] or 0)
        ragg["is_match"] += int(row["is_match"] or 0)
    for row in cur.execute(model_query, (run_id,)).fetchall():
        key = (row["variant"], row["model_id"])
        if key not in model_totals:
            model_totals[key] = {"trials_total": 0, "judge_validated": 0, "is_compared": 0, "is_match": 0, "ref_compared": 0, "ref_match": 0}
        agg = model_totals[key]
        for col in agg.keys():
            agg[col] += int(row[col] or 0)
    for row in cur.execute(
        """
        SELECT
          t.variant AS variant,
          t.model_id AS model_id,
          c.name AS condition_name,
          COUNT(*) AS n_expected,
          SUM(CASE WHEN o.parsed_answer_json IS NOT NULL AND trim(o.parsed_answer_json) != '' AND o.parsed_answer_json NOT LIKE '%[parse_error]%' THEN 1 ELSE 0 END) AS n_judge
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ?
        GROUP BY t.variant, t.model_id, c.name
        """,
        (run_id,),
    ).fetchall():
        key = (row["variant"], row["model_id"], row["condition_name"])
        if key not in model_condition_totals:
            model_condition_totals[key] = {"n_expected": 0, "n_judge": 0}
        model_condition_totals[key]["n_expected"] += int(row["n_expected"] or 0)
        model_condition_totals[key]["n_judge"] += int(row["n_judge"] or 0)
    conn.close()

if run_count == 0:
    print("No run databases found for pooled report.")
    raise SystemExit(1)

print()
print(paint("=" * 96, MAGENTA))
print(paint(f"{BOLD}Pooled Judge Summary Across Runs{RESET} | runs={run_count}", CYAN))
print(paint(f"Runs path: {runs_path}", YELLOW))
print(paint("=" * 96, MAGENTA))

dataset_overall_rows = []
condition_story_rows = []
overall_totals = {
    "trials_total": 0,
    "judge_validated": 0,
    "is_compared": 0,
    "is_match": 0,
    "ref_compared": 0,
    "ref_match": 0,
    "tp": 0,
    "tn": 0,
    "fp": 0,
    "fn": 0,
}
for dataset_name in dataset_names:
    condition_rows = []
    ds_tot = {"trials_total": 0, "judge_validated": 0, "is_compared": 0, "is_match": 0, "missing_outputs": 0, "empty_rows": 0, "parse_errors": 0, "ref_compared": 0, "ref_match": 0}
    for condition_name in condition_names:
        agg = dataset_condition_totals.get((dataset_name, condition_name), {"trials_total": 0, "judge_validated": 0, "is_compared": 0, "is_match": 0, "missing_outputs": 0, "empty_rows": 0, "parse_errors": 0, "ref_compared": 0, "ref_match": 0})
        for col in ds_tot.keys():
            ds_tot[col] += agg[col]
        row = [condition_name, agg["trials_total"], agg["judge_validated"], agg["is_compared"], pct(agg["judge_validated"], agg["trials_total"]), agg["is_match"], pct(agg["is_match"], agg["is_compared"]), agg["empty_rows"], agg["parse_errors"], agg["missing_outputs"]]
        if include_refusal:
            row.extend([agg["ref_compared"], agg["ref_match"], pct(agg["ref_match"], agg["ref_compared"])])
        condition_rows.append(row)
        condition_story_rows.append({
            "dataset": dataset_name,
            "condition": condition_name,
            "trials": agg["trials_total"],
            "judge_valid": agg["judge_validated"],
            "compared": agg["is_compared"],
            "is_match": agg["is_match"],
            "is_match_rate": (agg["is_match"] / agg["is_compared"]) if agg["is_compared"] > 0 else None,
            "judge_cov": (agg["judge_validated"] / agg["trials_total"]) if agg["trials_total"] > 0 else None,
            "parse_err": agg["parse_errors"],
            "missing": agg["missing_outputs"],
        })
    if detailed:
        print()
        print(paint(f"{BOLD}Dataset (pooled): {dataset_name}{RESET}", CYAN))
        hdr = ["condition", "trials", "judge_valid", "compared", "judge_cov", "is_match", "is_match_rate", "empty", "parse_err", "missing"]
        if include_refusal:
            hdr.extend(["ref_compared", "ref_match", "ref_match_rate"])
        print(make_table(hdr, condition_rows, right_align_cols=set(range(1, len(hdr)))))
    overall_row = [dataset_name, ds_tot["trials_total"], ds_tot["judge_validated"], ds_tot["is_compared"], pct(ds_tot["judge_validated"], ds_tot["trials_total"]), ds_tot["is_match"], pct(ds_tot["is_match"], ds_tot["is_compared"]), ds_tot["empty_rows"], ds_tot["parse_errors"], ds_tot["missing_outputs"]]
    if include_refusal:
        overall_row.extend([ds_tot["ref_compared"], ds_tot["ref_match"], pct(ds_tot["ref_match"], ds_tot["ref_compared"])])
    dataset_overall_rows.append(overall_row)
    overall_totals["trials_total"] += ds_tot["trials_total"]
    overall_totals["judge_validated"] += ds_tot["judge_validated"]
    overall_totals["is_compared"] += ds_tot["is_compared"]
    overall_totals["is_match"] += ds_tot["is_match"]
    overall_totals["ref_compared"] += ds_tot["ref_compared"]
    overall_totals["ref_match"] += ds_tot["ref_match"]

for agg in dataset_condition_totals.values():
    overall_totals["tp"] += int(agg.get("tp", 0))
    overall_totals["tn"] += int(agg.get("tn", 0))
    overall_totals["fp"] += int(agg.get("fp", 0))
    overall_totals["fn"] += int(agg.get("fn", 0))

overall_judge_cov = (overall_totals["judge_validated"] / overall_totals["trials_total"]) if overall_totals["trials_total"] > 0 else None
overall_match_rate = (overall_totals["is_match"] / overall_totals["is_compared"]) if overall_totals["is_compared"] > 0 else None
overall_ref_match_rate = (overall_totals["ref_match"] / overall_totals["ref_compared"]) if overall_totals["ref_compared"] > 0 else None
overall_gate = gate_status(
    overall_judge_cov,
    overall_match_rate,
    overall_totals["is_compared"],
    ready_min_cov,
    ready_min_match,
    ready_min_n,
)

print()
print(paint(f"{BOLD}Readiness Gate Scorecard (pooled){RESET}", CYAN))
scorecard_row = [
    "pooled_all_runs",
    overall_totals["trials_total"],
    overall_totals["judge_validated"],
    overall_totals["is_compared"],
    pct(overall_totals["judge_validated"], overall_totals["trials_total"]),
    pct(overall_totals["is_compared"], overall_totals["trials_total"]),
    pct(overall_totals["is_match"], overall_totals["is_compared"]),
]
scorecard_headers = ["scope", "trials", "judge_valid", "compared", "judge_cov", "compare_cov", "is_match_rate"]
if include_refusal:
    scorecard_headers.append("ref_match_rate")
    scorecard_row.append(pct(overall_totals["ref_match"], overall_totals["ref_compared"]))
scorecard_headers.append("gate_status")
scorecard_row.append(overall_gate)
print(make_table(scorecard_headers, [scorecard_row], right_align_cols=set(range(1, len(scorecard_headers) - 1))))
print(paint(f"  gates: judge_cov>={ready_min_cov*100:.1f}% | is_match_rate>={ready_min_match*100:.1f}% | compared>={ready_min_n}", YELLOW))

print()
print(paint(f"{BOLD}Dataset Overall (pooled){RESET}", CYAN))
overall_hdr = ["dataset", "trials", "judge_valid", "compared", "judge_cov", "is_match", "is_match_rate", "empty", "parse_err", "missing"]
if include_refusal:
    overall_hdr.extend(["ref_compared", "ref_match", "ref_match_rate"])
print(make_table(overall_hdr, dataset_overall_rows, right_align_cols=set(range(1, len(overall_hdr)))))

print()
print(paint(f"{BOLD}Dataset x Condition Readiness (pooled){RESET}", CYAN))
cell_rows = []
for r in condition_story_rows:
    status = gate_status(
        r["judge_cov"],
        r["is_match_rate"],
        r["compared"],
        ready_min_cov,
        ready_min_match,
        max(10, ready_min_n // 12),
    )
    cell_rows.append([
        status,
        r["dataset"],
        r["condition"],
        r["trials"],
        r["judge_valid"],
        r["compared"],
        "n/a" if r["judge_cov"] is None else f"{100.0 * r['judge_cov']:.1f}%",
        "n/a" if r["is_match_rate"] is None else f"{100.0 * r['is_match_rate']:.1f}%",
        r["is_match"],
        r["parse_err"],
        r["missing"],
    ])
cell_rows.sort(key=lambda x: (0 if x[0] == "AT_RISK" else 1, float(x[7].strip("%")) if x[7] != "n/a" else 999.0))
print(make_table(
    ["flag", "dataset", "condition", "trials", "judge_valid", "compared", "judge_cov", "is_match_rate", "is_match", "parse_err", "missing"],
    cell_rows if detailed else cell_rows[:18],
    right_align_cols={3, 4, 5, 6, 7, 8, 9, 10},
))

print()
print(paint(f"{BOLD}LLM-judged cells (dataset × condition, pooled across runs){RESET}", CYAN))
print(paint("  llm_judge_valid = trials with valid LLM is_correct in parsed_answer_json (same rule as judge_valid).", YELLOW))
llm_cell_rows_pooled = []
for r in condition_story_rows:
    tr = int(r["trials"] or 0)
    jv = int(r["judge_valid"] or 0)
    if tr <= 0:
        st = "EMPTY"
    elif jv >= tr:
        st = "FULL"
    elif jv > 0:
        st = "PARTIAL"
    else:
        st = "NONE"
    llm_cell_rows_pooled.append([r["dataset"], r["condition"], tr, jv, pct(jv, tr), st])
llm_cell_rows_pooled.sort(key=lambda x: (x[0], x[1]))
print(make_table(
    ["dataset", "condition", "trials", "llm_judge_valid", "judge_cov", "cell_status"],
    llm_cell_rows_pooled,
    right_align_cols={2, 3, 4},
))
full_cp = sum(1 for row in llm_cell_rows_pooled if row[5] == "FULL")
part_cp = sum(1 for row in llm_cell_rows_pooled if row[5] == "PARTIAL")
none_cp = sum(1 for row in llm_cell_rows_pooled if row[5] == "NONE")
print(
    paint(
        f"  summary: FULL={full_cp} PARTIAL={part_cp} NONE={none_cp} (of {len(llm_cell_rows_pooled)} cells)",
        GREEN,
    )
)

print()
print(paint(f"{BOLD}Cross-Run Stability (dataset){RESET}", CYAN))
stability_rows = []
for dataset_name in dataset_names:
    cov_vals = []
    mr_vals = []
    for r_id, by_ds in run_dataset_metrics.items():
        ds = by_ds.get(dataset_name)
        if not ds:
            continue
        if ds["trials_total"] > 0:
            cov_vals.append(ds["judge_validated"] / ds["trials_total"])
        if ds["is_compared"] > 0:
            mr_vals.append(ds["is_match"] / ds["is_compared"])
    if not cov_vals and not mr_vals:
        continue
    cov_mean = sum(cov_vals) / len(cov_vals) if cov_vals else None
    cov_min = min(cov_vals) if cov_vals else None
    cov_max = max(cov_vals) if cov_vals else None
    mr_mean = sum(mr_vals) / len(mr_vals) if mr_vals else None
    mr_min = min(mr_vals) if mr_vals else None
    mr_max = max(mr_vals) if mr_vals else None
    mr_delta = (mr_max - mr_min) if (mr_max is not None and mr_min is not None) else None
    stability_rows.append([
        dataset_name,
        "n/a" if cov_mean is None else f"{100.0 * cov_mean:.1f}%",
        "n/a" if cov_min is None else f"{100.0 * cov_min:.1f}%",
        "n/a" if cov_max is None else f"{100.0 * cov_max:.1f}%",
        "n/a" if mr_mean is None else f"{100.0 * mr_mean:.1f}%",
        "n/a" if mr_min is None else f"{100.0 * mr_min:.1f}%",
        "n/a" if mr_max is None else f"{100.0 * mr_max:.1f}%",
        "n/a" if mr_delta is None else f"{100.0 * mr_delta:.1f}%",
    ])
stability_rows.sort(key=lambda x: float(x[7].strip("%")) if x[7] != "n/a" else -1, reverse=True)
print(make_table(
    ["dataset", "cov_mean", "cov_min", "cov_max", "match_mean", "match_min", "match_max", "max_delta_match"],
    stability_rows,
    right_align_cols={1, 2, 3, 4, 5, 6, 7},
))

model_rows = []
for key in sorted(model_totals.keys()):
    variant, model_id = key
    agg = model_totals[key]
    row = [variant, model_id, agg["trials_total"], agg["judge_validated"], pct(agg["judge_validated"], agg["trials_total"]), agg["is_compared"], agg["is_match"], pct(agg["is_match"], agg["is_compared"])]
    if include_refusal:
        row.extend([agg["ref_compared"], agg["ref_match"], pct(agg["ref_match"], agg["ref_compared"])])
    model_rows.append(row)
print()
print(paint(f"{BOLD}Model Summary (pooled){RESET}", CYAN))
model_hdr = ["variant", "model_id", "trials_total", "judge_validated", "judge_cov", "is_compared", "is_match", "is_match_rate"]
if include_refusal:
    model_hdr.extend(["ref_compared", "ref_match", "ref_match_rate"])
print(make_table(model_hdr, model_rows, right_align_cols=set(range(2, len(model_hdr)))))

print()
print(paint(f"{BOLD}Publication-Gating Coverage (pooled){RESET}", CYAN))
CORE_CONDITIONS = {"control", "asch_history_5", "authoritative_bias"}
gate_by_model = {}
for (variant, model_id, cond_name), vals in model_condition_totals.items():
    key = (variant, model_id)
    if key not in gate_by_model:
        gate_by_model[key] = {"n_expected_3": 0, "n_judge_3": 0, "n_expected_12": 0, "n_judge_12": 0}
    g = gate_by_model[key]
    g["n_expected_12"] += vals["n_expected"]
    g["n_judge_12"] += vals["n_judge"]
    if cond_name in CORE_CONDITIONS:
        g["n_expected_3"] += vals["n_expected"]
        g["n_judge_3"] += vals["n_judge"]

gate_rows = []
for key in sorted(gate_by_model.keys()):
    variant, model_id = key
    g = gate_by_model[key]
    cov3 = (g["n_judge_3"] / g["n_expected_3"]) if g["n_expected_3"] > 0 else None
    cov12 = (g["n_judge_12"] / g["n_expected_12"]) if g["n_expected_12"] > 0 else None
    gate_rows.append([
        variant,
        model_id,
        g["n_expected_3"],
        g["n_judge_3"],
        "n/a" if cov3 is None else f"{100.0 * cov3:.1f}%",
        "READY" if (cov3 is not None and cov3 >= ready_min_cov) else "AT_RISK",
        g["n_expected_12"],
        g["n_judge_12"],
        "n/a" if cov12 is None else f"{100.0 * cov12:.1f}%",
        "READY" if (cov12 is not None and cov12 >= ready_min_cov) else "AT_RISK",
    ])
print(make_table(
    ["variant", "model_id", "n_expected_3", "n_judge_3", "cov_3cond", "gate_3cond", "n_expected_12", "n_judge_12", "cov_12cond", "gate_12cond"],
    gate_rows,
    right_align_cols={2, 3, 4, 6, 7, 8},
))

print()
print(paint(f"{BOLD}Manual-vs-Judge Drift Breakdown (pooled){RESET}", CYAN))
tp = overall_totals["tp"]
tn = overall_totals["tn"]
fp = overall_totals["fp"]
fn = overall_totals["fn"]
den = tp + tn + fp + fn
precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
drift_rows = [[
    tp,
    tn,
    fp,
    fn,
    pct(tp, den),
    pct(tn, den),
    pct(fp, den),
    pct(fn, den),
    "n/a" if precision is None else f"{100.0 * precision:.1f}%",
    "n/a" if recall is None else f"{100.0 * recall:.1f}%",
]]
print(make_table(
    ["tp", "tn", "fp", "fn", "tp_rate", "tn_rate", "fp_rate", "fn_rate", "precision", "recall"],
    drift_rows,
    right_align_cols=set(range(10)),
))
if fp > fn:
    print(paint("  drift_signal=FP-heavy (judge more likely to over-call correctness).", YELLOW))
elif fn > fp:
    print(paint("  drift_signal=FN-heavy (judge more likely to under-call correctness).", YELLOW))
else:
    print(paint("  drift_signal=balanced FP/FN.", YELLOW))
PY
}

if [[ "$MODE" == "all" ]]; then
  echo "Reporting all runs in: $RUNS_PATH"
  shopt -s nullglob
  found_any=0
  for dir in "$RUNS_PATH"/*; do
    [[ -d "$dir" ]] || continue
    db="${dir}/simulation.db"
    [[ -f "$db" ]] || continue
    found_any=1
    if [[ "$SHOW_PER_RUN" == "1" ]]; then
      run_id="$(basename "$dir" | cut -d'_' -f3-)"
      [[ -n "$run_id" ]] || continue
      report_one "$run_id" "$db"
    fi
  done
  if [[ $found_any -eq 0 ]]; then
    echo "No run databases found under: $RUNS_PATH"
    exit 1
  fi
  report_pooled_all
else
  shopt -s nullglob
  matches=()
  for dir in "$RUNS_PATH"/*"$RUN_ID"*; do
    [[ -d "$dir" ]] || continue
    db="${dir}/simulation.db"
    [[ -f "$db" ]] || continue
    matches+=("$db")
  done
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Error: no simulation.db found for run_id=$RUN_ID under $RUNS_PATH"
    exit 1
  fi
  report_one "$RUN_ID" "${matches[0]}"
fi
