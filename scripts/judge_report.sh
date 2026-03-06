#!/usr/bin/env bash
# Judge agreement report for one or all runs under runs_latest/runs.
# Prints one formatted table per dataset with one row per configured condition.
#
# Usage:
#   ./judge_report.sh
#   ./judge_report.sh <run_id>
#   ./judge_report.sh --all [runs_dir]
#   ./judge_report.sh --config <suite_config.json> [--all [runs_dir] | <run_id> [runs_dir]]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_PATH="${REPO_ROOT}/runs_latest/runs"
SUITE_CONFIG="${REPO_ROOT}/experiments/olmo_conformity/configs/suite_expanded_temp0.0.json"
MODE="all"
RUN_ID=""

usage() {
  echo "Usage:"
  echo "  ./judge_report.sh"
  echo "  ./judge_report.sh <run_id> [runs_dir]"
  echo "  ./judge_report.sh --all [runs_dir]"
  echo "  ./judge_report.sh --config <suite_config.json> [--all [runs_dir] | <run_id> [runs_dir]]"
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

  python - "$db" "$run_id" "$SUITE_CONFIG" <<'PY'
import json
import os
import sqlite3
import sys

db_path, run_id, suite_config_path = sys.argv[1], sys.argv[2], sys.argv[3]

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

overall = cur.execute(
    """
    SELECT COUNT(*) AS n
    FROM conformity_trials t
    WHERE t.run_id = ?
    """,
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
SELECT
  c.name AS condition_name,
  COUNT(*) AS total_trials,
  SUM(CASE WHEN o.output_id IS NULL THEN 1 ELSE 0 END) AS missing_outputs,
  SUM(CASE WHEN o.parsed_answer_json IS NULL OR o.parsed_answer_json = '' THEN 1 ELSE 0 END) AS empty_rows,
  SUM(CASE WHEN o.parsed_answer_json LIKE '%[parse_error]%' THEN 1 ELSE 0 END) AS parse_errors,
  SUM(
    CASE
      WHEN o.parsed_answer_json IS NOT NULL
        AND o.parsed_answer_json != ''
        AND o.parsed_answer_json NOT LIKE '%[parse_error]%'
        AND json_extract(o.parsed_answer_json, '$.is_correct') IS NOT NULL
      THEN 1 ELSE 0
    END
  ) AS valid_rows,
  SUM(
    CASE
      WHEN o.parsed_answer_json IS NOT NULL
        AND o.parsed_answer_json != ''
        AND o.parsed_answer_json NOT LIKE '%[parse_error]%'
        AND json_extract(o.parsed_answer_json, '$.is_correct') IS NOT NULL
        AND CAST(o.is_correct AS INT) = CAST(json_extract(o.parsed_answer_json, '$.is_correct') AS INT)
      THEN 1 ELSE 0
    END
  ) AS matches,
  SUM(
    CASE
      WHEN o.parsed_answer_json IS NOT NULL
        AND o.parsed_answer_json != ''
        AND o.parsed_answer_json NOT LIKE '%[parse_error]%'
        AND json_extract(o.parsed_answer_json, '$.is_correct') IS NOT NULL
        AND CAST(o.is_correct AS INT) != CAST(json_extract(o.parsed_answer_json, '$.is_correct') AS INT)
      THEN 1 ELSE 0
    END
  ) AS mismatches
FROM conformity_trials t
JOIN conformity_conditions c ON c.condition_id = t.condition_id
JOIN conformity_items i ON i.item_id = t.item_id
JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
WHERE t.run_id = ? AND d.name = ?
GROUP BY c.name
"""

for dataset_name in dataset_names:
    rows = cur.execute(dataset_query, (run_id, dataset_name)).fetchall()
    by_condition = {row["condition_name"]: row for row in rows}

    pretty_rows = []
    ds_total = 0
    ds_valid = 0
    ds_match = 0
    ds_mismatch = 0
    ds_empty = 0
    ds_parse = 0
    ds_missing = 0

    for condition_name in condition_names:
        row = by_condition.get(condition_name)
        if row is None:
            total_trials = 0
            missing_outputs = 0
            empty_rows = 0
            parse_errors = 0
            valid_rows = 0
            matches = 0
            mismatches = 0
        else:
            total_trials = int(row["total_trials"] or 0)
            missing_outputs = int(row["missing_outputs"] or 0)
            empty_rows = int(row["empty_rows"] or 0)
            parse_errors = int(row["parse_errors"] or 0)
            valid_rows = int(row["valid_rows"] or 0)
            matches = int(row["matches"] or 0)
            mismatches = int(row["mismatches"] or 0)

        ds_total += total_trials
        ds_valid += valid_rows
        ds_match += matches
        ds_mismatch += mismatches
        ds_empty += empty_rows
        ds_parse += parse_errors
        ds_missing += missing_outputs

        if valid_rows > 0:
            match_rate = f"{(100.0 * matches / valid_rows):.1f}%"
        else:
            match_rate = "n/a"

        pretty_rows.append([
            condition_name,
            total_trials,
            valid_rows,
            matches,
            mismatches,
            match_rate,
            empty_rows,
            parse_errors,
            missing_outputs,
        ])

    print()
    print(paint(f"{BOLD}Dataset: {dataset_name}{RESET}", CYAN))
    print(
        paint(
            f"  trials={ds_total} | valid_judged={ds_valid} | match={ds_match} | mismatch={ds_mismatch} | empty={ds_empty} | parse_error={ds_parse} | missing_output={ds_missing}",
            GREEN,
        )
    )
    print(
        make_table(
            headers=[
                "condition",
                "trials",
                "valid",
                "match",
                "mismatch",
                "match_rate",
                "empty",
                "parse_err",
                "missing",
            ],
            rows=pretty_rows,
            right_align_cols={1, 2, 3, 4, 5, 6, 7, 8},
        )
    )

conn.close()
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
    run_id="$(basename "$dir" | cut -d'_' -f3-)"
    [[ -n "$run_id" ]] || continue
    found_any=1
    report_one "$run_id" "$db"
  done
  if [[ $found_any -eq 0 ]]; then
    echo "No run databases found under: $RUNS_PATH"
    exit 1
  fi
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
