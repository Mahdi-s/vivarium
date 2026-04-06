#!/usr/bin/env python3
"""
Extract all trials where manual (heuristic) labels disagree with
LLM judge labels across all experiment databases.

Sources:
  - runs/            (cross-family: Llama, GPT, Gemini, Grok, Claude, OLMo-32B)
  - runs/think/      (OLMo-7B-Think variants)
  - runs_latest/runs/ (OLMo-7B: base, instruct, instruct_sft, instruct_dpo only)

Output: judge_mismatches.csv
"""

import sqlite3
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "judge_mismatches.csv"

# ── Collect all database paths ────────────────────────────────────
def collect_db_paths():
    dbs = []
    source_labels = {}

    # 1. runs/ (cross-family models)
    runs_dir = ROOT / "runs"
    for db_path in sorted(runs_dir.glob("*/simulation.db")):
        # Skip think/ subdirectory — handled separately
        if "think" not in str(db_path.parent):
            dbs.append(db_path)
            source_labels[str(db_path)] = "runs"

    # 2. runs/think/ (OLMo-7B-Think)
    think_dir = ROOT / "runs" / "think"
    for db_path in sorted(think_dir.glob("*/simulation.db")):
        dbs.append(db_path)
        source_labels[str(db_path)] = "runs/think"

    # 3. runs_latest/runs/ (OLMo-7B variants)
    latest_dir = ROOT / "runs_latest" / "runs"
    for db_path in sorted(latest_dir.glob("*/simulation.db")):
        dbs.append(db_path)
        source_labels[str(db_path)] = "runs_latest"

    return dbs, source_labels


# Variants to include from runs_latest (OLMo-7B family)
RUNS_LATEST_VARIANTS = {"base", "instruct", "instruct_sft", "instruct_dpo"}


QUERY = """
SELECT
    t.trial_id,
    t.run_id,
    t.model_id,
    t.variant,
    t.temperature,
    t.item_id,
    c.name                     AS condition_name,
    c.params_json              AS condition_params,
    i.question                 AS question_text,
    i.ground_truth_text,
    json_extract(i.source_json, '$.wrong_answer') AS wrong_answer_text,
    d.name                     AS dataset_name,
    i.domain,
    p.system_prompt,
    p.user_prompt,
    o.raw_text                 AS raw_output,
    o.parsed_answer_text       AS parsed_answer,
    o.is_correct               AS manual_is_correct,
    o.refusal_flag             AS manual_refusal_flag,
    o.parsed_answer_json,
    json_extract(o.parsed_answer_json, '$.is_correct')             AS judge_is_correct,
    json_extract(o.parsed_answer_json, '$.refusal_flag')           AS judge_refusal_flag,
    json_extract(o.parsed_answer_json, '$.wrong_answer_endorsed')  AS judge_wrong_answer_endorsed,
    json_extract(o.parsed_answer_json, '$.notes')                  AS judge_notes,
    json_extract(o.parsed_answer_json, '$._llm_judge.judge_model') AS judge_model
FROM conformity_trials t
JOIN conformity_outputs o      ON o.trial_id = t.trial_id
JOIN conformity_conditions c   ON c.condition_id = t.condition_id
JOIN conformity_items i        ON i.item_id = t.item_id
LEFT JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
LEFT JOIN conformity_prompts p ON p.trial_id = t.trial_id
WHERE o.parsed_answer_json IS NOT NULL
  AND (
      o.is_correct   != json_extract(o.parsed_answer_json, '$.is_correct')
   OR o.refusal_flag  != json_extract(o.parsed_answer_json, '$.refusal_flag')
  )
"""


CSV_COLUMNS = [
    "trial_id",
    "run_id",
    "source_dir",
    "db_path",
    "model_id",
    "variant",
    "temperature",
    "item_id",
    "dataset_name",
    "domain",
    "condition_name",
    "condition_params",
    "question_text",
    "ground_truth",
    "wrong_answer",
    "system_prompt",
    "user_prompt",
    "raw_output",
    "parsed_answer",
    "manual_is_correct",
    "manual_refusal_flag",
    "judge_is_correct",
    "judge_refusal_flag",
    "judge_wrong_answer_endorsed",
    "judge_notes",
    "judge_model",
    "mismatch_type",
    "parsed_answer_json",
]


def classify_mismatch(manual_correct, manual_refusal, judge_correct, judge_refusal):
    """Classify the type of disagreement."""
    parts = []
    if manual_correct != judge_correct:
        if manual_correct == 0 and judge_correct == 1:
            parts.append("is_correct: manual=0 judge=1 (judge more lenient)")
        elif manual_correct == 1 and judge_correct == 0:
            parts.append("is_correct: manual=1 judge=0 (judge stricter)")
        else:
            parts.append(f"is_correct: manual={manual_correct} judge={judge_correct}")
    if manual_refusal != judge_refusal:
        if manual_refusal == 0 and judge_refusal == 1:
            parts.append("refusal: manual=0 judge=1 (judge sees refusal)")
        elif manual_refusal == 1 and judge_refusal == 0:
            parts.append("refusal: manual=1 judge=0 (judge misses refusal)")
        else:
            parts.append(f"refusal: manual={manual_refusal} judge={judge_refusal}")
    return " | ".join(parts) if parts else "unknown"


def extract_from_db(db_path, source_label):
    """Extract mismatch rows from a single database."""
    rows = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(QUERY)
        for r in cursor:
            variant = r["variant"]
            # For runs_latest, only include the 4 target variants
            if source_label == "runs_latest" and variant not in RUNS_LATEST_VARIANTS:
                continue

            mismatch_type = classify_mismatch(
                r["manual_is_correct"], r["manual_refusal_flag"],
                r["judge_is_correct"], r["judge_refusal_flag"],
            )

            rows.append({
                "trial_id": r["trial_id"],
                "run_id": r["run_id"],
                "source_dir": source_label,
                "db_path": str(db_path.relative_to(ROOT)),
                "model_id": r["model_id"],
                "variant": r["variant"],
                "temperature": r["temperature"],
                "item_id": r["item_id"],
                "dataset_name": r["dataset_name"],
                "domain": r["domain"],
                "condition_name": r["condition_name"],
                "condition_params": r["condition_params"],
                "question_text": r["question_text"],
                "ground_truth": r["ground_truth_text"],
                "wrong_answer": r["wrong_answer_text"],
                "system_prompt": r["system_prompt"],
                "user_prompt": r["user_prompt"],
                "raw_output": r["raw_output"],
                "parsed_answer": r["parsed_answer"],
                "manual_is_correct": r["manual_is_correct"],
                "manual_refusal_flag": r["manual_refusal_flag"],
                "judge_is_correct": r["judge_is_correct"],
                "judge_refusal_flag": r["judge_refusal_flag"],
                "judge_wrong_answer_endorsed": r["judge_wrong_answer_endorsed"],
                "judge_notes": r["judge_notes"],
                "judge_model": r["judge_model"],
                "mismatch_type": mismatch_type,
                "parsed_answer_json": r["parsed_answer_json"],
            })
        conn.close()
    except Exception as e:
        print(f"  ERROR processing {db_path}: {e}", file=sys.stderr)
    return rows


def main():
    dbs, source_labels = collect_db_paths()
    print(f"Found {len(dbs)} databases to scan")

    all_rows = []
    for db_path in dbs:
        source = source_labels[str(db_path)]
        rows = extract_from_db(db_path, source)
        model_info = ""
        if rows:
            model_info = f" ({rows[0]['model_id']} / {rows[0]['variant']})"
        print(f"  {source}/{db_path.parent.name}: {len(rows)} mismatches{model_info}")
        all_rows.extend(rows)

    # Deduplicate by trial_id (some trials may appear in multiple DBs)
    seen = set()
    unique_rows = []
    for row in all_rows:
        if row["trial_id"] not in seen:
            seen.add(row["trial_id"])
            unique_rows.append(row)

    print(f"\nTotal mismatches: {len(all_rows)} (unique: {len(unique_rows)})")

    # Write CSV
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"Written to: {OUT}")
    print(f"File size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
