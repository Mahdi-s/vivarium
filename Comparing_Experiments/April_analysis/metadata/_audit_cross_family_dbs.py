"""
One-shot audit script for the 18 new cross-family + ablation DBs.

Dumps one row per (db_path, condition_name) cell with schema:
  db_path, model_id, variant, temperature, condition_name,
  n_rows, n_judged, median_raw_text_len, max_raw_text_len,
  judge_prompt_version, judge_model, item_count, experiment_group

Output: Comparing_Experiments/April_analysis/metadata/cross_family_db_audit.csv

Run from repo root:
  python "Comparing_Experiments/April_analysis/metadata/_audit_cross_family_dbs.py"

This is an E0 blocking precondition per the expansion plan — it must
succeed (100% judge coverage, expected row counts, no runs_latest/
sources) before anything downstream runs.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# 16 cross-family DBs + 2 ablation DBs = 18 total
CROSS_FAMILY_DBS = [
    ("runs/20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d/simulation.db", "cross_family_main"),
    ("runs/20260327_152926_7db9896e-9e3b-439f-88e3-74fe25ea2bad/simulation.db", "cross_family_main"),
    ("runs/20260327_152936_1c2e5cb6-0372-4835-bbb7-7230c55517e4/simulation.db", "cross_family_main"),
    ("runs/20260327_152944_62187f52-7a7e-4db0-a269-d14d8e887b1b/simulation.db", "cross_family_main"),
    ("runs/20260327_154349_1899a883-82e4-45f3-833a-d6403cf1ac95/simulation.db", "cross_family_main"),
    ("runs/20260327_154401_70860876-c5c2-4445-a59d-e44ae8094887/simulation.db", "cross_family_main"),
    ("runs/20260327_154412_3a0404f7-bd47-4b25-b2e2-5501e550566f/simulation.db", "cross_family_main"),
    ("runs/20260327_154419_49d07104-c14c-4a8b-a013-ae0783c5f3e8/simulation.db", "cross_family_main"),
    ("runs/20260327_154428_485ddc2d-6cae-4715-835e-76ab72e38159/simulation.db", "cross_family_main"),
    ("runs/20260327_154435_c2ce0f85-8f67-40f2-a82d-e136927cf6f5/simulation.db", "cross_family_main"),
    ("runs/20260327_224321_e043fbf6-27eb-410c-8da7-bc0f9172ab0b/simulation.db", "cross_family_main"),
    ("runs/20260327_224336_d71e75b1-17c5-4789-8ee7-13b29ef18359/simulation.db", "cross_family_main"),
    ("runs/20260327_224348_25056752-7081-449e-9a44-ad090b566107/simulation.db", "cross_family_main"),
    ("runs/20260327_224357_157a6a9e-13de-4bdb-bdd2-54d761498f24/simulation.db", "cross_family_main"),
    ("runs/20260327_224413_c07ede3a-16ac-4b47-ac42-4f6ad8dd8370/simulation.db", "cross_family_main"),
    ("runs/20260327_224422_eb63d212-77fe-46ef-965b-7777cc232f1f/simulation.db", "cross_family_main"),
    ("runs/20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5/simulation.db", "cross_family_main"),
    ("runs/20260327_224603_3ecdc9b7-49db-4625-b90e-fc3745b9224e/simulation.db", "cross_family_main"),
    ("runs/20260329_211511_5be5ada7-64be-4cbd-9024-aacbcaf233e3/simulation.db", "cross_family_main"),
    ("runs/20260329_211518_21556460-5f97-4a23-8c54-4a1f999ba619/simulation.db", "cross_family_main"),
    ("runs/20260329_235403_e8a90500-25cd-469f-a138-197c338fddaf/simulation.db", "ablation_system_prompt_removed"),
    ("runs/20260329_235408_ef72529e-5e82-463f-8b8b-b2a6c7decd3c/simulation.db", "ablation_system_prompt_removed"),
]


def audit_db(rel_path: str, experiment_group: str) -> list[dict]:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        raise SystemExit(f"[audit] missing DB: {rel_path}")

    uri = f"file:{abs_path}?mode=ro"
    rows: list[dict] = []
    with sqlite3.connect(uri, uri=True) as conn:
        cur = conn.cursor()

        # Assert no runs_latest/ contamination
        if rel_path.startswith("runs_latest/"):
            raise SystemExit(f"[audit] CF3 violation: runs_latest path {rel_path}")

        # Use the same first-output-per-trial dedup as _april_load_db_rows
        # in src/vivarium/analytics/behavioral.py (CTE picks MIN(created_at)
        # per trial, which for retry trials is the judged output).
        cur.execute(
            """
            WITH first_output_ids AS (
                SELECT MIN(o.output_id) AS output_id, o.trial_id
                FROM conformity_outputs o
                JOIN (
                    SELECT trial_id, MIN(created_at) AS min_ca
                    FROM conformity_outputs GROUP BY trial_id
                ) fo ON fo.trial_id = o.trial_id AND fo.min_ca = o.created_at
                GROUP BY o.trial_id
            )
            SELECT
                t.model_id,
                t.variant,
                t.temperature,
                cc.name AS condition_name,
                COUNT(*) AS n_rows,
                SUM(CASE WHEN json_extract(co.parsed_answer_json, '$._llm_judge') IS NOT NULL THEN 1 ELSE 0 END) AS n_judged,
                COUNT(DISTINCT t.item_id) AS item_count,
                GROUP_CONCAT(DISTINCT json_extract(co.parsed_answer_json, '$._llm_judge.prompt_version')) AS prompt_versions,
                GROUP_CONCAT(DISTINCT json_extract(co.parsed_answer_json, '$._llm_judge.judge_model')) AS judge_models
            FROM conformity_trials t
            JOIN conformity_conditions cc ON cc.condition_id = t.condition_id
            JOIN first_output_ids foi ON foi.trial_id = t.trial_id
            JOIN conformity_outputs co ON co.output_id = foi.output_id
            GROUP BY t.model_id, t.variant, t.temperature, cc.name
            ORDER BY t.model_id, t.temperature, cc.name
            """
        )
        cells = cur.fetchall()

        # Fetch raw_text lengths per cell (using same dedup)
        for (model_id, variant, temperature, condition_name,
             n_rows, n_judged, item_count, prompt_versions, judge_models) in cells:
            cur.execute(
                """
                WITH first_output_ids AS (
                    SELECT MIN(o.output_id) AS output_id, o.trial_id
                    FROM conformity_outputs o
                    JOIN (
                        SELECT trial_id, MIN(created_at) AS min_ca
                        FROM conformity_outputs GROUP BY trial_id
                    ) fo ON fo.trial_id = o.trial_id AND fo.min_ca = o.created_at
                    GROUP BY o.trial_id
                )
                SELECT LENGTH(co.raw_text)
                FROM conformity_trials t
                JOIN conformity_conditions cc ON cc.condition_id = t.condition_id
                JOIN first_output_ids foi ON foi.trial_id = t.trial_id
                JOIN conformity_outputs co ON co.output_id = foi.output_id
                WHERE t.model_id = ? AND t.variant = ?
                  AND t.temperature = ? AND cc.name = ?
                """,
                (model_id, variant, temperature, condition_name),
            )
            lens = [r[0] for r in cur.fetchall() if r[0] is not None]
            med_len = int(statistics.median(lens)) if lens else 0
            max_len = max(lens) if lens else 0

            rows.append({
                "db_path": rel_path,
                "experiment_group": experiment_group,
                "model_id": model_id,
                "variant": variant,
                "temperature": float(temperature),
                "condition_name": condition_name,
                "n_rows": n_rows,
                "n_judged": n_judged,
                "item_count": item_count,
                "median_raw_text_len": med_len,
                "max_raw_text_len": max_len,
                "judge_prompt_version": prompt_versions or "",
                "judge_model": judge_models or "",
            })
    return rows


def main() -> int:
    out_path = REPO_ROOT / "Comparing_Experiments/April_analysis/metadata/cross_family_db_audit.csv"
    all_rows: list[dict] = []
    for rel_path, group in CROSS_FAMILY_DBS:
        all_rows.extend(audit_db(rel_path, group))

    fieldnames = [
        "db_path", "experiment_group", "model_id", "variant",
        "temperature", "condition_name", "n_rows", "n_judged",
        "item_count", "median_raw_text_len", "max_raw_text_len",
        "judge_prompt_version", "judge_model",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # Summary to stderr
    cross_family_rows = [r for r in all_rows if r["experiment_group"] == "cross_family_main"]
    ablation_rows = [r for r in all_rows if r["experiment_group"] == "ablation_system_prompt_removed"]

    print(f"[audit] wrote {len(all_rows)} rows to {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"[audit]   cross_family_main: {len(cross_family_rows)} cells "
          f"({sum(r['n_rows'] for r in cross_family_rows)} trials)", file=sys.stderr)
    print(f"[audit]   ablation_system_prompt_removed: {len(ablation_rows)} cells "
          f"({sum(r['n_rows'] for r in ablation_rows)} trials)", file=sys.stderr)

    # Hard assertions (E0 preconditions)
    #
    # CF4 note: Instruct MCQ models legitimately emit 1-5 char answers
    # ("A", "0.01 C"), so the Think-style median > 2000 canary does not
    # apply. For cross-family the canary is just non-empty response with
    # non-zero max length (catches OpenRouter empty-response bugs).
    bad = []
    for r in all_rows:
        if r["n_judged"] != r["n_rows"]:
            bad.append(f"CF2 judge coverage: {r['db_path']} {r['condition_name']}: "
                       f"{r['n_judged']}/{r['n_rows']}")
        if r["n_rows"] != 400:
            bad.append(f"CF1 row count: {r['db_path']} {r['condition_name']}: {r['n_rows']} != 400")
        if r["max_raw_text_len"] <= 0:
            bad.append(f"CF4 empty-response canary: {r['db_path']} {r['condition_name']}: "
                       f"max_len={r['max_raw_text_len']}")
    if bad:
        print("[audit] FAIL — preconditions violated:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1

    print("[audit] PASS — all CF1/CF2/CF4 preconditions hold", file=sys.stderr)
    print(f"[audit]   expected grand total: 32000 + 1600 = 33600", file=sys.stderr)
    actual_trials = sum(r["n_rows"] for r in all_rows)
    print(f"[audit]   actual grand total: {actual_trials}", file=sys.stderr)
    assert actual_trials == 33600, f"expected 33600, got {actual_trials}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
