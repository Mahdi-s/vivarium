#!/usr/bin/env python3
"""
Fix legacy OLMo conformity DBs where unlabeled opinion items were stored with
ground_truth_text == "None" (string) instead of NULL.

Why this matters:
- These items should NOT contribute to accuracy/error_rate metrics.
- When stored as "None", the runner evaluates correctness against the literal
  string "None", producing is_correct=0 for every opinion trial, contaminating
  behavioral summaries.

This script:
1) Sets conformity_items.ground_truth_text = NULL for social_conventions items
2) Sets conformity_outputs.is_correct = NULL for trials on those items

It is safe to rerun (idempotent).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


def _iter_dbs(runs_dir: Path) -> Iterable[Tuple[str, Path]]:
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        db = run_dir / "simulation.db"
        if db.exists():
            yield run_dir.name, db


def fix_db(db_path: Path, *, dataset_name: str = "social_conventions_minimal") -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Identify dataset ids
        ds_ids = [
            str(r["dataset_id"])
            for r in conn.execute(
                "SELECT dataset_id FROM conformity_datasets WHERE name = ?;",
                (dataset_name,),
            ).fetchall()
        ]
        if not ds_ids:
            return {"datasets": 0, "items_fixed": 0, "outputs_fixed": 0}

        # Fix items
        items_fixed = conn.execute(
            f"""
            UPDATE conformity_items
            SET ground_truth_text = NULL
            WHERE dataset_id IN ({','.join(['?'] * len(ds_ids))})
              AND (ground_truth_text IS NULL OR ground_truth_text = 'None' OR lower(ground_truth_text) = 'null');
            """,
            ds_ids,
        ).rowcount

        # Fix outputs
        outputs_fixed = conn.execute(
            f"""
            UPDATE conformity_outputs
            SET is_correct = NULL
            WHERE trial_id IN (
                SELECT t.trial_id
                FROM conformity_trials t
                JOIN conformity_items i ON i.item_id = t.item_id
                WHERE i.dataset_id IN ({','.join(['?'] * len(ds_ids))})
            );
            """,
            ds_ids,
        ).rowcount

        conn.commit()
        return {"datasets": len(ds_ids), "items_fixed": int(items_fixed or 0), "outputs_fixed": int(outputs_fixed or 0)}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs-hpc", help="Directory containing run folders (each with simulation.db)")
    ap.add_argument("--dataset-name", type=str, default="social_conventions_minimal", help="Dataset name to treat as unlabeled opinion")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    total_items = 0
    total_outputs = 0
    n_dbs = 0
    for run_name, db in _iter_dbs(runs_dir):
        stats = fix_db(db, dataset_name=str(args.dataset_name))
        if stats["datasets"] == 0:
            continue
        n_dbs += 1
        total_items += stats["items_fixed"]
        total_outputs += stats["outputs_fixed"]
        print(f"{run_name}: items_fixed={stats['items_fixed']} outputs_fixed={stats['outputs_fixed']}")

    print(f"\nDone. dbs_touched={n_dbs} items_fixed_total={total_items} outputs_fixed_total={total_outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

