#!/usr/bin/env python3
"""
V7 Publication Item-Set Generator.

Computes the intersection of items that have *complete LLM-judge coverage*
across all temperature runs — meaning every included model has a valid
parsed_answer_json label for every one of the 3 behavioral conditions.

Only judge-labeled conditions are considered:
  Behavioral (judge-labeled):  control, asch_history_5, authoritative_bias

Excluded models (incomplete judge coverage):
  rl_zero    — 0 judge labels at T=0.0, T=0.2, T=0.8, T=1.0
  think_dpo  — only 575/1200 labeled at T=0.8 (48%); causes item intersection
               to collapse to 191 items from 4 of 8 datasets

Produces under --out-dir:
  item_set.csv               — publication manifest (400 items)
  judge_coverage_by_model.csv — per-(temperature, model) judge-label counts
  item_set_summary.txt       — human-readable summary with counts and reasoning

Usage:
  python 'Analysis Scripts/generate_v7_publication_item_set.py' \\
      --runs-dir runs_latest/runs \\
      --metadata Comparing_Experiments/runs_metadata_v6.json \\
      --out-dir Comparing_Experiments/v7_publication
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only the 3 conditions for which judge labels exist in the database
JUDGE_CONDITIONS = ("control", "asch_history_5", "authoritative_bias")

# Variants excluded from V7 due to incomplete judge-label coverage
EXCLUDED_VARIANTS = ("rl_zero", "think_dpo")

N_JUDGE_CONDITIONS = len(JUDGE_CONDITIONS)
N_SUITE_DATASETS = 8
INTENDED_MAX_ITEMS = 50  # per-dataset cap; the V7 item set should hit this exactly


def discover_runs(runs_dir: Path, metadata_path: Path) -> list[tuple[str, str, Path]]:
    """Return [(label, run_id, db_path)] ordered by temperature (ascending)."""
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {runs_dir}")

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    out: list[tuple[str, str, Path]] = []
    for label, info in meta.get("experiments", {}).items():
        run_id = str(info.get("run_id", ""))
        run_dir_name = str(info.get("run_dir", ""))
        if not run_id or not run_dir_name:
            continue
        db_path = runs_dir / run_dir_name / "simulation.db"
        if not db_path.is_file():
            print(f"  [warn] DB not found for temp={label}: {db_path}", file=sys.stderr)
            continue
        out.append((label, run_id, db_path))
    return sorted(out, key=lambda x: float(x[0]))


def query_run_judge_coverage(
    db_path: Path,
    run_id: str,
    excluded_variants: tuple[str, ...],
    judge_conditions: tuple[str, ...],
) -> dict[str, Any]:
    """
    For a single run, return:
      fully_covered_items: set of item_ids where ALL included models have
                           valid parsed_answer_json for ALL judge_conditions
      n_included_models:   number of distinct model variants after exclusion
      per_model_coverage:  [{variant, model_id, n_expected, n_judge_labeled}]
      per_dataset:         {dataset_name: {item_id, ...}} for fully_covered_items
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    excl_ph = ",".join("?" * len(excluded_variants))
    cond_ph = ",".join("?" * len(judge_conditions))

    try:
        # How many included models does this run have?
        n_models_row = conn.execute(
            f"SELECT COUNT(DISTINCT variant) AS n FROM conformity_trials "
            f"WHERE run_id = ? AND variant NOT IN ({excl_ph})",
            [run_id, *excluded_variants],
        ).fetchone()
        n_included_models: int = n_models_row["n"] if n_models_row else 0

        expected_labels_per_item = n_included_models * len(judge_conditions)

        # Per-(item) count of valid judge labels (across all included models × conditions)
        item_rows = conn.execute(
            f"""
            SELECT
                t.item_id,
                d.name AS dataset_name,
                COUNT(*) AS n_judge
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_items i ON i.item_id = t.item_id
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({cond_ph})
              AND t.variant NOT IN ({excl_ph})
              AND o.parsed_answer_json IS NOT NULL
              AND trim(o.parsed_answer_json) != ''
              AND o.parsed_answer_json NOT LIKE '%parse_error%'
            GROUP BY t.item_id, d.name
            """,
            [run_id, *judge_conditions, *excluded_variants],
        ).fetchall()

        fully_covered: set[str] = set()
        per_dataset: dict[str, set[str]] = {}
        for r in item_rows:
            if r["n_judge"] >= expected_labels_per_item:
                fully_covered.add(r["item_id"])
                per_dataset.setdefault(r["dataset_name"], set()).add(r["item_id"])

        # Per-model judge coverage stats
        model_rows = conn.execute(
            f"""
            SELECT
                t.variant,
                t.model_id,
                COUNT(*) AS n_trials,
                SUM(
                    CASE WHEN o.parsed_answer_json IS NOT NULL
                          AND trim(o.parsed_answer_json) != ''
                          AND o.parsed_answer_json NOT LIKE '%parse_error%'
                    THEN 1 ELSE 0 END
                ) AS n_judge
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({cond_ph})
              AND t.variant NOT IN ({excl_ph})
            GROUP BY t.variant, t.model_id
            ORDER BY t.variant
            """,
            [run_id, *judge_conditions, *excluded_variants],
        ).fetchall()

        per_model: list[dict[str, Any]] = []
        for r in model_rows:
            n_exp = r["n_trials"]
            n_j = r["n_judge"]
            per_model.append({
                "variant": r["variant"],
                "model_id": r["model_id"],
                "n_expected": n_exp,
                "n_judge_labeled": n_j,
                "coverage_pct": round(100.0 * n_j / n_exp, 1) if n_exp else 0.0,
            })

        return {
            "fully_covered_items": fully_covered,
            "n_included_models": n_included_models,
            "expected_labels_per_item": expected_labels_per_item,
            "per_model_coverage": per_model,
            "per_dataset": per_dataset,
        }
    finally:
        conn.close()


def query_item_metadata(db_path: Path, item_ids: set[str]) -> list[dict[str, str]]:
    """Return [{item_id, dataset_name, domain, ground_truth_text}] for item_ids."""
    if not item_ids:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ph = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT DISTINCT
                i.item_id,
                d.name AS dataset_name,
                i.domain,
                i.ground_truth_text
            FROM conformity_items i
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            WHERE i.item_id IN ({ph})
            """,
            list(item_ids),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate V7 publication item set: intersection of items with complete "
            "LLM-judge coverage for all included models across all temperature runs."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Directory containing run folders (each with simulation.db).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to runs_metadata JSON (v5 or v6).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Comparing_Experiments/v7_publication",
        help="Output directory for item_set.csv and summary.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = (REPO_ROOT / args.runs_dir).resolve()

    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_file():
        metadata_path = (REPO_ROOT / args.metadata).resolve()
    if not metadata_path.is_file():
        print(f"Metadata not found: {args.metadata}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("V7 Publication Item-Set Generator")
    print(f"  runs_dir:          {runs_dir}")
    print(f"  metadata:          {metadata_path}")
    print(f"  out_dir:           {out_dir}")
    print(f"  Judge conditions:  {N_JUDGE_CONDITIONS}  ({', '.join(JUDGE_CONDITIONS)})")
    print(f"  Excluded variants: {', '.join(EXCLUDED_VARIANTS)}")
    print("=" * 72)

    runs = discover_runs(runs_dir, metadata_path)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    print(f"\n{len(runs)} run(s) found. Computing judge-label coverage per run...")
    run_stats: list[dict[str, Any]] = []
    all_coverage_rows: list[dict[str, Any]] = []

    for label, run_id, db_path in runs:
        stats = query_run_judge_coverage(db_path, run_id, EXCLUDED_VARIANTS, JUDGE_CONDITIONS)
        stats["label"] = label
        stats["run_id"] = run_id
        stats["db_path"] = db_path
        run_stats.append(stats)

        n_full = len(stats["fully_covered_items"])
        print(
            f"  T={label}: {stats['n_included_models']} models included, "
            f"{n_full} items with complete judge coverage "
            f"(need {stats['expected_labels_per_item']} labels/item)"
        )
        for m in stats["per_model_coverage"]:
            all_coverage_rows.append({
                "temperature": label,
                "variant": m["variant"],
                "model_id": m["model_id"],
                "n_expected": m["n_expected"],
                "n_judge_labeled": m["n_judge_labeled"],
                "coverage_pct": m["coverage_pct"],
            })

    # Intersection across all runs: items fully covered at every temperature
    all_sets = [s["fully_covered_items"] for s in run_stats]
    intersection: set[str] = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    print(f"\nJudge-complete items in ALL {len(runs)} runs: {len(intersection)}")

    # Per-dataset breakdown
    meta_rows = query_item_metadata(run_stats[0]["db_path"], intersection)
    dataset_to_items: dict[str, set[str]] = {}
    for r in meta_rows:
        dataset_to_items.setdefault(r["dataset_name"], set()).add(r["item_id"])

    print("\nPer-dataset counts in V7 item set:")
    dataset_counts: dict[str, int] = {}
    for ds, items in sorted(dataset_to_items.items()):
        dataset_counts[ds] = len(items)
        print(f"  {ds:<40} {len(items):>4} items")

    min_per_dataset = min(dataset_counts.values()) if dataset_counts else 0
    n_included_models = run_stats[0]["n_included_models"]  # same across runs
    judge_trials_per_model_per_temp = len(intersection) * N_JUDGE_CONDITIONS
    total_pub_trials = len(intersection) * N_JUDGE_CONDITIONS * n_included_models * len(runs)

    print(f"\n  Total items:                           {len(intersection)}")
    print(f"  Min items per dataset:                 {min_per_dataset}")
    print(f"  Included models per run:               {n_included_models}")
    print(f"  Judge-labeled trials/model/temp:       {judge_trials_per_model_per_temp}")
    print(f"  Total judge-labeled trials (all temps):{total_pub_trials:,}")

    # ── Write item_set.csv ──────────────────────────────────────────────────
    item_set_path = out_dir / "item_set.csv"
    meta_by_id = {r["item_id"]: r for r in meta_rows}
    with open(item_set_path, "w", encoding="utf-8") as f:
        f.write("item_id,dataset_name,domain,ground_truth_text\n")
        for item_id in sorted(intersection):
            r = meta_by_id.get(item_id, {})
            ds = r.get("dataset_name", "")
            domain = r.get("domain", "")
            gt = (r.get("ground_truth_text") or "").replace('"', '""')
            f.write(f'"{item_id}","{ds}","{domain}","{gt}"\n')
    print(f"\nItem set written:  {item_set_path}  ({len(intersection)} items)")

    # ── Write judge_coverage_by_model.csv ───────────────────────────────────
    coverage_path = out_dir / "judge_coverage_by_model.csv"
    with open(coverage_path, "w", encoding="utf-8") as f:
        f.write("temperature,variant,model_id,n_expected,n_judge_labeled,coverage_pct\n")
        for row in all_coverage_rows:
            complete = "Y" if row["n_judge_labeled"] == row["n_expected"] else "N"
            f.write(
                f"{row['temperature']},{row['variant']},{row['model_id']},"
                f"{row['n_expected']},{row['n_judge_labeled']},{row['coverage_pct']},{complete}\n"
            )
    print(f"Judge coverage CSV:{coverage_path}")

    # ── Write item_set_summary.txt ──────────────────────────────────────────
    analysis_cmd = (
        "python 'Analysis Scripts/expanded_suite_behavioral_breakdown.py' \\\n"
        "  --runs-dir runs_latest/runs \\\n"
        "  --metadata Comparing_Experiments/runs_metadata_v6.json \\\n"
        f"  --publication-item-set {out_dir.relative_to(REPO_ROOT) / 'item_set.csv'} \\\n"
        f"  --out-dir {out_dir.relative_to(REPO_ROOT)} \\\n"
        "  --use-judge-labels \\\n"
        f"  --exclude-variants {' '.join(EXCLUDED_VARIANTS)} \\\n"
        "  --publication"
    )

    summary_path = out_dir / "item_set_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("V7 Publication Item-Set Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("SCOPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Judge-labeled conditions ({N_JUDGE_CONDITIONS}): {', '.join(JUDGE_CONDITIONS)}\n")
        f.write(f"Excluded variants: {', '.join(EXCLUDED_VARIANTS)}\n\n")

        f.write("EXCLUSION REASONING\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  rl_zero  (OLMo-3-7B-RL-Zero-Math):  0 judge labels at T=0.0, 0.2, 0.8, 1.0;\n"
            "           excluded by user request.\n"
        )
        f.write(
            "  think_dpo (OLMo-3-7B-Think-DPO):    575/1200 judge labels at T=0.8 (48%);\n"
            "           including it collapses the item intersection to 191 items\n"
            "           from only 4 of 8 datasets.\n\n"
        )

        f.write("ITEM SET\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total items: {len(intersection)}\n")
        f.write(f"Datasets:    {len(dataset_counts)}\n\n")
        f.write("Per-dataset counts:\n")
        for ds, cnt in sorted(dataset_counts.items()):
            f.write(f"  {ds:<40} {cnt:>4} items\n")
        f.write(f"\nMin items per dataset: {min_per_dataset}\n")
        f.write(f"(Intended max_items_per_dataset: {INTENDED_MAX_ITEMS})\n\n")

        f.write("JUDGE-LABEL COVERAGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Included models per run: {n_included_models}\n")
        f.write(f"Conditions with judge labels: {N_JUDGE_CONDITIONS} ({', '.join(JUDGE_CONDITIONS)})\n")
        f.write(f"Judge-labeled trials per model per temperature: "
                f"{judge_trials_per_model_per_temp}  ({len(intersection)} items × {N_JUDGE_CONDITIONS} conditions)\n")
        f.write(f"Total judge-labeled trials across all {len(runs)} temperatures: "
                f"{total_pub_trials:,}\n\n")

        f.write("Per-run fully-covered item counts:\n")
        for s in run_stats:
            f.write(f"  T={s['label']}: {len(s['fully_covered_items'])} items fully judge-covered\n")
        f.write(f"\nCross-temperature intersection: {len(intersection)} items\n\n")

        f.write("RECOMMENDED ANALYSIS COMMAND\n")
        f.write("-" * 40 + "\n")
        f.write(analysis_cmd + "\n\n")

        f.write("NOTE: The 9 extra suite conditions (tone, mitigation, format, authority)\n")
        f.write("have no judge labels. They are excluded from V7. Pass --include-extra-conditions\n")
        f.write("in a future run once judge labels for those conditions are available.\n")

    print(f"Summary written:   {summary_path}")

    print("\n" + "=" * 72)
    print("DONE. V7 item set ready.")
    print(f"  Items: {len(intersection)} ({len(dataset_counts)} datasets, {min_per_dataset} items/dataset min)")
    print(f"  Models: {n_included_models} (excluded: {', '.join(EXCLUDED_VARIANTS)})")
    print(f"  Judge-labeled trials per model per temp: {judge_trials_per_model_per_temp}")
    print()
    print("Recommended analysis command:")
    print(analysis_cmd)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
