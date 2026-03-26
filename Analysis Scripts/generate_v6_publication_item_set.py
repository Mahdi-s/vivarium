#!/usr/bin/env python3
"""
V6 Publication Item-Set Generator.

Computes the intersection of items that have LLM outputs across all temperature
runs, covering all 12 suite conditions. Produces:
  - Comparing_Experiments/v6_publication/item_set.csv  (publication manifest)
  - Prints a summary report: recommended max_items_per_dataset, per-run counts,
    and highlighted missing outputs.

Usage:
  python generate_v6_publication_item_set.py \\
      --runs-dir runs_latest/runs \\
      --metadata Comparing_Experiments/runs_metadata_v5.json \\
      --out-dir Comparing_Experiments/v6_publication

All 12 suite conditions are used when computing the intersection:
  Behavioral:  control, asch_history_5, authoritative_bias
  Tone:        asch_zhu_unbiased_unanimous_{plain,neutral,confident,uncertain}
  Mitigation:  asch_zhu_unbiased_da, asch_zhu_unbiased_qd
  Format:      asch_zhu_unbiased_diverse_plain
  Authority:   authority_zhu_unbiased_trust, authority_zhu_unbiased_trust_da
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Full suite conditions (all 12)
BEHAVIORAL_CONDITIONS = ("control", "asch_history_5", "authoritative_bias")
TONE_CONDITIONS = (
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
)
MITIGATION_CONDITIONS = ("asch_zhu_unbiased_da", "asch_zhu_unbiased_qd")
FORMAT_CONDITIONS = ("asch_zhu_unbiased_diverse_plain",)
AUTHORITY_CONDITIONS = ("authority_zhu_unbiased_trust", "authority_zhu_unbiased_trust_da")

ALL_SUITE_CONDITIONS = (
    BEHAVIORAL_CONDITIONS
    + TONE_CONDITIONS
    + MITIGATION_CONDITIONS
    + FORMAT_CONDITIONS
    + AUTHORITY_CONDITIONS
)

N_SUITE_CONDITIONS = len(ALL_SUITE_CONDITIONS)
N_SUITE_MODELS = 8
N_SUITE_DATASETS = 8
INTENDED_MAX_ITEMS = 50  # intended per-dataset cap for all temps


def discover_runs(runs_dir: Path, metadata_path: Path) -> list[tuple[str, str, Path]]:
    """Return (label, run_id, db_path) ordered by label (temperature string)."""
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


def query_run_items(db_path: Path, run_id: str) -> dict[str, Any]:
    """
    Return per-run stats:
      items_with_output: set of item_ids that have ≥1 output in any suite condition
      per_dataset:       {dataset_name: {item_id, ...}} for items with output
      n_trials:          total trial count
      n_with_output:     trials with output (all conditions)
      n_missing:         trials without output
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Total trials
        n_trials = conn.execute(
            "SELECT COUNT(*) AS n FROM conformity_trials WHERE run_id = ?;",
            (run_id,),
        ).fetchone()["n"]

        n_with_output = conn.execute(
            """
            SELECT COUNT(DISTINCT t.trial_id) AS n
            FROM conformity_trials t
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?;
            """,
            (run_id,),
        ).fetchone()["n"]

        # Items that have at least one output in a suite condition
        placeholders = ",".join("?" * len(ALL_SUITE_CONDITIONS))
        rows = conn.execute(
            f"""
            SELECT DISTINCT
                i.item_id,
                d.name AS dataset_name
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_items i ON i.item_id = t.item_id
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({placeholders});
            """,
            [run_id, *ALL_SUITE_CONDITIONS],
        ).fetchall()

        per_dataset: dict[str, set[str]] = {}
        items_with_output: set[str] = set()
        for r in rows:
            items_with_output.add(r["item_id"])
            per_dataset.setdefault(r["dataset_name"], set()).add(r["item_id"])

        return {
            "n_trials": n_trials,
            "n_with_output": n_with_output,
            "n_missing": n_trials - n_with_output,
            "items_with_output": items_with_output,
            "per_dataset": per_dataset,
        }
    finally:
        conn.close()


def query_item_metadata(db_path: Path, run_id: str, item_ids: set[str]) -> list[dict[str, str]]:
    """Return [{item_id, dataset_name, domain, ground_truth_text}] for given item_ids."""
    if not item_ids:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT DISTINCT
                i.item_id,
                d.name AS dataset_name,
                i.domain,
                i.ground_truth_text
            FROM conformity_items i
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            WHERE i.item_id IN ({placeholders});
            """,
            list(item_ids),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V6 publication item set: intersection across all temperature runs."
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
        default="Comparing_Experiments/v6_publication",
        help="Output directory for item_set.csv and summary (default: Comparing_Experiments/v6_publication).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
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
    print("V6 Publication Item-Set Generator")
    print(f"  runs_dir:  {runs_dir}")
    print(f"  metadata:  {metadata_path}")
    print(f"  out_dir:   {out_dir}")
    print(f"  Conditions used: {N_SUITE_CONDITIONS}")
    print("=" * 72)

    runs = discover_runs(runs_dir, metadata_path)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    print(f"\n{len(runs)} run(s) found:")
    run_stats: list[dict[str, Any]] = []
    for label, run_id, db_path in runs:
        stats = query_run_items(db_path, run_id)
        stats["label"] = label
        stats["run_id"] = run_id
        stats["db_path"] = db_path
        run_stats.append(stats)
        print(
            f"  T={label}: {stats['n_trials']:>7} trials, "
            f"{stats['n_with_output']:>7} with output, "
            f"{stats['n_missing']:>7} missing output"
        )

    # Intersection: item_ids present (with output) in ALL runs
    all_item_sets = [s["items_with_output"] for s in run_stats]
    intersection: set[str] = all_item_sets[0].copy()
    for s in all_item_sets[1:]:
        intersection &= s

    print(f"\nItems with output in ALL runs: {len(intersection)}")
    print(f"  (items lost from union due to missing outputs in at least one run: "
          f"{len(set.union(*all_item_sets)) - len(intersection)})")

    # Per-dataset breakdown of the intersection
    # Use first run's metadata to get dataset names for intersection items
    meta_rows = query_item_metadata(run_stats[0]["db_path"], run_stats[0]["run_id"], intersection)
    dataset_to_items: dict[str, set[str]] = {}
    for r in meta_rows:
        dataset_to_items.setdefault(r["dataset_name"], set()).add(r["item_id"])

    print("\nPer-dataset counts in intersection:")
    dataset_counts: dict[str, int] = {}
    for ds, items in sorted(dataset_to_items.items()):
        dataset_counts[ds] = len(items)
        print(f"  {ds:<40} {len(items):>4} items")

    min_per_dataset = min(dataset_counts.values()) if dataset_counts else 0
    recommended_max = min(min_per_dataset, INTENDED_MAX_ITEMS)
    print(f"\n  Minimum items per dataset in intersection: {min_per_dataset}")
    print(f"  Suite intended max_items_per_dataset:      {INTENDED_MAX_ITEMS}")
    print(f"  Recommended V6 publication max_items_per_dataset: {recommended_max}")

    total_pub_trials = len(intersection) * N_SUITE_CONDITIONS * N_SUITE_MODELS
    print(f"\n  Total publication trials (intersection × {N_SUITE_CONDITIONS} conds × {N_SUITE_MODELS} models): {total_pub_trials:,}")

    # Highlight missing outputs within the intersection
    print("\nMissing outputs within intersection (per run):")
    any_missing_in_pub = False
    for s in run_stats:
        missing_in_pub = intersection - s["items_with_output"]
        flag = " *** MISSING ***" if missing_in_pub else ""
        print(f"  T={s['label']}: {len(missing_in_pub)} items in intersection without output{flag}")
        if missing_in_pub:
            any_missing_in_pub = True

    # Write item_set.csv
    item_set_path = out_dir / "item_set.csv"
    with open(item_set_path, "w", encoding="utf-8") as f:
        f.write("item_id,dataset_name,domain,ground_truth_text\n")
        meta_by_id = {r["item_id"]: r for r in meta_rows}
        for item_id in sorted(intersection):
            r = meta_by_id.get(item_id, {})
            ds = r.get("dataset_name", "")
            domain = r.get("domain", "")
            gt = (r.get("ground_truth_text") or "").replace('"', '""')
            f.write(f'"{item_id}","{ds}","{domain}","{gt}"\n')
    print(f"\nItem set written: {item_set_path} ({len(intersection)} items)")

    # Write missing_outputs_note.csv
    note_path = out_dir / "missing_outputs_note.csv"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("temperature,run_id,n_trials,n_with_output,n_missing_output\n")
        for s in run_stats:
            f.write(f"{s['label']},{s['run_id']},{s['n_trials']},{s['n_with_output']},{s['n_missing']}\n")
    print(f"Missing outputs note written: {note_path}")

    # Write a short text summary
    summary_path = out_dir / "item_set_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("V6 Publication Item-Set Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Suite conditions used: {N_SUITE_CONDITIONS}\n")
        f.write(f"  Behavioral ({len(BEHAVIORAL_CONDITIONS)}): {', '.join(BEHAVIORAL_CONDITIONS)}\n")
        f.write(f"  Tone ({len(TONE_CONDITIONS)}): {', '.join(TONE_CONDITIONS)}\n")
        f.write(f"  Mitigation ({len(MITIGATION_CONDITIONS)}): {', '.join(MITIGATION_CONDITIONS)}\n")
        f.write(f"  Format ({len(FORMAT_CONDITIONS)}): {', '.join(FORMAT_CONDITIONS)}\n")
        f.write(f"  Authority ({len(AUTHORITY_CONDITIONS)}): {', '.join(AUTHORITY_CONDITIONS)}\n\n")
        f.write(f"Suite models: {N_SUITE_MODELS}\n")
        f.write(f"Suite datasets: {N_SUITE_DATASETS}\n")
        f.write(f"Intended max_items_per_dataset: {INTENDED_MAX_ITEMS}\n\n")
        f.write("Per-run trial counts:\n")
        for s in run_stats:
            f.write(
                f"  T={s['label']}: {s['n_trials']} trials, "
                f"{s['n_with_output']} with output, "
                f"{s['n_missing']} MISSING OUTPUT\n"
            )
        f.write(f"\nIntersection size (items with output in ALL runs): {len(intersection)}\n")
        f.write(f"Recommended V6 publication max_items_per_dataset: {recommended_max}\n")
        f.write(f"Total V6 publication trials: {total_pub_trials:,} "
                f"({len(intersection)} items × {N_SUITE_CONDITIONS} conds × {N_SUITE_MODELS} models)\n\n")
        f.write("Per-dataset counts in intersection:\n")
        for ds, cnt in sorted(dataset_counts.items()):
            f.write(f"  {ds:<40} {cnt:>4}\n")
        if any_missing_in_pub:
            f.write("\nWARNING: Some runs have items in the intersection without output.\n")
            f.write("These will be excluded when loading behavioral data with output-only JOINs.\n")
    print(f"Summary written: {summary_path}")

    print("\n" + "=" * 72)
    print("DONE. Use item_set.csv with --publication-item-set in the behavioral analysis.")
    print(f"Recommended max_items_per_dataset: {recommended_max}")
    if any_missing_in_pub:
        print("WARNING: Some intersection items have missing outputs in one or more runs.")
        print("  These will be automatically excluded by output JOIN in the analysis pipeline.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
