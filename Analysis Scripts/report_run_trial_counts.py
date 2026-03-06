#!/usr/bin/env python3
"""
Report trial and output counts per run, per condition, and highlight mismatches.

Usage:
  python report_run_trial_counts.py /path/to/runs_dir
  python report_run_trial_counts.py /path/to/runs_dir --metadata Comparing_Experiments/runs_metadata_v6.json
  python report_run_trial_counts.py /path/to/runs_dir --metadata Comparing_Experiments/runs_metadata_v6.json --csv
  python report_run_trial_counts.py /path/to/runs_dir --metadata Comparing_Experiments/runs_metadata_v6.json --save-csv out.csv

With --metadata: only runs listed in the JSON are included; label is temperature.
Without: all immediate subdirs containing simulation.db are included; label is run_dir name.

Output includes:
  - RUN SUMMARY: n_trials, n_with_output, n_missing_output per run (and sum).
  - TRIALS PER CONDITION: condition x run matrix.
  - MISSING OUTPUTS HIGHLIGHT: per-run missing output counts and V6 completion guidance.
  - MISMATCHES: runs with fewer trials than the maximum.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_runs(runs_dir: Path, metadata_path: Path | None) -> list[tuple[str, str, Path]]:
    """
    Return list of (label, run_id, db_path).
    Label is temperature string if metadata given, else run_dir name.
    """
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {runs_dir}")

    if metadata_path and metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        experiments = meta.get("experiments", {})
        out: list[tuple[str, str, Path]] = []
        for label, info in experiments.items():
            run_id = str(info.get("run_id", ""))
            run_dir_name = str(info.get("run_dir", ""))
            if not run_id or not run_dir_name:
                continue
            run_dir = runs_dir / run_dir_name
            db_path = run_dir / "simulation.db"
            if db_path.is_file():
                out.append((label, run_id, db_path))
        return out

    # No metadata: discover all subdirs with simulation.db
    out = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        db_path = sub / "simulation.db"
        if not db_path.is_file():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT run_id FROM runs LIMIT 1;").fetchone()
            conn.close()
            run_id = str(row["run_id"]) if row else sub.name
        except Exception:
            run_id = sub.name
        out.append((sub.name, run_id, db_path))
    return out


def get_run_counts(db_path: Path, run_id: str) -> dict[str, Any]:
    """Query trial and output counts by condition and model. Return dict with totals and breakdown."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Totals
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM conformity_trials WHERE run_id = ?;",
            (run_id,),
        ).fetchone()
        n_trials = int(row["n"]) if row else 0

        row = conn.execute(
            """
            SELECT COUNT(DISTINCT t.trial_id) AS n
            FROM conformity_trials t
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?;
            """,
            (run_id,),
        ).fetchone()
        n_with_output = int(row["n"]) if row else 0

        # By condition, model_id, variant
        rows = conn.execute(
            """
            SELECT
              c.name AS condition_name,
              t.model_id,
              t.variant,
              COUNT(*) AS n_trials,
              SUM(CASE WHEN o.output_id IS NOT NULL THEN 1 ELSE 0 END) AS n_with_output
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
            GROUP BY c.name, t.model_id, t.variant
            ORDER BY c.name, t.model_id, t.variant;
            """,
            (run_id,),
        ).fetchall()

        breakdown = [
            {
                "condition_name": r["condition_name"],
                "model_id": r["model_id"],
                "variant": r["variant"],
                "n_trials": r["n_trials"],
                "n_with_output": r["n_with_output"],
            }
            for r in rows
        ]

        return {
            "n_trials": n_trials,
            "n_with_output": n_with_output,
            "n_missing_output": n_trials - n_with_output,
            "breakdown": breakdown,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report trial/output counts per run and per condition; highlight mismatches."
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Path to directory containing run folders (each with simulation.db)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional JSON path: experiments.run_id/run_dir per ID; only these runs are used and labeled by ID (e.g. temperature).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print summary as CSV to stdout instead of formatted table.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Write summary CSV to this file path (in addition to the formatted report).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    metadata_path = Path(args.metadata).expanduser() if args.metadata else None
    if args.metadata and not metadata_path.is_file():
        metadata_path = (REPO_ROOT / args.metadata).resolve()
    if args.metadata and not metadata_path.is_file():
        print(f"Metadata file not found: {args.metadata}", file=sys.stderr)
        return 1

    try:
        runs = discover_runs(runs_dir, metadata_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    # Collect data for each run
    id_width = max(len(l) for l, _, _ in runs)
    id_width = max(id_width, 6)

    summary_rows: list[dict[str, Any]] = []
    all_breakdowns: list[tuple[str, list[dict]]] = []
    per_run_data: list[tuple[str, str, Path, dict]] = []

    for label, run_id, db_path in runs:
        data = get_run_counts(db_path, run_id)
        per_run_data.append((label, run_id, db_path, data))
        summary_rows.append({
            "id": label,
            "run_id": run_id[:8] + "…",
            "n_trials": data["n_trials"],
            "n_with_output": data["n_with_output"],
            "n_missing_output": data["n_missing_output"],
        })
        all_breakdowns.append((label, data["breakdown"]))

    # Mismatch: by (condition, model_id, variant) compute max count; flag runs that are below
    key_to_max: dict[tuple[str, str, str], int] = {}
    key_to_run_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for _label, _rid, _path, data in per_run_data:
        for b in data["breakdown"]:
            key = (b["condition_name"], b["model_id"], b["variant"])
            key_to_max[key] = max(key_to_max.get(key, 0), b["n_trials"])
            if key not in key_to_run_counts:
                key_to_run_counts[key] = {}
            key_to_run_counts[key][_label] = b["n_trials"]

    all_labels = [r["id"] for r in summary_rows]
    runs_with_shortage: set[str] = set()
    for key, by_run in key_to_run_counts.items():
        m = key_to_max[key]
        for run_label in all_labels:
            count = by_run.get(run_label, 0)
            if count < m:
                runs_with_shortage.add(run_label)

    total_trials = sum(r["n_trials"] for r in summary_rows)
    total_out = sum(r["n_with_output"] for r in summary_rows)
    total_miss = sum(r["n_missing_output"] for r in summary_rows)

    # ----- CSV output (stdout) -----
    if args.csv:
        print("id,run_id,n_trials,n_with_output,n_missing_output")
        for r in summary_rows:
            print(f"{r['id']},{r['run_id']},{r['n_trials']},{r['n_with_output']},{r['n_missing_output']}")
        return 0

    # ----- Save CSV to file if requested (and continue to formatted report) -----
    if args.save_csv:
        save_path = Path(args.save_csv).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("id,run_id,n_trials,n_with_output,n_missing_output\n")
            for r in summary_rows:
                f.write(f"{r['id']},{r['run_id']},{r['n_trials']},{r['n_with_output']},{r['n_missing_output']}\n")
        print(f"Summary CSV saved: {save_path}")

    # ----- Formatted report -----
    print("=" * 72)
    print("RUN SUMMARY (all IDs in folder)")
    print("=" * 72)
    fmt = f"  {{:<{id_width}}}  {{:>10}}  {{:>12}}  {{:>14}}"
    print(fmt.format("ID", "n_trials", "n_with_out", "n_missing_out"))
    print("-" * 72)
    for r in summary_rows:
        missing_flag = "  *** MISSING ***" if r["n_missing_output"] > 0 else ""
        print(fmt.format(r["id"], r["n_trials"], r["n_with_output"], r["n_missing_output"]) + missing_flag)
    print("-" * 72)
    print(fmt.format("(sum)", total_trials, total_out, total_miss))
    print()

    # Per-condition aggregate (one table: condition x run -> count)
    conditions = sorted({b["condition_name"] for _, _, _, data in per_run_data for b in data["breakdown"]})
    if conditions:
        print("TRIALS PER CONDITION (by run)")
        print("-" * 72)
        c_width = max(len(c) for c in conditions)
        c_width = max(c_width, 10)
        # Shorten run labels for table: use first 20 chars if many runs
        short_ids = [r["id"][:20] + "…" if len(r["id"]) > 20 else r["id"] for r in summary_rows]
        col_w = min(id_width, 22)
        header = f"  {{:<{c_width}}}".format("condition") + "  " + "  ".join(f"{s:>{col_w}}" for s in short_ids)
        print(header)
        print("-" * 72)
        for cond in conditions:
            row_vals = []
            for label, _, _, data in per_run_data:
                n = sum(b["n_trials"] for b in data["breakdown"] if b["condition_name"] == cond)
                row_vals.append(n)
            line = f"  {{:<{c_width}}}".format(cond[:c_width]) + "  " + "  ".join(f"{v:>{col_w}}" for v in row_vals)
            print(line)
        print()

    # Missing outputs highlight
    runs_with_missing = [r for r in summary_rows if r["n_missing_output"] > 0]
    print("MISSING OUTPUTS HIGHLIGHT")
    print("-" * 72)
    if runs_with_missing:
        print(f"  {len(runs_with_missing)} run(s) have trials without LLM outputs:")
        for r in runs_with_missing:
            pct = 100.0 * r["n_missing_output"] / r["n_trials"] if r["n_trials"] else 0.0
            print(f"    *** T={r['id']}: {r['n_missing_output']:,} missing  "
                  f"({pct:.1f}% of {r['n_trials']:,} trials)")
        print()
        print(f"  Total missing across all runs: {total_miss:,}")
        print()
        print("  V6 Publication approach:")
        print("    1. Generate balanced item set (intersection across runs):")
        print("       python 'Analysis Scripts/generate_v6_publication_item_set.py' \\")
        print("           --runs-dir <RUNS_DIR> --metadata Comparing_Experiments/runs_metadata_v6.json \\")
        print("           --out-dir Comparing_Experiments/v6_publication")
        print()
        print("    2. Run behavioral breakdown with all 12 conditions + publication filter:")
        print("       python 'Analysis Scripts/expanded_suite_behavioral_breakdown.py' \\")
        print("           --runs-dir <RUNS_DIR> --metadata Comparing_Experiments/runs_metadata_v6.json \\")
        print("           --publication-item-set Comparing_Experiments/v6_publication/item_set.csv \\")
        print("           --out-dir Comparing_Experiments/v6_publication --use-judge-labels")
        print()
        print("    3. (Optional) Backfill missing outputs on HPC:")
        print("       python -m vivarium olmo-conformity-complete-suite \\")
        print("           --runs-dir <RUNS_DIR> --metadata Comparing_Experiments/runs_metadata_v6.json")
    else:
        print("  None — all runs have complete LLM outputs.")
    print()

    # Mismatches: runs with total trials below max
    max_trials = max(r["n_trials"] for r in summary_rows)
    runs_below_max = [r["id"] for r in summary_rows if r["n_trials"] < max_trials]
    print("MISMATCHES (trial count below max)")
    print("-" * 72)
    print(f"  Max trials in any run: {max_trials}. Runs with fewer total trials: {len(runs_below_max)}")
    if runs_below_max:
        for run_label in sorted(runs_below_max):
            r = next(x for x in summary_rows if x["id"] == run_label)
            print(f"    - {run_label}  ({r['n_trials']} trials, {max_trials - r['n_trials']} short)")
        print()
        print("  Runs also below max for some (condition×model×variant):", len(runs_with_shortage))
    else:
        print("  None (all runs have the same total trial count).")
    print()

    print(f"TOTAL MISSING OUTPUTS (to be calculated/backfilled): {total_miss:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
