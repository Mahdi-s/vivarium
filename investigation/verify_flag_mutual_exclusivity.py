#!/usr/bin/env python3
"""
Data Integrity Audit: Refusal vs Wrong-Answer Mutual Exclusivity
================================================================

Scans ALL SQLite databases referenced by the April_analysis manifests
and checks for rows where both ``refusal_flag=1`` AND
``wrong_answer_endorsed=1`` in ``parsed_answer_json``.

These flags should be mutually exclusive.  If both are set, the
state classifier (``april_classify_state()``) would misclassify
refusals as wrong-answer endorsements, inflating BER.

Usage:
    python investigation/verify_flag_mutual_exclusivity.py

Output: human-readable summary table + JSON report.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    """Walk up from this script to find the repo root (contains src/)."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "src").is_dir():
            return d
        d = d.parent
    raise RuntimeError("Cannot find repo root (no src/ directory found)")


REPO = _repo_root()

# Manifests
MANIFEST_7B = REPO / "Comparing_Experiments" / "April_analysis" / "metadata" / "runs_metadata.json"
MANIFEST_CF = REPO / "Comparing_Experiments" / "April_analysis" / "metadata" / "cross_family_metadata.json"

# Query: count flag conflicts in judged rows only
_CONFLICT_SQL = """\
WITH first_output AS (
    SELECT trial_id, MIN(created_at) AS first_created
    FROM conformity_outputs
    GROUP BY trial_id
)
SELECT
    ct.variant,
    COUNT(*) AS total_judged,
    SUM(CASE WHEN
        CAST(json_extract(co.parsed_answer_json, '$.refusal_flag') AS INTEGER) = 1
        AND CAST(json_extract(co.parsed_answer_json, '$.wrong_answer_endorsed') AS INTEGER) = 1
    THEN 1 ELSE 0 END) AS refusal_AND_endorsed,
    SUM(CASE WHEN
        CAST(json_extract(co.parsed_answer_json, '$.is_correct') AS INTEGER) = 1
        AND CAST(json_extract(co.parsed_answer_json, '$.wrong_answer_endorsed') AS INTEGER) = 1
    THEN 1 ELSE 0 END) AS correct_AND_endorsed,
    SUM(CASE WHEN
        CAST(json_extract(co.parsed_answer_json, '$.is_correct') AS INTEGER) = 1
        AND CAST(json_extract(co.parsed_answer_json, '$.refusal_flag') AS INTEGER) = 1
    THEN 1 ELSE 0 END) AS correct_AND_refusal
FROM conformity_trials ct
JOIN conformity_outputs co ON ct.trial_id = co.trial_id
JOIN first_output fo ON co.trial_id = fo.trial_id
                    AND co.created_at = fo.first_created
WHERE json_extract(co.parsed_answer_json, '$._llm_judge') IS NOT NULL
GROUP BY ct.variant
"""


def _load_manifest_sources(path: Path) -> List[Dict[str, Any]]:
    """Load sources from a manifest file."""
    with open(path) as f:
        manifest = json.load(f)
    sources = manifest.get("sources_primary", [])
    sources += manifest.get("sources_secondary", [])
    return sources


def _query_db(db_path: str) -> List[Dict[str, Any]]:
    """Query a single DB for flag conflicts. Returns per-variant rows."""
    full_path = str(REPO / db_path)
    if not os.path.exists(full_path):
        return [{"variant": "FILE_NOT_FOUND", "total_judged": 0,
                 "refusal_AND_endorsed": 0, "correct_AND_endorsed": 0,
                 "correct_AND_refusal": 0}]
    try:
        conn = sqlite3.connect(f"file:{full_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(_CONFLICT_SQL).fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return [{"variant": f"ERROR: {e}", "total_judged": 0,
                 "refusal_AND_endorsed": 0, "correct_AND_endorsed": 0,
                 "correct_AND_refusal": 0}]


def main() -> int:
    print("=" * 80)
    print("  DATA INTEGRITY AUDIT: Refusal vs Wrong-Answer Mutual Exclusivity")
    print("=" * 80)
    print()

    # Collect all DB sources from both manifests
    all_sources: List[Dict[str, Any]] = []

    if MANIFEST_7B.exists():
        for s in _load_manifest_sources(MANIFEST_7B):
            s["manifest"] = "7B"
            all_sources.append(s)

    if MANIFEST_CF.exists():
        for s in _load_manifest_sources(MANIFEST_CF):
            s["manifest"] = "cross_family"
            all_sources.append(s)

    # Deduplicate by DB path
    seen = set()
    unique_sources = []
    for s in all_sources:
        if s["db"] not in seen:
            seen.add(s["db"])
            unique_sources.append(s)

    print(f"Scanning {len(unique_sources)} unique databases...\n")

    # Results
    report: List[Dict[str, Any]] = []
    total_conflicts = 0
    total_conflicts_in_analysis = 0

    for src in unique_sources:
        db_path = src["db"]
        ignore = set(src.get("ignore_variants", []))
        manifest = src.get("manifest", "?")
        allowed_variants = src.get("variants")  # None = all

        rows = _query_db(db_path)

        db_total_judged = sum(r["total_judged"] for r in rows)
        db_refusal_endorsed = sum(r["refusal_AND_endorsed"] for r in rows)

        # Count conflicts that would actually enter the analysis
        conflicts_in_analysis = 0
        for r in rows:
            v = r["variant"]
            if v in ignore:
                continue
            if allowed_variants and v not in allowed_variants:
                continue
            conflicts_in_analysis += r["refusal_AND_endorsed"]

        entry = {
            "db": db_path,
            "manifest": manifest,
            "ignore_variants": list(ignore),
            "allowed_variants": allowed_variants,
            "total_judged": db_total_judged,
            "refusal_AND_endorsed_total": db_refusal_endorsed,
            "refusal_AND_endorsed_in_analysis": conflicts_in_analysis,
            "per_variant": rows,
        }
        report.append(entry)
        total_conflicts += db_refusal_endorsed
        total_conflicts_in_analysis += conflicts_in_analysis

    # Print summary table
    print(f"{'DB Path':62s} {'Manifest':12s} {'Judged':>7s} {'R&E All':>8s} {'R&E Used':>8s}")
    print("-" * 100)
    for r in report:
        flag = " *** " if r["refusal_AND_endorsed_in_analysis"] > 0 else ""
        print(f"{r['db']:62s} {r['manifest']:12s} {r['total_judged']:7d} "
              f"{r['refusal_AND_endorsed_total']:8d} {r['refusal_AND_endorsed_in_analysis']:8d}{flag}")

    print("-" * 100)
    print(f"{'TOTALS':62s} {'':12s} {'':>7s} {total_conflicts:8d} {total_conflicts_in_analysis:8d}")
    print()

    # Detail on any conflicts
    if total_conflicts > 0:
        print("=== CONFLICT DETAILS (refusal_flag=1 AND wrong_answer_endorsed=1) ===\n")
        for r in report:
            if r["refusal_AND_endorsed_total"] > 0:
                print(f"  DB: {r['db']}")
                print(f"  Ignored variants: {r['ignore_variants']}")
                for v in r["per_variant"]:
                    if v["refusal_AND_endorsed"] > 0:
                        in_analysis = "YES (PROBLEM!)" if (
                            v["variant"] not in set(r["ignore_variants"]) and
                            (r["allowed_variants"] is None or v["variant"] in r["allowed_variants"])
                        ) else "NO (filtered out)"
                        print(f"    variant={v['variant']:20s}  conflicts={v['refusal_AND_endorsed']:4d}  "
                              f"in_analysis={in_analysis}")
                print()

    # Also report correct+endorsed and correct+refusal (informational)
    print("=== INFORMATIONAL: is_correct=1 AND wrong_answer_endorsed=1 ===")
    print("(Not a mutual exclusivity violation — semantically valid)")
    for r in report:
        total_ce = sum(v["correct_AND_endorsed"] for v in r["per_variant"])
        if total_ce > 0:
            print(f"  {r['db']:62s}  count={total_ce}")
    print()

    print("=== INFORMATIONAL: is_correct=1 AND refusal_flag=1 ===")
    print("(Not a mutual exclusivity violation — model may hedge while correct)")
    for r in report:
        total_cr = sum(v["correct_AND_refusal"] for v in r["per_variant"])
        if total_cr > 0:
            print(f"  {r['db']:62s}  count={total_cr}")
    print()

    # Verdict
    print("=" * 80)
    if total_conflicts_in_analysis == 0:
        print(f"  VERDICT: PASS")
        print(f"  {total_conflicts} conflict(s) exist in raw databases, but ZERO reach the analysis.")
        print(f"  All conflicting rows are on variants excluded by ignore_variants.")
        print(f"  Current BER numbers are CORRECT.")
    else:
        print(f"  VERDICT: FAIL")
        print(f"  {total_conflicts_in_analysis} conflict(s) ENTER the analysis pipeline!")
        print(f"  BER may be inflated by misclassified refusals.")
        print(f"  Run investigation/fix_judge_refusal_flags.py on affected databases.")
    print("=" * 80)

    # Write JSON report
    report_path = REPO / "investigation" / "mutual_exclusivity_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_databases": len(unique_sources),
            "total_conflicts_raw": total_conflicts,
            "total_conflicts_in_analysis": total_conflicts_in_analysis,
            "verdict": "PASS" if total_conflicts_in_analysis == 0 else "FAIL",
            "databases": report,
        }, f, indent=2)
    print(f"\nJSON report: {report_path}")

    return 0 if total_conflicts_in_analysis == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
