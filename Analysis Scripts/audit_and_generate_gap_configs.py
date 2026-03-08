#!/usr/bin/env python3
"""
Audit Database Schema Completeness & Generate Gap-Fill Configs

This script:
1. Audits all temperature runs to find missing (variant, condition) cells
2. Generates minimal suite configs to fill only the gaps
3. Can optionally update metadata to track completion status

Usage:
    python audit_and_generate_gap_configs.py --audit
    python audit_and_generate_gap_configs.py --generate-gaps
    python audit_and_generate_gap_configs.py --both
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configuration
DEFAULT_RUNS_DIR = Path("runs_latest/runs")
DEFAULT_CONFIG_DIR = Path("experiments/olmo_conformity/configs")
DEFAULT_METADATA_PATH = Path("Comparing_Experiments/runs_metadata_v6.json")
BASE_SUITE_PATH = Path("experiments/olmo_conformity/configs/suite_expanded_temp0.0.json")


@dataclass
class Gap:
    """Represents a missing (variant, condition_id, condition_name) cell."""
    temperature: float
    variant: str
    condition_id: str
    condition_name: str

    def __repr__(self) -> str:
        return f"{self.variant:20s} × {self.condition_name:45s} @ T={self.temperature}"


def audit_database(runs_dir: Path, metadata_path: Path) -> Dict[float, List[Gap]]:
    """
    Audit all databases to find missing cells.

    Returns:
        {temperature: [Gap, ...]}
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    gap_analysis: Dict[float, List[Gap]] = defaultdict(list)

    for temp_key, temp_entry in sorted(metadata.get("experiments", {}).items()):
        temp = temp_entry.get("temperature")
        run_dir = runs_dir / temp_entry.get("run_dir", "")
        db_path = run_dir / "simulation.db"

        if not db_path.exists():
            print(f"T={temp}: DB not found at {db_path}")
            continue

        db = sqlite3.connect(db_path)
        try:
            # Get all possible (condition, variant) combinations
            cursor = db.execute("""
                SELECT DISTINCT c.condition_id, c.name, t.variant
                FROM conformity_conditions c
                CROSS JOIN (SELECT DISTINCT variant FROM conformity_trials) t
                ORDER BY c.name, t.variant
            """)
            all_combos = {(row[0], row[1], row[2]) for row in cursor.fetchall()}

            # Get actual combinations with data
            cursor = db.execute("""
                SELECT DISTINCT c.condition_id, c.name, t.variant
                FROM conformity_trials t
                JOIN conformity_conditions c ON t.condition_id = c.condition_id
            """)
            actual = {(row[0], row[1], row[2]) for row in cursor.fetchall()}

            # Find gaps
            gaps = []
            for cond_id, cond_name, variant in all_combos:
                if (cond_id, cond_name, variant) not in actual:
                    gaps.append(Gap(
                        temperature=temp,
                        variant=variant,
                        condition_id=cond_id,
                        condition_name=cond_name
                    ))

            if gaps:
                gap_analysis[temp] = gaps
        finally:
            db.close()

    return gap_analysis


def generate_gap_fill_configs(
    gap_analysis: Dict[float, List[Gap]],
    base_suite_path: Path,
    output_dir: Path,
) -> Dict[float, Path]:
    """
    Generate minimal suite configs for gap-filling.

    Returns:
        {temperature: path_to_config}
    """
    with open(base_suite_path) as f:
        base_suite = json.load(f)

    output_paths = {}

    for temp, gaps in sorted(gap_analysis.items()):
        if not gaps:
            continue

        # Group gaps by variant
        variants_to_fill = {gap.variant for gap in gaps}
        conditions_to_fill = {gap.condition_name for gap in gaps}

        # Build targeted suite config
        config = base_suite.copy()
        config["suite_name"] = "olmo_conformity_surgical_gap_fill"
        config["suite_version"] = "v6_gaps"
        config["description"] = f"Surgical completion: {len(variants_to_fill)} variant(s) × {len(conditions_to_fill)} condition(s) at T={temp}"

        # Filter models to only those with gaps
        config["models"] = [
            m for m in config.get("models", [])
            if m.get("variant") in variants_to_fill
        ]

        # Set temperature to this specific value
        for model in config["models"]:
            model["temperatures"] = [temp]

        # Filter conditions to only those with gaps
        config["conditions"] = [
            c for c in config.get("conditions", [])
            if c.get("name") in conditions_to_fill
        ]

        # Save
        temp_str = str(temp).replace(".", "p")
        output_path = output_dir / f"suite_surgical_gap_fill_t{temp_str}.json"

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        output_paths[temp] = output_path
        print(f"Generated: {output_path}")
        print(f"  Variants: {sorted(variants_to_fill)}")
        print(f"  Conditions: {len(conditions_to_fill)}")
        print(f"  Estimated trials: {len(variants_to_fill) * len(conditions_to_fill) * 8}")
        print()

    return output_paths


def print_audit_summary(gap_analysis: Dict[float, List[Gap]]) -> None:
    """Print formatted audit results."""
    print("\n" + "="*80)
    print("DATABASE COMPLETENESS AUDIT")
    print("="*80)

    total_gaps = 0
    for temp in sorted(gap_analysis.keys()):
        gaps = gap_analysis[temp]
        total_gaps += len(gaps)

        if gaps:
            print(f"\nT={temp}: {len(gaps)} GAPS")
            # Group by variant
            by_variant = defaultdict(list)
            for gap in gaps:
                by_variant[gap.variant].append(gap.condition_name)

            for variant in sorted(by_variant.keys()):
                conditions = by_variant[variant]
                print(f"  {variant}: {len(conditions)} missing conditions")
                if len(conditions) <= 10:
                    for cond in sorted(conditions):
                        print(f"    - {cond}")
        else:
            print(f"T={temp}: ✓ COMPLETE")

    print(f"\n{'='*80}")
    print(f"TOTAL GAPS: {total_gaps} cells")
    print(f"ESTIMATED COST TO FILL: {total_gaps * 8} trials (negligible)")
    print(f"{'='*80}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit database completeness and generate gap-fill configs"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Base directory for runs (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Path to runs metadata JSON (default: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Where to save generated configs (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--base-suite",
        type=Path,
        default=BASE_SUITE_PATH,
        help=f"Base suite config to clone from (default: {BASE_SUITE_PATH})",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Only audit, don't generate configs",
    )
    parser.add_argument(
        "--generate-gaps",
        action="store_true",
        help="Only generate gap-fill configs, don't audit",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Do both audit and generation (default if neither specified)",
    )

    args = parser.parse_args()

    # Determine what to do
    do_audit = args.audit or (not args.generate_gaps and not args.both)
    do_generate = args.generate_gaps or args.both

    # Run audit
    if do_audit:
        print("Auditing databases...")
        gap_analysis = audit_database(args.runs_dir, args.metadata)
        print_audit_summary(gap_analysis)

        if do_generate:
            print("Generating gap-fill configs...")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_paths = generate_gap_fill_configs(
                gap_analysis,
                args.base_suite,
                args.output_dir,
            )

            if config_paths:
                print("\n" + "="*80)
                print("TO RUN GAP FILLS:")
                print("="*80)
                for temp, config_path in sorted(config_paths.items()):
                    print(f"\nT={temp}:")
                    print(f"  python experiments/olmo_conformity/configs/run_expanded_experiments.py \\")
                    print(f"    --runs-dir {args.runs_dir} \\")
                    print(f"    --suite {config_path} \\")
                    print(f"    --phase trials --runs-only")
            else:
                print("\n✓ No gaps found. Database is complete.")

    return 0


if __name__ == "__main__":
    exit(main())
