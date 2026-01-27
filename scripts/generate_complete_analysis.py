#!/usr/bin/env python3
"""
Complete Post-Processing Analysis Generator for Olmo Conformity Experiments.

This script orchestrates all post-processing analyses including:
1. Executing existing analytics functions (logit lens, interventions, probes)
2. Adding statistical tests and correlations
3. Generating comprehensive summary tables
4. Creating token-level analyses

Usage:
    python scripts/generate_complete_analysis.py --run-dir runs/20251217_002021_b2cc39a5-3d9d-444d-8489-bb74d6946973
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aam.analytics import (
    compute_behavioral_metrics,
    generate_behavioral_graphs,
    export_behavioral_logs,
    compute_probe_metrics,
    generate_probe_graphs,
    export_probe_logs,
    compute_intervention_metrics,
    generate_intervention_graphs,
    export_intervention_logs,
    compute_judgeval_metrics,
    generate_judgeval_graphs,
    export_judgeval_logs,
    compute_think_metrics,
    generate_think_graphs,
    export_think_logs,
    compute_token_metrics,
    generate_token_graphs,
    export_token_logs,
    compute_all_correlations,
)
from aam.analytics.utils import load_simulation_db, get_run_metadata, check_missing_prerequisites, save_metrics_json, save_table_csv, ensure_logs_dir


def extract_run_id(run_dir: str) -> str:
    """Extract run_id from run directory name."""
    dir_name = Path(run_dir).name
    # Format: YYYYMMDD_HHMMSS_run-id
    if "_" in dir_name:
        parts = dir_name.split("_")
        if len(parts) >= 3:
            return "_".join(parts[2:])  # Everything after timestamp
    return dir_name


def main():
    parser = argparse.ArgumentParser(description="Generate complete post-processing analysis")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--run-id", help="Run ID (auto-detected from directory if not provided)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Load database
    print(f"Loading database from: {run_dir}")
    try:
        db = load_simulation_db(run_dir)
    except Exception as e:
        print(f"Error loading database: {e}")
        return 1

    # Extract run_id
    run_id = args.run_id or extract_run_id(run_dir)
    print(f"Using run_id: {run_id}")

    # Check prerequisites
    print("\nChecking prerequisites...")
    prerequisites = check_missing_prerequisites(db, run_id)
    for name, exists in prerequisites.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {exists}")

    all_figures = {}
    all_logs = {}

    # 1. Behavioral Analysis
    print("\n" + "=" * 80)
    print("1. Behavioral Analysis")
    print("=" * 80)
    try:
        behavioral_metrics = compute_behavioral_metrics(db, run_id, run_dir)
        behavioral_figures = generate_behavioral_graphs(db, run_id, run_dir, behavioral_metrics)
        behavioral_logs = export_behavioral_logs(db, run_id, run_dir, behavioral_metrics)
        all_figures.update(behavioral_figures)
        all_logs.update(behavioral_logs)
        print(f"  Generated {len(behavioral_figures)} figures")
        print(f"  Exported {len(behavioral_logs)} log files")
    except Exception as e:
        print(f"  Error in behavioral analysis: {e}")

    # 2. Probe Analysis
    print("\n" + "=" * 80)
    print("2. Probe Projection Analysis")
    print("=" * 80)
    try:
        probe_metrics = compute_probe_metrics(db, run_id, run_dir)
        probe_figures = generate_probe_graphs(db, run_id, run_dir, probe_metrics)
        probe_logs = export_probe_logs(db, run_id, run_dir, probe_metrics)
        all_figures.update(probe_figures)
        all_logs.update(probe_logs)
        print(f"  Generated {len(probe_figures)} figures")
        print(f"  Exported {len(probe_logs)} log files")
    except Exception as e:
        print(f"  Error in probe analysis: {e}")

    # 3. Intervention Analysis
    print("\n" + "=" * 80)
    print("3. Intervention Analysis")
    print("=" * 80)
    try:
        intervention_metrics = compute_intervention_metrics(db, run_id, run_dir)
        intervention_figures = generate_intervention_graphs(db, run_id, run_dir, intervention_metrics)
        intervention_logs = export_intervention_logs(db, run_id, run_dir, intervention_metrics)
        all_figures.update(intervention_figures)
        all_logs.update(intervention_logs)
        print(f"  Generated {len(intervention_figures)} figures")
        print(f"  Exported {len(intervention_logs)} log files")
    except Exception as e:
        print(f"  Error in intervention analysis: {e}")

    # 4. Judge Eval Analysis
    print("\n" + "=" * 80)
    print("4. Judge Eval Analysis")
    print("=" * 80)
    try:
        judgeval_metrics = compute_judgeval_metrics(db, run_id, run_dir)
        if judgeval_metrics.get("statistics", {}).get("n_scores", 0) > 0:
            judgeval_figures = generate_judgeval_graphs(db, run_id, run_dir, judgeval_metrics)
            judgeval_logs = export_judgeval_logs(db, run_id, run_dir, judgeval_metrics)
            all_figures.update(judgeval_figures)
            all_logs.update(judgeval_logs)
            print(f"  Generated {len(judgeval_figures)} figures")
            print(f"  Exported {len(judgeval_logs)} log files")
        else:
            print("  No Judge Eval scores found, skipping")
    except Exception as e:
        print(f"  Error in judge eval analysis: {e}")

    # 5. Logit Lens Analysis (Think Tokens)
    print("\n" + "=" * 80)
    print("5. Logit Lens Analysis")
    print("=" * 80)
    try:
        think_metrics = compute_think_metrics(db, run_id, run_dir)
        think_figures = generate_think_graphs(db, run_id, run_dir, think_metrics)
        think_logs = export_think_logs(db, run_id, run_dir, think_metrics)
        all_figures.update(think_figures)
        all_logs.update(think_logs)
        print(f"  Generated {len(think_figures)} figures")
        print(f"  Exported {len(think_logs)} log files")
    except Exception as e:
        print(f"  Error in logit lens analysis: {e}")

    # 6. Token-Level Analysis
    print("\n" + "=" * 80)
    print("6. Token-Level Analysis")
    print("=" * 80)
    try:
        token_metrics = compute_token_metrics(db, run_id, run_dir)
        token_figures = generate_token_graphs(db, run_id, run_dir, token_metrics)
        token_logs = export_token_logs(db, run_id, run_dir, token_metrics)
        all_figures.update(token_figures)
        all_logs.update(token_logs)
        print(f"  Generated {len(token_figures)} figures")
        print(f"  Exported {len(token_logs)} log files")
    except Exception as e:
        print(f"  Error in token analysis: {e}")

    # 7. Cross-Analysis Correlations
    print("\n" + "=" * 80)
    print("7. Cross-Analysis Correlations")
    print("=" * 80)
    try:
        all_correlations = compute_all_correlations(db, run_id)
        
        # Save correlations
        paths = ensure_logs_dir(run_dir)
        correlations_path = os.path.join(paths["logs_dir"], "correlations.json")
        save_metrics_json(all_correlations, correlations_path)
        all_logs["correlations"] = correlations_path
        
        # Export correlation tables
        for corr_type, corr_data in all_correlations.items():
            if "correlations" in corr_data and len(corr_data["correlations"]) > 0:
                csv_path = os.path.join(paths["tables_dir"], f"correlations_{corr_type}.csv")
                save_table_csv(corr_data["correlations"], csv_path)
                all_logs[f"correlations_{corr_type}"] = csv_path
        
        print(f"  Computed {len(all_correlations)} correlation types")
        print(f"  Saved to: {correlations_path}")
    except Exception as e:
        print(f"  Error in correlation analysis: {e}")

    # 8. Generate Summary Tables
    print("\n" + "=" * 80)
    print("8. Generating Summary Tables")
    print("=" * 80)
    try:
        paths = ensure_logs_dir(run_dir)
        
        # Collect all metrics for summary
        summary_tables = {}
        
        # Behavioral summary
        try:
            behavioral_metrics = compute_behavioral_metrics(db, run_id, run_dir)
            if "accuracy_by_condition" in behavioral_metrics.get("metrics", {}):
                summary_tables["behavioral_accuracy"] = behavioral_metrics["metrics"]["accuracy_by_condition"]
            if "sycophancy_rate" in behavioral_metrics.get("metrics", {}):
                summary_tables["sycophancy_rate"] = behavioral_metrics["metrics"]["sycophancy_rate"]
        except Exception as e:
            print(f"  Warning: Could not generate behavioral summary: {e}")
        
        # Probe summary
        try:
            probe_metrics = compute_probe_metrics(db, run_id, run_dir)
            if "statistical_tests" in probe_metrics.get("metrics", {}):
                summary_tables["probe_statistical_tests"] = probe_metrics["metrics"]["statistical_tests"]
        except Exception as e:
            print(f"  Warning: Could not generate probe summary: {e}")
        
        # Intervention summary
        try:
            intervention_metrics = compute_intervention_metrics(db, run_id, run_dir)
            if "statistical_tests" in intervention_metrics.get("metrics", {}):
                summary_tables["intervention_statistical_tests"] = intervention_metrics["metrics"]["statistical_tests"]
            if "effect_size" in intervention_metrics.get("metrics", {}):
                summary_tables["intervention_effect_size"] = intervention_metrics["metrics"]["effect_size"]
        except Exception as e:
            print(f"  Warning: Could not generate intervention summary: {e}")
        
        # Save summary tables
        for table_name, table_data in summary_tables.items():
            if table_data:
                csv_path = os.path.join(paths["tables_dir"], f"summary_{table_name}.csv")
                save_table_csv(table_data, csv_path)
                all_logs[f"summary_{table_name}"] = csv_path
        
        print(f"  Generated {len(summary_tables)} summary tables")
    except Exception as e:
        print(f"  Error generating summary tables: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total figures generated: {len(all_figures)}")
    print(f"Total log files exported: {len(all_logs)}")
    
    # Save summary
    summary_path = os.path.join(run_dir, "artifacts", "analysis_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "figures": all_figures,
        "logs": all_logs,
        "prerequisites": prerequisites,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nAnalysis summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
