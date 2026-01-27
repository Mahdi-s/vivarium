"""
End-to-end experiment orchestration.

This module provides a single function to run the complete Olmo conformity experiment:
1. Behavioral trials
2. Probe training
3. Projection computation
4. Interventions (optional)
5. Analysis and reporting
6. Scientific report generation (validity verification)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aam.analytics.reporting import ScientificReportGenerator
from aam.persistence import TraceDb, TraceDbConfig

from .analysis import generate_core_figures
from .intervention import run_intervention_sweep
from .runner import RunPaths, run_suite
from .vector_analysis import run_truth_social_vector_analysis


@dataclass
class ExperimentConfig:
    """Configuration for full experiment run."""
    suite_config_path: str
    runs_dir: str
    run_id: Optional[str] = None
    
    # Trial execution
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit_enabled: bool = True
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    rate_limit_max_concurrent: int = 10
    
    # Activation capture
    capture_activations: bool = False
    capture_layers: Optional[List[int]] = None
    capture_components: Optional[List[str]] = None
    capture_dtype: str = "float16"
    
    # Probe training
    truth_probe_dataset_path: Optional[str] = None
    social_probe_dataset_path: Optional[str] = None
    probe_layers: Optional[List[int]] = None
    probe_component: str = "hook_resid_post"
    probe_token_position: int = -1
    
    # Interventions
    run_interventions: bool = False
    intervention_layers: Optional[List[int]] = None
    intervention_alphas: Optional[List[float]] = None
    social_probe_artifact_path: Optional[str] = None
    social_probe_id: Optional[str] = None
    
    # Analysis
    generate_reports: bool = True
    run_vector_analysis: bool = False
    temperature: float = 0.0  # Added to track config temperature


def run_full_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete Olmo conformity experiment workflow.
    
    Returns dict with:
    - run_id: Experiment run ID
    - run_paths: Paths to output directories
    - trial_stats: Statistics about trials
    - probe_ids: IDs of trained probes
    - intervention_stats: Statistics about interventions
    - analysis_artifacts: Paths to generated analysis files
    """
    results: Dict[str, Any] = {
        "run_id": None,
        "run_paths": None,
        "trial_stats": {},
        "probe_ids": {},
        "intervention_stats": {},
        "analysis_artifacts": {},
    }
    
    print("="*60)
    print("Olmo Conformity Experiment - Full Workflow")
    print("="*60)
    
    # Step 1: Run behavioral trials
    print("\n[Step 1/5] Running behavioral trials...")
    paths = run_suite(
        suite_config_path=config.suite_config_path,
        runs_dir=config.runs_dir,
        run_id=config.run_id,
        api_base=config.api_base,
        api_key=config.api_key,
        rate_limit_enabled=config.rate_limit_enabled,
        rate_limit_rpm=config.rate_limit_rpm,
        rate_limit_tpm=config.rate_limit_tpm,
        rate_limit_max_concurrent=config.rate_limit_max_concurrent,
        capture_activations=config.capture_activations,
        capture_layers=config.capture_layers,
        capture_components=config.capture_components,
        capture_dtype=config.capture_dtype,
    )
    
    results["run_id"] = paths.run_dir.split("_")[-1] if "_" in paths.run_dir else None
    results["run_paths"] = {
        "run_dir": paths.run_dir,
        "db_path": paths.db_path,
        "artifacts_dir": paths.artifacts_dir,
        "figures_dir": paths.figures_dir,
        "tables_dir": paths.tables_dir,
    }
    
    # Get trial stats
    trace_db = TraceDb(TraceDbConfig(db_path=paths.db_path))
    trace_db.connect()
    trace_db.init_schema()
    
    trial_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM conformity_trials WHERE run_id = ?;",
        (results["run_id"],)
    ).fetchone()[0]
    
    results["trial_stats"] = {
        "total_trials": trial_count,
        "run_dir": paths.run_dir,
    }
    print(f"  Completed {trial_count} trials")
    
    # Step 2: Train probes (if datasets provided)
    # SCIENTIFIC RIGOR: Train probes PER VARIANT to prevent cross-model probe leakage
    if config.truth_probe_dataset_path:
        print("\n[Step 2/5] Training probes (per variant for scientific rigor)...")
        probe_layers = config.probe_layers or list(range(32))
        
        # Get all unique (model_id, variant) pairs from trials
        variant_rows = trace_db.conn.execute(
            """
            SELECT DISTINCT model_id, variant
            FROM conformity_trials
            WHERE run_id = ?
            ORDER BY variant;
            """,
            (results["run_id"],)
        ).fetchall()
        
        if not variant_rows:
            print("  Warning: No trials found, skipping probe training")
        else:
            # Train probes for each variant separately
            results["probe_ids"] = {}
            
            for vr in variant_rows:
                model_id = vr["model_id"]
                variant = vr["variant"]
                
                print(f"\n  Training probes for variant: {variant}")
                vector_results = run_truth_social_vector_analysis(
                    trace_db=trace_db,
                    run_id=results["run_id"],
                    model_id=model_id,
                    variant=variant,  # CRITICAL: Train per variant
                    truth_probe_dataset_path=config.truth_probe_dataset_path,
                    social_probe_dataset_path=config.social_probe_dataset_path,
                    layers=probe_layers,
                    component=config.probe_component,
                    token_position=config.probe_token_position,
                    dtype=config.capture_dtype,
                    artifacts_dir=paths.artifacts_dir,
                    temperature=config.temperature,  # Pass temperature from config
                )
            
                results["probe_ids"][variant] = {
                    "truth_probe_id": vector_results["truth_probe_id"],
                    "social_probe_id": vector_results.get("social_probe_id"),
                }
                results["analysis_artifacts"].update(vector_results.get("analysis_artifacts", {}))
                print(f"    Trained truth probe: {vector_results['truth_probe_id']}")
                if vector_results.get("social_probe_id"):
                    print(f"    Trained social probe: {vector_results['social_probe_id']}")
            
            print(f"\n  Trained probes for {len(variant_rows)} variant(s)")
    else:
        print("\n[Step 2/5] Skipping probe training (no dataset provided)")
    
    # Step 3: Run interventions (if enabled)
    # Note: Interventions should also be per-variant to maintain scientific rigor
    if config.run_interventions and config.social_probe_artifact_path and config.social_probe_id:
        print("\n[Step 3/5] Running interventions...")
        intervention_layers = config.intervention_layers or [15, 16, 17, 18, 19, 20]
        intervention_alphas = config.intervention_alphas or [0.5, 1.0, 2.0]
        
        # Get model_id from first trial
        model_row = trace_db.conn.execute(
            "SELECT model_id FROM conformity_trials WHERE run_id = ? LIMIT 1;",
            (results["run_id"],)
        ).fetchone()
        
        if model_row:
            model_id = model_row["model_id"]
            inserted = run_intervention_sweep(
                trace_db=trace_db,
                run_id=results["run_id"],
                model_id=model_id,
                probe_artifact_path=config.social_probe_artifact_path,
                social_probe_id=config.social_probe_id,
                target_layers=intervention_layers,
                component_hook=config.probe_component,
                alpha_values=intervention_alphas,
                max_new_tokens=64,
                temperature=config.temperature,
            )
            
            results["intervention_stats"] = {
                "intervention_results_inserted": inserted,
            }
            print(f"  Completed {inserted} intervention trials")
        else:
            print("  Warning: No trials found, skipping interventions")
    elif config.run_interventions and results.get("probe_ids"):
        # Use per-variant probes if available
        print("\n[Step 3/5] Running per-variant interventions...")
        intervention_layers = config.intervention_layers or [15, 16, 17, 18, 19, 20]
        intervention_alphas = config.intervention_alphas or [0.5, 1.0, 2.0]
        
        results["intervention_stats"] = {"by_variant": {}}
        total_inserted = 0
        
        for variant, probe_ids in results.get("probe_ids", {}).items():
            if not probe_ids.get("social_probe_id"):
                continue
            
            # Get model_id for this variant
            model_row = trace_db.conn.execute(
                "SELECT model_id FROM conformity_trials WHERE run_id = ? AND variant = ? LIMIT 1;",
                (results["run_id"], variant)
            ).fetchone()
            
            if not model_row:
                continue
            
            model_id = model_row["model_id"]
            # Artifact path includes variant suffix
            artifact_path = os.path.join(paths.artifacts_dir, f"social_probe_{variant}.safetensors")
            
            if not os.path.exists(artifact_path):
                print(f"  Warning: Social probe artifact not found for {variant}, skipping")
                continue
            
            inserted = run_intervention_sweep(
                trace_db=trace_db,
                run_id=results["run_id"],
                model_id=model_id,
                probe_artifact_path=artifact_path,
                social_probe_id=probe_ids["social_probe_id"],
                target_layers=intervention_layers,
                component_hook=config.probe_component,
                alpha_values=intervention_alphas,
                max_new_tokens=64,
                temperature=config.temperature,
            )
            
            results["intervention_stats"]["by_variant"][variant] = inserted
            total_inserted += inserted
            print(f"    {variant}: {inserted} intervention trials")
        
        results["intervention_stats"]["total_inserted"] = total_inserted
        print(f"  Completed {total_inserted} total intervention trials")
    else:
        print("\n[Step 3/5] Skipping interventions (not enabled or missing probe)")
    
    # Step 4: Generate reports
    if config.generate_reports:
        print("\n[Step 4/5] Generating analysis reports...")
        figures = generate_core_figures(
            trace_db=trace_db,
            run_id=results["run_id"],
            run_dir=paths.run_dir,
        )
        results["analysis_artifacts"].update(figures)
        print(f"  Generated {len(figures)} figure(s)")
    else:
        print("\n[Step 4/5] Skipping report generation")
    
    # Step 5: Vector analysis (if enabled)
    if config.run_vector_analysis and config.truth_probe_dataset_path:
        print("\n[Step 5/6] Running vector analysis...")
        # This was already done in Step 2 if probes were trained
        if results["probe_ids"].get("truth_probe_id"):
            print("  Vector analysis completed in Step 2")
        else:
            print("  Warning: Probes not trained, skipping vector analysis")
    else:
        print("\n[Step 5/6] Skipping vector analysis")
    
    # Step 6: Generate Scientific Report
    print("\n[Step 6/6] Generating Scientific Report...")
    try:
        reporter = ScientificReportGenerator(Path(paths.run_dir))
        report = reporter.generate()
        report_path = Path(paths.artifacts_dir) / "scientific_report.json"
        report.save(str(report_path))
        results["scientific_report_path"] = str(report_path)
        results["scientific_report_summary"] = {
            "integrity_verified": report.integrity_verified,
            "dual_stack_risk": report.dual_stack_risk,
            "anomalies_count": len(report.anomalies),
            "metrics": report.metrics,
        }
        print(f"  Report saved to: {report_path}")
        print(f"  Integrity verified: {report.integrity_verified}")
        if report.dual_stack_risk:
            print("  ⚠ WARNING: Dual-stack risk detected (different model weights for inference vs. probing)")
        if report.anomalies:
            print(f"  Anomalies detected ({len(report.anomalies)}):")
            for anomaly in report.anomalies[:3]:
                print(f"    - {anomaly}")
            if len(report.anomalies) > 3:
                print(f"    ... and {len(report.anomalies) - 3} more")
        reporter.close()
    except Exception as e:
        print(f"  Warning: Scientific report generation failed: {e}")
        results["scientific_report_error"] = str(e)
    
    trace_db.close()
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Run ID: {results['run_id']}")
    print(f"Run Directory: {paths.run_dir}")
    print(f"Database: {paths.db_path}")
    print(f"Artifacts: {paths.artifacts_dir}")
    if "scientific_report_path" in results:
        print(f"Scientific Report: {results['scientific_report_path']}")
    
    return results
