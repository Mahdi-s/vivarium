"""
Intervention analytics for Olmo Conformity Experiment.

Implements Figure 6 (Intervention Impact) and intervention effect metrics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas and matplotlib are required for intervention analytics")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.analytics.statistics import compute_ttest, compute_effect_size_binary
from aam.persistence import TraceDb


def compute_intervention_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute intervention metrics: flip-to-truth rate, effect size, success rate.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Check if interventions exist
    intervention_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM conformity_interventions WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    
    if intervention_count == 0:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No interventions found for this run"},
        }
    
    # Load intervention results
    df = pd.read_sql_query(
        """
        SELECT 
            r.result_id,
            r.trial_id,
            r.intervention_id,
            r.flipped_to_truth,
            i.name AS intervention_name,
            i.alpha,
            i.target_layers_json,
            t.variant,
            c.name AS condition_name
        FROM conformity_intervention_results r
        JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
        JOIN conformity_trials t ON t.trial_id = r.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE i.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No intervention results found"},
        }
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Flip-to-truth rate by alpha and layer
    flip_rate = (
        df.groupby(["variant", "intervention_name", "alpha"])["flipped_to_truth"]
        .agg(flip_rate="mean", n_trials="count")
        .reset_index()
    )
    metrics["metrics"]["flip_to_truth_rate"] = flip_rate.to_dict("records")
    
    # Effect size: before/after correctness comparison
    # Load before/after outputs
    before_after = pd.read_sql_query(
        """
        SELECT 
            r.result_id,
            r.flipped_to_truth,
            o_before.is_correct AS correctness_before,
            o_after.is_correct AS correctness_after,
            i.name AS intervention_name,
            i.alpha,
            t.variant
        FROM conformity_intervention_results r
        JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
        JOIN conformity_outputs o_before ON o_before.output_id = r.output_id_before
        JOIN conformity_outputs o_after ON o_after.output_id = r.output_id_after
        JOIN conformity_trials t ON t.trial_id = r.trial_id
        WHERE i.run_id = ? AND o_before.is_correct IS NOT NULL AND o_after.is_correct IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if not before_after.empty:
        before_after["effect"] = before_after["correctness_after"] - before_after["correctness_before"]
        effect_size = (
            before_after.groupby(["variant", "intervention_name", "alpha"])["effect"]
            .agg(mean_effect="mean", std_effect="std", n_trials="count")
            .reset_index()
        )
        metrics["metrics"]["effect_size"] = effect_size.to_dict("records")
        
        # Intervention success rate (fraction that improved)
        before_after["improved"] = (before_after["effect"] > 0).astype(int)
        success_rate = (
            before_after.groupby(["variant", "intervention_name", "alpha"], as_index=False)
            ["improved"]
            .mean()
            .rename(columns={"improved": "success_rate"})
        )
        metrics["metrics"]["intervention_success_rate"] = success_rate.to_dict("records")
        
        # Statistical tests: compare before vs after
        statistical_tests = []
        for variant in before_after["variant"].unique():
            variant_data = before_after[before_after["variant"] == variant]
            
            before_values = variant_data["correctness_before"].tolist()
            after_values = variant_data["correctness_after"].tolist()
            
            if len(before_values) >= 2 and len(after_values) >= 2:
                ttest_result = compute_ttest(before_values, after_values, alternative="two-sided")
                statistical_tests.append({
                    "variant": variant,
                    "comparison": "before_vs_after",
                    **ttest_result,
                })
        
        # Compare different alpha values
        for variant in before_after["variant"].unique():
            variant_data = before_after[before_after["variant"] == variant]
            alphas = sorted(variant_data["alpha"].unique())
            
            for i, alpha1 in enumerate(alphas):
                for alpha2 in alphas[i+1:]:
                    alpha1_data = variant_data[variant_data["alpha"] == alpha1]["effect"].tolist()
                    alpha2_data = variant_data[variant_data["alpha"] == alpha2]["effect"].tolist()
                    
                    if len(alpha1_data) >= 2 and len(alpha2_data) >= 2:
                        ttest_result = compute_ttest(alpha1_data, alpha2_data)
                        statistical_tests.append({
                            "variant": variant,
                            "comparison": f"alpha_{alpha1}_vs_{alpha2}",
                            **ttest_result,
                        })
        
        if statistical_tests:
            metrics["metrics"]["statistical_tests"] = statistical_tests
    
    metrics["statistics"] = {
        "n_interventions": intervention_count,
        "n_results": len(df),
        "variants": sorted(df["variant"].unique().tolist()) if not df.empty else [],
    }
    
    return metrics


def generate_intervention_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate intervention visualizations (Figure 6 + supporting graphs).
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_intervention_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load intervention results
    df = pd.read_sql_query(
        """
        SELECT 
            r.flipped_to_truth,
            i.name AS intervention_name,
            i.alpha,
            t.variant,
            o_before.is_correct AS correctness_before,
            o_after.is_correct AS correctness_after
        FROM conformity_intervention_results r
        JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
        JOIN conformity_trials t ON t.trial_id = r.trial_id
        JOIN conformity_outputs o_before ON o_before.output_id = r.output_id_before
        JOIN conformity_outputs o_after ON o_after.output_id = r.output_id_after
        WHERE i.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return figures
    
    # Figure 6: Intervention Impact (Before/After)
    # Flip rate by alpha
    if "flip_to_truth_rate" in metrics["metrics"]:
        flip_data = pd.DataFrame(metrics["metrics"]["flip_to_truth_rate"])
        
        for variant in flip_data["variant"].unique():
            variant_data = flip_data[flip_data["variant"] == variant]
            
            fig, ax = create_figure(size_key="single")
            
            for intervention in variant_data["intervention_name"].unique():
                int_data = variant_data[variant_data["intervention_name"] == intervention].sort_values("alpha")
                ax.plot(
                    int_data["alpha"],
                    int_data["flip_rate"],
                    marker="o",
                    label=intervention,
                    linewidth=2,
                )
            
            ax.set_xlabel("Alpha (Intervention Strength)", fontsize=14)
            ax.set_ylabel("Flip-to-Truth Rate", fontsize=14)
            ax.set_title(f"Figure 6: Intervention Impact - Flip Rate ({variant})", fontsize=16, fontweight="bold")
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig_path = os.path.join(paths["figures_dir"], f"figure6_intervention_impact_{variant}")
            saved = save_figure(fig, fig_path)
            figures[f"figure6_intervention_impact_{variant}"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    # Before/After correctness comparison (paired bar chart)
    if not df.empty and "correctness_before" in df.columns and "correctness_after" in df.columns:
        before_after_mean = (
            df.groupby(["variant", "intervention_name"], as_index=False)
            .agg({
                "correctness_before": "mean",
                "correctness_after": "mean",
            })
        )
        
        for variant in before_after_mean["variant"].unique():
            variant_data = before_after_mean[before_after_mean["variant"] == variant]
            
            fig, ax = create_figure(size_key="single")
            
            x = range(len(variant_data))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], variant_data["correctness_before"], width, label="Before", alpha=0.7)
            ax.bar([i + width/2 for i in x], variant_data["correctness_after"], width, label="After", alpha=0.7)
            
            ax.set_xlabel("Intervention", fontsize=14)
            ax.set_ylabel("Correctness", fontsize=14)
            ax.set_title(f"Before/After Correctness Comparison ({variant})", fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(variant_data["intervention_name"], rotation=45, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            
            fig_path = os.path.join(paths["figures_dir"], f"before_after_comparison_{variant}")
            saved = save_figure(fig, fig_path)
            figures[f"before_after_comparison_{variant}"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    return figures


def export_intervention_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export intervention metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_intervention_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_interventions.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "flip_to_truth_rate" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "flip_to_truth_rate.csv")
        save_table_csv(metrics["metrics"]["flip_to_truth_rate"], csv_path)
        csv_paths["flip_to_truth_rate"] = csv_path
    
    if "effect_size" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "effect_size.csv")
        save_table_csv(metrics["metrics"]["effect_size"], csv_path)
        csv_paths["effect_size"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
