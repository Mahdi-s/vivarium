"""
Probe analytics for Olmo Conformity Experiment.

Implements Figure 2 (Truth vs Social Signal Across Layers) and probe validation metrics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from collections import Counter

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas and matplotlib are required for probe analytics")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.analytics.statistics import compute_ttest
from aam.persistence import TraceDb


def compute_probe_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute probe metrics: TVP, SVP, collision points, probe validation.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Check if probes exist
    probe_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM conformity_probes WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    
    if probe_count == 0:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No probes found for this run"},
        }
    
    # Load probe projections
    df = pd.read_sql_query(
        """
        SELECT 
            p.projection_id,
            p.trial_id,
            p.probe_id,
            p.layer_index,
            p.value_float,
            pr.probe_kind,
            pr.component,
            t.variant,
            c.name AS condition_name
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE pr.run_id = ?
        ORDER BY p.trial_id, p.probe_id, p.layer_index
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No probe projections found"},
        }
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Separate truth and social projections
    truth_projections = df[df["probe_kind"] == "truth"].copy()
    social_projections = df[df["probe_kind"] == "social"].copy()
    
    # Compute mean projections by layer
    if not truth_projections.empty:
        tvp_by_layer = (
            truth_projections.groupby(["variant", "condition_name", "layer_index"])["value_float"]
            .agg(mean_tvp="mean", std_tvp="std", n="count")
            .reset_index()
        )
        metrics["metrics"]["truth_vector_projection"] = tvp_by_layer.to_dict("records")
    
    if not social_projections.empty:
        svp_by_layer = (
            social_projections.groupby(["variant", "condition_name", "layer_index"])["value_float"]
            .agg(mean_svp="mean", std_svp="std", n="count")
            .reset_index()
        )
        metrics["metrics"]["social_vector_projection"] = svp_by_layer.to_dict("records")
    
    # Collision detection: find layers where SVP > TVP
    if not truth_projections.empty and not social_projections.empty:
        # Merge on (trial_id, layer_index)
        merged = truth_projections.merge(
            social_projections,
            on=["trial_id", "layer_index"],
            suffixes=("_truth", "_social"),
            how="inner",
        )
        merged["collision"] = (merged["value_float_social"] > merged["value_float_truth"]).astype(int)
        
        # Find first collision layer per trial
        collision_layers = (
            merged[merged["collision"] == 1]
            .groupby("trial_id", as_index=False)
            .agg({"layer_index": "min"})
            .rename(columns={"layer_index": "first_collision_layer"})
        )
        
        if not collision_layers.empty:
            metrics["metrics"]["collision_layers"] = collision_layers.to_dict("records")
    
    # Probe validation metrics (from probe metadata)
    probe_metrics = pd.read_sql_query(
        """
        SELECT 
            probe_id,
            probe_kind,
            layers_json,
            metrics_json
        FROM conformity_probes
        WHERE run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    validation_data = []
    for _, row in probe_metrics.iterrows():
        try:
            metrics_json = json.loads(row["metrics_json"])
            validation_data.append({
                "probe_id": row["probe_id"],
                "probe_kind": row["probe_kind"],
                "train_accuracy": metrics_json.get("train_accuracy"),
                "test_accuracy": metrics_json.get("test_accuracy"),
            })
        except:
            pass
    
    if validation_data:
        metrics["metrics"]["probe_validation"] = validation_data
    
    # Statistical tests: compare probe projections across conditions
    statistical_tests = []
    if not df.empty:
        conditions = sorted(df["condition_name"].unique().tolist())
        if len(conditions) >= 2:
            for probe_kind in df["probe_kind"].unique():
                probe_data = df[df["probe_kind"] == probe_kind].copy()
                
                # Compare each pair of conditions
                for i, cond1 in enumerate(conditions):
                    for cond2 in conditions[i+1:]:
                        cond1_data = probe_data[probe_data["condition_name"] == cond1]["value_float"].tolist()
                        cond2_data = probe_data[probe_data["condition_name"] == cond2]["value_float"].tolist()
                        
                        if len(cond1_data) >= 2 and len(cond2_data) >= 2:
                            ttest_result = compute_ttest(cond1_data, cond2_data)
                            statistical_tests.append({
                                "probe_kind": probe_kind,
                                "condition1": cond1,
                                "condition2": cond2,
                                **ttest_result,
                            })
    
    if statistical_tests:
        metrics["metrics"]["statistical_tests"] = statistical_tests
    
    metrics["statistics"] = {
        "n_probes": probe_count,
        "n_projections": len(df),
        "probe_kinds": sorted(df["probe_kind"].unique().tolist()) if not df.empty else [],
    }
    
    return metrics


def generate_probe_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate probe visualizations (Figure 2 + supporting graphs).
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_probe_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load projection data
    df = pd.read_sql_query(
        """
        SELECT 
            p.layer_index,
            p.value_float,
            pr.probe_kind,
            t.variant,
            c.name AS condition_name
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE pr.run_id = ?
        ORDER BY p.layer_index
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return figures
    
    # Figure 2: Truth vs Social Signal Across Layers (Line Plot)
    truth_data = df[df["probe_kind"] == "truth"].copy()
    social_data = df[df["probe_kind"] == "social"].copy()
    
    if not truth_data.empty and not social_data.empty:
        # Compute mean by layer, variant, condition
        truth_mean = (
            truth_data.groupby(["variant", "condition_name", "layer_index"], as_index=False)
            ["value_float"]
            .mean()
        )
        social_mean = (
            social_data.groupby(["variant", "condition_name", "layer_index"], as_index=False)
            ["value_float"]
            .mean()
        )
        
        # Plot for each variant
        for variant in truth_mean["variant"].unique():
            fig, ax = create_figure(size_key="dense")
            
            variant_truth = truth_mean[truth_mean["variant"] == variant]
            variant_social = social_mean[social_mean["variant"] == variant]
            
            colors = get_color_palette(len(variant_truth["condition_name"].unique()))
            color_map = {cond: colors[i] for i, cond in enumerate(sorted(variant_truth["condition_name"].unique()))}
            
            # Plot truth projections
            for condition in variant_truth["condition_name"].unique():
                cond_data = variant_truth[variant_truth["condition_name"] == condition].sort_values("layer_index")
                ax.plot(
                    cond_data["layer_index"],
                    cond_data["value_float"],
                    label=f"Truth ({condition})",
                    color=color_map[condition],
                    linestyle="-",
                    linewidth=2,
                )
            
            # Plot social projections
            for condition in variant_social["condition_name"].unique():
                cond_data = variant_social[variant_social["condition_name"] == condition].sort_values("layer_index")
                ax.plot(
                    cond_data["layer_index"],
                    cond_data["value_float"],
                    label=f"Social ({condition})",
                    color=color_map[condition],
                    linestyle="--",
                    linewidth=2,
                )
            
            ax.set_xlabel("Layer Index", fontsize=14)
            ax.set_ylabel("Projection Value", fontsize=14)
            ax.set_title(f"Figure 2: Truth vs Social Signal Across Layers ({variant})", fontsize=16, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            
            fig_path = os.path.join(paths["figures_dir"], f"figure2_truth_vs_social_{variant}")
            saved = save_figure(fig, fig_path)
            figures[f"figure2_truth_vs_social_{variant}"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    # Collision point heatmap (social - truth by layer/condition)
    if not truth_data.empty and not social_data.empty:
        # Merge to compute difference
        merged = truth_data.merge(
            social_data,
            on=["variant", "condition_name", "layer_index"],
            suffixes=("_truth", "_social"),
            how="inner",
        )
        merged["svp_minus_tvp"] = merged["value_float_social"] - merged["value_float_truth"]
        
        collision_mean = (
            merged.groupby(["variant", "condition_name", "layer_index"], as_index=False)
            ["svp_minus_tvp"]
            .mean()
        )
        
        if not collision_mean.empty:
            # Pivot for heatmap
            for variant in collision_mean["variant"].unique():
                variant_data = collision_mean[collision_mean["variant"] == variant]
                pivot = variant_data.pivot(
                    index="layer_index",
                    columns="condition_name",
                    values="svp_minus_tvp",
                )
                
                fig, ax = create_figure(size_key="single")
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", interpolation="nearest")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                ax.set_xlabel("Condition", fontsize=14)
                ax.set_ylabel("Layer Index", fontsize=14)
                ax.set_title(f"Social - Truth Projection Difference ({variant})", fontsize=16)
                plt.colorbar(im, ax=ax, label="SVP - TVP")
                
                fig_path = os.path.join(paths["figures_dir"], f"collision_heatmap_{variant}")
                saved = save_figure(fig, fig_path)
                figures[f"collision_heatmap_{variant}"] = saved.get("png", saved.get("pdf", ""))
                plt.close(fig)
    
    return figures


def export_probe_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export probe metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_probe_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_probes.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "truth_vector_projection" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "truth_vector_projection.csv")
        save_table_csv(metrics["metrics"]["truth_vector_projection"], csv_path)
        csv_paths["truth_vector_projection"] = csv_path
    
    if "social_vector_projection" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "social_vector_projection.csv")
        save_table_csv(metrics["metrics"]["social_vector_projection"], csv_path)
        csv_paths["social_vector_projection"] = csv_path
    
    if "probe_validation" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "probe_validation.csv")
        save_table_csv(metrics["metrics"]["probe_validation"], csv_path)
        csv_paths["probe_validation"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
