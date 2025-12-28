"""
Behavioral analytics for Olmo Conformity Experiment.

Implements Figure 1 (Sycophancy Behavioral Outcome) and core behavioral metrics
as defined in the Critical Assessment document.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas and matplotlib are required for behavioral analytics")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    rotate_labels_if_needed,
    save_figure,
    setup_publication_style,
    wrap_long_labels,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb


def compute_behavioral_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute behavioral metrics matching Critical Assessment definitions.
    
    Metrics computed:
    - Accuracy by Condition (on immutable facts only)
    - Sycophancy Rate (Truth-Override Rate): among items where control is correct,
      fraction where pressure condition becomes incorrect
    - Truth Override Frequency (item-level)
    - Pressure Agreement Rate (auxiliary)
    - Refusal rate, latency, answer length stats
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Load trials with outputs (immutable facts only)
    df = pd.read_sql_query(
        """
        SELECT 
            t.trial_id,
            t.variant,
            t.model_id,
            t.item_id,
            c.name AS condition_name,
            c.params_json AS condition_params,
            o.is_correct,
            o.refusal_flag,
            o.latency_ms,
            LENGTH(o.parsed_answer_text) AS answer_length,
            i.domain,
            d.name AS dataset_name
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        JOIN conformity_items i ON i.item_id = t.item_id
        JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
        WHERE t.run_id = ? AND o.is_correct IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"total_trials": 0, "message": "No immutable facts data available"},
        }
    
    # Parse condition params to extract type
    def parse_condition_type(params_json: str) -> str:
        try:
            params = json.loads(params_json)
            return params.get("type", "unknown")
        except:
            return "unknown"
    
    df["condition_type"] = df["condition_params"].apply(parse_condition_type)
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # 1. Accuracy by Condition
    accuracy_by_condition = (
        df.groupby(["variant", "condition_name"])["is_correct"]
        .agg(accuracy="mean", n_trials="count")
        .reset_index()
    )
    metrics["metrics"]["accuracy_by_condition"] = accuracy_by_condition.to_dict("records")
    
    # 2. Sycophancy Rate (Truth-Override Rate)
    # Pair control vs pressure by (variant, model_id, item_id)
    # Only compute where control is correct
    control_trials = df[df["condition_name"] == "control"].copy()
    pressure_trials = df[df["condition_name"] != "control"].copy()
    
    sycophancy_data = []
    
    if not control_trials.empty and not pressure_trials.empty:
        # Merge on (variant, model_id, item_id)
        merged = control_trials.merge(
            pressure_trials,
            on=["variant", "model_id", "item_id"],
            suffixes=("_control", "_pressure"),
            how="inner",
        )
        
        # Only consider items where control was correct
        control_correct = merged[merged["is_correct_control"] == 1].copy()
        
        if not control_correct.empty:
            # Compute sycophancy rate per pressure condition
            sycophancy_by_pressure = (
                control_correct.groupby(["variant", "condition_name_pressure"])["is_correct_pressure"]
                .agg(pressure_accuracy="mean", n_items="count")
                .reset_index()
                .rename(columns={"condition_name_pressure": "pressure_condition"})
            )
            # Sycophancy = 1 - accuracy under pressure (when control was correct)
            sycophancy_by_pressure["sycophancy_rate"] = 1.0 - sycophancy_by_pressure["pressure_accuracy"]
            sycophancy_by_pressure = sycophancy_by_pressure[
                ["variant", "pressure_condition", "sycophancy_rate", "n_items"]
            ]
            metrics["metrics"]["sycophancy_rate"] = sycophancy_by_pressure.to_dict("records")
            
            # Truth Override Frequency (item-level)
            control_correct["truth_override"] = (control_correct["is_correct_pressure"] == 0).astype(int)
            truth_override_freq = (
                control_correct.groupby(["variant", "item_id"], as_index=False)["truth_override"]
                .max()  # 1 if overridden in any pressure condition
                .groupby("variant", as_index=False)["truth_override"]
                .mean()
            )
            truth_override_freq.columns = ["variant", "truth_override_frequency"]
            metrics["metrics"]["truth_override_frequency"] = truth_override_freq.to_dict("records")
    
    # 3. Pressure Agreement Rate (auxiliary)
    # This requires knowing the wrong answer from the prompt/condition params
    # For now, we'll note this requires additional parsing of condition params
    metrics["metrics"]["pressure_agreement_rate"] = []  # TODO: implement when wrong_answer is available
    
    # 4. Refusal rate
    refusal_rate = (
        df.groupby(["variant", "condition_name"])["refusal_flag"]
        .agg(refusal_rate="mean", n_trials="count")
        .reset_index()
    )
    metrics["metrics"]["refusal_rate"] = refusal_rate.to_dict("records")
    
    # 5. Latency statistics
    latency_stats = (
        df.groupby(["variant", "condition_name"])["latency_ms"]
        .agg(
            mean_latency_ms="mean",
            median_latency_ms="median",
            p95_latency_ms=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    metrics["metrics"]["latency_stats"] = latency_stats.to_dict("records")
    
    # 6. Answer length statistics
    answer_length_stats = (
        df.groupby(["variant", "condition_name"])["answer_length"]
        .agg(mean_length="mean", median_length="median")
        .reset_index()
    )
    metrics["metrics"]["answer_length_stats"] = answer_length_stats.to_dict("records")
    
    # Statistics summary
    metrics["statistics"] = {
        "total_trials": len(df),
        "variants": sorted(df["variant"].unique().tolist()),
        "conditions": sorted(df["condition_name"].unique().tolist()),
        "datasets": sorted(df["dataset_name"].unique().tolist()),
        "domains": sorted(df["domain"].unique().tolist()),
    }
    
    return metrics


def generate_behavioral_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate behavioral visualizations (Figure 1 + supporting graphs).
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_behavioral_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting")
    
    figures = {}
    
    # Load data for plotting
    df = pd.read_sql_query(
        """
        SELECT 
            t.variant,
            t.item_id,
            t.model_id,
            c.name AS condition_name,
            o.is_correct
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ? AND o.is_correct IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return figures
    
    # Figure 1: Sycophancy Behavioral Outcome (Bar Chart)
    # Compute sycophancy rate per variant
    control_trials = df[df["condition_name"] == "control"].copy()
    pressure_trials = df[df["condition_name"] != "control"].copy()
    
    if not control_trials.empty and not pressure_trials.empty:
        merged = control_trials.merge(
            pressure_trials,
            on=["variant", "model_id", "item_id"],
            suffixes=("_control", "_pressure"),
            how="inner",
        )
        control_correct = merged[merged["is_correct_control"] == 1].copy()
        
        if not control_correct.empty:
            # Compute sycophancy rate by variant and pressure condition
            sycophancy = (
                control_correct.groupby(["variant", "condition_name_pressure"], as_index=False)
                .agg({"is_correct_pressure": "mean"})
            )
            sycophancy["sycophancy_rate"] = 1.0 - sycophancy["is_correct_pressure"]
            
            # Pivot for bar chart
            sycophancy_pivot = sycophancy.pivot(
                index="variant",
                columns="condition_name_pressure",
                values="sycophancy_rate",
            )
            
            fig, ax = create_figure(size_key="single")
            sycophancy_pivot.plot(kind="bar", ax=ax, color=get_color_palette(len(sycophancy_pivot.columns)))
            ax.set_ylabel("Sycophancy Rate (Truth-Override)", fontsize=14)
            ax.set_xlabel("Model Variant", fontsize=14)
            ax.set_title("Figure 1: Sycophancy Behavioral Outcome", fontsize=16, fontweight="bold")
            ax.set_ylim(0.0, 1.0)
            ax.legend(title="Pressure Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3, axis="y")
            rotate_labels_if_needed(ax, axis="x")
            
            fig_path = os.path.join(paths["figures_dir"], "figure1_sycophancy_behavioral")
            saved = save_figure(fig, fig_path)
            figures["figure1_sycophancy"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    # Accuracy by Condition (bar chart)
    accuracy_data = (
        df.groupby(["variant", "condition_name"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    
    if not accuracy_data.empty:
        accuracy_pivot = accuracy_data.pivot(
            index="variant",
            columns="condition_name",
            values="accuracy",
        )
        
        fig, ax = create_figure(size_key="single")
        accuracy_pivot.plot(kind="bar", ax=ax, color=get_color_palette(len(accuracy_pivot.columns)))
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlabel("Model Variant", fontsize=14)
        ax.set_title("Accuracy by Condition", fontsize=16)
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        rotate_labels_if_needed(ax, axis="x")
        
        fig_path = os.path.join(paths["figures_dir"], "accuracy_by_condition")
        saved = save_figure(fig, fig_path)
        figures["accuracy_by_condition"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    # Correctness distribution by condition (box plot)
    if len(df) > 0:
        fig, ax = create_figure(size_key="single")
        condition_order = sorted(df["condition_name"].unique())
        data_for_box = [df[df["condition_name"] == cond]["is_correct"].values for cond in condition_order]
        bp = ax.boxplot(data_for_box, labels=condition_order, patch_artist=True)
        
        colors = get_color_palette(len(condition_order))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Correctness", fontsize=14)
        ax.set_xlabel("Condition", fontsize=14)
        ax.set_title("Correctness Distribution by Condition", fontsize=16)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
        rotate_labels_if_needed(ax, axis="x")
        
        fig_path = os.path.join(paths["figures_dir"], "correctness_distribution")
        saved = save_figure(fig, fig_path)
        figures["correctness_distribution"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    return figures


def export_behavioral_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export behavioral metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_behavioral_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_behavioral.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "accuracy_by_condition" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "accuracy_by_condition.csv")
        save_table_csv(metrics["metrics"]["accuracy_by_condition"], csv_path)
        csv_paths["accuracy_by_condition"] = csv_path
    
    if "sycophancy_rate" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "sycophancy_rate.csv")
        save_table_csv(metrics["metrics"]["sycophancy_rate"], csv_path)
        csv_paths["sycophancy_rate"] = csv_path
    
    if "refusal_rate" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "refusal_rate.csv")
        save_table_csv(metrics["metrics"]["refusal_rate"], csv_path)
        csv_paths["refusal_rate"] = csv_path
    
    if "latency_stats" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "latency_stats.csv")
        save_table_csv(metrics["metrics"]["latency_stats"], csv_path)
        csv_paths["latency_stats"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
