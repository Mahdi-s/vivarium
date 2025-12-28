"""
Judge Eval analytics for Olmo Conformity Experiment.

Required for subjective queries and correlation analysis.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise RuntimeError("pandas, matplotlib, and numpy are required for judgeval analytics")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb


def compute_judgeval_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute Judge Eval metrics: conformity, truthfulness, rationalization scores.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Check if Judge Eval scores exist
    judgeval_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_outputs o
        JOIN conformity_trials t ON t.trial_id = o.trial_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        (run_id,),
    ).fetchone()[0]
    
    if judgeval_count == 0:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No Judge Eval scores found for this run"},
        }
    
    # Load Judge Eval scores
    df = pd.read_sql_query(
        """
        SELECT 
            t.trial_id,
            t.variant,
            c.name AS condition_name,
            json_extract(o.parsed_answer_json, '$.conformity') AS conformity_score,
            json_extract(o.parsed_answer_json, '$.truthfulness') AS truthfulness_score,
            json_extract(o.parsed_answer_json, '$.rationalization') AS rationalization_score,
            o.is_correct
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No Judge Eval data available"},
        }
    
    # Convert scores to numeric
    for col in ["conformity_score", "truthfulness_score", "rationalization_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Mean scores by variant/condition
    mean_scores = (
        df.groupby(["variant", "condition_name"], as_index=False)
        .agg({
            "conformity_score": "mean",
            "truthfulness_score": "mean",
            "rationalization_score": "mean",
        })
    )
    metrics["metrics"]["mean_scores_by_variant_condition"] = mean_scores.to_dict("records")
    
    # Correlation: Judge Eval vs actual correctness
    if "is_correct" in df.columns and df["is_correct"].notna().any():
        correctness_corr = df[["conformity_score", "truthfulness_score", "is_correct"]].corr()
        metrics["metrics"]["correlation_with_correctness"] = {
            "conformity": float(correctness_corr.loc["conformity_score", "is_correct"]),
            "truthfulness": float(correctness_corr.loc["truthfulness_score", "is_correct"]),
        }
    
    # Judge Eval accuracy (how well it predicts correctness)
    if "is_correct" in df.columns and df["is_correct"].notna().any():
        # Use truthfulness as predictor
        df_clean = df[df["truthfulness_score"].notna() & df["is_correct"].notna()].copy()
        if not df_clean.empty:
            # Threshold at 0.5
            df_clean["predicted_correct"] = (df_clean["truthfulness_score"] > 0.5).astype(int)
            accuracy = (df_clean["predicted_correct"] == df_clean["is_correct"]).mean()
            metrics["metrics"]["judgeval_accuracy"] = float(accuracy)
    
    metrics["statistics"] = {
        "n_scores": len(df),
        "variants": sorted(df["variant"].unique().tolist()) if not df.empty else [],
        "conditions": sorted(df["condition_name"].unique().tolist()) if not df.empty else [],
    }
    
    return metrics


def generate_judgeval_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate Judge Eval visualizations.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_judgeval_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load Judge Eval data
    df = pd.read_sql_query(
        """
        SELECT 
            t.variant,
            c.name AS condition_name,
            json_extract(o.parsed_answer_json, '$.conformity') AS conformity_score,
            json_extract(o.parsed_answer_json, '$.truthfulness') AS truthfulness_score,
            json_extract(o.parsed_answer_json, '$.rationalization') AS rationalization_score,
            o.is_correct
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return figures
    
    # Convert to numeric
    for col in ["conformity_score", "truthfulness_score", "rationalization_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Conformity scores by variant/condition (bar chart)
    if "conformity_score" in df.columns and df["conformity_score"].notna().any():
        conformity_mean = (
            df.groupby(["variant", "condition_name"], as_index=False)
            ["conformity_score"]
            .mean()
        )
        
        conformity_pivot = conformity_mean.pivot(
            index="variant",
            columns="condition_name",
            values="conformity_score",
        )
        
        fig, ax = create_figure(size_key="single")
        conformity_pivot.plot(kind="bar", ax=ax, color=get_color_palette(len(conformity_pivot.columns)))
        ax.set_ylabel("Mean Conformity Score (Judge Eval)", fontsize=14)
        ax.set_xlabel("Model Variant", fontsize=14)
        ax.set_title("Conformity Scores by Variant/Condition", fontsize=16)
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        
        fig_path = os.path.join(paths["figures_dir"], "judgeval_conformity_scores")
        saved = save_figure(fig, fig_path)
        figures["judgeval_conformity_scores"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    # Truthfulness vs Correctness correlation (scatter plot)
    if "truthfulness_score" in df.columns and "is_correct" in df.columns:
        df_clean = df[df["truthfulness_score"].notna() & df["is_correct"].notna()].copy()
        
        if not df_clean.empty:
            fig, ax = create_figure(size_key="single")
            ax.scatter(df_clean["truthfulness_score"], df_clean["is_correct"], alpha=0.5, s=50)
            ax.set_xlabel("Judge Eval Truthfulness Score", fontsize=14)
            ax.set_ylabel("Actual Correctness", fontsize=14)
            ax.set_title("Judge Eval Truthfulness vs Actual Correctness", fontsize=16)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Add correlation line if possible
            if len(df_clean) > 1:
                z = np.polyfit(df_clean["truthfulness_score"], df_clean["is_correct"], 1)
                p = np.poly1d(z)
                ax.plot(df_clean["truthfulness_score"], p(df_clean["truthfulness_score"]), "r--", alpha=0.5, linewidth=2)
            
            fig_path = os.path.join(paths["figures_dir"], "judgeval_truthfulness_correlation")
            saved = save_figure(fig, fig_path)
            figures["judgeval_truthfulness_correlation"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    # Rationalization score distribution (histogram)
    if "rationalization_score" in df.columns and df["rationalization_score"].notna().any():
        df_rational = df[df["rationalization_score"].notna()].copy()
        
        if not df_rational.empty:
            fig, ax = create_figure(size_key="single")
            ax.hist(df_rational["rationalization_score"], bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Rationalization Score", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            ax.set_title("Rationalization Score Distribution", fontsize=16)
            ax.grid(True, alpha=0.3, axis="y")
            
            fig_path = os.path.join(paths["figures_dir"], "judgeval_rationalization_distribution")
            saved = save_figure(fig, fig_path)
            figures["judgeval_rationalization_distribution"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    return figures


def export_judgeval_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export Judge Eval metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_judgeval_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_judgeval.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "mean_scores_by_variant_condition" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "judgeval_scores_by_variant_condition.csv")
        save_table_csv(metrics["metrics"]["mean_scores_by_variant_condition"], csv_path)
        csv_paths["judgeval_scores"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
