"""
Think token analytics for Olmo Conformity Experiment.

Implements Figure 5 (Think Trajectory / Logit Lens Plot) for Think variants.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas and matplotlib are required for think token analytics")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb


def compute_think_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute think token metrics: logit lens, rationalization detection.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Check if think tokens exist
    think_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_think_tokens tt
        JOIN conformity_trials t ON t.trial_id = tt.trial_id
        WHERE t.run_id = ?
        """,
        (run_id,),
    ).fetchone()[0]
    
    logit_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_logit_lens ll
        JOIN conformity_trials t ON t.trial_id = ll.trial_id
        WHERE t.run_id = ?
        """,
        (run_id,),
    ).fetchone()[0]
    
    if think_count == 0 and logit_count == 0:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No think tokens or logit lens data found for this run"},
        }
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Load logit lens data
    if logit_count > 0:
        df_logit = pd.read_sql_query(
            """
            SELECT 
                ll.logit_id,
                ll.trial_id,
                ll.layer_index,
                ll.token_index,
                ll.topk_json,
                t.variant,
                c.name AS condition_name
            FROM conformity_logit_lens ll
            JOIN conformity_trials t ON t.trial_id = ll.trial_id
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            WHERE t.run_id = ?
            ORDER BY ll.trial_id, ll.layer_index, ll.token_index
            """,
            trace_db.conn,
            params=(run_id,),
        )
        
        if not df_logit.empty:
            # Parse topk_json to extract top predictions
            def parse_topk(topk_json: str) -> Dict[str, Any]:
                try:
                    return json.loads(topk_json)
                except:
                    return {}
            
            df_logit["topk"] = df_logit["topk_json"].apply(parse_topk)
            metrics["metrics"]["logit_lens_data"] = df_logit.to_dict("records")
    
    # Load think tokens
    if think_count > 0:
        df_think = pd.read_sql_query(
            """
            SELECT 
                tt.think_id,
                tt.trial_id,
                tt.token_index,
                tt.token_text,
                t.variant,
                c.name AS condition_name
            FROM conformity_think_tokens tt
            JOIN conformity_trials t ON t.trial_id = tt.trial_id
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            WHERE t.run_id = ?
            ORDER BY tt.trial_id, tt.token_index
            """,
            trace_db.conn,
            params=(run_id,),
        )
        
        if not df_think.empty:
            # Rationalization detection (placeholder - would need more sophisticated analysis)
            metrics["metrics"]["think_tokens_data"] = df_think.to_dict("records")
    
    metrics["statistics"] = {
        "n_think_tokens": think_count,
        "n_logit_lens": logit_count,
    }
    
    return metrics


def generate_think_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate think token visualizations (Figure 5 + supporting graphs).
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_think_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load logit lens data
    df_logit = pd.read_sql_query(
        """
        SELECT 
            ll.trial_id,
            ll.layer_index,
            ll.token_index,
            ll.topk_json,
            t.variant,
            c.name AS condition_name
        FROM conformity_logit_lens ll
        JOIN conformity_trials t ON t.trial_id = ll.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ?
        ORDER BY ll.trial_id, ll.layer_index, ll.token_index
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_logit.empty:
        return figures
    
    # Figure 5: Logit Lens Across Layers (Line Chart)
    # Extract top prediction probability from topk_json
    def get_top_prob(topk_json: str) -> float:
        try:
            topk = json.loads(topk_json)
            if isinstance(topk, list) and len(topk) > 0:
                return float(topk[0].get("prob", 0.0))
            elif isinstance(topk, dict) and "top" in topk:
                return float(topk["top"].get("prob", 0.0))
        except:
            pass
        return 0.0
    
    df_logit["top_prob"] = df_logit["topk_json"].apply(get_top_prob)
    
    # Plot logit lens for each variant
    for variant in df_logit["variant"].unique():
        variant_data = df_logit[df_logit["variant"] == variant].copy()
        
        # Average across trials and token positions for each layer
        logit_mean = (
            variant_data.groupby(["condition_name", "layer_index"], as_index=False)
            ["top_prob"]
            .mean()
        )
        
        fig, ax = create_figure(size_key="single")
        
        colors = get_color_palette(len(logit_mean["condition_name"].unique()))
        color_map = {cond: colors[i] for i, cond in enumerate(sorted(logit_mean["condition_name"].unique()))}
        
        for condition in logit_mean["condition_name"].unique():
            cond_data = logit_mean[logit_mean["condition_name"] == condition].sort_values("layer_index")
            ax.plot(
                cond_data["layer_index"],
                cond_data["top_prob"],
                label=condition,
                color=color_map[condition],
                marker="o",
                linewidth=2,
            )
        
        ax.set_xlabel("Layer Index", fontsize=14)
        ax.set_ylabel("Top Prediction Probability", fontsize=14)
        ax.set_title(f"Figure 5: Logit Lens Across Layers ({variant})", fontsize=16, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = os.path.join(paths["figures_dir"], f"figure5_logit_lens_{variant}")
        saved = save_figure(fig, fig_path)
        figures[f"figure5_logit_lens_{variant}"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    return figures


def export_think_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export think token metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_think_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_think.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables (if data is not too large)
    csv_paths = {}
    
    if "logit_lens_data" in metrics["metrics"] and len(metrics["metrics"]["logit_lens_data"]) < 10000:
        csv_path = os.path.join(paths["tables_dir"], "logit_lens_by_layer.csv")
        save_table_csv(metrics["metrics"]["logit_lens_data"], csv_path)
        csv_paths["logit_lens"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
