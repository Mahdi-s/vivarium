"""
Token-level analysis for Olmo Conformity Experiment.

Analyzes logit lens token predictions, token evolution patterns, and token prediction accuracy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas, numpy, and matplotlib are required for token analysis")

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb
from collections import Counter
import os


def compute_token_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute token-level metrics from logit lens data.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed token metrics
    """
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
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No logit lens data found"},
        }
    
    # Load actual outputs for comparison
    df_outputs = pd.read_sql_query(
        """
        SELECT 
            t.trial_id,
            o.parsed_answer_text,
            o.raw_text
        FROM conformity_trials t
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Parse topk_json to extract tokens
    def parse_topk(topk_json: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(topk_json)
        except:
            return []
    
    df_logit["topk"] = df_logit["topk_json"].apply(parse_topk)
    
    # Extract top token and probability
    def get_top_token(topk: List[Dict[str, Any]]) -> Optional[str]:
        if isinstance(topk, list) and len(topk) > 0:
            return topk[0].get("token")
        return None
    
    def get_top_prob(topk: List[Dict[str, Any]]) -> float:
        if isinstance(topk, list) and len(topk) > 0:
            return float(topk[0].get("prob", 0.0))
        return 0.0
    
    df_logit["top_token"] = df_logit["topk"].apply(get_top_token)
    df_logit["top_prob"] = df_logit["topk"].apply(get_top_prob)
    
    # Top token frequency by layer and condition
    token_freq = []
    for layer in sorted(df_logit["layer_index"].unique()):
        layer_data = df_logit[df_logit["layer_index"] == layer]
        for condition in layer_data["condition_name"].unique():
            cond_data = layer_data[layer_data["condition_name"] == condition]
            top_tokens = cond_data["top_token"].dropna().tolist()
            
            if len(top_tokens) > 0:
                token_counts = Counter(top_tokens)
                for token, count in token_counts.most_common(10):
                    token_freq.append({
                        "layer_index": int(layer),
                        "condition_name": condition,
                        "token": token,
                        "frequency": int(count),
                        "proportion": float(count / len(top_tokens)),
                    })
    
    if token_freq:
        metrics["metrics"]["token_frequency"] = token_freq
    
    # Token prediction accuracy (simplified: check if top token appears in output)
    if not df_outputs.empty:
        accuracy_data = []
        merged = df_logit.merge(df_outputs, on="trial_id", how="inner")
        
        for layer in sorted(merged["layer_index"].unique()):
            layer_data = merged[merged["layer_index"] == layer]
            
            # Check if top token appears in parsed answer
            def token_in_output(row: pd.Series) -> bool:
                top_token = row.get("top_token", "")
                answer = str(row.get("parsed_answer_text", "")).lower()
                if top_token and answer:
                    # Simple check: token appears in answer (word boundary aware)
                    return top_token.lower() in answer.split()
                return False
            
            layer_data["token_match"] = layer_data.apply(token_in_output, axis=1)
            
            for condition in layer_data["condition_name"].unique():
                cond_data = layer_data[layer_data["condition_name"] == condition]
                if len(cond_data) > 0:
                    accuracy = cond_data["token_match"].mean()
                    accuracy_data.append({
                        "layer_index": int(layer),
                        "condition_name": condition,
                        "token_prediction_accuracy": float(accuracy),
                        "n_trials": int(len(cond_data)),
                    })
        
        if accuracy_data:
            metrics["metrics"]["token_prediction_accuracy"] = accuracy_data
    
    # Token evolution patterns (how top token changes across layers)
    evolution_data = []
    for trial_id in df_logit["trial_id"].unique():
        trial_data = df_logit[df_logit["trial_id"] == trial_id].sort_values("layer_index")
        if len(trial_data) >= 2:
            top_tokens = trial_data["top_token"].dropna().tolist()
            if len(top_tokens) >= 2:
                # Count token transitions
                transitions = []
                for i in range(len(top_tokens) - 1):
                    transitions.append((top_tokens[i], top_tokens[i+1]))
                
                evolution_data.append({
                    "trial_id": trial_id,
                    "n_transitions": len(transitions),
                    "unique_tokens": len(set(top_tokens)),
                    "transitions": transitions[:10],  # Limit to first 10
                })
    
    if evolution_data:
        metrics["metrics"]["token_evolution"] = evolution_data[:100]  # Limit to 100 trials
    
    metrics["statistics"] = {
        "n_logit_lens_rows": len(df_logit),
        "n_trials": len(df_logit["trial_id"].unique()),
        "layers": sorted(df_logit["layer_index"].unique().tolist()),
    }
    
    return metrics


def generate_token_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate token-level visualizations.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping figure_name -> path
    """
    if metrics is None:
        metrics = compute_token_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load logit lens data
    df_logit = pd.read_sql_query(
        """
        SELECT 
            ll.trial_id,
            ll.layer_index,
            ll.topk_json,
            t.variant,
            c.name AS condition_name
        FROM conformity_logit_lens ll
        JOIN conformity_trials t ON t.trial_id = ll.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ?
        ORDER BY ll.trial_id, ll.layer_index
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_logit.empty:
        return figures
    
    # Parse topk_json
    def get_top_token(topk_json: str) -> Optional[str]:
        try:
            topk = json.loads(topk_json)
            if isinstance(topk, list) and len(topk) > 0:
                return topk[0].get("token")
        except:
            pass
        return None
    
    df_logit["top_token"] = df_logit["topk_json"].apply(get_top_token)
    
    # Token frequency heatmap by layer and condition
    if "token_frequency" in metrics.get("metrics", {}):
        token_freq_df = pd.DataFrame(metrics["metrics"]["token_frequency"])
        
        if not token_freq_df.empty:
            # Pivot for heatmap: layer x condition x top token
            # For simplicity, show top 5 tokens per layer/condition
            top_tokens = token_freq_df.nlargest(50, "frequency")["token"].unique()[:10]
            
            for condition in token_freq_df["condition_name"].unique():
                cond_data = token_freq_df[token_freq_df["condition_name"] == condition]
                
                # Create matrix: layer x token
                layers = sorted(cond_data["layer_index"].unique())
                matrix_data = []
                
                for layer in layers:
                    layer_tokens = cond_data[cond_data["layer_index"] == layer]
                    row = []
                    for token in top_tokens:
                        token_row = layer_tokens[layer_tokens["token"] == token]
                        if not token_row.empty:
                            row.append(float(token_row.iloc[0]["frequency"]))
                        else:
                            row.append(0.0)
                    matrix_data.append(row)
                
                if matrix_data:
                    fig, ax = create_figure(size_key="single")
                    im = ax.imshow(matrix_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
                    ax.set_xticks(range(len(top_tokens)))
                    ax.set_xticklabels(top_tokens, rotation=45, ha="right")
                    ax.set_yticks(range(len(layers)))
                    ax.set_yticklabels(layers)
                    ax.set_xlabel("Top Token", fontsize=14)
                    ax.set_ylabel("Layer Index", fontsize=14)
                    ax.set_title(f"Token Frequency Heatmap - {condition}", fontsize=16)
                    plt.colorbar(im, ax=ax, label="Frequency")
                    
                    fig_path = os.path.join(paths["figures_dir"], f"token_frequency_heatmap_{condition}")
                    saved = save_figure(fig, fig_path)
                    figures[f"token_frequency_heatmap_{condition}"] = saved.get("png", saved.get("pdf", ""))
                    plt.close(fig)
    
    # Token prediction accuracy by layer
    if "token_prediction_accuracy" in metrics.get("metrics", {}):
        accuracy_df = pd.DataFrame(metrics["metrics"]["token_prediction_accuracy"])
        
        if not accuracy_df.empty:
            fig, ax = create_figure(size_key="single")
            
            for condition in accuracy_df["condition_name"].unique():
                cond_data = accuracy_df[accuracy_df["condition_name"] == condition].sort_values("layer_index")
                ax.plot(
                    cond_data["layer_index"],
                    cond_data["token_prediction_accuracy"],
                    label=condition,
                    marker="o",
                    linewidth=2,
                )
            
            ax.set_xlabel("Layer Index", fontsize=14)
            ax.set_ylabel("Token Prediction Accuracy", fontsize=14)
            ax.set_title("Token Prediction Accuracy by Layer", fontsize=16)
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig_path = os.path.join(paths["figures_dir"], "token_prediction_accuracy")
            saved = save_figure(fig, fig_path)
            figures["token_prediction_accuracy"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    return figures


def export_token_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export token metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_token_metrics(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_tokens.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "token_frequency" in metrics.get("metrics", {}):
        csv_path = os.path.join(paths["tables_dir"], "token_frequency.csv")
        save_table_csv(metrics["metrics"]["token_frequency"], csv_path)
        csv_paths["token_frequency"] = csv_path
    
    if "token_prediction_accuracy" in metrics.get("metrics", {}):
        csv_path = os.path.join(paths["tables_dir"], "token_prediction_accuracy.csv")
        save_table_csv(metrics["metrics"]["token_prediction_accuracy"], csv_path)
        csv_paths["token_prediction_accuracy"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
