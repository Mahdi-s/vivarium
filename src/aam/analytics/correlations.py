"""
Cross-analysis correlation functions for Olmo Conformity Experiment.

Computes correlations between different analysis dimensions:
- Probe projections vs behavioral outcomes
- Probe projections vs judge eval scores
- Logit lens vs probe projections
- Intervention effects vs probe projections
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise RuntimeError("pandas and numpy are required for correlation analysis")

from aam.analytics.statistics import compute_correlation
from aam.persistence import TraceDb


def compute_probe_behavioral_correlations(
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Compute correlations between probe projections and behavioral correctness.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with correlation results for truth and social probes
    """
    # Load probe projections with behavioral outcomes
    df = pd.read_sql_query(
        """
        SELECT 
            p.layer_index,
            p.value_float,
            pr.probe_kind,
            t.variant,
            c.name AS condition_name,
            o.is_correct
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE pr.run_id = ? AND o.is_correct IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {"correlations": [], "message": "No data available"}
    
    results = []
    
    # Correlations by probe kind and layer
    for probe_kind in df["probe_kind"].unique():
        probe_data = df[df["probe_kind"] == probe_kind].copy()
        
        for layer in sorted(probe_data["layer_index"].unique()):
            layer_data = probe_data[probe_data["layer_index"] == layer]
            
            if len(layer_data) < 2:
                continue
            
            corr_result = compute_correlation(
                layer_data["value_float"].tolist(),
                layer_data["is_correct"].tolist(),
            )
            
            if corr_result["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "layer_index": int(layer),
                    **corr_result,
                })
    
    # Overall correlation (averaged across layers)
    for probe_kind in df["probe_kind"].unique():
        probe_data = df[df["probe_kind"] == probe_kind].copy()
        
        # Average projection per trial across layers
        trial_avg = probe_data.groupby(["trial_id"])["value_float"].mean().reset_index()
        trial_avg = trial_avg.merge(
            probe_data[["trial_id", "is_correct"]].drop_duplicates(),
            on="trial_id",
        )
        
        if len(trial_avg) >= 2:
            corr_result = compute_correlation(
                trial_avg["value_float"].tolist(),
                trial_avg["is_correct"].tolist(),
            )
            
            if corr_result["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "layer_index": None,  # Average across layers
                    **corr_result,
                })
    
    return {"correlations": results}


def compute_probe_judgeval_correlations(
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Compute correlations between probe projections and judge eval scores.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with correlation results
    """
    # Load probe projections with judge eval scores
    df = pd.read_sql_query(
        """
        SELECT 
            p.layer_index,
            p.value_float,
            pr.probe_kind,
            t.variant,
            c.name AS condition_name,
            json_extract(o.parsed_answer_json, '$.conformity') AS conformity_score,
            json_extract(o.parsed_answer_json, '$.truthfulness') AS truthfulness_score,
            json_extract(o.parsed_answer_json, '$.rationalization') AS rationalization_score
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE pr.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {"correlations": [], "message": "No judge eval data available"}
    
    # Convert judge eval scores to numeric
    for col in ["conformity_score", "truthfulness_score", "rationalization_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    results = []
    
    # Correlations by probe kind
    for probe_kind in df["probe_kind"].unique():
        probe_data = df[df["probe_kind"] == probe_kind].copy()
        
        # Average projection per trial across layers
        trial_avg = probe_data.groupby(["trial_id"])["value_float"].mean().reset_index()
        trial_avg = trial_avg.merge(
            probe_data[["trial_id", "conformity_score", "truthfulness_score", "rationalization_score"]].drop_duplicates(),
            on="trial_id",
        )
        
        # Correlation with conformity score
        if trial_avg["conformity_score"].notna().sum() >= 2:
            corr_conv = compute_correlation(
                trial_avg["value_float"].tolist(),
                trial_avg["conformity_score"].dropna().tolist(),
            )
            if corr_conv["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "judge_metric": "conformity",
                    **corr_conv,
                })
        
        # Correlation with truthfulness score
        if trial_avg["truthfulness_score"].notna().sum() >= 2:
            corr_truth = compute_correlation(
                trial_avg["value_float"].tolist(),
                trial_avg["truthfulness_score"].dropna().tolist(),
            )
            if corr_truth["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "judge_metric": "truthfulness",
                    **corr_truth,
                })
    
    return {"correlations": results}


def compute_logit_lens_probe_correlations(
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Compute correlations between logit lens top predictions and probe projections.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with correlation results
    """
    # Load logit lens data
    df_logit = pd.read_sql_query(
        """
        SELECT 
            ll.trial_id,
            ll.layer_index,
            ll.topk_json,
            t.variant
        FROM conformity_logit_lens ll
        JOIN conformity_trials t ON t.trial_id = ll.trial_id
        WHERE t.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_logit.empty:
        return {"correlations": [], "message": "No logit lens data available"}
    
    # Extract top probability from logit lens
    def get_top_prob(topk_json: str) -> float:
        try:
            topk = json.loads(topk_json)
            if isinstance(topk, list) and len(topk) > 0:
                return float(topk[0].get("prob", 0.0))
        except:
            pass
        return 0.0
    
    df_logit["top_prob"] = df_logit["topk_json"].apply(get_top_prob)
    
    # Load probe projections
    df_probe = pd.read_sql_query(
        """
        SELECT 
            p.trial_id,
            p.layer_index,
            p.value_float,
            pr.probe_kind
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        WHERE pr.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_probe.empty:
        return {"correlations": [], "message": "No probe projection data available"}
    
    # Merge on trial_id and layer_index
    merged = df_logit.merge(
        df_probe,
        on=["trial_id", "layer_index"],
        how="inner",
    )
    
    if merged.empty:
        return {"correlations": [], "message": "No overlapping data"}
    
    results = []
    
    # Correlations by probe kind and layer
    for probe_kind in merged["probe_kind"].unique():
        probe_data = merged[merged["probe_kind"] == probe_kind].copy()
        
        for layer in sorted(probe_data["layer_index"].unique()):
            layer_data = probe_data[probe_data["layer_index"] == layer]
            
            if len(layer_data) < 2:
                continue
            
            corr_result = compute_correlation(
                layer_data["top_prob"].tolist(),
                layer_data["value_float"].tolist(),
            )
            
            if corr_result["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "layer_index": int(layer),
                    **corr_result,
                })
    
    return {"correlations": results}


def compute_intervention_probe_correlations(
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Compute correlations between intervention effects and probe projections.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with correlation results
    """
    # Load intervention results
    df_int = pd.read_sql_query(
        """
        SELECT 
            r.trial_id,
            r.flipped_to_truth,
            r.before_correct,
            r.after_correct,
            i.alpha,
            i.name AS intervention_name
        FROM conformity_intervention_results r
        JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
        JOIN conformity_trials t ON t.trial_id = r.trial_id
        WHERE i.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_int.empty:
        return {"correlations": [], "message": "No intervention data available"}
    
    # Compute intervention effect
    df_int["effect"] = df_int["after_correct"] - df_int["before_correct"]
    
    # Load probe projections (average across layers)
    df_probe = pd.read_sql_query(
        """
        SELECT 
            p.trial_id,
            p.value_float,
            pr.probe_kind
        FROM conformity_probe_projections p
        JOIN conformity_probes pr ON pr.probe_id = p.probe_id
        JOIN conformity_trials t ON t.trial_id = p.trial_id
        WHERE pr.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_probe.empty:
        return {"correlations": [], "message": "No probe projection data available"}
    
    # Average probe projections per trial
    probe_avg = df_probe.groupby(["trial_id", "probe_kind"])["value_float"].mean().reset_index()
    
    # Merge with intervention results
    merged = df_int.merge(probe_avg, on="trial_id", how="inner")
    
    if merged.empty:
        return {"correlations": [], "message": "No overlapping data"}
    
    results = []
    
    # Correlations by probe kind
    for probe_kind in merged["probe_kind"].unique():
        probe_data = merged[merged["probe_kind"] == probe_kind].copy()
        
        # Correlation with intervention effect
        if len(probe_data) >= 2:
            corr_effect = compute_correlation(
                probe_data["value_float"].tolist(),
                probe_data["effect"].tolist(),
            )
            
            if corr_effect["correlation"] is not None:
                results.append({
                    "probe_kind": probe_kind,
                    "metric": "intervention_effect",
                    **corr_effect,
                })
            
            # Correlation with flip to truth
            if probe_data["flipped_to_truth"].notna().sum() >= 2:
                corr_flip = compute_correlation(
                    probe_data["value_float"].tolist(),
                    probe_data["flipped_to_truth"].tolist(),
                )
                
                if corr_flip["correlation"] is not None:
                    results.append({
                        "probe_kind": probe_kind,
                        "metric": "flipped_to_truth",
                        **corr_flip,
                    })
    
    return {"correlations": results}


def compute_all_correlations(
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Compute all cross-analysis correlations.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with all correlation results
    """
    return {
        "probe_behavioral": compute_probe_behavioral_correlations(trace_db, run_id),
        "probe_judgeval": compute_probe_judgeval_correlations(trace_db, run_id),
        "logit_lens_probe": compute_logit_lens_probe_correlations(trace_db, run_id),
        "intervention_probe": compute_intervention_probe_correlations(trace_db, run_id),
    }
