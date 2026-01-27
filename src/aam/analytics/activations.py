"""
Activation analytics for Olmo Conformity Experiment.

Implements Figure 4 (PCA/UMAP of Activation Space) prerequisites and capture coverage logs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise RuntimeError("pandas, matplotlib, and numpy are required for activation analytics")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb


def compute_activation_stats(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute activation statistics: file count, layers captured, memory usage.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        
    Returns:
        Dict with computed metrics
    """
    # Load activation metadata
    df = pd.read_sql_query(
        """
        SELECT 
            record_id,
            layer_index,
            component,
            token_position,
            shard_file_path,
            shape_json,
            dtype,
            time_step,
            agent_id
        FROM activation_metadata
        WHERE run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No activation metadata found for this run"},
        }
    
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "statistics": {},
    }
    
    # Activation file count and sizes
    unique_files = df["shard_file_path"].nunique()
    metrics["metrics"]["activation_file_count"] = int(unique_files)
    
    # Layers captured
    layers_captured = sorted(df["layer_index"].unique().tolist())
    metrics["metrics"]["layers_captured"] = layers_captured
    
    # Component coverage
    components = df["component"].unique().tolist()
    metrics["metrics"]["components_captured"] = components
    
    # Statistics by layer
    layer_stats = (
        df.groupby("layer_index", as_index=False)
        .agg({
            "record_id": "count",
            "component": lambda x: x.nunique(),
        })
        .rename(columns={"record_id": "n_records", "component": "n_components"})
    )
    metrics["metrics"]["layer_statistics"] = layer_stats.to_dict("records")
    
    # Memory usage estimate (rough - would need actual file sizes)
    metrics["statistics"] = {
        "total_records": len(df),
        "unique_files": unique_files,
        "layers_captured": len(layers_captured),
        "components": components,
        "time_steps": sorted(df["time_step"].unique().tolist()),
    }
    
    return metrics


def generate_activation_embeddings(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    layer_index: Optional[int] = None,
    method: str = "pca",
    n_components: int = 2,
) -> Dict[str, Any]:
    """
    Generate dimensionality reduction embeddings for activation space (Figure 4).
    
    Note: This requires loading actual activation tensors from safetensors files.
    For now, this is a placeholder that documents the expected interface.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        layer_index: Layer to analyze (None = last layer)
        method: "pca", "tsne", or "umap"
        n_components: Number of dimensions (2 for visualization)
        
    Returns:
        Dict with embedding data and metadata
    """
    # Load activation metadata
    df_meta = pd.read_sql_query(
        """
        SELECT 
            record_id,
            layer_index,
            component,
            shard_file_path,
            tensor_key,
            time_step,
            agent_id
        FROM activation_metadata
        WHERE run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df_meta.empty:
        return {
            "run_id": run_id,
            "embedding": None,
            "metadata": {"message": "No activation metadata found"},
        }
    
    # Filter by layer if specified
    if layer_index is not None:
        df_meta = df_meta[df_meta["layer_index"] == layer_index]
    
    if df_meta.empty:
        return {
            "run_id": run_id,
            "embedding": None,
            "metadata": {"message": f"No activations found for layer {layer_index}"},
        }
    
    # TODO: Load actual activation tensors from safetensors files
    # This requires:
    # 1. Loading safetensors files using safetensors library
    # 2. Extracting tensors by tensor_key
    # 3. Reshaping/flattening to vectors
    # 4. Applying dimensionality reduction
    
    return {
        "run_id": run_id,
        "embedding": None,
        "metadata": {
            "message": "Activation tensor loading not yet implemented",
            "n_records": len(df_meta),
            "layer_index": layer_index,
            "method": method,
        },
    }


def generate_activation_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    layer_index: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate activation visualizations (Figure 4 + coverage graphs).
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        layer_index: Layer to visualize (None = last layer)
        
    Returns:
        Dict mapping figure_name -> path
    """
    paths = ensure_logs_dir(run_dir)
    setup_publication_style()
    
    figures = {}
    
    # Load activation metadata
    df = pd.read_sql_query(
        """
        SELECT 
            layer_index,
            component,
            time_step
        FROM activation_metadata
        WHERE run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    
    if df.empty:
        return figures
    
    # Coverage heatmap: layers x components
    coverage = df.groupby(["layer_index", "component"], as_index=False).size()
    coverage_pivot = coverage.pivot(
        index="layer_index",
        columns="component",
        values="size",
    ).fillna(0)
    
    if not coverage_pivot.empty:
        fig, ax = create_figure(size_key="single")
        im = ax.imshow(coverage_pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(len(coverage_pivot.columns)))
        ax.set_xticklabels(coverage_pivot.columns)
        ax.set_yticks(range(len(coverage_pivot.index)))
        ax.set_yticklabels(coverage_pivot.index)
        ax.set_xlabel("Component", fontsize=14)
        ax.set_ylabel("Layer Index", fontsize=14)
        ax.set_title("Activation Capture Coverage", fontsize=16)
        plt.colorbar(im, ax=ax, label="Number of Records")
        
        fig_path = os.path.join(paths["figures_dir"], "activation_coverage")
        saved = save_figure(fig, fig_path)
        figures["activation_coverage"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    # Figure 4 placeholder: PCA/UMAP scatter
    # This requires actual tensor loading (see generate_activation_embeddings)
    # For now, we'll create a note that this requires tensor loading
    
    return figures


def export_activation_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Export activation metrics to JSON and CSV files.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        metrics: Pre-computed metrics (if None, will compute)
        
    Returns:
        Dict mapping log_type -> path
    """
    if metrics is None:
        metrics = compute_activation_stats(trace_db, run_id, run_dir)
    
    paths = ensure_logs_dir(run_dir)
    
    # Save JSON metrics
    json_path = os.path.join(paths["logs_dir"], "metrics_activations.json")
    save_metrics_json(metrics, json_path)
    
    # Save CSV tables
    csv_paths = {}
    
    if "layer_statistics" in metrics["metrics"]:
        csv_path = os.path.join(paths["tables_dir"], "activation_metadata.csv")
        save_table_csv(metrics["metrics"]["layer_statistics"], csv_path)
        csv_paths["activation_metadata"] = csv_path
    
    return {
        "metrics_json": json_path,
        **csv_paths,
    }
