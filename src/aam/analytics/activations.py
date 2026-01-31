"""
Activation analytics for Olmo Conformity Experiment.

Implements:
- Figure 4 (PCA/UMAP of Activation Space) prerequisites and capture coverage logs
- Condition-wise activation comparison (without probe training)
- Activation statistics and similarity analysis
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise RuntimeError("pandas, matplotlib, and numpy are required for activation analytics")

try:
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import cosine as cosine_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

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


# ============================================================================
# ACTIVATION COMPARISON FUNCTIONS (WITHOUT PROBE TRAINING)
# ============================================================================

def load_activation_vectors(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    layer_index: int,
    component: str = "resid_post",
    max_vectors: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load activation vectors from safetensors files for a specific layer.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        layer_index: Layer to load activations from
        component: Component name (e.g., "resid_post", "attn_out")
        max_vectors: Maximum number of vectors to load (None = all)
        
    Returns:
        Tuple of (vectors: np.ndarray [N, d_model], metadata: pd.DataFrame)
    """
    if not HAS_SAFETENSORS:
        raise RuntimeError("safetensors library required. Install with: pip install safetensors")
    
    # Query activation metadata
    query = """
    SELECT 
        am.record_id,
        am.shard_file_path,
        am.tensor_key,
        am.shape_json,
        am.time_step,
        am.agent_id,
        t.trial_id,
        t.variant,
        c.name AS condition_name
    FROM activation_metadata am
    JOIN conformity_trial_steps ts ON am.time_step = ts.time_step AND am.agent_id = ts.agent_id
    JOIN conformity_trials t ON ts.trial_id = t.trial_id
    JOIN conformity_conditions c ON t.condition_id = c.condition_id
    WHERE am.run_id = ? AND am.layer_index = ? AND am.component = ?
    """
    
    params = [run_id, layer_index, component]
    if max_vectors is not None:
        query += " LIMIT ?"
        params.append(max_vectors)
    
    df = pd.read_sql_query(query, trace_db.conn, params=params)
    
    if df.empty:
        return np.array([]), pd.DataFrame()
    
    # Load vectors from safetensors files
    vectors = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            shard_path = row["shard_file_path"]
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(run_dir, shard_path)
            
            if not os.path.exists(shard_path):
                continue
            
            tensors = load_safetensors(shard_path)
            tensor_key = row["tensor_key"]
            
            if tensor_key in tensors:
                vec = tensors[tensor_key].detach().cpu().numpy().flatten()
                vectors.append(vec)
                valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not load tensor {row['tensor_key']}: {e}")
            continue
    
    if not vectors:
        return np.array([]), pd.DataFrame()
    
    # Stack vectors and filter metadata
    vectors_array = np.stack(vectors, axis=0)
    metadata = df.iloc[valid_indices].reset_index(drop=True)
    
    return vectors_array, metadata


def compute_activation_comparison(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    layer_indices: Optional[List[int]] = None,
    component: str = "resid_post",
) -> Dict[str, Any]:
    """
    Compare activation statistics across conditions WITHOUT training probes.
    
    This function loads activations and computes:
    - Mean activation norms by condition
    - Variance analysis
    - Layer-wise activation evolution
    - Cosine similarity between condition centroids
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        run_dir: Run directory path
        layer_indices: Layers to analyze (None = auto-select representative layers)
        component: Activation component (default: "resid_post")
        
    Returns:
        Dict with comparison metrics and statistics
    """
    if layer_indices is None:
        # Default: analyze layers at 25%, 50%, 75%, and 100% depth
        # First, find available layers
        df_layers = pd.read_sql_query(
            "SELECT DISTINCT layer_index FROM activation_metadata WHERE run_id = ?",
            trace_db.conn,
            params=(run_id,),
        )
        if df_layers.empty:
            return {"error": "No activation metadata found"}
        
        all_layers = sorted(df_layers["layer_index"].tolist())
        n_layers = len(all_layers)
        if n_layers >= 4:
            indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
            layer_indices = [all_layers[i] for i in indices if i < n_layers]
        else:
            layer_indices = all_layers
    
    results = {
        "run_id": run_id,
        "layers_analyzed": layer_indices,
        "component": component,
        "layer_stats": {},
        "condition_comparison": {},
    }
    
    conditions = ["control", "asch_history_5", "authoritative_bias"]
    
    for layer_idx in layer_indices:
        print(f"  Analyzing layer {layer_idx}...")
        vectors, metadata = load_activation_vectors(
            trace_db, run_id, run_dir, layer_idx, component
        )
        
        if len(vectors) == 0:
            results["layer_stats"][layer_idx] = {"error": "No vectors loaded"}
            continue
        
        layer_result = {
            "n_vectors": len(vectors),
            "vector_dim": vectors.shape[1] if len(vectors) > 0 else 0,
            "conditions": {},
        }
        
        # Compute statistics by condition
        condition_centroids = {}
        
        for condition in conditions:
            mask = metadata["condition_name"] == condition
            cond_vectors = vectors[mask]
            
            if len(cond_vectors) == 0:
                continue
            
            # Compute norms
            norms = np.linalg.norm(cond_vectors, axis=1)
            
            # Compute centroid
            centroid = np.mean(cond_vectors, axis=0)
            condition_centroids[condition] = centroid
            
            # Compute variance
            variance = np.var(cond_vectors, axis=0).mean()  # Mean variance across dimensions
            
            layer_result["conditions"][condition] = {
                "n_samples": len(cond_vectors),
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "mean_variance": float(variance),
            }
        
        # Compute cosine similarities between condition centroids
        if len(condition_centroids) >= 2 and HAS_SCIPY:
            similarities = {}
            cond_list = list(condition_centroids.keys())
            for i, c1 in enumerate(cond_list):
                for c2 in cond_list[i + 1:]:
                    sim = 1 - cosine_distance(condition_centroids[c1], condition_centroids[c2])
                    similarities[f"{c1}_vs_{c2}"] = float(sim)
            layer_result["centroid_similarities"] = similarities
        
        results["layer_stats"][layer_idx] = layer_result
    
    # Statistical tests comparing conditions (if scipy available)
    if HAS_SCIPY and len(results["layer_stats"]) > 0:
        for layer_idx, layer_data in results["layer_stats"].items():
            if "conditions" not in layer_data:
                continue
            
            cond_data = layer_data["conditions"]
            
            # Compare control vs asch_history_5 norms
            if "control" in cond_data and "asch_history_5" in cond_data:
                control_norm = cond_data["control"]["mean_norm"]
                asch_norm = cond_data["asch_history_5"]["mean_norm"]
                diff = asch_norm - control_norm
                results["condition_comparison"][f"layer_{layer_idx}_norm_diff"] = {
                    "control_vs_asch": float(diff),
                    "description": f"Mean norm difference: Asch - Control = {diff:.4f}",
                }
    
    return results


def plot_activation_comparison(
    comparison_results: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate visualizations for activation comparison.
    
    Args:
        comparison_results: Output from compute_activation_comparison()
        output_dir: Directory to save figures
        
    Returns:
        Dict mapping figure name to path
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}
    
    layer_stats = comparison_results.get("layer_stats", {})
    if not layer_stats:
        return figures
    
    # Figure 1: Mean activation norms by layer and condition
    layers = sorted(layer_stats.keys())
    conditions = ["control", "asch_history_5", "authoritative_bias"]
    condition_labels = {"control": "Control", "asch_history_5": "Asch", "authoritative_bias": "Authority"}
    condition_colors = {"control": "#4363d8", "asch_history_5": "#e6194B", "authoritative_bias": "#3cb44b"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layers))
    width = 0.25
    
    for i, condition in enumerate(conditions):
        norms = []
        for layer in layers:
            if layer in layer_stats and "conditions" in layer_stats[layer]:
                cond_data = layer_stats[layer]["conditions"].get(condition, {})
                norms.append(cond_data.get("mean_norm", 0))
            else:
                norms.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, norms, width, label=condition_labels.get(condition, condition),
               color=condition_colors.get(condition, "gray"), alpha=0.8)
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Activation Norm")
    ax.set_title("Activation Norms by Layer and Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    fig_path = os.path.join(output_dir, "activation_norms_by_condition.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    figures["activation_norms"] = fig_path
    
    # Figure 2: Centroid similarities heatmap
    has_similarities = any(
        "centroid_similarities" in layer_stats.get(layer, {})
        for layer in layers
    )
    
    if has_similarities:
        similarity_data = []
        for layer in layers:
            if "centroid_similarities" in layer_stats.get(layer, {}):
                sims = layer_stats[layer]["centroid_similarities"]
                row = {"layer": layer}
                row.update(sims)
                similarity_data.append(row)
        
        if similarity_data:
            df_sim = pd.DataFrame(similarity_data)
            sim_cols = [c for c in df_sim.columns if c != "layer"]
            
            if sim_cols:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                matrix = df_sim[sim_cols].values
                im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
                
                ax.set_xticks(np.arange(len(sim_cols)))
                ax.set_xticklabels([s.replace("_vs_", " vs ") for s in sim_cols], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(df_sim)))
                ax.set_yticklabels(df_sim["layer"].tolist())
                ax.set_xlabel("Condition Pair")
                ax.set_ylabel("Layer Index")
                ax.set_title("Cosine Similarity Between Condition Centroids")
                
                plt.colorbar(im, ax=ax, label="Cosine Similarity")
                
                # Add text annotations
                for i in range(len(df_sim)):
                    for j in range(len(sim_cols)):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
                
                fig_path = os.path.join(output_dir, "centroid_similarities.png")
                plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close()
                figures["centroid_similarities"] = fig_path
    
    return figures


def generate_activation_comparison_report(
    comparison_results: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Generate a markdown report of activation comparison analysis.
    
    Args:
        comparison_results: Output from compute_activation_comparison()
        output_path: Path to save the report
        
    Returns:
        The report content as a string
    """
    lines = [
        "# Activation Comparison Report",
        "",
        "Analysis of internal activations across experimental conditions (without probe training).",
        "",
        "## Overview",
        "",
        f"- Run ID: {comparison_results.get('run_id', 'Unknown')}",
        f"- Layers analyzed: {comparison_results.get('layers_analyzed', [])}",
        f"- Component: {comparison_results.get('component', 'Unknown')}",
        "",
        "## Layer-wise Statistics",
        "",
    ]
    
    layer_stats = comparison_results.get("layer_stats", {})
    
    for layer_idx, layer_data in sorted(layer_stats.items()):
        if "error" in layer_data:
            lines.append(f"### Layer {layer_idx}: {layer_data['error']}")
            continue
        
        lines.append(f"### Layer {layer_idx}")
        lines.append("")
        lines.append(f"- Vectors: {layer_data.get('n_vectors', 0)}")
        lines.append(f"- Dimension: {layer_data.get('vector_dim', 0)}")
        lines.append("")
        
        conditions = layer_data.get("conditions", {})
        if conditions:
            lines.append("| Condition | N | Mean Norm | Std Norm | Mean Variance |")
            lines.append("|-----------|---|-----------|----------|---------------|")
            
            for cond, stats in conditions.items():
                lines.append(
                    f"| {cond} | {stats.get('n_samples', 0)} | "
                    f"{stats.get('mean_norm', 0):.4f} | "
                    f"{stats.get('std_norm', 0):.4f} | "
                    f"{stats.get('mean_variance', 0):.6f} |"
                )
            lines.append("")
        
        similarities = layer_data.get("centroid_similarities", {})
        if similarities:
            lines.append("**Centroid Similarities:**")
            for pair, sim in similarities.items():
                lines.append(f"- {pair.replace('_vs_', ' vs ')}: {sim:.4f}")
            lines.append("")
    
    # Summary
    lines.extend([
        "## Interpretation",
        "",
        "This analysis examines how model activations differ across experimental conditions:",
        "",
        "- **Activation Norms**: Higher norms may indicate stronger internal representations",
        "- **Centroid Similarities**: Values close to 1.0 indicate similar activation patterns between conditions",
        "- **Variance**: Higher variance may indicate less consistent internal processing",
        "",
        "If activations differ significantly between control and pressure conditions, this suggests",
        "the model's internal state is affected by social pressure, not just its outputs.",
        "",
    ])
    
    report = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(report)
    
    return report
