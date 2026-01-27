"""
Generate comprehensive cross-temperature and cross-model visualizations.

This script loads data from multiple runs (different temperatures) and creates
unified visualizations showing:
1. Truth vs Social projections across layers (by temperature, model family, condition)
2. Sycophancy rates across temperatures
3. Turn layer detection across temperatures
4. Vector collision patterns by temperature

Usage:
    python "Analysis Scripts/generate_cross_temperature_figures.py" \
        --run-ids 20260124_133539_66ddd916-d61c-4b5d-8ece-594ecd23a983 \
                  20260124_194416_f21e76a6-270c-4347-8a87-dcde3db4b371 \
                  20260124_230102_0af03fbc-d576-4afa-9815-b37a11f57631 \
        --output-dir ./cross_temp_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
except ImportError:
    raise RuntimeError("pandas, matplotlib, and numpy are required for this script")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aam.analytics.plotting_style import (
    create_figure,
    get_color_palette,
    save_figure,
    setup_publication_style,
)
from aam.analytics.utils import load_simulation_db
from aam.persistence import TraceDb


@dataclass
class RunData:
    """Container for run data with metadata."""
    run_id: str
    run_dir: str
    temperature: float
    db: TraceDb


def find_run_dir_for_run_id(*, run_id: str, runs_dir: str) -> str:
    """Find the most recently modified run directory matching run_id."""
    base = Path(runs_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"runs_dir not found: {base}")

    matches: list[Path] = []
    # Handle extended run IDs with -temp0 suffix
    base_run_id = run_id.replace("-temp0", "")
    
    for p in base.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        # Extract run_id from directory name (format: timestamp_timestamp_runid or timestamp_timestamp_runid-temp0)
        # Try to match the base run_id
        if n == run_id or n.endswith(run_id):
            matches.append(p)
        elif n.endswith(f"{base_run_id}-temp0") or n.endswith(base_run_id):
            matches.append(p)
        # Also try extracting run_id from directory name
        parts = n.split("_")
        if len(parts) >= 3:
            extracted_id = "_".join(parts[2:])
            if extracted_id == run_id or extracted_id == base_run_id:
                matches.append(p)
            elif extracted_id.endswith(f"-temp0") and extracted_id.replace("-temp0", "") == base_run_id:
                matches.append(p)

    if not matches:
        raise FileNotFoundError(f"No run directory found for run_id={run_id!r} (base={base_run_id!r}) under {str(base)!r}")

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matches[0])


def extract_temperature_from_db(db: TraceDb, run_id: str) -> float:
    """Extract temperature from conformity_trials table."""
    row = db.conn.execute(
        """
        SELECT DISTINCT temperature
        FROM conformity_trials
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    
    if row is None:
        # Try to extract from run config
        run_row = db.conn.execute(
            "SELECT config_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run_row:
            try:
                config = json.loads(run_row["config_json"])
                temp = config.get("run", {}).get("temperature", 0.0)
                return float(temp)
            except:
                pass
        return 0.0
    
    return float(row["temperature"])


def load_runs_data(run_ids: List[str], runs_dir: str) -> List[RunData]:
    """Load data from multiple runs."""
    runs_data = []
    
    for run_id in run_ids:
        # Handle extended run IDs with -temp0 suffix
        base_run_id = run_id.replace("-temp0", "")
        try:
            run_dir = find_run_dir_for_run_id(run_id=run_id, runs_dir=runs_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping run {run_id}")
            continue
        
        try:
            db = load_simulation_db(run_dir)
            # Try to get run_id from database first
            db_run_id_row = db.conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            
            if db_run_id_row:
                actual_run_id = db_run_id_row["run_id"]
            else:
                # Fall back to base_run_id
                actual_run_id = base_run_id
            
            temperature = extract_temperature_from_db(db, actual_run_id)
            
            runs_data.append(RunData(
                run_id=actual_run_id,
                run_dir=run_dir,
                temperature=temperature,
                db=db,
            ))
        except Exception as e:
            print(f"Warning: Failed to load run {run_id}: {e}, skipping")
            continue
    
    return runs_data


def get_temperature_color(temperature: float) -> str:
    """Get a consistent color for a temperature value."""
    temp_colors = {
        0.0: "#1f77b4",  # blue
        0.5: "#ff7f0e",  # orange
        1.0: "#d62728",  # red
    }
    return temp_colors.get(temperature, "#7f7f7f")


def get_variant_linestyle(variant: str) -> str:
    """Get a consistent linestyle for a model variant."""
    variant_styles = {
        "base": "-",
        "instruct": "--",
        "instruct_sft": "-.",
        "think": ":",
        "think_sft": (0, (3, 1, 1, 1)),
        "rl_zero": (0, (5, 5)),
    }
    return variant_styles.get(variant, "-")


def get_variant_display_name(variant: str) -> str:
    """Get a display name for a model variant."""
    display_names = {
        "base": "Base",
        "instruct": "Instruct",
        "instruct_sft": "Instruct-SFT",
        "think": "Think",
        "think_sft": "Think-SFT",
        "rl_zero": "RL-Zero",
    }
    return display_names.get(variant, variant)


def generate_tug_of_war_figure(
    runs_data: List[RunData],
    output_dir: str,
    condition_filter: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate Figure 2-style Truth vs Social projections across layers,
    showing all temperatures and model families in unified plots.
    
    Args:
        runs_data: List of RunData objects
        output_dir: Output directory for figures
        condition_filter: Optional condition name to filter (e.g., "asch_history_5")
    
    Returns:
        Dict mapping figure_name -> path
    """
    setup_publication_style()
    figures = {}
    
    # Collect all projection data
    all_projections = []
    
    for run in runs_data:
        # Get probe IDs
        truth_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'truth'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        social_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'social'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        if not truth_probe or not social_probe:
            continue
        
        truth_probe_id = truth_probe["probe_id"]
        social_probe_id = social_probe["probe_id"]
        
        # Load projections
        query = """
            SELECT 
                p.layer_index,
                p.value_float,
                pr.probe_kind,
                t.variant,
                c.name AS condition_name,
                t.temperature
            FROM conformity_probe_projections p
            JOIN conformity_probes pr ON pr.probe_id = p.probe_id
            JOIN conformity_trials t ON t.trial_id = p.trial_id
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            WHERE pr.run_id = ? AND pr.probe_id IN (?, ?)
        """
        
        params = [run.run_id, truth_probe_id, social_probe_id]
        if condition_filter:
            query += " AND c.name = ?"
            params.append(condition_filter)
        
        query += " ORDER BY p.layer_index"
        
        df = pd.read_sql_query(query, run.db.conn, params=params)
        
        if not df.empty:
            df["run_temperature"] = run.temperature
            all_projections.append(df)
    
    if not all_projections:
        return figures
    
    combined_df = pd.concat(all_projections, ignore_index=True)
    
    # Separate truth and social
    truth_data = combined_df[combined_df["probe_kind"] == "truth"].copy()
    social_data = combined_df[combined_df["probe_kind"] == "social"].copy()
    
    if truth_data.empty or social_data.empty:
        return figures
    
    # Compute means by layer, variant, temperature, condition
    truth_mean = (
        truth_data.groupby(["variant", "run_temperature", "condition_name", "layer_index"], as_index=False)
        ["value_float"]
        .agg(mean_value="mean", std_value="std", n="count")
    )
    
    social_mean = (
        social_data.groupby(["variant", "run_temperature", "condition_name", "layer_index"], as_index=False)
        ["value_float"]
        .agg(mean_value="mean", std_value="std", n="count")
    )
    
    # Create separate plots for each condition
    conditions = sorted(truth_mean["condition_name"].unique())
    
    for condition in conditions:
        if condition_filter and condition != condition_filter:
            continue
        
        # Filter data for this condition
        cond_truth = truth_mean[truth_mean["condition_name"] == condition]
        cond_social = social_mean[social_mean["condition_name"] == condition]
        
        if cond_truth.empty or cond_social.empty:
            continue
        
        # Create figure with subplots for each variant
        variants = sorted(cond_truth["variant"].unique())
        n_variants = len(variants)
        
        # Use a grid layout: 2 columns, rows as needed
        n_cols = 2
        n_rows = (n_variants + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), constrained_layout=True)
        if n_variants == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, variant in enumerate(variants):
            ax = axes[idx]
            
            var_truth = cond_truth[cond_truth["variant"] == variant]
            var_social = cond_social[cond_social["variant"] == variant]
            
            # Plot each temperature
            temperatures = sorted(var_truth["run_temperature"].unique())
            
            for temp in temperatures:
                temp_truth = var_truth[var_truth["run_temperature"] == temp].sort_values("layer_index")
                temp_social = var_social[var_social["run_temperature"] == temp].sort_values("layer_index")
                
                color = get_temperature_color(temp)
                linestyle = get_variant_linestyle(variant)
                
                # Plot truth (solid line)
                if not temp_truth.empty:
                    ax.plot(
                        temp_truth["layer_index"],
                        temp_truth["mean_value"],
                        label=f"Truth (T={temp})",
                        color=color,
                        linestyle="-",
                        linewidth=2.5,
                        marker="o" if len(temp_truth) <= 32 else None,
                        markersize=4,
                    )
                
                # Plot social (dashed line)
                if not temp_social.empty:
                    ax.plot(
                        temp_social["layer_index"],
                        temp_social["mean_value"],
                        label=f"Social (T={temp})",
                        color=color,
                        linestyle="--",
                        linewidth=2.5,
                        marker="s" if len(temp_social) <= 32 else None,
                        markersize=4,
                        alpha=0.8,
                    )
            
            ax.set_xlabel("Layer Index", fontsize=12)
            ax.set_ylabel("Projection Value", fontsize=12)
            ax.set_title(f"{get_variant_display_name(variant)} - {condition}", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
            
            # Add vertical line at typical turn layer (20-24)
            ax.axvspan(20, 24, alpha=0.1, color="gray", label="Typical Turn Layer")
        
        # Hide unused subplots
        for idx in range(n_variants, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(
            f"Truth vs Social Projections Across Layers\n{condition}",
            fontsize=16,
            fontweight="bold",
            y=1.0,
        )
        
        fig_path = os.path.join(output_dir, f"tug_of_war_{condition.replace(' ', '_').lower()}")
        saved = save_figure(fig, fig_path)
        figures[f"tug_of_war_{condition}"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)
    
    return figures


def generate_sycophancy_by_temperature(
    runs_data: List[RunData],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate sycophancy rate comparison across temperatures.
    
    Returns:
        Dict mapping figure_name -> path
    """
    setup_publication_style()
    figures = {}
    
    all_sycophancy = []
    
    for run in runs_data:
        # Load behavioral data
        df = pd.read_sql_query(
            """
            SELECT 
                t.variant,
                t.item_id,
                t.model_id,
                c.name AS condition_name,
                o.is_correct,
                o.raw_text,
                t.temperature
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ? AND o.is_correct IS NOT NULL
            """,
            run.db.conn,
            params=(run.run_id,),
        )
        
        if df.empty:
            continue
        
        # Mark empty responses
        df["is_empty_response"] = df["raw_text"].apply(lambda x: x is None or (isinstance(x, str) and x.strip() == ""))
        df_non_empty = df[~df["is_empty_response"]].copy()
        
        # Compute sycophancy
        control_trials = df_non_empty[df_non_empty["condition_name"] == "control"].copy()
        pressure_trials = df_non_empty[
            df_non_empty["condition_name"].isin(["asch_history_5", "authoritative_bias"])
        ].copy()
        
        if not control_trials.empty and not pressure_trials.empty:
            merged = control_trials.merge(
                pressure_trials,
                on=["variant", "model_id", "item_id"],
                suffixes=("_control", "_pressure"),
                how="inner",
            )
            
            control_correct = merged[merged["is_correct_control"] == 1].copy()
            
            if not control_correct.empty:
                sycophancy = (
                    control_correct.groupby(["variant", "condition_name_pressure"], as_index=False)
                    .agg({"is_correct_pressure": ["mean", "count"]})
                )
                sycophancy.columns = ["variant", "pressure_condition", "pressure_accuracy", "n_items"]
                sycophancy["sycophancy_rate"] = 1.0 - sycophancy["pressure_accuracy"]
                sycophancy["temperature"] = run.temperature
                all_sycophancy.append(sycophancy)
    
    if not all_sycophancy:
        return figures
    
    combined = pd.concat(all_sycophancy, ignore_index=True)
    
    # Create grouped bar chart
    fig, ax = create_figure(size_key="wide")
    
    # Pivot for grouped bars
    pivot = combined.pivot_table(
        index="variant",
        columns=["pressure_condition", "temperature"],
        values="sycophancy_rate",
        fill_value=0.0,
    )
    
    # Create grouped bars
    variants = sorted(combined["variant"].unique())
    conditions = sorted(combined["pressure_condition"].unique())
    temperatures = sorted(combined["temperature"].unique())
    
    x = np.arange(len(variants))
    width = 0.35 / len(conditions) / len(temperatures)
    
    for cond_idx, condition in enumerate(conditions):
        for temp_idx, temp in enumerate(temperatures):
            offset = (cond_idx * len(temperatures) + temp_idx) * width - (len(conditions) * len(temperatures) * width) / 2 + width / 2
            
            values = []
            for variant in variants:
                try:
                    val = pivot.loc[variant, (condition, temp)]
                except KeyError:
                    val = 0.0
                values.append(val)
            
            color = get_temperature_color(temp)
            alpha = 0.7 if condition == "asch_history_5" else 0.9
            
            ax.bar(
                x + offset,
                values,
                width,
                label=f"{condition} (T={temp})",
                color=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
    
    ax.set_xlabel("Model Variant", fontsize=14)
    ax.set_ylabel("Sycophancy Rate", fontsize=14)
    ax.set_title("Sycophancy Rate Across Temperatures and Conditions", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([get_variant_display_name(v) for v in variants], rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    
    fig_path = os.path.join(output_dir, "sycophancy_by_temperature")
    saved = save_figure(fig, fig_path)
    figures["sycophancy_by_temperature"] = saved.get("png", saved.get("pdf", ""))
    plt.close(fig)
    
    return figures


def generate_turn_layer_by_temperature(
    runs_data: List[RunData],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate turn layer analysis across temperatures.
    
    Returns:
        Dict mapping figure_name -> path
    """
    setup_publication_style()
    figures = {}
    
    all_turn_layers = []
    
    for run in runs_data:
        # Get probe IDs
        truth_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'truth'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        social_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'social'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        if not truth_probe or not social_probe:
            continue
        
        truth_probe_id = truth_probe["probe_id"]
        social_probe_id = social_probe["probe_id"]
        
        # Find first collision layer per trial
        df = pd.read_sql_query(
            """
            WITH truth AS (
              SELECT trial_id, layer_index, value_float AS truth_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            social AS (
              SELECT trial_id, layer_index, value_float AS social_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            merged AS (
              SELECT
                t.trial_id,
                t.layer_index,
                (s.social_val - t.truth_val) AS diff
              FROM truth t
              JOIN social s
                ON s.trial_id = t.trial_id AND s.layer_index = t.layer_index
            ),
            collisions AS (
              SELECT trial_id, MIN(layer_index) AS first_collision_layer
              FROM merged
              WHERE diff > 0
              GROUP BY trial_id
            )
            SELECT
              c.trial_id,
              c.first_collision_layer,
              tr.variant,
              cond.name AS condition_name,
              tr.temperature
            FROM collisions c
            JOIN conformity_trials tr ON tr.trial_id = c.trial_id
            JOIN conformity_conditions cond ON cond.condition_id = tr.condition_id
            WHERE tr.run_id = ?
            ORDER BY c.first_collision_layer ASC;
            """,
            run.db.conn,
            params=(truth_probe_id, social_probe_id, run.run_id),
        )
        
        if not df.empty:
            df["run_temperature"] = run.temperature
            all_turn_layers.append(df)
    
    if not all_turn_layers:
        return figures
    
    combined = pd.concat(all_turn_layers, ignore_index=True)
    
    # Create violin/box plot showing turn layer distribution
    fig, ax = create_figure(size_key="wide")
    
    variants = sorted(combined["variant"].unique())
    temperatures = sorted(combined["run_temperature"].unique())
    conditions = sorted(combined["condition_name"].unique())
    
    # Filter to pressure conditions only
    pressure_conditions = [c for c in conditions if c in ["asch_history_5", "authoritative_bias"]]
    
    if not pressure_conditions:
        return figures
    
    # Create grouped positions
    positions = []
    labels = []
    data_groups = []
    
    pos = 0
    for variant in variants:
        for condition in pressure_conditions:
            for temp in temperatures:
                subset = combined[
                    (combined["variant"] == variant) &
                    (combined["condition_name"] == condition) &
                    (combined["run_temperature"] == temp)
                ]
                
                if not subset.empty:
                    positions.append(pos)
                    labels.append(f"{get_variant_display_name(variant)}\n{condition}\nT={temp}")
                    data_groups.append(subset["first_collision_layer"].values)
                    pos += 1
    
    if not data_groups:
        return figures
    
    # Create box plot
    bp = ax.boxplot(
        data_groups,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=False,
    )
    
    # Color boxes by temperature
    color_idx = 0
    for patch in bp["boxes"]:
        # Determine temperature from position
        idx = color_idx // len(pressure_conditions) % len(temperatures)
        temp = temperatures[idx]
        patch.set_facecolor(get_temperature_color(temp))
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
        color_idx += 1
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("First Collision Layer (Turn Layer)", fontsize=14)
    ax.set_title("Turn Layer Distribution Across Temperatures", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    fig_path = os.path.join(output_dir, "turn_layer_by_temperature")
    saved = save_figure(fig, fig_path)
    figures["turn_layer_by_temperature"] = saved.get("png", saved.get("pdf", ""))
    plt.close(fig)
    
    return figures


def generate_vector_difference_heatmap(
    runs_data: List[RunData],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate heatmap showing (Social - Truth) difference across layers, temperatures, and variants.
    
    Returns:
        Dict mapping figure_name -> path
    """
    setup_publication_style()
    figures = {}
    
    all_differences = []
    
    for run in runs_data:
        # Get probe IDs
        truth_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'truth'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        social_probe = run.db.conn.execute(
            """
            SELECT probe_id FROM conformity_probes
            WHERE run_id = ? AND probe_kind = 'social'
            ORDER BY created_at DESC LIMIT 1
            """,
            (run.run_id,),
        ).fetchone()
        
        if not truth_probe or not social_probe:
            continue
        
        truth_probe_id = truth_probe["probe_id"]
        social_probe_id = social_probe["probe_id"]
        
        # Compute difference
        df = pd.read_sql_query(
            """
            WITH truth AS (
              SELECT trial_id, layer_index, value_float AS truth_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            social AS (
              SELECT trial_id, layer_index, value_float AS social_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            )
            SELECT
              tr.variant,
              cond.name AS condition_name,
              t.layer_index,
              (s.social_val - t.truth_val) AS diff,
              tr.temperature
            FROM truth t
            JOIN social s
              ON s.trial_id = t.trial_id AND s.layer_index = t.layer_index
            JOIN conformity_trials tr ON tr.trial_id = t.trial_id
            JOIN conformity_conditions cond ON cond.condition_id = tr.condition_id
            WHERE tr.run_id = ? AND cond.name IN ('control', 'asch_history_5', 'authoritative_bias')
            """,
            run.db.conn,
            params=(truth_probe_id, social_probe_id, run.run_id),
        )
        
        if not df.empty:
            df["run_temperature"] = run.temperature
            all_differences.append(df)
    
    if not all_differences:
        return figures
    
    combined = pd.concat(all_differences, ignore_index=True)
    
    # Create heatmap for each variant and condition combination
    variants = sorted(combined["variant"].unique())
    conditions = sorted(combined["condition_name"].unique())
    
    for variant in variants:
        for condition in conditions:
            var_cond_data = combined[
                (combined["variant"] == variant) &
                (combined["condition_name"] == condition)
            ]
            
            if var_cond_data.empty:
                continue
            
            # Pivot: layers x temperatures
            pivot = var_cond_data.pivot_table(
                index="layer_index",
                columns="run_temperature",
                values="diff",
                aggfunc="mean",
            )
            
            if pivot.empty:
                continue
            
            fig, ax = create_figure(size_key="single")
            
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                cmap="RdBu_r",
                interpolation="nearest",
            )
            
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"T={t}" for t in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Temperature", fontsize=14)
            ax.set_ylabel("Layer Index", fontsize=14)
            ax.set_title(
                f"Social - Truth Difference\n{get_variant_display_name(variant)} - {condition}",
                fontsize=16,
                fontweight="bold",
            )
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("SVP - TVP", fontsize=12)
            
            fig_path = os.path.join(
                output_dir,
                f"vector_difference_heatmap_{variant}_{condition.replace(' ', '_').lower()}",
            )
            saved = save_figure(fig, fig_path)
            figures[f"vector_difference_heatmap_{variant}_{condition}"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cross-temperature and cross-model visualizations"
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="List of run IDs to analyze",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(REPO_ROOT / "runs"),
        help="Base runs/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Optional condition filter (e.g., 'asch_history_5')",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load runs data
    print(f"Loading data from {len(args.run_ids)} runs...")
    runs_data = load_runs_data(args.run_ids, args.runs_dir)
    
    print(f"Loaded {len(runs_data)} runs with temperatures: {[r.temperature for r in runs_data]}")
    
    # Generate figures
    all_figures = {}
    
    print("Generating tug-of-war figures...")
    tug_figures = generate_tug_of_war_figure(runs_data, args.output_dir, args.condition)
    all_figures.update(tug_figures)
    
    print("Generating sycophancy by temperature figure...")
    syc_figures = generate_sycophancy_by_temperature(runs_data, args.output_dir)
    all_figures.update(syc_figures)
    
    print("Generating turn layer by temperature figure...")
    turn_figures = generate_turn_layer_by_temperature(runs_data, args.output_dir)
    all_figures.update(turn_figures)
    
    print("Generating vector difference heatmaps...")
    heatmap_figures = generate_vector_difference_heatmap(runs_data, args.output_dir)
    all_figures.update(heatmap_figures)
    
    # Print results
    print("\nGenerated figures:")
    for name, path in all_figures.items():
        print(f"  {name}={path}")
    
    # Close databases
    for run in runs_data:
        run.db.close()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
