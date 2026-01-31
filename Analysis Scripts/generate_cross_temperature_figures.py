"""
Cross-Temperature Paper Figures Generator

Generates 3 composite paper-quality figures from three temperature runs (T=0, T=0.5, T=1):
- Figure 1: Behavioral composite (error rates / social pressure effect across variants×temperatures)
- Figure 2: Mechanism / Turn-Layer (SVP-TVP across layers, per variant, with temperature curves)
- Figure 3: Intervention composite (flip-to-truth rate vs alpha, per variant, with temperature curves)

Usage:
    python generate_cross_temperature_figures.py [--out-dir OUTPUT_DIR]

The run IDs are configured in the RUNS dict below. Modify as needed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "cross_temp_figures_output"

# Run directories and IDs for all three temperatures
RUNS = {
    0.0: {
        "dir": "20260127_211450_73b34738-b76e-4c55-8653-74b497b1989b",
        "id": "73b34738-b76e-4c55-8653-74b497b1989b",
    },
    0.5: {
        "dir": "20260127_231128_4e6cd5a7-af59-4fe2-ae8d-c9bcc2f57c00",
        "id": "4e6cd5a7-af59-4fe2-ae8d-c9bcc2f57c00",
    },
    1.0: {
        "dir": "20260127_222205_f1c7ed74-2561-4c52-9279-3d3269fcb7f3",
        "id": "f1c7ed74-2561-4c52-9279-3d3269fcb7f3",
    },
}

TEMPERATURES = [0.0, 0.5, 1.0]
TEMP_LABELS = {0.0: "T=0", 0.5: "T=0.5", 1.0: "T=1"}
TEMP_COLORS = {0.0: "#2C7BB6", 0.5: "#2CA25F", 1.0: "#D7191C"}  # Colorblind-safe: blue, green, red
TEMP_LINESTYLES = {0.0: '-', 0.5: '--', 1.0: ':'}  # Different line styles to distinguish overlapping lines
TEMP_MARKERS = {0.0: 'o', 0.5: 's', 1.0: '^'}  # Different markers

BEHAVIORAL_CONDITIONS = ('control', 'asch_history_5', 'authoritative_bias')
CONDITION_LABELS = {
    'control': 'Control',
    'asch_history_5': 'Asch (5)',
    'authoritative_bias': 'Authority',
}

VARIANTS = ['base', 'instruct', 'instruct_sft', 'think', 'think_sft', 'rl_zero']
VARIANT_LABELS = {
    'base': 'Base',
    'instruct': 'Instruct',
    'instruct_sft': 'Instruct-SFT',
    'think': 'Think',
    'think_sft': 'Think-SFT',
    'rl_zero': 'RL-Zero',
}

# Model ID to variant mapping
MODEL_TO_VARIANT = {
    "allenai/Olmo-3-1025-7B": "base",
    "allenai/Olmo-3-7B-Instruct": "instruct",
    "allenai/Olmo-3-7B-Instruct-SFT": "instruct_sft",
    "allenai/Olmo-3-7B-Think": "think",
    "allenai/Olmo-3-7B-Think-SFT": "think_sft",
    "allenai/Olmo-3-7B-RL-Zero-Math": "rl_zero",
}

# ============================================================================
# SETUP
# ============================================================================

def setup_style():
    """Setup publication-quality plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def connect_db(runs_dir: Path, run_dir: str) -> sqlite3.Connection:
    """Connect to simulation database with row factory."""
    db_path = runs_dir / run_dir / "simulation.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# DATA LOADING
# ============================================================================

def load_behavioral_data(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Load behavioral trial data from a single run (one row per trial, using first/original output)."""
    query = """
    WITH first_outputs AS (
        SELECT trial_id, MIN(created_at) AS min_created_at
        FROM conformity_outputs
        GROUP BY trial_id
    ),
    first_output_ids AS (
        SELECT MIN(o.output_id) AS output_id, o.trial_id
        FROM conformity_outputs o
        JOIN first_outputs fo ON fo.trial_id = o.trial_id AND fo.min_created_at = o.created_at
        GROUP BY o.trial_id
    )
    SELECT 
        t.trial_id,
        t.variant,
        c.name AS condition_name,
        i.ground_truth_text,
        o.is_correct,
        o.refusal_flag
    FROM conformity_trials t
    JOIN conformity_conditions c ON t.condition_id = c.condition_id
    JOIN conformity_items i ON t.item_id = i.item_id
    JOIN first_output_ids foi ON foi.trial_id = t.trial_id
    JOIN conformity_outputs o ON o.output_id = foi.output_id
    WHERE t.run_id = ?
      AND c.name IN (?, ?, ?);
    """
    df = pd.read_sql_query(query, conn, params=[run_id, *BEHAVIORAL_CONDITIONS])
    return df


def load_probe_projections(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Load probe projections (may be from probe-capture or behavioral trials)."""
    query = """
    SELECT 
        p.trial_id,
        p.probe_id,
        p.layer_index,
        p.value_float,
        pr.probe_kind,
        pr.model_id,
        t.variant,
        c.name AS condition_name
    FROM conformity_probe_projections p
    JOIN conformity_probes pr ON pr.probe_id = p.probe_id
    JOIN conformity_trials t ON t.trial_id = p.trial_id
    JOIN conformity_conditions c ON c.condition_id = t.condition_id
    WHERE pr.run_id = ?
    ORDER BY p.trial_id, p.layer_index;
    """
    df = pd.read_sql_query(query, conn, params=[run_id])
    return df


def load_intervention_data(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Load intervention results from a single run."""
    query = """
    SELECT 
        r.result_id,
        r.trial_id,
        r.flipped_to_truth,
        i.alpha,
        i.name AS intervention_name,
        t.variant,
        c.name AS condition_name
    FROM conformity_intervention_results r
    JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
    JOIN conformity_trials t ON t.trial_id = r.trial_id
    JOIN conformity_conditions c ON c.condition_id = t.condition_id
    WHERE i.run_id = ?;
    """
    df = pd.read_sql_query(query, conn, params=[run_id])
    return df


def load_all_data(runs_dir: Path) -> Dict[str, Any]:
    """Load all data from all three temperature runs."""
    data = {
        'behavioral': {},
        'projections': {},
        'interventions': {},
    }
    
    for temp, run_info in RUNS.items():
        conn = connect_db(runs_dir, run_info["dir"])
        run_id = run_info["id"]
        
        data['behavioral'][temp] = load_behavioral_data(conn, run_id)
        data['projections'][temp] = load_probe_projections(conn, run_id)
        data['interventions'][temp] = load_intervention_data(conn, run_id)
        
        conn.close()
    
    return data


# ============================================================================
# FIGURE 1: BEHAVIORAL COMPOSITE
# ============================================================================

def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute error and refusal rates by condition and variant."""
    agg = df.groupby(['condition_name', 'variant']).agg(
        n_trials=('trial_id', 'count'),
        n_correct=('is_correct', 'sum'),
        n_refusals=('refusal_flag', 'sum'),
    ).reset_index()
    
    agg['error_rate'] = 1 - (agg['n_correct'] / agg['n_trials'])
    agg['refusal_rate'] = agg['n_refusals'] / agg['n_trials']
    
    return agg


def generate_figure1_behavioral(data: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate Figure 1: Behavioral composite.
    
    Layout: 3 heatmaps (one per condition) showing error rates as variants×temperature.
    """
    rates_by_temp = {temp: compute_rates(df) for temp, df in data['behavioral'].items()}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    for idx, condition in enumerate(BEHAVIORAL_CONDITIONS):
        ax = axes[idx]
        
        # Build matrix: rows=variants, cols=temperatures
        matrix = np.zeros((len(VARIANTS), len(TEMPERATURES)))
        for i, variant in enumerate(VARIANTS):
            for j, temp in enumerate(TEMPERATURES):
                df = rates_by_temp[temp]
                row = df[(df['condition_name'] == condition) & (df['variant'] == variant)]
                if not row.empty:
                    matrix[i, j] = row['error_rate'].iloc[0]
        
        # Heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.4, vmax=1.0)
        
        # X-axis labels (temperature)
        ax.set_xticks(range(len(TEMPERATURES)))
        ax.set_xticklabels([TEMP_LABELS[t] for t in TEMPERATURES])
        ax.set_xlabel('Temperature')
        
        # Y-axis labels (model variants) - only on leftmost panel
        ax.set_yticks(range(len(VARIANTS)))
        if idx == 0:
            ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS])
            ax.set_ylabel('Model Variant')
        else:
            ax.set_yticklabels([])
        
        ax.set_title(CONDITION_LABELS.get(condition, condition), fontweight='bold')
        
        # Annotations
        for i in range(len(VARIANTS)):
            for j in range(len(TEMPERATURES)):
                val = matrix[i, j]
                color = 'white' if val > 0.7 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', 
                       fontsize=9, color=color, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='Error Rate')
    
    plt.suptitle('Figure 1: Error Rates Across Models, Conditions, and Temperatures', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Save
    output_path = output_dir / 'figure1_behavioral_composite'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}.png/pdf")
    return str(output_path)


def generate_figure1b_social_pressure(data: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate Figure 1b: Social pressure effect (Asch - Control) by variant and temperature.
    """
    rates_by_temp = {temp: compute_rates(df) for temp, df in data['behavioral'].items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(VARIANTS))
    width = 0.25
    
    for i, temp in enumerate(TEMPERATURES):
        df = rates_by_temp[temp]
        effects = []
        for variant in VARIANTS:
            control = df[(df['condition_name'] == 'control') & (df['variant'] == variant)]
            asch = df[(df['condition_name'] == 'asch_history_5') & (df['variant'] == variant)]
            if not control.empty and not asch.empty:
                effect = asch['error_rate'].iloc[0] - control['error_rate'].iloc[0]
            else:
                effect = 0
            effects.append(effect)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, effects, width, label=TEMP_LABELS[temp], 
                     color=TEMP_COLORS[temp], alpha=0.9)
    
    ax.set_ylabel('Social Pressure Effect\n(Asch Error Rate − Control Error Rate)')
    ax.set_xlabel('Model Variant')
    ax.set_title('Figure 1b: Social Pressure Effect by Temperature', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=30, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.25, 0.2)
    
    # Save
    output_path = output_dir / 'figure1b_social_pressure_effect'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}.png/pdf")
    return str(output_path)


# ============================================================================
# FIGURE 2: MECHANISM / TURN-LAYER
# ============================================================================

def generate_figure2_mechanism(data: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate Figure 2: Mechanism / Turn-Layer composite.
    
    Layout: 2×3 grid (one panel per variant) showing SVP-TVP difference across layers
    for each temperature.
    
    NOTE: Currently uses probe-capture trial projections. For the full story,
    we need projections computed on behavioral (Asch) trials.
    """
    # Check if we have any projection data
    has_projections = any(len(df) > 0 for df in data['projections'].values())
    
    if not has_projections:
        print("  WARNING: No probe projection data available for mechanism figure")
        return ""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, variant in enumerate(VARIANTS):
        ax = axes[idx]
        
        for temp in TEMPERATURES:
            df = data['projections'][temp]
            if df.empty:
                continue
            
            # Map model_id to variant
            df = df.copy()
            df['mapped_variant'] = df['model_id'].map(MODEL_TO_VARIANT)
            
            # Filter to this variant (by model mapping, since probe trials may have variant="huggingface")
            variant_df = df[df['mapped_variant'] == variant]
            
            if variant_df.empty:
                continue
            
            # Separate truth and social projections
            truth_df = variant_df[variant_df['probe_kind'].str.contains('truth', case=False, na=False)]
            social_df = variant_df[variant_df['probe_kind'].str.contains('social', case=False, na=False)]
            
            if truth_df.empty or social_df.empty:
                continue
            
            # Compute mean by layer
            truth_mean = truth_df.groupby('layer_index')['value_float'].mean()
            social_mean = social_df.groupby('layer_index')['value_float'].mean()
            
            # Compute difference (SVP - TVP)
            layers = sorted(set(truth_mean.index) & set(social_mean.index))
            if not layers:
                continue
            
            diff = [social_mean[l] - truth_mean[l] for l in layers]
            
            ax.plot(layers, diff, 
                   linestyle=TEMP_LINESTYLES[temp],
                   marker=TEMP_MARKERS[temp],
                   label=TEMP_LABELS[temp], 
                   color=TEMP_COLORS[temp], 
                   linewidth=2, 
                   markersize=5,
                   alpha=0.8)
        
        # Reference line at 0 (turn point)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_title(VARIANT_LABELS[variant], fontweight='bold')
        if idx >= 3:
            ax.set_xlabel('Layer Index')
        if idx % 3 == 0:
            ax.set_ylabel('Social − Truth Projection')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('Figure 2: Truth vs Social Vector Difference Across Layers\n'
                 '(Positive = Social dominates; Turn Layer where line crosses 0)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Add note about data source
    fig.text(0.5, -0.02, 
             'Note: Lines overlap because probe projections are computed on identical input texts across temperatures.\n'
             'Temperature affects generation, not input activations. Different line styles used for visibility.',
             ha='center', fontsize=9, style='italic', alpha=0.7)
    
    # Save
    output_path = output_dir / 'figure2_mechanism_turn_layer'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}.png/pdf")
    return str(output_path)


# ============================================================================
# FIGURE 3: INTERVENTION COMPOSITE
# ============================================================================

def generate_figure3_intervention(data: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate Figure 3: Intervention composite.
    
    Layout: 2×3 grid (one panel per variant) showing flip-to-truth rate vs alpha
    for each temperature.
    """
    # Check which temperatures have intervention data
    temps_with_data = [t for t, df in data['interventions'].items() if len(df) > 0]
    
    if not temps_with_data:
        print("  WARNING: No intervention data available for any temperature")
        return ""
    
    print(f"  Intervention data available for: {[TEMP_LABELS[t] for t in temps_with_data]}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, variant in enumerate(VARIANTS):
        ax = axes[idx]
        
        has_data_for_variant = False
        
        for temp in TEMPERATURES:
            df = data['interventions'][temp]
            if df.empty:
                continue
            
            variant_df = df[df['variant'] == variant]
            if variant_df.empty:
                continue
            
            # Compute flip rate by alpha
            flip_by_alpha = variant_df.groupby('alpha').agg(
                flip_rate=('flipped_to_truth', 'mean'),
                n=('result_id', 'count')
            ).reset_index()
            
            if flip_by_alpha.empty:
                continue
            
            has_data_for_variant = True
            ax.plot(flip_by_alpha['alpha'], flip_by_alpha['flip_rate'], 
                   'o-', label=TEMP_LABELS[temp], color=TEMP_COLORS[temp], 
                   linewidth=2, markersize=6)
        
        ax.set_title(VARIANT_LABELS[variant], fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        
        if idx >= 3:
            ax.set_xlabel('Intervention Strength (α)')
        if idx % 3 == 0:
            ax.set_ylabel('Flip-to-Truth Rate')
        
        if idx == 0 and has_data_for_variant:
            ax.legend(loc='upper left', fontsize=9)
        
        if not has_data_for_variant:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, alpha=0.5)
    
    plt.suptitle('Figure 3: Causal Intervention Effect\n'
                 '(Flip-to-Truth Rate by Social Vector Subtraction Strength)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Add note about missing temperatures
    missing_temps = [TEMP_LABELS[t] for t in TEMPERATURES if t not in temps_with_data]
    if missing_temps:
        fig.text(0.5, -0.02, 
                 f'Note: Intervention data missing for {", ".join(missing_temps)}.',
                 ha='center', fontsize=9, style='italic', alpha=0.7)
    
    # Save
    output_path = output_dir / 'figure3_intervention_composite'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}.png/pdf")
    return str(output_path)


# ============================================================================
# FIGURE 4: TEMPERATURE SCATTER (Item-Level Conformity)
# ============================================================================

CONDITION_COLORS = {
    'control': '#2CA02C',           # Green
    'asch_history_5': '#D62728',    # Red  
    'authoritative_bias': '#1F77B4', # Blue
}

VARIANT_MARKERS = {
    'base': 'o',
    'instruct': 's',
    'instruct_sft': '^',
    'think': 'D',
    'think_sft': 'v',
    'rl_zero': 'P',
}


def generate_figure4_temperature_scatter(data: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate Figure 4: Temperature Effect on Error Rates.
    
    Scatter plots comparing error rates between temperatures, with:
    - Color = Condition (Control/Asch/Authority)
    - Marker shape = Model variant
    - y=x line shows where no temperature effect would be
    
    Points above the line = higher error at higher temperature
    Points below the line = lower error at higher temperature
    """
    rates_by_temp = {temp: compute_rates(df) for temp, df in data['behavioral'].items()}
    
    # Create 1x3 layout: T=0 vs T=0.5, T=0 vs T=1, T=0.5 vs T=1
    temp_pairs = [(0.0, 0.5), (0.0, 1.0), (0.5, 1.0)]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    
    # Make room at the top for legends
    fig.subplots_adjust(top=0.75, wspace=0.25)
    
    for ax_idx, (t_low, t_high) in enumerate(temp_pairs):
        ax = axes[ax_idx]
        
        df_low = rates_by_temp[t_low]
        df_high = rates_by_temp[t_high]
        
        # Plot each condition × variant combination
        for condition in BEHAVIORAL_CONDITIONS:
            for variant in VARIANTS:
                row_low = df_low[(df_low['condition_name'] == condition) & (df_low['variant'] == variant)]
                row_high = df_high[(df_high['condition_name'] == condition) & (df_high['variant'] == variant)]
                
                if row_low.empty or row_high.empty:
                    continue
                
                x = row_low['error_rate'].iloc[0]
                y = row_high['error_rate'].iloc[0]
                
                ax.scatter(x, y, 
                          c=CONDITION_COLORS[condition],
                          marker=VARIANT_MARKERS[variant],
                          s=120,
                          alpha=0.8,
                          edgecolors='white',
                          linewidths=0.5)
        
        # y=x reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='y=x (no effect)')
        
        # Formatting
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_aspect('equal')
        ax.set_xlabel(f'Error Rate ({TEMP_LABELS[t_low]})')
        ax.set_ylabel(f'Error Rate ({TEMP_LABELS[t_high]})')
        ax.set_title(f'{TEMP_LABELS[t_low]} vs {TEMP_LABELS[t_high]}', fontweight='bold')
        
        # Add region annotations
        ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0], alpha=0.05, color='red')
        ax.fill_between([0.4, 1.0], [0.4, 0.4], [0.4, 1.0], alpha=0.05, color='green')
        
        if ax_idx == 2:
            ax.text(0.95, 0.45, 'Higher T\nimproves', ha='right', va='bottom', fontsize=8, alpha=0.6)
            ax.text(0.45, 0.95, 'Higher T\nhurts', ha='left', va='top', fontsize=8, alpha=0.6)
    
    # Create legend with condition colors and variant markers
    from matplotlib.lines import Line2D
    
    # Condition legend
    condition_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS[c], 
                                markersize=10, label=CONDITION_LABELS[c]) 
                        for c in BEHAVIORAL_CONDITIONS]
    
    # Variant legend  
    variant_handles = [Line2D([0], [0], marker=VARIANT_MARKERS[v], color='gray', 
                              markersize=8, linestyle='', label=VARIANT_LABELS[v])
                      for v in VARIANTS]
    
    # Add legends in a horizontal row at the top, outside the plot area
    legend1 = fig.legend(handles=condition_handles, loc='upper left', bbox_to_anchor=(0.08, 0.99),
                        title='Condition', fontsize=9, title_fontsize=10, framealpha=0.95,
                        ncol=1, columnspacing=0.5)
    legend2 = fig.legend(handles=variant_handles, loc='upper left', bbox_to_anchor=(0.22, 0.99),
                        title='Model', fontsize=9, title_fontsize=10, ncol=3, framealpha=0.95,
                        columnspacing=0.8)
    fig.add_artist(legend1)
    
    plt.suptitle('Figure 4: Temperature Effect on Error Rates\n'
                 '(Points above diagonal = temperature increases errors)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'figure4_temperature_scatter'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}.png/pdf")
    return str(output_path)


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def generate_summary_table(data: Dict[str, Any], output_dir: Path) -> str:
    """Generate a summary CSV with key metrics across all temperatures."""
    rows = []
    
    for temp in TEMPERATURES:
        df = data['behavioral'][temp]
        rates = compute_rates(df)
        
        for _, row in rates.iterrows():
            rows.append({
                'temperature': temp,
                'variant': row['variant'],
                'condition': row['condition_name'],
                'n_trials': row['n_trials'],
                'error_rate': row['error_rate'],
                'refusal_rate': row['refusal_rate'],
            })
    
    summary_df = pd.DataFrame(rows)
    output_path = output_dir / 'cross_temperature_summary.csv'
    summary_df.to_csv(output_path, index=False)
    
    print(f"  Saved: {output_path}")
    return str(output_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate cross-temperature paper figures')
    parser.add_argument('--runs-dir', type=str, default=str(DEFAULT_RUNS_DIR),
                       help='Base directory containing run folders')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for figures')
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Cross-Temperature Paper Figures Generator")
    print("="*60)
    
    # Setup
    setup_style()
    
    # Load data
    print("\n[1] Loading data from all runs...")
    data = load_all_data(runs_dir)
    
    for temp in TEMPERATURES:
        n_behavioral = len(data['behavioral'][temp])
        n_projections = len(data['projections'][temp])
        n_interventions = len(data['interventions'][temp])
        print(f"  {TEMP_LABELS[temp]}: {n_behavioral} behavioral, {n_projections} projections, {n_interventions} interventions")
    
    # Generate figures
    print("\n[2] Generating Figure 1: Behavioral composite...")
    generate_figure1_behavioral(data, output_dir)
    generate_figure1b_social_pressure(data, output_dir)
    
    print("\n[3] Generating Figure 2: Mechanism / Turn-Layer...")
    generate_figure2_mechanism(data, output_dir)
    
    print("\n[4] Generating Figure 3: Intervention composite...")
    generate_figure3_intervention(data, output_dir)
    
    print("\n[5] Generating Figure 4: Temperature scatter...")
    generate_figure4_temperature_scatter(data, output_dir)
    
    print("\n[6] Generating summary table...")
    generate_summary_table(data, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
