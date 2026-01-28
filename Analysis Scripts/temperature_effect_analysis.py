"""
Temperature Effect Analysis for Olmo Conformity Experiments

Analyzes two runs comparing T=0 (deterministic) vs T=1 (stochastic) decoding
to understand how temperature affects social conformity in LLMs.

Run IDs:
- T=0: 73b34738-b76e-4c55-8653-74b497b1989b
- T=1: f1c7ed74-2561-4c52-9279-3d3269fcb7f3
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configuration
RUNS_DIR = Path.home() / "repos" / "abstractAgentMachine" / "runs"
OUTPUT_DIR = Path.home() / "repos" / "abstractAgentMachine" / "Analysis Scripts" / "temperature_analysis_output"

RUN_T0 = "20260127_211450_73b34738-b76e-4c55-8653-74b497b1989b"
RUN_T1 = "20260127_222205_f1c7ed74-2561-4c52-9279-3d3269fcb7f3"

RUN_ID_T0 = "73b34738-b76e-4c55-8653-74b497b1989b"
RUN_ID_T1 = "f1c7ed74-2561-4c52-9279-3d3269fcb7f3"

# Behavioral conditions (exclude probe capture conditions)
BEHAVIORAL_CONDITIONS = ('control', 'asch_history_5', 'authoritative_bias')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""
    run_id: str
    temperature: float
    seed: int
    created_at: float
    suite_name: str
    models: List[Dict[str, str]]
    conditions: List[Dict[str, Any]]
    datasets: List[Dict[str, str]]
    

def connect_db(run_dir: str) -> sqlite3.Connection:
    """Connect to simulation database with row factory."""
    db_path = RUNS_DIR / run_dir / "simulation.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_run_config(conn: sqlite3.Connection) -> RunConfig:
    """Load and parse run configuration from database."""
    row = conn.execute(
        "SELECT run_id, seed, created_at, config_json FROM runs LIMIT 1;"
    ).fetchone()
    
    config = json.loads(row["config_json"])
    suite = config.get("suite_config", {})
    run_params = suite.get("run", {})
    
    return RunConfig(
        run_id=row["run_id"],
        temperature=run_params.get("temperature", 0.0),
        seed=row["seed"],
        created_at=row["created_at"],
        suite_name=suite.get("suite_name", "unknown"),
        models=suite.get("models", []),
        conditions=suite.get("conditions", []),
        datasets=suite.get("datasets", []),
    )


def get_behavioral_data(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """
    Extract behavioral data for conformity trials.
    
    Returns DataFrame with columns:
    - trial_id, model_id, variant, condition_name, condition_params
    - question, ground_truth_text, domain
    - raw_text, parsed_answer_text, is_correct, refusal_flag
    - latency_ms, token_usage_json
    """
    query = """
    SELECT 
        t.trial_id,
        t.model_id,
        t.variant,
        c.name AS condition_name,
        c.params_json AS condition_params,
        i.question,
        i.ground_truth_text,
        i.domain,
        o.raw_text,
        o.parsed_answer_text,
        o.is_correct,
        o.refusal_flag,
        o.latency_ms,
        o.token_usage_json,
        o.parsed_answer_json
    FROM conformity_trials t
    JOIN conformity_conditions c ON t.condition_id = c.condition_id
    JOIN conformity_items i ON t.item_id = i.item_id
    JOIN conformity_outputs o ON t.trial_id = o.trial_id
    WHERE t.run_id = ?
      AND c.name IN (?, ?, ?);
    """
    
    df = pd.read_sql_query(query, conn, params=[run_id, *BEHAVIORAL_CONDITIONS])
    return df


def get_probe_data(conn: sqlite3.Connection, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract probe projections if available.
    
    Returns:
    - probes_df: DataFrame of probe metadata
    - projections_df: DataFrame of layer-wise projections
    """
    # Get probes (using actual schema)
    probes_query = """
    SELECT 
        probe_id,
        probe_kind,
        component,
        model_id,
        layers_json,
        metrics_json,
        created_at
    FROM conformity_probes
    WHERE run_id = ?;
    """
    try:
        probes_df = pd.read_sql_query(probes_query, conn, params=[run_id])
    except Exception as e:
        print(f"  Warning: Could not load probes: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    if probes_df.empty:
        return probes_df, pd.DataFrame()
    
    # Get projections
    projections_query = """
    SELECT 
        pp.trial_id,
        pp.probe_id,
        pp.layer_index,
        pp.token_index,
        pp.value_float,
        p.probe_kind,
        p.component
    FROM conformity_probe_projections pp
    JOIN conformity_probes p ON pp.probe_id = p.probe_id
    WHERE p.run_id = ?;
    """
    try:
        projections_df = pd.read_sql_query(projections_query, conn, params=[run_id])
    except Exception as e:
        print(f"  Warning: Could not load projections: {e}")
        return probes_df, pd.DataFrame()
    
    return probes_df, projections_df


def get_intervention_data(conn: sqlite3.Connection, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract intervention results if available.
    
    Returns:
    - interventions_df: DataFrame of intervention definitions
    - results_df: DataFrame of intervention results
    """
    # Use actual schema (no vector_kind column)
    interventions_query = """
    SELECT 
        intervention_id,
        name,
        alpha,
        target_layers_json,
        component,
        notes,
        created_at
    FROM conformity_interventions
    WHERE run_id = ?;
    """
    try:
        interventions_df = pd.read_sql_query(interventions_query, conn, params=[run_id])
    except Exception as e:
        print(f"  Warning: Could not load interventions: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    if interventions_df.empty:
        return interventions_df, pd.DataFrame()
    
    results_query = """
    SELECT 
        ir.trial_id,
        ir.intervention_id,
        ir.flipped_to_truth,
        ir.original_is_correct,
        ir.intervened_is_correct,
        ir.original_output,
        ir.intervened_output,
        i.name AS intervention_name,
        i.alpha,
        i.target_layers_json
    FROM conformity_intervention_results ir
    JOIN conformity_interventions i ON ir.intervention_id = i.intervention_id
    WHERE i.run_id = ?;
    """
    try:
        results_df = pd.read_sql_query(results_query, conn, params=[run_id])
    except Exception as e:
        print(f"  Warning: Could not load intervention results: {e}")
        return interventions_df, pd.DataFrame()
    
    return interventions_df, results_df


def calculate_conformity_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate conformity rates by condition and model.
    
    Returns DataFrame with:
    - condition, variant, n_trials, n_correct, n_incorrect, n_empty
    - accuracy, conformity_rate, refusal_rate
    - mean_latency, std_latency
    """
    # Identify empty responses
    df = df.copy()
    df['is_empty'] = df['raw_text'].isna() | (df['raw_text'] == '')
    
    agg = df.groupby(['condition_name', 'variant']).agg(
        n_trials=('trial_id', 'count'),
        n_correct=('is_correct', 'sum'),
        n_refusals=('refusal_flag', 'sum'),
        n_empty=('is_empty', 'sum'),
        mean_latency=('latency_ms', 'mean'),
        std_latency=('latency_ms', 'std'),
    ).reset_index()
    
    agg['n_incorrect'] = agg['n_trials'] - agg['n_correct']
    agg['accuracy'] = agg['n_correct'] / agg['n_trials']
    agg['conformity_rate'] = 1 - agg['accuracy']
    agg['refusal_rate'] = agg['n_refusals'] / agg['n_trials']
    agg['empty_rate'] = agg['n_empty'] / agg['n_trials']
    
    return agg


def calculate_conformity_susceptibility(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate conformity susceptibility as (control_accuracy - asch_accuracy) / control_accuracy.
    """
    # Pivot to get control and asch columns
    pivot = rates_df.pivot(index='variant', columns='condition_name', values='accuracy').reset_index()
    
    if 'control' in pivot.columns and 'asch_history_5' in pivot.columns:
        pivot['susceptibility_asch'] = (pivot['control'] - pivot['asch_history_5']) / pivot['control']
    
    if 'control' in pivot.columns and 'authoritative_bias' in pivot.columns:
        pivot['susceptibility_auth'] = (pivot['control'] - pivot['authoritative_bias']) / pivot['control']
    
    return pivot


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for proportions.
    
    Returns (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportions.
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def compare_conformity_rates(df_t0: pd.DataFrame, df_t1: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical comparison of conformity rates between T=0 and T=1.
    
    Returns DataFrame with:
    - condition, variant
    - rate_t0, rate_t1, difference
    - ci_lower, ci_upper
    - chi2, p_value, effect_size
    """
    results = []
    
    for condition in BEHAVIORAL_CONDITIONS:
        for variant in df_t0['variant'].unique():
            row_t0 = df_t0[(df_t0['condition_name'] == condition) & (df_t0['variant'] == variant)]
            row_t1 = df_t1[(df_t1['condition_name'] == condition) & (df_t1['variant'] == variant)]
            
            if row_t0.empty or row_t1.empty:
                continue
            
            n_t0 = int(row_t0['n_trials'].iloc[0])
            correct_t0 = int(row_t0['n_correct'].iloc[0])
            n_t1 = int(row_t1['n_trials'].iloc[0])
            correct_t1 = int(row_t1['n_correct'].iloc[0])
            
            rate_t0 = 1 - (correct_t0 / n_t0) if n_t0 > 0 else 0
            rate_t1 = 1 - (correct_t1 / n_t1) if n_t1 > 0 else 0
            
            # Wilson CI for difference (approximation)
            ci_t0 = wilson_ci(n_t0 - correct_t0, n_t0)
            ci_t1 = wilson_ci(n_t1 - correct_t1, n_t1)
            
            # Chi-square test
            table = [[correct_t0, n_t0 - correct_t0], [correct_t1, n_t1 - correct_t1]]
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(table)
                if min(expected.flatten()) < 5:
                    # Use Fisher's exact for small samples
                    _, p_value = stats.fisher_exact(table)
                    chi2 = np.nan
            except Exception:
                chi2, p_value = np.nan, np.nan
            
            # Effect size
            effect = cohens_h(rate_t1, rate_t0) if rate_t0 > 0 else np.nan
            
            results.append({
                'condition': condition,
                'variant': variant,
                'rate_t0': rate_t0,
                'rate_t1': rate_t1,
                'difference': rate_t1 - rate_t0,
                'ci_t0_lower': ci_t0[0],
                'ci_t0_upper': ci_t0[1],
                'ci_t1_lower': ci_t1[0],
                'ci_t1_upper': ci_t1[1],
                'chi2': chi2,
                'p_value': p_value,
                'effect_size_h': effect,
                'n_t0': n_t0,
                'n_t1': n_t1,
            })
    
    return pd.DataFrame(results)


def analyze_probe_trajectories(projections_df: pd.DataFrame, behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze truth vs social vector trajectories across layers.
    
    Returns DataFrame with layer-wise statistics.
    """
    if projections_df.empty:
        return pd.DataFrame()
    
    # Join with behavioral data to get correctness
    merged = projections_df.merge(
        behavioral_df[['trial_id', 'is_correct', 'condition_name', 'variant']],
        on='trial_id',
        how='left'
    )
    
    # Aggregate by layer, probe_kind, condition, correctness
    agg = merged.groupby(['layer_index', 'probe_kind', 'condition_name', 'is_correct']).agg(
        mean_projection=('value_float', 'mean'),
        std_projection=('value_float', 'std'),
        n_samples=('value_float', 'count'),
    ).reset_index()
    
    # Calculate standard error
    agg['se_projection'] = agg['std_projection'] / np.sqrt(agg['n_samples'])
    
    return agg


def identify_turn_layer(projections_df: pd.DataFrame, behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the Turn Layer for each trial.
    
    Turn Layer = first layer where P_social > P_truth for conforming (incorrect) trials.
    
    Returns DataFrame with trial_id, turn_layer, and metadata.
    """
    if projections_df.empty:
        return pd.DataFrame()
    
    # Pivot to get truth and social side by side
    pivot = projections_df.pivot_table(
        index=['trial_id', 'layer_index'],
        columns='probe_kind',
        values='value_float',
        aggfunc='mean'
    ).reset_index()
    
    if 'truth' not in pivot.columns or 'social' not in pivot.columns:
        return pd.DataFrame()
    
    # Find turn layer for each trial
    turn_layers = []
    for trial_id in pivot['trial_id'].unique():
        trial_data = pivot[pivot['trial_id'] == trial_id].sort_values('layer_index')
        
        turn_layer = None
        for _, row in trial_data.iterrows():
            if row['social'] > row['truth']:
                turn_layer = int(row['layer_index'])
                break
        
        turn_layers.append({
            'trial_id': trial_id,
            'turn_layer': turn_layer,
        })
    
    turn_df = pd.DataFrame(turn_layers)
    
    # Merge with behavioral data
    turn_df = turn_df.merge(
        behavioral_df[['trial_id', 'is_correct', 'condition_name', 'variant']],
        on='trial_id',
        how='left'
    )
    
    return turn_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_conformity_rates_by_condition(rates_t0: pd.DataFrame, rates_t1: pd.DataFrame, 
                                        output_path: Path) -> None:
    """
    Figure 1: Conformity Rates by Condition and Temperature
    Grouped bar chart with error bars showing 95% CI.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    
    variants = sorted(rates_t0['variant'].unique())
    
    for idx, variant in enumerate(variants):
        ax = axes[idx]
        
        data_t0 = rates_t0[rates_t0['variant'] == variant].set_index('condition_name')
        data_t1 = rates_t1[rates_t1['variant'] == variant].set_index('condition_name')
        
        conditions = list(BEHAVIORAL_CONDITIONS)
        x = np.arange(len(conditions))
        width = 0.35
        
        # Calculate Wilson CIs
        ci_t0 = [wilson_ci(int(data_t0.loc[c, 'n_trials'] - data_t0.loc[c, 'n_correct']), 
                          int(data_t0.loc[c, 'n_trials'])) 
                 for c in conditions if c in data_t0.index]
        ci_t1 = [wilson_ci(int(data_t1.loc[c, 'n_trials'] - data_t1.loc[c, 'n_correct']), 
                          int(data_t1.loc[c, 'n_trials'])) 
                 for c in conditions if c in data_t1.index]
        
        rates_0 = [data_t0.loc[c, 'conformity_rate'] if c in data_t0.index else 0 
                   for c in conditions]
        rates_1 = [data_t1.loc[c, 'conformity_rate'] if c in data_t1.index else 0 
                   for c in conditions]
        
        # Error bars (symmetric for simplicity)
        err_0 = [(ci[1] - ci[0]) / 2 for ci in ci_t0] if ci_t0 else [0] * len(conditions)
        err_1 = [(ci[1] - ci[0]) / 2 for ci in ci_t1] if ci_t1 else [0] * len(conditions)
        
        bars1 = ax.bar(x - width/2, rates_0, width, label='T=0', 
                       yerr=err_0, capsize=3, color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, rates_1, width, label='T=1', 
                       yerr=err_1, capsize=3, color='coral', alpha=0.8)
        
        ax.set_ylabel('Conformity Rate' if idx % 3 == 0 else '')
        ax.set_title(f'{variant}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'Asch (5)', 'Authority'], rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right')
    
    # Hide unused subplots
    for idx in range(len(variants), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Figure 1: Conformity Rates by Condition and Temperature', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_conformity_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure1_conformity_rates.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def plot_probe_trajectories(trajectories_df: pd.DataFrame, temperature: float,
                           output_path: Path, filename_suffix: str = "") -> None:
    """
    Figure 2: Truth vs Social Projection Trajectories
    Line plot showing layer-wise projection values.
    """
    if trajectories_df.empty:
        print("No probe trajectory data available for Figure 2")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    conditions = list(BEHAVIORAL_CONDITIONS)
    colors = {'truth': 'forestgreen', 'social': 'crimson'}
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        cond_data = trajectories_df[trajectories_df['condition_name'] == condition]
        
        for probe_kind in ['truth', 'social']:
            probe_data = cond_data[cond_data['probe_kind'] == probe_kind]
            if probe_data.empty:
                continue
            
            # Aggregate across correctness for overall trajectory
            agg = probe_data.groupby('layer_index').agg(
                mean=('mean_projection', 'mean'),
                se=('se_projection', 'mean'),
            ).reset_index()
            
            ax.plot(agg['layer_index'], agg['mean'], 
                   label=f'P_{probe_kind}', color=colors[probe_kind], linewidth=2)
            ax.fill_between(agg['layer_index'], 
                           agg['mean'] - agg['se'], 
                           agg['mean'] + agg['se'],
                           color=colors[probe_kind], alpha=0.2)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Projection Value' if idx == 0 else '')
        ax.set_title(f'{condition.replace("_", " ").title()}')
        ax.legend(loc='best')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Figure 2: Truth vs Social Projection Trajectories (T={temperature})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / f'figure2_trajectories{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'figure2_trajectories{filename_suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {output_path}")


def plot_turn_layer_distribution(turn_df_t0: pd.DataFrame, turn_df_t1: pd.DataFrame,
                                  output_path: Path) -> None:
    """
    Figure 3: Turn Layer Distribution
    Histogram/violin comparing T=0 vs T=1 and correct vs incorrect.
    """
    if turn_df_t0.empty and turn_df_t1.empty:
        print("No turn layer data available for Figure 3")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: T=0 vs T=1
    ax = axes[0]
    data_plot = []
    
    if not turn_df_t0.empty:
        valid_t0 = turn_df_t0[turn_df_t0['turn_layer'].notna()]
        for _, row in valid_t0.iterrows():
            data_plot.append({'Temperature': 'T=0', 'Turn Layer': row['turn_layer']})
    
    if not turn_df_t1.empty:
        valid_t1 = turn_df_t1[turn_df_t1['turn_layer'].notna()]
        for _, row in valid_t1.iterrows():
            data_plot.append({'Temperature': 'T=1', 'Turn Layer': row['turn_layer']})
    
    if data_plot:
        plot_df = pd.DataFrame(data_plot)
        sns.violinplot(data=plot_df, x='Temperature', y='Turn Layer', ax=ax, palette=['steelblue', 'coral'])
        ax.set_title('Turn Layer by Temperature')
    
    # Panel B: Correct vs Incorrect
    ax = axes[1]
    data_plot = []
    
    for df, temp in [(turn_df_t0, 'T=0'), (turn_df_t1, 'T=1')]:
        if df.empty:
            continue
        valid = df[df['turn_layer'].notna()]
        for _, row in valid.iterrows():
            correctness = 'Correct' if row['is_correct'] else 'Incorrect'
            data_plot.append({'Correctness': correctness, 'Turn Layer': row['turn_layer']})
    
    if data_plot:
        plot_df = pd.DataFrame(data_plot)
        sns.violinplot(data=plot_df, x='Correctness', y='Turn Layer', ax=ax, 
                       palette=['forestgreen', 'tomato'])
        ax.set_title('Turn Layer by Response Correctness')
    
    plt.suptitle('Figure 3: Turn Layer Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_turn_layer.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_turn_layer.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_path}")


def plot_temperature_effect_scatter(behavioral_t0: pd.DataFrame, behavioral_t1: pd.DataFrame,
                                     output_path: Path) -> None:
    """
    Figure 4: Temperature Effect Visualization
    Scatter plot of T=0 vs T=1 conformity rate per item.
    """
    # Calculate per-item conformity rates
    def get_item_rates(df: pd.DataFrame) -> Dict[str, float]:
        return df.groupby('question').agg(
            conformity_rate=('is_correct', lambda x: 1 - x.mean())
        )['conformity_rate'].to_dict()
    
    rates_t0 = get_item_rates(behavioral_t0)
    rates_t1 = get_item_rates(behavioral_t1)
    
    # Align items
    common_items = set(rates_t0.keys()) & set(rates_t1.keys())
    
    if not common_items:
        print("No common items found for Figure 4")
        return
    
    # Get condition for each item (use T=0 as reference)
    item_conditions = behavioral_t0.groupby('question')['condition_name'].first().to_dict()
    
    x = [rates_t0[item] for item in common_items]
    y = [rates_t1[item] for item in common_items]
    conditions = [item_conditions.get(item, 'unknown') for item in common_items]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    condition_colors = {'control': 'forestgreen', 'asch_history_5': 'crimson', 
                        'authoritative_bias': 'darkorange', 'unknown': 'gray'}
    
    for cond in set(conditions):
        mask = [c == cond for c in conditions]
        x_cond = [x[i] for i in range(len(x)) if mask[i]]
        y_cond = [y[i] for i in range(len(y)) if mask[i]]
        ax.scatter(x_cond, y_cond, label=cond.replace('_', ' ').title(), 
                  alpha=0.7, s=80, c=condition_colors.get(cond, 'gray'))
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    
    ax.set_xlabel('Conformity Rate (T=0)')
    ax.set_ylabel('Conformity Rate (T=1)')
    ax.set_title('Figure 4: Temperature Effect on Item-Level Conformity')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_temperature_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_temperature_scatter.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {output_path}")


def plot_intervention_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 5: Intervention Results
    Before/after accuracy comparison and flip rates.
    """
    if results_df.empty:
        print("No intervention data available for Figure 5")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Flip rate by intervention
    ax = axes[0]
    flip_rates = results_df.groupby('intervention_name').agg(
        flip_rate=('flipped_to_truth', 'mean'),
        n_trials=('trial_id', 'count'),
    ).reset_index()
    
    ax.bar(flip_rates['intervention_name'], flip_rates['flip_rate'], 
           color='steelblue', alpha=0.8)
    ax.set_ylabel('Flip Rate (Conformity → Truth)')
    ax.set_xlabel('Intervention')
    ax.set_title('Flip Rate by Intervention Type')
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel B: Before/after accuracy
    ax = axes[1]
    before = results_df['original_is_correct'].mean()
    after = results_df['intervened_is_correct'].mean()
    
    x = ['Before', 'After']
    heights = [before, after]
    colors = ['tomato', 'forestgreen']
    
    ax.bar(x, heights, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Before/After Intervention')
    ax.set_ylim(0, 1)
    
    plt.suptitle('Figure 5: Intervention Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'figure5_interventions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_interventions.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {output_path}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis() -> Dict[str, Any]:
    """
    Run complete analysis pipeline and return results dict.
    """
    print("=" * 80)
    print("TEMPERATURE EFFECT ANALYSIS: T=0 vs T=1")
    print("=" * 80)
    
    results = {}
    
    # ==========================================================================
    # PHASE 1: DATA VALIDATION AND SUMMARY STATISTICS
    # ==========================================================================
    print("\n### Phase 1: Data Validation and Summary Statistics ###")
    
    # Connect to databases
    conn_t0 = connect_db(RUN_T0)
    conn_t1 = connect_db(RUN_T1)
    
    # Load configurations
    config_t0 = load_run_config(conn_t0)
    config_t1 = load_run_config(conn_t1)
    
    print(f"\nRun T=0: {config_t0.run_id}")
    print(f"  Temperature: {config_t0.temperature}")
    print(f"  Suite: {config_t0.suite_name}")
    print(f"  Seed: {config_t0.seed}")
    print(f"  Models: {len(config_t0.models)}")
    
    print(f"\nRun T=1: {config_t1.run_id}")
    print(f"  Temperature: {config_t1.temperature}")
    print(f"  Suite: {config_t1.suite_name}")
    print(f"  Seed: {config_t1.seed}")
    print(f"  Models: {len(config_t1.models)}")
    
    results['config_t0'] = config_t0
    results['config_t1'] = config_t1
    
    # ==========================================================================
    # PHASE 2: BEHAVIORAL ANALYSIS
    # ==========================================================================
    print("\n### Phase 2: Behavioral Analysis ###")
    
    # Get behavioral data
    behavioral_t0 = get_behavioral_data(conn_t0, config_t0.run_id)
    behavioral_t1 = get_behavioral_data(conn_t1, config_t1.run_id)
    
    print(f"\nT=0 trials: {len(behavioral_t0)}")
    print(f"T=1 trials: {len(behavioral_t1)}")
    
    # Calculate conformity rates
    rates_t0 = calculate_conformity_rates(behavioral_t0)
    rates_t1 = calculate_conformity_rates(behavioral_t1)
    
    print("\nConformity Rates T=0:")
    print(rates_t0[['condition_name', 'variant', 'accuracy', 'conformity_rate', 'n_trials']])
    
    print("\nConformity Rates T=1:")
    print(rates_t1[['condition_name', 'variant', 'accuracy', 'conformity_rate', 'n_trials']])
    
    # Calculate susceptibility
    susceptibility_t0 = calculate_conformity_susceptibility(rates_t0)
    susceptibility_t1 = calculate_conformity_susceptibility(rates_t1)
    
    print("\nConformity Susceptibility T=0:")
    print(susceptibility_t0)
    
    print("\nConformity Susceptibility T=1:")
    print(susceptibility_t1)
    
    # Statistical comparison
    comparison = compare_conformity_rates(rates_t0, rates_t1)
    print("\nTemperature Effect (T=1 - T=0):")
    print(comparison[['condition', 'variant', 'rate_t0', 'rate_t1', 'difference', 'p_value', 'effect_size_h']])
    
    results['behavioral_t0'] = behavioral_t0
    results['behavioral_t1'] = behavioral_t1
    results['rates_t0'] = rates_t0
    results['rates_t1'] = rates_t1
    results['susceptibility_t0'] = susceptibility_t0
    results['susceptibility_t1'] = susceptibility_t1
    results['comparison'] = comparison
    
    # ==========================================================================
    # PHASE 3: MECHANISTIC ANALYSIS
    # ==========================================================================
    print("\n### Phase 3: Mechanistic Analysis ###")
    
    # Get probe data
    probes_t0, projections_t0 = get_probe_data(conn_t0, config_t0.run_id)
    probes_t1, projections_t1 = get_probe_data(conn_t1, config_t1.run_id)
    
    print(f"\nT=0 probes: {len(probes_t0)}, projections: {len(projections_t0)}")
    print(f"T=1 probes: {len(probes_t1)}, projections: {len(projections_t1)}")
    
    # Analyze trajectories
    trajectories_t0 = analyze_probe_trajectories(projections_t0, behavioral_t0)
    trajectories_t1 = analyze_probe_trajectories(projections_t1, behavioral_t1)
    
    # Identify turn layers
    turn_layers_t0 = identify_turn_layer(projections_t0, behavioral_t0)
    turn_layers_t1 = identify_turn_layer(projections_t1, behavioral_t1)
    
    if not turn_layers_t0.empty:
        print(f"\nT=0 Turn Layer stats:")
        print(f"  Mean: {turn_layers_t0['turn_layer'].mean():.2f}")
        print(f"  Std: {turn_layers_t0['turn_layer'].std():.2f}")
    
    if not turn_layers_t1.empty:
        print(f"\nT=1 Turn Layer stats:")
        print(f"  Mean: {turn_layers_t1['turn_layer'].mean():.2f}")
        print(f"  Std: {turn_layers_t1['turn_layer'].std():.2f}")
    
    results['probes_t0'] = probes_t0
    results['probes_t1'] = probes_t1
    results['projections_t0'] = projections_t0
    results['projections_t1'] = projections_t1
    results['trajectories_t0'] = trajectories_t0
    results['trajectories_t1'] = trajectories_t1
    results['turn_layers_t0'] = turn_layers_t0
    results['turn_layers_t1'] = turn_layers_t1
    
    # ==========================================================================
    # PHASE 4: MODEL COMPARISON
    # ==========================================================================
    print("\n### Phase 4: Model Comparison ###")
    
    # Compare Instruct vs Think
    for temp, rates in [('T=0', rates_t0), ('T=1', rates_t1)]:
        print(f"\n{temp} Model Comparison (Asch condition):")
        asch_rates = rates[rates['condition_name'] == 'asch_history_5'][['variant', 'conformity_rate', 'n_trials']]
        print(asch_rates.to_string(index=False))
    
    # ==========================================================================
    # PHASE 5: INTERVENTION ANALYSIS
    # ==========================================================================
    print("\n### Phase 5: Intervention Analysis ###")
    
    interventions_t0, intervention_results_t0 = get_intervention_data(conn_t0, config_t0.run_id)
    interventions_t1, intervention_results_t1 = get_intervention_data(conn_t1, config_t1.run_id)
    
    print(f"\nT=0 interventions: {len(interventions_t0)}, results: {len(intervention_results_t0)}")
    print(f"T=1 interventions: {len(interventions_t1)}, results: {len(intervention_results_t1)}")
    
    results['interventions_t0'] = interventions_t0
    results['interventions_t1'] = interventions_t1
    results['intervention_results_t0'] = intervention_results_t0
    results['intervention_results_t1'] = intervention_results_t1
    
    # Close connections
    conn_t0.close()
    conn_t1.close()
    
    # ==========================================================================
    # GENERATE VISUALIZATIONS
    # ==========================================================================
    print("\n### Generating Visualizations ###")
    
    plot_conformity_rates_by_condition(rates_t0, rates_t1, OUTPUT_DIR)
    plot_probe_trajectories(trajectories_t0, 0.0, OUTPUT_DIR, "_t0")
    plot_probe_trajectories(trajectories_t1, 1.0, OUTPUT_DIR, "_t1")
    plot_turn_layer_distribution(turn_layers_t0, turn_layers_t1, OUTPUT_DIR)
    plot_temperature_effect_scatter(behavioral_t0, behavioral_t1, OUTPUT_DIR)
    
    if not intervention_results_t0.empty:
        plot_intervention_results(intervention_results_t0, OUTPUT_DIR)
    elif not intervention_results_t1.empty:
        plot_intervention_results(intervention_results_t1, OUTPUT_DIR)
    
    # ==========================================================================
    # SAVE INTERMEDIATE DATA
    # ==========================================================================
    print("\n### Saving Intermediate Data ###")
    
    rates_t0.to_csv(OUTPUT_DIR / 'rates_t0.csv', index=False)
    rates_t1.to_csv(OUTPUT_DIR / 'rates_t1.csv', index=False)
    comparison.to_csv(OUTPUT_DIR / 'temperature_comparison.csv', index=False)
    behavioral_t0.to_csv(OUTPUT_DIR / 'behavioral_t0.csv', index=False)
    behavioral_t1.to_csv(OUTPUT_DIR / 'behavioral_t1.csv', index=False)
    
    print(f"\nData saved to {OUTPUT_DIR}")
    
    return results


if __name__ == "__main__":
    results = run_analysis()
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
