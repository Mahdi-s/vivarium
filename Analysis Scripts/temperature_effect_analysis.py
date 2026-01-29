"""
Temperature Effect Analysis for Olmo Conformity Experiments

Analyzes three runs comparing T=0 (deterministic), T=0.5 (moderate), and T=1 (stochastic) 
decoding to understand how temperature affects social conformity in LLMs.

Run IDs:
- T=0.0: 73b34738-b76e-4c55-8653-74b497b1989b
- T=0.5: 4e6cd5a7-af59-4fe2-ae8d-c9bcc2f57c00
- T=1.0: f1c7ed74-2561-4c52-9279-3d3269fcb7f3
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

# Configuration: repo-relative paths so script works regardless of repo directory name
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
RUNS_DIR = _REPO_ROOT / "runs"
OUTPUT_DIR = _SCRIPT_DIR / "temperature_analysis_output"

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
TEMP_COLORS = {0.0: "steelblue", 0.5: "mediumseagreen", 1.0: "coral"}

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
    """Extract behavioral data for conformity trials."""
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
    """Extract probe projections if available."""
    probes_query = """
    SELECT probe_id, probe_kind, component, model_id, layers_json, metrics_json, created_at
    FROM conformity_probes WHERE run_id = ?;
    """
    try:
        probes_df = pd.read_sql_query(probes_query, conn, params=[run_id])
    except Exception as e:
        print(f"  Warning: Could not load probes: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    if probes_df.empty:
        return probes_df, pd.DataFrame()
    
    projections_query = """
    SELECT pp.trial_id, pp.probe_id, pp.layer_index, pp.token_index, pp.value_float,
           p.probe_kind, p.component
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


def calculate_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate error rates by condition and model."""
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
    agg['error_rate'] = 1 - agg['accuracy']
    agg['refusal_rate'] = agg['n_refusals'] / agg['n_trials']
    agg['empty_rate'] = agg['n_empty'] / agg['n_trials']
    
    return agg


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for proportions."""
    if total == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(max(0.001, min(0.999, p1))))
    phi2 = 2 * np.arcsin(np.sqrt(max(0.001, min(0.999, p2))))
    return phi1 - phi2


def compare_rates_pairwise(rates_dict: Dict[float, pd.DataFrame]) -> pd.DataFrame:
    """Statistical comparison of error rates across all temperature pairs."""
    results = []
    
    temp_pairs = [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]
    
    for t1, t2 in temp_pairs:
        df1 = rates_dict[t1]
        df2 = rates_dict[t2]
        
        for condition in BEHAVIORAL_CONDITIONS:
            for variant in df1['variant'].unique():
                row1 = df1[(df1['condition_name'] == condition) & (df1['variant'] == variant)]
                row2 = df2[(df2['condition_name'] == condition) & (df2['variant'] == variant)]
                
                if row1.empty or row2.empty:
                    continue
                
                n1 = int(row1['n_trials'].iloc[0])
                correct1 = int(row1['n_correct'].iloc[0])
                n2 = int(row2['n_trials'].iloc[0])
                correct2 = int(row2['n_correct'].iloc[0])
                
                rate1 = 1 - (correct1 / n1) if n1 > 0 else 0
                rate2 = 1 - (correct2 / n2) if n2 > 0 else 0
                
                ci1 = wilson_ci(n1 - correct1, n1)
                ci2 = wilson_ci(n2 - correct2, n2)
                
                # Chi-square or Fisher's exact test
                table = [[correct1, n1 - correct1], [correct2, n2 - correct2]]
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(table)
                    if min(expected.flatten()) < 5:
                        _, p_value = stats.fisher_exact(table)
                        chi2 = np.nan
                except Exception:
                    chi2, p_value = np.nan, np.nan
                
                effect = cohens_h(rate2, rate1)
                
                results.append({
                    'comparison': f"T={t1} vs T={t2}",
                    't_low': t1,
                    't_high': t2,
                    'condition': condition,
                    'variant': variant,
                    'rate_low': rate1,
                    'rate_high': rate2,
                    'difference': rate2 - rate1,
                    'ci_low_lower': ci1[0],
                    'ci_low_upper': ci1[1],
                    'ci_high_lower': ci2[0],
                    'ci_high_upper': ci2[1],
                    'chi2': chi2,
                    'p_value': p_value,
                    'effect_size_h': effect,
                    'n_low': n1,
                    'n_high': n2,
                })
    
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_error_rates_by_condition_3temp(rates_dict: Dict[float, pd.DataFrame], 
                                         output_path: Path) -> None:
    """
    Figure 1: Error Rates by Condition and Temperature (3 temperatures)
    Grouped bar chart with error bars showing 95% CI.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharey=True)
    axes = axes.flatten()
    
    variants = sorted(rates_dict[0.0]['variant'].unique())
    
    for idx, variant in enumerate(variants):
        ax = axes[idx]
        
        conditions = list(BEHAVIORAL_CONDITIONS)
        x = np.arange(len(conditions))
        width = 0.25
        
        for i, temp in enumerate(TEMPERATURES):
            data = rates_dict[temp][rates_dict[temp]['variant'] == variant].set_index('condition_name')
            
            rates = [data.loc[c, 'error_rate'] if c in data.index else 0 for c in conditions]
            
            # Calculate CIs
            cis = [wilson_ci(int(data.loc[c, 'n_incorrect']), int(data.loc[c, 'n_trials'])) 
                   if c in data.index else (0, 0) for c in conditions]
            err = [(ci[1] - ci[0]) / 2 for ci in cis]
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, rates, width, label=TEMP_LABELS[temp],
                         yerr=err, capsize=2, color=TEMP_COLORS[temp], alpha=0.85)
        
        ax.set_ylabel('Error Rate' if idx % 3 == 0 else '')
        ax.set_title(f'{variant}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'Asch (5)', 'Authority'], rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='_nolegend_')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    for idx in range(len(variants), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Figure 1: Error Rates by Condition and Temperature\n(T=0: Deterministic, T=0.5: Moderate, T=1: Stochastic)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_error_rates_3temp.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure1_error_rates_3temp.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def plot_temperature_curve(rates_dict: Dict[float, pd.DataFrame], output_path: Path) -> None:
    """
    Figure 2: Temperature-Error Rate Curves
    Line plot showing how error rate changes with temperature for each model/condition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    condition_titles = {
        'control': 'Control (No Pressure)',
        'asch_history_5': 'Asch (5 Confederates)',
        'authoritative_bias': 'Authoritative Bias'
    }
    
    variants = sorted(rates_dict[0.0]['variant'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    variant_colors = dict(zip(variants, colors))
    
    temps = np.array(TEMPERATURES)
    
    for idx, condition in enumerate(BEHAVIORAL_CONDITIONS):
        ax = axes[idx]
        
        for variant in variants:
            rates = []
            cis_low = []
            cis_high = []
            
            for temp in TEMPERATURES:
                data = rates_dict[temp]
                row = data[(data['condition_name'] == condition) & (data['variant'] == variant)]
                if not row.empty:
                    rate = row['error_rate'].iloc[0]
                    n = int(row['n_trials'].iloc[0])
                    n_err = int(row['n_incorrect'].iloc[0])
                    ci = wilson_ci(n_err, n)
                    rates.append(rate)
                    cis_low.append(ci[0])
                    cis_high.append(ci[1])
                else:
                    rates.append(np.nan)
                    cis_low.append(np.nan)
                    cis_high.append(np.nan)
            
            rates = np.array(rates)
            cis_low = np.array(cis_low)
            cis_high = np.array(cis_high)
            
            ax.plot(temps, rates, 'o-', label=variant, color=variant_colors[variant], 
                   linewidth=2, markersize=8)
            ax.fill_between(temps, cis_low, cis_high, alpha=0.15, color=variant_colors[variant])
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Error Rate' if idx == 0 else '')
        ax.set_title(condition_titles.get(condition, condition))
        ax.set_xticks(TEMPERATURES)
        ax.set_xticklabels(['0.0', '0.5', '1.0'])
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        if idx == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
    plt.suptitle('Figure 2: Error Rate vs Temperature by Condition', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_temperature_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_temperature_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {output_path}")


def plot_social_pressure_effect(rates_dict: Dict[float, pd.DataFrame], output_path: Path) -> None:
    """
    Figure 3: Social Pressure Effect by Temperature
    Shows the *additional* error rate from social pressure (Asch - Control).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    variants = sorted(rates_dict[0.0]['variant'].unique())
    x = np.arange(len(variants))
    width = 0.25
    
    for i, temp in enumerate(TEMPERATURES):
        data = rates_dict[temp]
        
        effects = []
        for variant in variants:
            control_row = data[(data['condition_name'] == 'control') & (data['variant'] == variant)]
            asch_row = data[(data['condition_name'] == 'asch_history_5') & (data['variant'] == variant)]
            
            if not control_row.empty and not asch_row.empty:
                control_rate = control_row['error_rate'].iloc[0]
                asch_rate = asch_row['error_rate'].iloc[0]
                effect = asch_rate - control_rate
            else:
                effect = 0
            effects.append(effect)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, effects, width, label=TEMP_LABELS[temp],
                     color=TEMP_COLORS[temp], alpha=0.85)
    
    ax.set_ylabel('Social Pressure Effect\n(Asch Error Rate − Control Error Rate)')
    ax.set_xlabel('Model Variant')
    ax.set_title('Figure 3: Social Pressure Effect by Temperature\n(Positive = Pressure Increases Errors)')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.2, 0.25)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_social_pressure_effect.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_social_pressure_effect.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_path}")


def plot_refusal_rates(rates_dict: Dict[float, pd.DataFrame], output_path: Path) -> None:
    """
    Figure 4: Refusal Rates by Condition and Temperature
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    condition_titles = {
        'control': 'Control',
        'asch_history_5': 'Asch (5)',
        'authoritative_bias': 'Authority'
    }
    
    variants = sorted(rates_dict[0.0]['variant'].unique())
    x = np.arange(len(variants))
    width = 0.25
    
    for idx, condition in enumerate(BEHAVIORAL_CONDITIONS):
        ax = axes[idx]
        
        for i, temp in enumerate(TEMPERATURES):
            data = rates_dict[temp]
            cond_data = data[data['condition_name'] == condition].set_index('variant')
            
            rates = [cond_data.loc[v, 'refusal_rate'] if v in cond_data.index else 0 
                    for v in variants]
            
            offset = (i - 1) * width
            ax.bar(x + offset, rates, width, label=TEMP_LABELS[temp] if idx == 0 else '_nolegend_',
                  color=TEMP_COLORS[temp], alpha=0.85)
        
        ax.set_ylabel('Refusal Rate' if idx == 0 else '')
        ax.set_title(condition_titles.get(condition, condition))
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha='right')
        ax.set_ylim(0, 0.5)
        
        if idx == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle('Figure 4: Refusal Rates by Condition and Temperature', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_refusal_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_refusal_rates.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {output_path}")


def plot_heatmap_error_rates(rates_dict: Dict[float, pd.DataFrame], output_path: Path) -> None:
    """
    Figure 5: Heatmap of Error Rates (Model × Condition × Temperature)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    condition_titles = {
        'control': 'Control',
        'asch_history_5': 'Asch (5)',
        'authoritative_bias': 'Authority'
    }
    
    variants = sorted(rates_dict[0.0]['variant'].unique())
    
    for idx, condition in enumerate(BEHAVIORAL_CONDITIONS):
        ax = axes[idx]
        
        # Build matrix: rows = variants, cols = temperatures
        matrix = np.zeros((len(variants), len(TEMPERATURES)))
        
        for i, variant in enumerate(variants):
            for j, temp in enumerate(TEMPERATURES):
                data = rates_dict[temp]
                row = data[(data['condition_name'] == condition) & (data['variant'] == variant)]
                if not row.empty:
                    matrix[i, j] = row['error_rate'].iloc[0]
        
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.3, vmax=1.0)
        
        ax.set_xticks(np.arange(len(TEMPERATURES)))
        ax.set_xticklabels([TEMP_LABELS[t] for t in TEMPERATURES])
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels(variants if idx == 0 else [])
        ax.set_title(condition_titles.get(condition, condition))
        ax.set_xlabel('Temperature')
        if idx == 0:
            ax.set_ylabel('Model Variant')
        
        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(TEMPERATURES)):
                text = ax.text(j, i, f'{matrix[i, j]:.1%}',
                              ha='center', va='center', fontsize=9,
                              color='white' if matrix[i, j] > 0.65 else 'black')
    
    plt.colorbar(im, ax=axes, label='Error Rate', shrink=0.8)
    plt.suptitle('Figure 5: Error Rate Heatmap (Model × Condition × Temperature)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'figure5_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {output_path}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis() -> Dict[str, Any]:
    """Run complete analysis pipeline for all three temperatures."""
    print("=" * 80)
    print("TEMPERATURE EFFECT ANALYSIS: T=0 vs T=0.5 vs T=1")
    print("=" * 80)
    
    results = {}
    configs = {}
    behavioral_data = {}
    rates_dict = {}
    
    # ==========================================================================
    # PHASE 1: LOAD ALL RUNS
    # ==========================================================================
    print("\n### Phase 1: Data Validation and Summary Statistics ###")
    
    for temp in TEMPERATURES:
        run_info = RUNS[temp]
        conn = connect_db(run_info["dir"])
        config = load_run_config(conn)
        
        print(f"\nRun T={temp}: {config.run_id}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Suite: {config.suite_name}")
        print(f"  Seed: {config.seed}")
        print(f"  Models: {len(config.models)}")
        
        configs[temp] = config
        
        # Get behavioral data
        behavioral = get_behavioral_data(conn, config.run_id)
        behavioral_data[temp] = behavioral
        print(f"  Trials: {len(behavioral)}")
        
        # Calculate rates
        rates = calculate_rates(behavioral)
        rates_dict[temp] = rates
        
        conn.close()
    
    results['configs'] = configs
    results['behavioral_data'] = behavioral_data
    results['rates'] = rates_dict
    
    # ==========================================================================
    # PHASE 2: ERROR RATE ANALYSIS
    # ==========================================================================
    print("\n### Phase 2: Error Rate Analysis ###")
    
    for temp in TEMPERATURES:
        print(f"\nError Rates T={temp}:")
        rates = rates_dict[temp]
        print(rates[['condition_name', 'variant', 'accuracy', 'error_rate', 'n_trials', 'n_refusals']].to_string(index=False))
    
    # ==========================================================================
    # PHASE 3: STATISTICAL COMPARISONS
    # ==========================================================================
    print("\n### Phase 3: Statistical Comparisons ###")
    
    comparison = compare_rates_pairwise(rates_dict)
    results['comparison'] = comparison
    
    # Show key comparisons
    print("\nKey Temperature Effects (Asch condition, |Δ| > 0.05):")
    asch_comp = comparison[(comparison['condition'] == 'asch_history_5') & 
                           (abs(comparison['difference']) > 0.05)]
    print(asch_comp[['comparison', 'variant', 'rate_low', 'rate_high', 
                     'difference', 'p_value', 'effect_size_h']].to_string(index=False))
    
    # ==========================================================================
    # PHASE 4: GENERATE VISUALIZATIONS
    # ==========================================================================
    print("\n### Phase 4: Generating Visualizations ###")
    
    plot_error_rates_by_condition_3temp(rates_dict, OUTPUT_DIR)
    plot_temperature_curve(rates_dict, OUTPUT_DIR)
    plot_social_pressure_effect(rates_dict, OUTPUT_DIR)
    plot_refusal_rates(rates_dict, OUTPUT_DIR)
    plot_heatmap_error_rates(rates_dict, OUTPUT_DIR)
    
    # ==========================================================================
    # PHASE 5: SAVE DATA
    # ==========================================================================
    print("\n### Phase 5: Saving Intermediate Data ###")
    
    for temp in TEMPERATURES:
        rates_dict[temp].to_csv(OUTPUT_DIR / f'rates_t{temp}.csv', index=False)
        behavioral_data[temp].to_csv(OUTPUT_DIR / f'behavioral_t{temp}.csv', index=False)
    
    comparison.to_csv(OUTPUT_DIR / 'temperature_comparison_all.csv', index=False)
    
    # Save combined rates for easy comparison
    combined = []
    for temp in TEMPERATURES:
        df = rates_dict[temp].copy()
        df['temperature'] = temp
        combined.append(df)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / 'rates_combined.csv', index=False)
    
    print(f"\nData saved to {OUTPUT_DIR}")
    
    results['combined_rates'] = combined_df
    
    return results


if __name__ == "__main__":
    results = run_analysis()
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
