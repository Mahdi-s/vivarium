"""
Jupyter Notebook for Olmo Conformity Experiment Analytics.

This script can be converted to a .ipynb notebook or run directly.
It provides an interactive data science exploration interface.
"""

# %%
# Setup and Imports
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from aam.persistence import TraceDb, TraceDbConfig
from aam.analytics import (
    compute_behavioral_metrics,
    generate_behavioral_graphs,
    export_behavioral_logs,
    compute_probe_metrics,
    generate_probe_graphs,
    export_probe_logs,
    compute_intervention_metrics,
    generate_intervention_graphs,
    export_intervention_logs,
    compute_judgeval_metrics,
    generate_judgeval_graphs,
    export_judgeval_logs,
    compute_think_metrics,
    generate_think_graphs,
    export_think_logs,
    compute_activation_stats,
    export_activation_logs,
)
from aam.analytics.utils import (
    load_simulation_db,
    get_run_metadata,
    check_missing_prerequisites,
    save_missing_prerequisites_log,
    ensure_logs_dir,
)

# %%
# Configuration
# Update this path to point to your run directory
import os
RUN_DIR = os.path.abspath("./runs/20251216_204108_4df7567b-c85c-4f9e-8ec4-4ec9f32a7ef0")

# Verify it exists
db_path = os.path.join(RUN_DIR, "simulation.db")
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database not found: {db_path}")
print(f"Using database: {db_path}")
# Or use: RUN_DIR = input("Enter run directory path: ")

# %%
# Section 1: Setup - Load Database and Display Run Info
print("=" * 80)
print("Section 1: Setup - Loading Database")
print("=" * 80)

db = load_simulation_db(RUN_DIR)
run_id = Path(RUN_DIR).name.split("_")[-1] if "_" in Path(RUN_DIR).name else Path(RUN_DIR).name

# Get run metadata
try:
    metadata = get_run_metadata(db, run_id)
    print(f"\nRun ID: {metadata['run_id']}")
    print(f"Seed: {metadata['seed']}")
    print(f"Created At: {metadata['created_at']}")
    print(f"\nConfig:")
    print(json.dumps(metadata['config'], indent=2))
except Exception as e:
    print(f"Warning: Could not load run metadata: {e}")
    # Try to infer run_id from directory
    run_id = os.path.basename(RUN_DIR)

# %%
# Section 2: Data Overview - Tables, Schemas, Row Counts
print("\n" + "=" * 80)
print("Section 2: Data Overview")
print("=" * 80)

# List all tables
tables_query = """
SELECT name FROM sqlite_master 
WHERE type='table' AND name LIKE 'conformity_%' OR name IN ('runs', 'trace', 'messages', 'activation_metadata')
ORDER BY name;
"""
tables = pd.read_sql_query(tables_query, db.conn)
print(f"\nAvailable tables ({len(tables)}):")
for table in tables['name']:
    count = db.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]
    print(f"  - {table}: {count} rows")

# %%
# Section 3: Behavioral Analysis
print("\n" + "=" * 80)
print("Section 3: Behavioral Analysis")
print("=" * 80)

behavioral_metrics = compute_behavioral_metrics(db, run_id, RUN_DIR)
print("\nBehavioral Metrics Summary:")
print(json.dumps(behavioral_metrics.get("statistics", {}), indent=2))

if "metrics" in behavioral_metrics and behavioral_metrics["metrics"]:
    print("\nKey Metrics:")
    if "accuracy_by_condition" in behavioral_metrics["metrics"]:
        df_acc = pd.DataFrame(behavioral_metrics["metrics"]["accuracy_by_condition"])
        print("\nAccuracy by Condition:")
        print(df_acc.to_string(index=False))
    
    if "sycophancy_rate" in behavioral_metrics["metrics"]:
        df_syc = pd.DataFrame(behavioral_metrics["metrics"]["sycophancy_rate"])
        print("\nSycophancy Rate (Truth-Override):")
        print(df_syc.to_string(index=False))

# Generate behavioral graphs
behavioral_figures = generate_behavioral_graphs(db, run_id, RUN_DIR, behavioral_metrics)
print(f"\nGenerated {len(behavioral_figures)} behavioral figures:")
for name, path in behavioral_figures.items():
    print(f"  - {name}: {path}")

# Export logs
behavioral_logs = export_behavioral_logs(db, run_id, RUN_DIR, behavioral_metrics)
print(f"\nExported behavioral logs:")
for name, path in behavioral_logs.items():
    print(f"  - {name}: {path}")

# %%
# Section 4: Judge Eval Analysis (if available)
print("\n" + "=" * 80)
print("Section 4: Judge Eval Analysis")
print("=" * 80)

judgeval_metrics = compute_judgeval_metrics(db, run_id, RUN_DIR)

if judgeval_metrics.get("statistics", {}).get("n_scores", 0) > 0:
    print(f"\nJudge Eval scores found: {judgeval_metrics['statistics']['n_scores']}")
    
    if "mean_scores_by_variant_condition" in judgeval_metrics["metrics"]:
        df_judge = pd.DataFrame(judgeval_metrics["metrics"]["mean_scores_by_variant_condition"])
        print("\nMean Judge Eval Scores by Variant/Condition:")
        print(df_judge.to_string(index=False))
    
    if "correlation_with_correctness" in judgeval_metrics["metrics"]:
        print("\nCorrelation with Correctness:")
        print(json.dumps(judgeval_metrics["metrics"]["correlation_with_correctness"], indent=2))
    
    # Generate graphs
    judgeval_figures = generate_judgeval_graphs(db, run_id, RUN_DIR, judgeval_metrics)
    print(f"\nGenerated {len(judgeval_figures)} Judge Eval figures:")
    for name, path in judgeval_figures.items():
        print(f"  - {name}: {path}")
    
    # Export logs
    judgeval_logs = export_judgeval_logs(db, run_id, RUN_DIR, judgeval_metrics)
    print(f"\nExported Judge Eval logs:")
    for name, path in judgeval_logs.items():
        print(f"  - {name}: {path}")
else:
    print("\nNo Judge Eval scores found in this run.")

# %%
# Section 5: Probe Analysis (if probes trained)
print("\n" + "=" * 80)
print("Section 5: Probe Analysis")
print("=" * 80)

probe_metrics = compute_probe_metrics(db, run_id, RUN_DIR)

if probe_metrics.get("statistics", {}).get("n_probes", 0) > 0:
    print(f"\nProbes found: {probe_metrics['statistics']['n_probes']}")
    print(f"Projections: {probe_metrics['statistics'].get('n_projections', 0)}")
    
    if "truth_vector_projection" in probe_metrics["metrics"]:
        df_tvp = pd.DataFrame(probe_metrics["metrics"]["truth_vector_projection"])
        print("\nTruth Vector Projection (sample):")
        print(df_tvp.head(10).to_string(index=False))
    
    # Generate graphs
    probe_figures = generate_probe_graphs(db, run_id, RUN_DIR, probe_metrics)
    print(f"\nGenerated {len(probe_figures)} probe figures:")
    for name, path in probe_figures.items():
        print(f"  - {name}: {path}")
    
    # Export logs
    probe_logs = export_probe_logs(db, run_id, RUN_DIR, probe_metrics)
    print(f"\nExported probe logs:")
    for name, path in probe_logs.items():
        print(f"  - {name}: {path}")
else:
    print("\nNo probes found in this run.")

# %%
# Section 6: Intervention Analysis (if interventions run)
print("\n" + "=" * 80)
print("Section 6: Intervention Analysis")
print("=" * 80)

intervention_metrics = compute_intervention_metrics(db, run_id, RUN_DIR)

if intervention_metrics.get("statistics", {}).get("n_interventions", 0) > 0:
    print(f"\nInterventions found: {intervention_metrics['statistics']['n_interventions']}")
    
    if "flip_to_truth_rate" in intervention_metrics["metrics"]:
        df_flip = pd.DataFrame(intervention_metrics["metrics"]["flip_to_truth_rate"])
        print("\nFlip-to-Truth Rate:")
        print(df_flip.to_string(index=False))
    
    # Generate graphs
    intervention_figures = generate_intervention_graphs(db, run_id, RUN_DIR, intervention_metrics)
    print(f"\nGenerated {len(intervention_figures)} intervention figures:")
    for name, path in intervention_figures.items():
        print(f"  - {name}: {path}")
    
    # Export logs
    intervention_logs = export_intervention_logs(db, run_id, RUN_DIR, intervention_metrics)
    print(f"\nExported intervention logs:")
    for name, path in intervention_logs.items():
        print(f"  - {name}: {path}")
else:
    print("\nNo interventions found in this run.")

# %%
# Section 7: Activation Inspection
print("\n" + "=" * 80)
print("Section 7: Activation Inspection")
print("=" * 80)

activation_metrics = compute_activation_stats(db, run_id, RUN_DIR)

if activation_metrics.get("statistics", {}).get("total_records", 0) > 0:
    print(f"\nActivation records: {activation_metrics['statistics']['total_records']}")
    print(f"Unique files: {activation_metrics['statistics'].get('unique_files', 0)}")
    print(f"Layers captured: {activation_metrics['statistics'].get('layers_captured', [])}")
    print(f"Components: {activation_metrics['statistics'].get('components', [])}")
    
    # Generate graphs
    activation_figures = generate_activation_graphs(db, run_id, RUN_DIR)
    print(f"\nGenerated {len(activation_figures)} activation figures:")
    for name, path in activation_figures.items():
        print(f"  - {name}: {path}")
    
    # Export logs
    activation_logs = export_activation_logs(db, run_id, RUN_DIR, activation_metrics)
    print(f"\nExported activation logs:")
    for name, path in activation_logs.items():
        print(f"  - {name}: {path}")
else:
    print("\nNo activation metadata found in this run.")

# %%
# Section 8: Missing Prerequisites Check
print("\n" + "=" * 80)
print("Section 8: Missing Prerequisites Check")
print("=" * 80)

missing = check_missing_prerequisites(db, run_id)
print("\nPrerequisites Status:")
for name, exists in missing.items():
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {'Present' if exists else 'Missing'}")

# Save missing prerequisites log
paths = ensure_logs_dir(RUN_DIR)
missing_log_path = os.path.join(paths["logs_dir"], "missing.json")
save_missing_prerequisites_log(missing, missing_log_path)
print(f"\nMissing prerequisites log saved to: {missing_log_path}")

# %%
# Section 9: Custom Queries
print("\n" + "=" * 80)
print("Section 9: Custom Queries")
print("=" * 80)
print("\nYou can now run custom SQL queries on the database.")
print("Example:")
print("  df = pd.read_sql_query('SELECT * FROM conformity_trials WHERE run_id = ?', db.conn, params=(run_id,))")
print("  print(df.head())")

# Example custom query
example_query = """
SELECT 
    t.variant,
    c.name AS condition_name,
    COUNT(*) AS n_trials,
    AVG(o.is_correct) AS mean_accuracy
FROM conformity_trials t
JOIN conformity_conditions c ON c.condition_id = t.condition_id
JOIN conformity_outputs o ON o.trial_id = t.trial_id
WHERE t.run_id = ? AND o.is_correct IS NOT NULL
GROUP BY t.variant, c.name
ORDER BY t.variant, c.name
"""
df_example = pd.read_sql_query(example_query, db.conn, params=(run_id,))
print("\nExample Query Result:")
print(df_example.to_string(index=False))

print("\n" + "=" * 80)
print("Analytics Complete!")
print("=" * 80)
