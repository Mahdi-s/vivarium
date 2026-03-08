# Commands to Complete Remaining Trials Locally

## Overview

The simulation is orchestrated through `experiments/olmo_conformity/configs/run_expanded_experiments.py`, which:
1. Runs the vivarium experiment runner for each temperature
2. Executes judge labeling on outputs
3. Runs post-hoc analysis

## Quick Start: Re-run Everything for All Temperatures

```bash
cd /path/to/your/repo

# Full pipeline: experiments + judge + post-hoc analysis
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs

# Or with keep-awake flag (macOS):
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --no-sleep
```

## By Phase: If You Want Fine-Grained Control

### Phase 1: Run the Simulation Trials Only

This generates the raw outputs for all models × conditions × temperatures.

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase trials \
  --force-rerun  # Use this to re-run even if metadata says "completed"
```

**For specific temperatures only** (e.g., just T=0.8 and T=1.0):

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase trials \
  --temps 0.8,1.0 \
  --force-rerun
```

**For HPC or custom model/runs directories:**

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir /custom/path/to/runs \
  --models-dir /custom/path/to/models \
  --phase trials
```

### Phase 2: Run Judge Labeling on Outputs

After Phase 1 completes, run the Gemma-3 judge model to label outputs:

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase judge
```

This internally runs:
```bash
python -m vivarium olmo-conformity-judgeval \
  --run-id {run_id} \
  --db {path/to/simulation.db}
```
for each completed run.

### Phase 3: Post-hoc Analysis

Generates interpretability probes and additional metrics:

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase posthoc
```

## Low-Level: Direct Vivarium Command (If Orchestrator Fails)

If the orchestrator has issues, you can invoke vivarium directly:

```bash
# Run a single temperature's trials
python -m vivarium olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_expanded_localtest_by_model.json \
  --runs-dir ./runs_latest/runs

# With Ollama (if using remote inference):
python -m vivarium olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_expanded_localtest_by_model.json \
  --runs-dir ./runs_latest/runs \
  --api-base http://localhost:11434/v1
```

## Understanding the Suite Configuration

The suite config (e.g., `suite_expanded_localtest_by_model.json`) defines:
- **Models**: Which model variants to run (base, instruct, instruct_sft, instruct_dpo, think, think_sft, think_dpo)
- **Conditions**: Which conformity conditions (control, asch_history_5, tone variants, mitigations, etc.)
- **Temperatures**: Which temperature values for each model (usually 0.0–1.0 in 0.2 increments)
- **Datasets**: Which evaluation datasets (knowledge, math, science, reasoning, truthfulness)

To inspect or customize:
```bash
cat experiments/olmo_conformity/configs/suite_expanded_localtest_by_model.json | head -50
```

## Useful Options

### Dry Run (Preview Without Executing)
```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --dry-run
```

### Skip Running, Only Generate Analysis
If trials are already complete but you want to regenerate reports/figures:
```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --skip-runs
```

### Only Regenerate Combined Cross-Temperature Analysis
```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --only-analysis
```

### Run Only Experiments (No Reports/Analysis)
```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase trials \
  --runs-only
```

## What Gets Generated

### After Phase 1 (Trials):
- `runs_latest/runs/{timestamp}_{run_id}/`
  - `simulation.db` – SQLite database with all trials, outputs, metadata
  - `config.json` – Suite config used
  - Timestamp directory names like `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215`

### After Phase 2 (Judge):
- `simulation.db` updated with judge labels in:
  - `conformity_outputs.parsed_answer_json` – Judge's parsed output (is_correct, wrong_answer_endorsed, refusal_flag, notes)

### After Phase 3 (Post-hoc):
- Additional probe projections, logit lens data, turn-layer analysis saved to `simulation.db`

### Metadata:
- `Comparing_Experiments/runs_metadata.json` – Master registry of all runs (temperature, run_id, run_dir, status)
- `Comparing_Experiments/analysis_log.txt` – Full pipeline execution log

## Environment Variables (Optional)

Set these to override paths if needed:
```bash
export VIVARIUM_HF_CACHE=/path/to/huggingface/cache
export VIVARIUM_ARTIFACTS_DIR=/path/to/runs
export PYTHONPATH=/path/to/repo/src
```

## Current Data Status

From `Comparing_Experiments/runs_metadata_v6.json`:
- **T=0.0**: 34,794 trials ✓ Complete
- **T=0.2**: 34,746 trials ✓ Complete
- **T=0.4**: 37,978 trials ✓ Complete
- **T=0.6**: 38,170 trials ✓ Complete
- **T=0.8**: 32,925 trials ✓ Complete
- **T=1.0**: 63,315 trials (24,707 missing outputs; run with max_items=200 instead of 50)

### Known Issue at T=1.0
The T=1.0 run was accidentally created with `max_items_per_dataset=200` instead of 50, creating extra unneeded trials. The analysis pipeline handles this via balanced item-set intersection.

## Example: Complete Workflow for Specific Temperatures

```bash
# Step 1: Run trials for T=0.8 and T=1.0 only (filling gaps if any)
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase trials \
  --temps 0.8,1.0 \
  --no-sleep

# Step 2: Run judge on all completed runs
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase judge

# Step 3: Run post-hoc analysis
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --phase posthoc

# Step 4: Generate publication item set (from Analysis Scripts)
cd Analysis\ Scripts
python generate_publication_item_set.py \
  --runs-dir ../runs_latest/runs \
  --out-dir ../Comparing_Experiments/final_publication
```

## Monitoring Progress

```bash
# Watch the log in real-time
tail -f Comparing_Experiments/analysis_log.txt

# Check run status
python Analysis\ Scripts/report_run_trial_counts.py \
  ./runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json
```

## Troubleshooting

**"no such column: model_name"**
→ Database schema issue; ensure you're using updated vivarium

**"Config not found"**
→ Check that your suite config path is correct and readable

**"Timed out waiting for metadata lock"**
→ Multiple processes writing metadata simultaneously; wait or use `--runs-only --temps X.X` to isolate

**Incomplete outputs**
→ Check `Comparing_Experiments/runs_metadata_v6.json` for `n_missing_output` field; re-run with `--force-rerun`
