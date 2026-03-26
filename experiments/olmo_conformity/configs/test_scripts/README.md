# Local test scripts (Olmo conformity)

Minimal local smoke-test setup.

## Suite

- **`suite_expanded_localtest_by_model.json`** – Single config defining the full experiment:
  - Models with per-model temperature lists
  - 1 item per dataset per condition
  - Execution order: model-first (each model runs all its temps before the next)

## Run trials

```bash
bash 01_run_trials.sh
```

Uses `run_expanded_experiments.py` with `--phase trials`. Metadata is written to `Comparing_Experiments/runs_metadata.json`.

Options (pass through to the script):
- `--dry-run` – Show what would run without executing
- `--no-sleep` – Keep Mac awake during runs (caffeinate)
- `--api-base http://localhost:11434/v1` – Use Ollama for inference

Env vars:
- `SUITE` – Override suite path (default: `suite_expanded_localtest_by_model.json`)
- `RUNS_DIR` – Override runs directory (default: `{repo}/runs`)

## Run posthoc interpretability

After trials complete, run the full posthoc interpretability pipeline (logit lens tug-of-war, contrastive vector steering, activation patching, probes, interventions, and reports):

```bash
RUN_ID=<uuid> bash 02_run_posthoc.sh
```

The `RUN_ID` is the UUID suffix from the run directory created by `01_run_trials.sh` (e.g. `runs/20260226_123456_abc123def/` → `RUN_ID=abc123def`).

Options (pass through to `run_interpretability_posthoc.py`):
- `--skip-activations` – Skip activation capture backfill (if already captured)
- `--skip-logit-lens-tug-of-war` – Skip P(truth) vs P(sycophantic) tug-of-war analysis
- `--skip-contrastive-steering` – Skip contrastive vector steering (RepE / CAA)
- `--skip-activation-patching` – Skip activation patching (causal tracing)
- `--skip-interventions` – Skip probe-based interventions
- `--skip-reports` – Skip figure/table regeneration
- `--steering-alphas "-2.0,-1.0,0.5,1.0,2.0,4.0"` – Override steering alpha sweep
- `--steering-layers "10,11,...,20"` – Override target layers for steering
- `--patching-layers "0,1,...,31"` – Override target layers for patching

Env vars:
- `RUN_ID` (required) – UUID of the completed trial run
- `RUNS_DIR` – Override runs directory (default: `{repo}/runs`)

## Next steps (after trials complete)

Judge and posthoc are built into `run_expanded_experiments.py`:

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py --phase judge
python experiments/olmo_conformity/configs/run_expanded_experiments.py --phase posthoc
```
