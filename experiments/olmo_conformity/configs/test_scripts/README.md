# Local test scripts (Olmo conformity)

This folder contains small local smoke-test configs and shell scripts.

- Suite configs here limit to **5 items per dataset** and cover **all models** across **all temperatures**.
- Scripts write a simple run manifest TSV (`runs_manifest.tsv`) in this folder to pass outputs between stages.
- By default, `01_run_trials_all_temps.sh` points HF cache at `models/huggingface_cache` (repo-local) via `AAM_HF_CACHE`.

Suggested usage:
- `bash 01_run_trials_all_temps.sh`
- `bash 02_judgeval_all_temps.sh` (optional)
- `bash 03_posthoc_all_temps.sh` (optional; requires local HF/torch tooling)
- `bash 04_report_all_temps.sh`
