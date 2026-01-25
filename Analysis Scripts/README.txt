Analysis Scripts
================

Standalone scripts for regenerating paper figures and tables from saved run artifacts.

Usage
-----

These scripts assume either:
- you installed the project (e.g. `pip install -e .`), OR
- you run them directly (they add `src/` to PYTHONPATH internally).

You must provide either:
- --run-dir: path to a run directory that contains `simulation.db`, or
- --run-id: the run UUID/suffix, in which case the script will search `--runs-dir` (default: ./runs).

Examples
--------

Generate all figures:
  python "Analysis Scripts/generate_all_figures.py" --run-dir runs/<timestamp>_<run_id>

Generate all figures by run_id lookup:
  python "Analysis Scripts/generate_all_figures.py" --run-id <run_id>

Generate individual figures:
  python "Analysis Scripts/generate_figure1_behavioral.py" --run-id <run_id>
  python "Analysis Scripts/generate_figure2_tug_of_war.py" --run-id <run_id>
  python "Analysis Scripts/generate_figure3_turn_layer.py" --run-id <run_id>
  python "Analysis Scripts/generate_figure4_intervention.py" --run-id <run_id>

Generate tables/logs:
  python "Analysis Scripts/generate_all_tables.py" --run-id <run_id>

Outputs
-------

Figures are written under:
  runs/<...>/artifacts/figures/

Tables/logs are written under:
  runs/<...>/artifacts/tables/
  runs/<...>/artifacts/logs/

