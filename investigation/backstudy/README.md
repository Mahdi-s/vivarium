# investigation/backstudy — evidence-based back-study (Sept 2026)

Purpose: test, on the existing trial data and with a local feedback loop, the hypothesis that *prompt structure* is what makes OLMo-3 abandon answers it knows — and identify a defensible research gap. Read in this order:

1. `RESEARCH_PROBLEM_METHOD.md` — the frameworks used to pick and justify the gap (problematization, strong inference, validity typology, Marr, Heilmeier).
2. `FINDINGS.md` — the numbers from the existing 201K OLMo + 45K cross-family trials, and the pilot.
3. `LITERATURE_LANDSCAPE.md` — 7-angle literature sweep (Sept 2026) with scoop table, open gaps G1–G9, unapplied theories, ranked framings.
4. `../RESEARCH_GAP_RECOMMENDATION.md` — the recommendation.

## Re-running

```bash
python3 investigation/backstudy/build_datasets.py          # raw SQLite -> data/olmo_trials.parquet, data/crossfamily_trials.parquet
python3 investigation/backstudy/analysis_core.py           # 4-state transitions, contrasts, ordering stability -> results/core.md
python3 investigation/backstudy/analysis_judge_null.py     # 5-state robustness -> results/five_state_*.csv
python3 investigation/backstudy/analysis_structure_items.py# feature decomposition, item susceptibility/Rasch, temperature
python3 investigation/backstudy/analysis_content.py        # what the model says when it abandons -> results/content_examples.md
python3 investigation/backstudy/analysis_validity.py       # truncation / judge-null / length checks
python3 investigation/backstudy/analysis_crossfamily.py    # cross-family + n-gram ablations -> results/crossfamily.md
```

Local model pilot (uses the repo's `.venv`, MPS, HF cache; system python segfaults on transformers' torchvision import):

```bash
.venv/bin/python investigation/backstudy/tools/belief_probe.py --model-id allenai/Olmo-3-7B-Instruct-SFT --items-per-dataset 5 --conditions all
investigation/backstudy/tools/run_pilot.sh 5 all           # sequential: SFT, Instruct, Base, DPO
python3 investigation/backstudy/analysis_belief_probe.py   # -> results/belief_probe.md
```

`belief_probe.py --dry-run` prints rendered prompts without loading a model. Outputs resume from `data/belief_probe/<variant>.jsonl`.

Label rules, condition structure and schema notes are in the docstring of `build_datasets.py`. No script modifies any run DB.

## External data (T7 drive)

`/Volumes/T7/conformity data analysis/dataset_analysis/` holds the full Dolci mixtures (instruct-SFT 2.15M rows, instruct-DPO 259,922 pairs with `preference_type`/`chosen_model`/`rejected_model`, instruct-RL 170K, think-SFT 2.27M, think-DPO 150K, think-RL 102K; 41 GB parquet) and the paper's audit outputs (`results/phase1…phase7_*.csv`; the audit scripts survive only as `__pycache__`). `Archive-runs_conformity.zip` on T7 contains the `runs_latest` DBs plus the HPC Think re-runs (`runs-think-hpc/`, 7B at 2,048 tokens; `runs-32B/` at 4,096) whose local copies lack the DBs.

```bash
python3 investigation/backstudy/analysis_dolci_policy.py   # DPO pairs: hedging/refusal/pushback/agreement, by preference type; model-pair confound
python3 investigation/backstudy/analysis_dolci_sft.py      # SFT sample: hedging, 'say you are unsure' instructions, definiteness by source
```
