# Supplementary: LLM Judge Labeling Audit (runs-hpc-full)

**Generated:** 2026-02-07 15:47:29

## 0) Executive Summary

- **No judge labels found** in `runs-hpc-full/runs` (0/23760 first-outputs have `parsed_answer_json`).
- Judge-based statistics cannot be computed from the provided artifacts yet.
- Paper numbers based on `conformity_outputs.is_correct` are unaffected (judge is supplementary validation).

## 1) Judge-Label Availability

Primary runs dir: `runs-hpc-full/runs`

|   temperature | run_id                               | run_dir                                              |   n_trials |   n_first_outputs |   n_judged | judged_pct   | error   |
|--------------:|:-------------------------------------|:-----------------------------------------------------|-----------:|------------------:|-----------:|:-------------|:--------|
|           0   | 56478e99-7607-4957-9f53-a53b73a7e9d4 | 20260203_170602_56478e99-7607-4957-9f53-a53b73a7e9d4 |       3960 |              3960 |          0 | 0.0%         |         |
|           0.2 | 99127619-fcc7-4fd4-ba3a-cc810610249f | 20260203_170602_99127619-fcc7-4fd4-ba3a-cc810610249f |       3960 |              3960 |          0 | 0.0%         |         |
|           0.4 | 271bb5b2-572d-4ecd-8577-b07a7cd10846 | 20260203_170602_271bb5b2-572d-4ecd-8577-b07a7cd10846 |       3960 |              3960 |          0 | 0.0%         |         |
|           0.6 | dda9d6b3-a516-41b3-a85a-b424de8f15d3 | 20260203_170602_dda9d6b3-a516-41b3-a85a-b424de8f15d3 |       3960 |              3960 |          0 | 0.0%         |         |
|           0.8 | eb777acc-3ab5-4f87-b073-249a50d25863 | 20260203_170602_eb777acc-3ab5-4f87-b073-249a50d25863 |       3960 |              3960 |          0 | 0.0%         |         |
|           1   | fa0b1d4f-d547-4094-b07c-4f9efc20f771 | 20260203_185521_fa0b1d4f-d547-4094-b07c-4f9efc20f771 |       3960 |              3960 |          0 | 0.0%         |         |

Secondary/probe runs dir: `runs-hpc-full/probe/runs`

|   temperature | run_id                               | run_dir                                              | n_trials   | n_first_outputs   | n_judged   | judged_pct   | error                            |
|--------------:|:-------------------------------------|:-----------------------------------------------------|:-----------|:------------------|:-----------|:-------------|:---------------------------------|
|           0   | 56478e99-7607-4957-9f53-a53b73a7e9d4 | 20260203_170602_56478e99-7607-4957-9f53-a53b73a7e9d4 | 4361       | 4360              | 0          | 0.0%         |                                  |
|           0.2 | 99127619-fcc7-4fd4-ba3a-cc810610249f | 20260203_170602_99127619-fcc7-4fd4-ba3a-cc810610249f | 4361       | 4360              | 0          | 0.0%         |                                  |
|           0.4 | 271bb5b2-572d-4ecd-8577-b07a7cd10846 | 20260203_170602_271bb5b2-572d-4ecd-8577-b07a7cd10846 | NA         | NA                | NA         | NA           | database disk image is malformed |
|           0.6 | dda9d6b3-a516-41b3-a85a-b424de8f15d3 | 20260203_170602_dda9d6b3-a516-41b3-a85a-b424de8f15d3 | 4860       | 4860              | 0          | 0.0%         |                                  |
|           0.8 | eb777acc-3ab5-4f87-b073-249a50d25863 | 20260203_170602_eb777acc-3ab5-4f87-b073-249a50d25863 | NA         | NA                | NA         | NA           | file is not a database           |
|           1   | fa0b1d4f-d547-4094-b07c-4f9efc20f771 | 20260203_185521_fa0b1d4f-d547-4094-b07c-4f9efc20f771 | 4361       | 4360              | 0          | 0.0%         |                                  |

## 2) Baseline (Rule-Based) Statistics From DB

- Factual correctness uses `conformity_outputs.is_correct` (first output per trial).
- Opinion wrong-answer endorsement uses the endorsement heuristic on `parsed_answer_text` (same as `scripts/audit_paper_numbers.py`).

### 2.1 Pooled pressure deltas (factual; pooled across temperatures)

| variant      |   delta_error_asch_pp |   delta_error_authority_pp |
|:-------------|----------------------:|---------------------------:|
| base         |                  0.58 |                       2.92 |
| instruct     |                  3.08 |                       7.08 |
| instruct_sft |                  8.42 |                       4.08 |
| think        |                 -3.42 |                      -1.58 |
| think_sft    |                 -4.08 |                      -2.67 |
| rl_zero      |                  1.08 |                       1.83 |

### 2.2 Temperature amplification (paired T=1 vs T=0; McNemar exact)

| condition          |   delta_error_pp |   p_mcnemar |   n_pairs |   b_1to0 |   c_0to1 |
|:-------------------|-----------------:|------------:|----------:|---------:|---------:|
| control            |             0.34 |     0.762   |      1169 |       51 |       47 |
| asch_history_5     |             2.34 |     0.0118  |      1196 |       72 |       44 |
| authoritative_bias |             3.39 |     0.00033 |      1180 |       80 |       40 |

### 2.3 Truth override (factual; pooled across temperatures)

| variant      | pressure_condition   |   n_items |   truth_override_rate |
|:-------------|:---------------------|----------:|----------------------:|
| base         | asch_history_5       |       311 |                 0.273 |
| base         | authoritative_bias   |       311 |                 0.328 |
| instruct     | asch_history_5       |       323 |                 0.251 |
| instruct     | authoritative_bias   |       323 |                 0.415 |
| instruct_sft | asch_history_5       |       322 |                 0.45  |
| instruct_sft | authoritative_bias   |       322 |                 0.339 |
| think        | asch_history_5       |       270 |                 0.13  |
| think        | authoritative_bias   |       270 |                 0.193 |
| think_sft    | asch_history_5       |       314 |                 0.096 |
| think_sft    | authoritative_bias   |       314 |                 0.118 |
| rl_zero      | asch_history_5       |        86 |                 0.674 |
| rl_zero      | authoritative_bias   |        86 |                 0.756 |

### 2.4 Opinion items: wrong-answer endorsement + refusal (rule-based)

|   temperature | variant      | condition_name     |   n_trials | wrong_answer_endorse_rate   | refusal_rate   |
|--------------:|:-------------|:-------------------|-----------:|:----------------------------|:---------------|
|           0   | base         | control            |         20 | 35.0%                       | 0.0%           |
|           0   | base         | asch_history_5     |         20 | 80.0%                       | 0.0%           |
|           0   | base         | authoritative_bias |         20 | 75.0%                       | 20.0%          |
|           0   | instruct     | control            |         20 | 35.0%                       | 0.0%           |
|           0   | instruct     | asch_history_5     |         20 | 70.0%                       | 10.0%          |
|           0   | instruct     | authoritative_bias |         20 | 30.0%                       | 70.0%          |
|           0   | instruct_sft | control            |         20 | 50.0%                       | 30.0%          |
|           0   | instruct_sft | asch_history_5     |         20 | 80.0%                       | 0.0%           |
|           0   | instruct_sft | authoritative_bias |         20 | 20.0%                       | 70.0%          |
|           0   | think        | control            |         20 | 10.0%                       | 70.0%          |
|           0   | think        | asch_history_5     |         20 | 25.0%                       | 15.0%          |
|           0   | think        | authoritative_bias |         20 | 15.0%                       | 45.0%          |
|           0   | think_sft    | control            |         20 | 10.0%                       | 65.0%          |
|           0   | think_sft    | asch_history_5     |         20 | 5.0%                        | 25.0%          |
|           0   | think_sft    | authoritative_bias |         20 | 10.0%                       | 35.0%          |
|           0   | rl_zero      | control            |         20 | 0.0%                        | 0.0%           |
|           0   | rl_zero      | asch_history_5     |         20 | 40.0%                       | 0.0%           |
|           0   | rl_zero      | authoritative_bias |         20 | 0.0%                        | 0.0%           |
|           0.2 | base         | control            |         20 | 30.0%                       | 0.0%           |
|           0.2 | base         | asch_history_5     |         20 | 80.0%                       | 0.0%           |
|           0.2 | base         | authoritative_bias |         20 | 55.0%                       | 25.0%          |
|           0.2 | instruct     | control            |         20 | 40.0%                       | 10.0%          |
|           0.2 | instruct     | asch_history_5     |         20 | 55.0%                       | 15.0%          |
|           0.2 | instruct     | authoritative_bias |         20 | 10.0%                       | 85.0%          |
|           0.2 | instruct_sft | control            |         20 | 35.0%                       | 40.0%          |
|           0.2 | instruct_sft | asch_history_5     |         20 | 80.0%                       | 0.0%           |
|           0.2 | instruct_sft | authoritative_bias |         20 | 15.0%                       | 70.0%          |
|           0.2 | think        | control            |         20 | 0.0%                        | 55.0%          |
|           0.2 | think        | asch_history_5     |         20 | 15.0%                       | 25.0%          |
|           0.2 | think        | authoritative_bias |         20 | 5.0%                        | 55.0%          |
|           0.2 | think_sft    | control            |         20 | 0.0%                        | 45.0%          |
|           0.2 | think_sft    | asch_history_5     |         20 | 15.0%                       | 25.0%          |
|           0.2 | think_sft    | authoritative_bias |         20 | 10.0%                       | 45.0%          |
|           0.2 | rl_zero      | control            |         20 | 10.0%                       | 0.0%           |
|           0.2 | rl_zero      | asch_history_5     |         20 | 10.0%                       | 0.0%           |
|           0.2 | rl_zero      | authoritative_bias |         20 | 10.0%                       | 0.0%           |
|           0.4 | base         | control            |         20 | 25.0%                       | 25.0%          |
|           0.4 | base         | asch_history_5     |         20 | 80.0%                       | 5.0%           |
|           0.4 | base         | authoritative_bias |         20 | 65.0%                       | 15.0%          |
|           0.4 | instruct     | control            |         20 | 40.0%                       | 15.0%          |
|           0.4 | instruct     | asch_history_5     |         20 | 45.0%                       | 10.0%          |
|           0.4 | instruct     | authoritative_bias |         20 | 30.0%                       | 50.0%          |
|           0.4 | instruct_sft | control            |         20 | 35.0%                       | 45.0%          |
|           0.4 | instruct_sft | asch_history_5     |         20 | 85.0%                       | 0.0%           |
|           0.4 | instruct_sft | authoritative_bias |         20 | 25.0%                       | 50.0%          |
|           0.4 | think        | control            |         20 | 15.0%                       | 40.0%          |
|           0.4 | think        | asch_history_5     |         20 | 5.0%                        | 15.0%          |
|           0.4 | think        | authoritative_bias |         20 | 15.0%                       | 55.0%          |
|           0.4 | think_sft    | control            |         20 | 20.0%                       | 50.0%          |
|           0.4 | think_sft    | asch_history_5     |         20 | 10.0%                       | 45.0%          |
|           0.4 | think_sft    | authoritative_bias |         20 | 20.0%                       | 35.0%          |
|           0.4 | rl_zero      | control            |         20 | 5.0%                        | 0.0%           |
|           0.4 | rl_zero      | asch_history_5     |         20 | 5.0%                        | 5.0%           |
|           0.4 | rl_zero      | authoritative_bias |         20 | 15.0%                       | 0.0%           |
|           0.6 | base         | control            |         20 | 25.0%                       | 25.0%          |
|           0.6 | base         | asch_history_5     |         20 | 75.0%                       | 0.0%           |
|           0.6 | base         | authoritative_bias |         20 | 65.0%                       | 5.0%           |
|           0.6 | instruct     | control            |         20 | 40.0%                       | 20.0%          |
|           0.6 | instruct     | asch_history_5     |         20 | 45.0%                       | 10.0%          |
|           0.6 | instruct     | authoritative_bias |         20 | 20.0%                       | 45.0%          |
|           0.6 | instruct_sft | control            |         20 | 40.0%                       | 45.0%          |
|           0.6 | instruct_sft | asch_history_5     |         20 | 95.0%                       | 0.0%           |
|           0.6 | instruct_sft | authoritative_bias |         20 | 40.0%                       | 40.0%          |
|           0.6 | think        | control            |         20 | 20.0%                       | 40.0%          |
|           0.6 | think        | asch_history_5     |         20 | 35.0%                       | 40.0%          |
|           0.6 | think        | authoritative_bias |         20 | 15.0%                       | 55.0%          |
|           0.6 | think_sft    | control            |         20 | 10.0%                       | 50.0%          |
|           0.6 | think_sft    | asch_history_5     |         20 | 10.0%                       | 40.0%          |
|           0.6 | think_sft    | authoritative_bias |         20 | 20.0%                       | 45.0%          |
|           0.6 | rl_zero      | control            |         20 | 15.0%                       | 0.0%           |
|           0.6 | rl_zero      | asch_history_5     |         20 | 25.0%                       | 0.0%           |
|           0.6 | rl_zero      | authoritative_bias |         20 | 5.0%                        | 0.0%           |
|           0.8 | base         | control            |         20 | 10.0%                       | 25.0%          |
|           0.8 | base         | asch_history_5     |         20 | 55.0%                       | 0.0%           |
|           0.8 | base         | authoritative_bias |         20 | 40.0%                       | 10.0%          |
|           0.8 | instruct     | control            |         20 | 35.0%                       | 30.0%          |
|           0.8 | instruct     | asch_history_5     |         20 | 45.0%                       | 10.0%          |
|           0.8 | instruct     | authoritative_bias |         20 | 60.0%                       | 30.0%          |
|           0.8 | instruct_sft | control            |         20 | 35.0%                       | 50.0%          |
|           0.8 | instruct_sft | asch_history_5     |         20 | 80.0%                       | 0.0%           |
|           0.8 | instruct_sft | authoritative_bias |         20 | 55.0%                       | 25.0%          |
|           0.8 | think        | control            |         20 | 10.0%                       | 45.0%          |
|           0.8 | think        | asch_history_5     |         20 | 10.0%                       | 35.0%          |
|           0.8 | think        | authoritative_bias |         20 | 25.0%                       | 55.0%          |
|           0.8 | think_sft    | control            |         20 | 20.0%                       | 35.0%          |
|           0.8 | think_sft    | asch_history_5     |         20 | 0.0%                        | 45.0%          |
|           0.8 | think_sft    | authoritative_bias |         20 | 20.0%                       | 35.0%          |
|           0.8 | rl_zero      | control            |         20 | 5.0%                        | 5.0%           |
|           0.8 | rl_zero      | asch_history_5     |         20 | 15.0%                       | 0.0%           |
|           0.8 | rl_zero      | authoritative_bias |         20 | 15.0%                       | 10.0%          |
|           1   | base         | control            |         20 | 25.0%                       | 30.0%          |
|           1   | base         | asch_history_5     |         20 | 45.0%                       | 10.0%          |
|           1   | base         | authoritative_bias |         20 | 40.0%                       | 20.0%          |
|           1   | instruct     | control            |         20 | 30.0%                       | 50.0%          |
|           1   | instruct     | asch_history_5     |         20 | 50.0%                       | 15.0%          |
|           1   | instruct     | authoritative_bias |         20 | 40.0%                       | 40.0%          |
|           1   | instruct_sft | control            |         20 | 20.0%                       | 50.0%          |
|           1   | instruct_sft | asch_history_5     |         20 | 90.0%                       | 0.0%           |
|           1   | instruct_sft | authoritative_bias |         20 | 55.0%                       | 10.0%          |
|           1   | think        | control            |         20 | 20.0%                       | 40.0%          |
|           1   | think        | asch_history_5     |         20 | 10.0%                       | 40.0%          |
|           1   | think        | authoritative_bias |         20 | 30.0%                       | 45.0%          |
|           1   | think_sft    | control            |         20 | 10.0%                       | 40.0%          |
|           1   | think_sft    | asch_history_5     |         20 | 20.0%                       | 45.0%          |
|           1   | think_sft    | authoritative_bias |         20 | 15.0%                       | 45.0%          |
|           1   | rl_zero      | control            |         20 | 10.0%                       | 0.0%           |
|           1   | rl_zero      | asch_history_5     |         20 | 25.0%                       | 0.0%           |
|           1   | rl_zero      | authoritative_bias |         20 | 20.0%                       | 0.0%           |

## 3) LLM Judge Statistics (From `parsed_answer_json`)

Expected schema (per first output):
```json
{ "truthfulness": 0.0, "conformity": 0.0, "rationalization": null, "notes": "...", "_llm_judge": {"prompt_version":"...","judge_model":"..."} }
```

**Status:** No judge JSON found in the provided DBs; judge-based sections cannot be populated yet.

### 3.1 How to generate judge labels (post-hoc)

For each run_id (per-temperature), run:
```bash
python experiments/olmo_conformity/configs/run_llm_judge_posthoc.py \
  --run-id <RUN_ID> \
  --runs-dir runs-hpc-full/runs \
  --max-concurrency 4 \
  --trial-scope behavioral-only
```

Then regenerate this report:
```bash
python scripts/audit_llm_judge_labeling.py --runs-dir runs-hpc-full/runs --out-md paper/JUDGE_REPORT.md
```

Notes:
- Requires Ollama + an installed judge model (see `src/aam/experiments/olmo_conformity/ollama_judge.py`).
- `olmo-conformity-judgeval` writes into `conformity_outputs.parsed_answer_json`.
