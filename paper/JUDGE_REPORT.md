# Supplementary: LLM Judge Labeling Audit (runs-hpc-full)

**Generated:** 2026-03-09 15:25:20

## 0) Executive Summary

- Judge coverage: **208088/215288** (**96.7%**).
- Judge produces **binary labels** (`is_correct`, `refusal_flag`, `wrong_answer_endorsed`) directly comparable to the rule-based system.
- Overall `is_correct` agreement: **78.6%** (Cohen's κ = 0.117, n=176627).

## 1) Judge-Label Availability

Primary runs dir: `runs_latest/runs`

|   temperature | run_id                               | run_dir                                              |   n_trials |   n_first_outputs |   n_judged | judged_pct   | error   |
|--------------:|:-------------------------------------|:-----------------------------------------------------|-----------:|------------------:|-----------:|:-------------|:--------|
|           0   | 9f240f89-e58e-423a-ae68-f990b31c84cd | 20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd |      34794 |             34794 |      34794 | 100.0%       |         |
|           0.2 | 46f0762a-210a-459a-8709-d24a0f194eb0 | 20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0 |      34746 |             34746 |      34746 | 100.0%       |         |
|           0.4 | bbd05985-d185-460a-b0aa-dd356d27ec94 | 20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94 |      37978 |             37978 |      37978 | 100.0%       |         |
|           0.6 | 86c72262-d1aa-41b5-9c22-d7b2e0570215 | 20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215 |      38170 |             38170 |      38170 | 100.0%       |         |
|           0.8 | 9369442d-d825-4cd0-81a1-8ed276c37814 | 20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814 |      34800 |             34800 |      31200 | 89.7%        |         |
|           1   | 9173bfae-4e8a-464f-8c9c-7ee91caa8b6e | 20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e |      34800 |             34800 |      31200 | 89.7%        |         |

## 2) Baseline (Rule-Based) Statistics From DB

- Factual correctness uses `conformity_outputs.is_correct` (first output per trial).
- Opinion wrong-answer endorsement uses the endorsement heuristic on `parsed_answer_text` (same as `scripts/audit_paper_numbers.py`).

### 2.1 Pooled pressure deltas (factual; pooled across temperatures)

| variant      |   delta_error_asch_pp |   delta_error_authority_pp |
|:-------------|----------------------:|---------------------------:|
| base         |                  2.05 |                       2.43 |
| instruct     |                  2.48 |                       6.19 |
| instruct_sft |                  4.95 |                       3.52 |
| think        |                 -2.52 |                       1.95 |
| think_sft    |                 -3.19 |                       1.14 |
| rl_zero      |                 -0.71 |                       0.29 |

### 2.2 Temperature amplification (paired T=1 vs T=0; McNemar exact)

| condition          |   delta_error_pp |   p_mcnemar |   n_pairs |   b_1to0 |   c_0to1 |
|:-------------------|-----------------:|------------:|----------:|---------:|---------:|
| control            |             1.28 |      0.0203 |      2587 |      112 |       79 |
| asch_history_5     |             1.14 |      0.0453 |      2712 |      128 |       97 |
| authoritative_bias |             0.15 |      0.819  |      2689 |       88 |       84 |

### 2.3 Truth override (factual; pooled across temperatures)

| variant      | pressure_condition   |   n_items |   truth_override_rate |
|:-------------|:---------------------|----------:|----------------------:|
| base         | asch_history_5       |       250 |                 0.516 |
| base         | authoritative_bias   |       250 |                 0.556 |
| instruct     | asch_history_5       |       263 |                 0.498 |
| instruct     | authoritative_bias   |       263 |                 0.681 |
| instruct_sft | asch_history_5       |       323 |                 0.523 |
| instruct_sft | authoritative_bias   |       323 |                 0.455 |
| think        | asch_history_5       |       155 |                 0.29  |
| think        | authoritative_bias   |       155 |                 0.542 |
| think_sft    | asch_history_5       |       203 |                 0.236 |
| think_sft    | authoritative_bias   |       203 |                 0.458 |
| rl_zero      | asch_history_5       |        60 |                 0.583 |
| rl_zero      | authoritative_bias   |        60 |                 0.583 |

### 2.4 Opinion items: wrong-answer endorsement + refusal (rule-based)

|   temperature | variant      | condition_name     |   n_trials | wrong_answer_endorse_rate   | refusal_rate   |
|--------------:|:-------------|:-------------------|-----------:|:----------------------------|:---------------|
|           0   | base         | control            |         50 | 0.0%                        | 4.0%           |
|           0   | base         | asch_history_5     |         50 | 40.0%                       | 2.0%           |
|           0   | base         | authoritative_bias |         50 | 72.0%                       | 8.0%           |
|           0   | instruct     | control            |         50 | 0.0%                        | 2.0%           |
|           0   | instruct     | asch_history_5     |         50 | 10.0%                       | 0.0%           |
|           0   | instruct     | authoritative_bias |         50 | 34.0%                       | 0.0%           |
|           0   | instruct_sft | control            |         50 | 0.0%                        | 4.0%           |
|           0   | instruct_sft | asch_history_5     |         50 | 84.0%                       | 0.0%           |
|           0   | instruct_sft | authoritative_bias |         50 | 42.0%                       | 6.0%           |
|           0   | think        | control            |         50 | 0.0%                        | 6.0%           |
|           0   | think        | asch_history_5     |         50 | 10.0%                       | 4.0%           |
|           0   | think        | authoritative_bias |         50 | 64.0%                       | 4.0%           |
|           0   | think_sft    | control            |         50 | 2.0%                        | 8.0%           |
|           0   | think_sft    | asch_history_5     |         50 | 20.0%                       | 2.0%           |
|           0   | think_sft    | authoritative_bias |         50 | 38.0%                       | 8.0%           |
|           0   | rl_zero      | control            |         50 | 0.0%                        | 2.0%           |
|           0   | rl_zero      | asch_history_5     |         50 | 10.0%                       | 0.0%           |
|           0   | rl_zero      | authoritative_bias |         50 | 38.0%                       | 2.0%           |
|           0.2 | base         | control            |         50 | 0.0%                        | 0.0%           |
|           0.2 | base         | asch_history_5     |         50 | 52.0%                       | 2.0%           |
|           0.2 | base         | authoritative_bias |         50 | 66.0%                       | 12.0%          |
|           0.2 | instruct     | control            |         50 | 0.0%                        | 0.0%           |
|           0.2 | instruct     | asch_history_5     |         50 | 14.0%                       | 0.0%           |
|           0.2 | instruct     | authoritative_bias |         50 | 24.0%                       | 0.0%           |
|           0.2 | instruct_sft | control            |         50 | 0.0%                        | 4.0%           |
|           0.2 | instruct_sft | asch_history_5     |         50 | 82.0%                       | 0.0%           |
|           0.2 | instruct_sft | authoritative_bias |         50 | 42.0%                       | 4.0%           |
|           0.2 | think        | control            |         50 | 0.0%                        | 4.0%           |
|           0.2 | think        | asch_history_5     |         50 | 8.0%                        | 2.0%           |
|           0.2 | think        | authoritative_bias |         50 | 50.0%                       | 6.0%           |
|           0.2 | think_sft    | control            |         50 | 0.0%                        | 10.0%          |
|           0.2 | think_sft    | asch_history_5     |         50 | 26.0%                       | 8.0%           |
|           0.2 | think_sft    | authoritative_bias |         50 | 44.0%                       | 4.0%           |
|           0.2 | rl_zero      | control            |         50 | 0.0%                        | 0.0%           |
|           0.2 | rl_zero      | asch_history_5     |         50 | 6.0%                        | 0.0%           |
|           0.2 | rl_zero      | authoritative_bias |         50 | 40.0%                       | 2.0%           |
|           0.4 | base         | control            |         50 | 0.0%                        | 4.0%           |
|           0.4 | base         | asch_history_5     |         50 | 44.0%                       | 0.0%           |
|           0.4 | base         | authoritative_bias |         50 | 54.0%                       | 10.0%          |
|           0.4 | instruct     | control            |         50 | 4.0%                        | 0.0%           |
|           0.4 | instruct     | asch_history_5     |         50 | 14.0%                       | 0.0%           |
|           0.4 | instruct     | authoritative_bias |         50 | 28.0%                       | 0.0%           |
|           0.4 | instruct_sft | control            |         50 | 0.0%                        | 4.0%           |
|           0.4 | instruct_sft | asch_history_5     |         50 | 82.0%                       | 0.0%           |
|           0.4 | instruct_sft | authoritative_bias |         50 | 34.0%                       | 0.0%           |
|           0.4 | think        | control            |         50 | 0.0%                        | 2.0%           |
|           0.4 | think        | asch_history_5     |         50 | 16.0%                       | 4.0%           |
|           0.4 | think        | authoritative_bias |         50 | 64.0%                       | 4.0%           |
|           0.4 | think_sft    | control            |         50 | 2.0%                        | 6.0%           |
|           0.4 | think_sft    | asch_history_5     |         50 | 20.0%                       | 6.0%           |
|           0.4 | think_sft    | authoritative_bias |         50 | 44.0%                       | 2.0%           |
|           0.4 | rl_zero      | control            |         50 | 0.0%                        | 2.0%           |
|           0.4 | rl_zero      | asch_history_5     |         50 | 8.0%                        | 0.0%           |
|           0.4 | rl_zero      | authoritative_bias |         50 | 6.0%                        | 4.0%           |
|           0.6 | base         | control            |         50 | 0.0%                        | 4.0%           |
|           0.6 | base         | asch_history_5     |         50 | 38.0%                       | 0.0%           |
|           0.6 | base         | authoritative_bias |         50 | 42.0%                       | 4.0%           |
|           0.6 | instruct     | control            |         50 | 2.0%                        | 0.0%           |
|           0.6 | instruct     | asch_history_5     |         50 | 12.0%                       | 0.0%           |
|           0.6 | instruct     | authoritative_bias |         50 | 22.0%                       | 0.0%           |
|           0.6 | instruct_sft | control            |         50 | 0.0%                        | 2.0%           |
|           0.6 | instruct_sft | asch_history_5     |         50 | 74.0%                       | 0.0%           |
|           0.6 | instruct_sft | authoritative_bias |         50 | 46.0%                       | 0.0%           |
|           0.6 | think        | control            |         50 | 0.0%                        | 12.0%          |
|           0.6 | think        | asch_history_5     |         50 | 16.0%                       | 6.0%           |
|           0.6 | think        | authoritative_bias |         50 | 66.0%                       | 4.0%           |
|           0.6 | think_sft    | control            |         50 | 0.0%                        | 16.0%          |
|           0.6 | think_sft    | asch_history_5     |         50 | 16.0%                       | 12.0%          |
|           0.6 | think_sft    | authoritative_bias |         50 | 50.0%                       | 6.0%           |
|           0.6 | rl_zero      | control            |         50 | 0.0%                        | 2.0%           |
|           0.6 | rl_zero      | asch_history_5     |         50 | 4.0%                        | 6.0%           |
|           0.6 | rl_zero      | authoritative_bias |         50 | 12.0%                       | 2.0%           |
|           0.8 | base         | control            |         50 | 0.0%                        | 0.0%           |
|           0.8 | base         | asch_history_5     |         50 | 34.0%                       | 0.0%           |
|           0.8 | base         | authoritative_bias |         50 | 24.0%                       | 6.0%           |
|           0.8 | instruct     | control            |         50 | 2.0%                        | 0.0%           |
|           0.8 | instruct     | asch_history_5     |         50 | 10.0%                       | 0.0%           |
|           0.8 | instruct     | authoritative_bias |         50 | 18.0%                       | 0.0%           |
|           0.8 | instruct_sft | control            |         50 | 2.0%                        | 2.0%           |
|           0.8 | instruct_sft | asch_history_5     |         50 | 78.0%                       | 0.0%           |
|           0.8 | instruct_sft | authoritative_bias |         50 | 48.0%                       | 0.0%           |
|           0.8 | think        | control            |         50 | 0.0%                        | 10.0%          |
|           0.8 | think        | asch_history_5     |         50 | 12.0%                       | 4.0%           |
|           0.8 | think        | authoritative_bias |         50 | 66.0%                       | 4.0%           |
|           0.8 | think_sft    | control            |         50 | 0.0%                        | 14.0%          |
|           0.8 | think_sft    | asch_history_5     |         50 | 12.0%                       | 0.0%           |
|           0.8 | think_sft    | authoritative_bias |         50 | 50.0%                       | 4.0%           |
|           0.8 | rl_zero      | control            |         50 | 0.0%                        | 0.0%           |
|           0.8 | rl_zero      | asch_history_5     |         50 | 24.0%                       | 0.0%           |
|           0.8 | rl_zero      | authoritative_bias |         50 | 22.0%                       | 0.0%           |
|           1   | base         | control            |         50 | 0.0%                        | 2.0%           |
|           1   | base         | asch_history_5     |         50 | 26.0%                       | 0.0%           |
|           1   | base         | authoritative_bias |         50 | 20.0%                       | 4.0%           |
|           1   | instruct     | control            |         50 | 0.0%                        | 0.0%           |
|           1   | instruct     | asch_history_5     |         50 | 14.0%                       | 0.0%           |
|           1   | instruct     | authoritative_bias |         50 | 18.0%                       | 0.0%           |
|           1   | instruct_sft | control            |         50 | 0.0%                        | 0.0%           |
|           1   | instruct_sft | asch_history_5     |         50 | 66.0%                       | 2.0%           |
|           1   | instruct_sft | authoritative_bias |         50 | 30.0%                       | 2.0%           |
|           1   | think        | control            |         50 | 0.0%                        | 10.0%          |
|           1   | think        | asch_history_5     |         50 | 20.0%                       | 4.0%           |
|           1   | think        | authoritative_bias |         50 | 76.0%                       | 2.0%           |
|           1   | think_sft    | control            |         50 | 2.0%                        | 10.0%          |
|           1   | think_sft    | asch_history_5     |         50 | 22.0%                       | 2.0%           |
|           1   | think_sft    | authoritative_bias |         50 | 64.0%                       | 2.0%           |
|           1   | rl_zero      | control            |         50 | 0.0%                        | 0.0%           |
|           1   | rl_zero      | asch_history_5     |         50 | 14.0%                       | 0.0%           |
|           1   | rl_zero      | authoritative_bias |         50 | 16.0%                       | 0.0%           |

## 3) LLM Judge Statistics (From `parsed_answer_json`)

Expected schema (per first output):
```json
{ "is_correct": 0|1, "refusal_flag": 0|1, "wrong_answer_endorsed": 0|1, "notes": "...", "_llm_judge": {"prompt_version":"...","judge_model":"..."} }
```

Judge prompt versions observed: 2026-03-01_local_v2, 2026-03-03_local_v3
Judge models observed: gemma3:12b, gemma3:1b, glm-5:cloud, gpt-oss:20b, qwen3:0.6b

### 3.2 Judge vs Manual: `is_correct` agreement

- **Overall agreement:** 78.6%
- **Cohen's κ:** 0.117
- **n:** 176627
- Confusion matrix: TP=5615, TN=133186, FP=26119, FN=11707

#### Per-cell breakdown

|   temperature | variant      | condition_name     |   n | agreement_rate   |   cohens_kappa |
|--------------:|:-------------|:-------------------|----:|:-----------------|---------------:|
|           0   | base         | asch_history_5     | 345 | 82.3%            |          0.328 |
|           0   | base         | authoritative_bias | 334 | 81.1%            |          0.277 |
|           0   | base         | control            | 294 | 67.3%            |          0.276 |
|           0   | instruct     | asch_history_5     | 324 | 68.2%            |          0.182 |
|           0   | instruct     | authoritative_bias | 312 | 67.6%            |          0.027 |
|           0   | instruct     | control            | 244 | 57.0%            |          0.199 |
|           0   | instruct_sft | asch_history_5     | 332 | 91.3%            |          0.633 |
|           0   | instruct_sft | authoritative_bias | 318 | 75.5%            |          0.276 |
|           0   | instruct_sft | control            | 280 | 62.9%            |          0.267 |
|           0   | rl_zero      | asch_history_5     | 317 | 71.9%            |          0.046 |
|           0   | rl_zero      | authoritative_bias | 283 | 78.1%            |          0.123 |
|           0   | rl_zero      | control            | 170 | 22.9%            |          0.01  |
|           0   | think        | asch_history_5     | 324 | 62.7%            |          0.188 |
|           0   | think        | authoritative_bias | 318 | 57.9%            |          0.021 |
|           0   | think        | control            | 243 | 32.5%            |          0.052 |
|           0   | think_sft    | asch_history_5     | 331 | 69.2%            |          0.311 |
|           0   | think_sft    | authoritative_bias | 321 | 63.6%            |          0.149 |
|           0   | think_sft    | control            | 234 | 39.7%            |          0.084 |
|           0.2 | base         | asch_history_5     | 342 | 82.2%            |          0.332 |
|           0.2 | base         | authoritative_bias | 328 | 80.5%            |          0.312 |
|           0.2 | base         | control            | 280 | 68.2%            |          0.285 |
|           0.2 | instruct     | asch_history_5     | 318 | 68.9%            |          0.186 |
|           0.2 | instruct     | authoritative_bias | 308 | 65.3%            |          0.1   |
|           0.2 | instruct     | control            | 251 | 57.8%            |          0.22  |
|           0.2 | instruct_sft | asch_history_5     | 339 | 91.7%            |          0.668 |
|           0.2 | instruct_sft | authoritative_bias | 318 | 78.6%            |          0.296 |
|           0.2 | instruct_sft | control            | 272 | 60.3%            |          0.241 |
|           0.2 | rl_zero      | asch_history_5     | 322 | 74.5%            |          0.156 |
|           0.2 | rl_zero      | authoritative_bias | 286 | 77.6%            |          0.07  |
|           0.2 | rl_zero      | control            | 157 | 21.7%            |          0.019 |
|           0.2 | think        | asch_history_5     | 329 | 63.8%            |          0.191 |
|           0.2 | think        | authoritative_bias | 323 | 55.4%            |          0.029 |
|           0.2 | think        | control            | 240 | 30.8%            |          0.04  |
|           0.2 | think_sft    | asch_history_5     | 334 | 68.9%            |          0.271 |
|           0.2 | think_sft    | authoritative_bias | 326 | 65.6%            |          0.188 |
|           0.2 | think_sft    | control            | 231 | 39.0%            |          0.08  |
|           0.4 | base         | asch_history_5     | 342 | 83.6%            |          0.437 |
|           0.4 | base         | authoritative_bias | 342 | 79.8%            |          0.244 |
|           0.4 | base         | control            | 288 | 68.4%            |          0.297 |
|           0.4 | instruct     | asch_history_5     | 323 | 67.8%            |          0.195 |
|           0.4 | instruct     | authoritative_bias | 315 | 66.3%            |          0.053 |
|           0.4 | instruct     | control            | 247 | 57.9%            |          0.201 |
|           0.4 | instruct_sft | asch_history_5     | 340 | 91.8%            |          0.663 |
|           0.4 | instruct_sft | authoritative_bias | 323 | 78.9%            |          0.253 |
|           0.4 | instruct_sft | control            | 279 | 65.9%            |          0.33  |
|           0.4 | rl_zero      | asch_history_5     | 318 | 67.3%            |          0.019 |
|           0.4 | rl_zero      | authoritative_bias | 287 | 69.3%            |          0.002 |
|           0.4 | rl_zero      | control            | 186 | 25.3%            |          0.021 |
|           0.4 | think        | asch_history_5     | 320 | 64.4%            |          0.177 |
|           0.4 | think        | authoritative_bias | 321 | 57.9%            |          0.072 |
|           0.4 | think        | control            | 244 | 30.7%            |          0.03  |
|           0.4 | think_sft    | asch_history_5     | 334 | 69.2%            |          0.293 |
|           0.4 | think_sft    | authoritative_bias | 326 | 62.3%            |          0.137 |
|           0.4 | think_sft    | control            | 238 | 39.9%            |          0.088 |
|           0.6 | base         | asch_history_5     | 337 | 83.1%            |          0.382 |
|           0.6 | base         | authoritative_bias | 337 | 81.3%            |          0.235 |
|           0.6 | base         | control            | 288 | 68.1%            |          0.287 |
|           0.6 | instruct     | asch_history_5     | 324 | 68.8%            |          0.218 |
|           0.6 | instruct     | authoritative_bias | 313 | 68.1%            |          0.07  |
|           0.6 | instruct     | control            | 244 | 59.8%            |          0.24  |
|           0.6 | instruct_sft | asch_history_5     | 338 | 92.3%            |          0.668 |
|           0.6 | instruct_sft | authoritative_bias | 330 | 81.8%            |          0.449 |
|           0.6 | instruct_sft | control            | 275 | 67.3%            |          0.338 |
|           0.6 | rl_zero      | asch_history_5     | 325 | 58.5%            |         -0.01  |
|           0.6 | rl_zero      | authoritative_bias | 288 | 67.0%            |         -0.006 |
|           0.6 | rl_zero      | control            | 187 | 26.7%            |         -0.01  |
|           0.6 | think        | asch_history_5     | 325 | 64.9%            |          0.174 |
|           0.6 | think        | authoritative_bias | 324 | 59.0%            |          0.048 |
|           0.6 | think        | control            | 233 | 30.9%            |          0.035 |
|           0.6 | think_sft    | asch_history_5     | 327 | 67.6%            |          0.28  |
|           0.6 | think_sft    | authoritative_bias | 332 | 64.5%            |          0.169 |
|           0.6 | think_sft    | control            | 239 | 40.2%            |          0.082 |
|           0.8 | base         | asch_history_5     | 350 | 66.3%            |          0.104 |
|           0.8 | base         | authoritative_bias | 348 | 64.9%            |          0.157 |
|           0.8 | base         | control            | 350 | 63.1%            |          0.14  |
|           0.8 | instruct     | asch_history_5     | 347 | 64.8%            |          0.139 |
|           0.8 | instruct     | authoritative_bias | 349 | 62.8%            |          0.049 |
|           0.8 | instruct     | control            | 345 | 63.8%            |          0.163 |
|           0.8 | instruct_sft | asch_history_5     | 349 | 83.7%            |          0.399 |
|           0.8 | instruct_sft | authoritative_bias | 348 | 69.3%            |          0.203 |
|           0.8 | instruct_sft | control            | 343 | 66.5%            |          0.292 |
|           0.8 | rl_zero      | asch_history_5     | 350 | 88.9%            |         -0.01  |
|           0.8 | rl_zero      | authoritative_bias | 349 | 87.4%            |         -0.043 |
|           0.8 | rl_zero      | control            | 350 | 86.6%            |          0.027 |
|           0.8 | think        | asch_history_5     | 346 | 57.8%            |          0.076 |
|           0.8 | think        | authoritative_bias | 348 | 59.8%            |          0.073 |
|           0.8 | think        | control            | 345 | 43.2%            |          0.046 |
|           0.8 | think_sft    | asch_history_5     | 349 | 59.6%            |          0.139 |
|           0.8 | think_sft    | authoritative_bias | 350 | 58.0%            |          0.075 |
|           0.8 | think_sft    | control            | 342 | 43.9%            |          0.054 |
|           1   | base         | asch_history_5     | 350 | 70.6%            |          0.196 |
|           1   | base         | authoritative_bias | 350 | 67.4%            |          0.124 |
|           1   | base         | control            | 350 | 64.6%            |          0.089 |
|           1   | instruct     | asch_history_5     | 347 | 62.5%            |          0.117 |
|           1   | instruct     | authoritative_bias | 348 | 64.7%            |          0.033 |
|           1   | instruct     | control            | 345 | 64.6%            |          0.197 |
|           1   | instruct_sft | asch_history_5     | 350 | 81.7%            |          0.36  |
|           1   | instruct_sft | authoritative_bias | 350 | 67.4%            |          0.199 |
|           1   | instruct_sft | control            | 346 | 68.8%            |          0.283 |
|           1   | rl_zero      | asch_history_5     | 324 | 87.7%            |          0.152 |
|           1   | rl_zero      | authoritative_bias | 320 | 86.9%            |          0.044 |
|           1   | rl_zero      | control            | 262 | 75.2%            |          0.125 |
|           1   | think        | asch_history_5     | 350 | 53.1%            |          0.054 |
|           1   | think        | authoritative_bias | 350 | 56.6%            |          0.063 |
|           1   | think        | control            | 338 | 40.2%            |          0.045 |
|           1   | think_sft    | asch_history_5     | 350 | 56.3%            |          0.085 |
|           1   | think_sft    | authoritative_bias | 350 | 58.3%            |          0.114 |
|           1   | think_sft    | control            | 340 | 45.9%            |          0.089 |

### 3.3 Judge vs Manual: `refusal_flag` agreement

- **Overall agreement:** 82.4%
- **Cohen's κ:** 0.094
- **n:** 208088

### 3.4 Judge vs Rule-based: `wrong_answer_endorsed` agreement

- **Overall agreement:** 68.0%
- **Cohen's κ:** 0.115
- **n:** 196626

#### Judge endorsement rate by cell

|   temperature | variant      | condition_name     |   n | judge_endorse_rate   |
|--------------:|:-------------|:-------------------|----:|:---------------------|
|           0   | base         | asch_history_5     | 395 | 61.3%                |
|           0   | base         | authoritative_bias | 390 | 57.7%                |
|           0   | base         | control            | 222 | 7.2%                 |
|           0   | instruct     | asch_history_5     | 375 | 28.0%                |
|           0   | instruct     | authoritative_bias | 366 | 36.3%                |
|           0   | instruct     | control            | 157 | 4.5%                 |
|           0   | instruct_sft | asch_history_5     | 384 | 65.9%                |
|           0   | instruct_sft | authoritative_bias | 367 | 54.2%                |
|           0   | instruct_sft | control            | 192 | 4.2%                 |
|           0   | rl_zero      | asch_history_5     | 368 | 22.3%                |
|           0   | rl_zero      | authoritative_bias | 339 | 24.8%                |
|           0   | rl_zero      | control            |  69 | 2.9%                 |
|           0   | think        | asch_history_5     | 377 | 24.9%                |
|           0   | think        | authoritative_bias | 373 | 18.5%                |
|           0   | think        | control            | 114 | 1.8%                 |
|           0   | think_sft    | asch_history_5     | 383 | 29.2%                |
|           0   | think_sft    | authoritative_bias | 374 | 20.9%                |
|           0   | think_sft    | control            | 125 | 2.4%                 |
|           0.2 | base         | asch_history_5     | 392 | 58.2%                |
|           0.2 | base         | authoritative_bias | 379 | 59.4%                |
|           0.2 | base         | control            | 208 | 9.1%                 |
|           0.2 | instruct     | asch_history_5     | 368 | 27.7%                |
|           0.2 | instruct     | authoritative_bias | 366 | 34.2%                |
|           0.2 | instruct     | control            | 158 | 3.8%                 |
|           0.2 | instruct_sft | asch_history_5     | 392 | 65.6%                |
|           0.2 | instruct_sft | authoritative_bias | 370 | 55.1%                |
|           0.2 | instruct_sft | control            | 189 | 3.7%                 |
|           0.2 | rl_zero      | asch_history_5     | 376 | 27.7%                |
|           0.2 | rl_zero      | authoritative_bias | 341 | 20.5%                |
|           0.2 | rl_zero      | control            |  65 | 6.2%                 |
|           0.2 | think        | asch_history_5     | 380 | 25.3%                |
|           0.2 | think        | authoritative_bias | 374 | 16.3%                |
|           0.2 | think        | control            | 108 | 0.0%                 |
|           0.2 | think_sft    | asch_history_5     | 384 | 30.7%                |
|           0.2 | think_sft    | authoritative_bias | 377 | 22.8%                |
|           0.2 | think_sft    | control            | 127 | 2.4%                 |
|           0.4 | base         | asch_history_5     | 393 | 59.0%                |
|           0.4 | base         | authoritative_bias | 393 | 56.7%                |
|           0.4 | base         | control            | 214 | 8.4%                 |
|           0.4 | instruct     | asch_history_5     | 375 | 26.7%                |
|           0.4 | instruct     | authoritative_bias | 372 | 38.4%                |
|           0.4 | instruct     | control            | 159 | 3.8%                 |
|           0.4 | instruct_sft | asch_history_5     | 390 | 66.7%                |
|           0.4 | instruct_sft | authoritative_bias | 376 | 58.8%                |
|           0.4 | instruct_sft | control            | 180 | 3.3%                 |
|           0.4 | rl_zero      | asch_history_5     | 368 | 19.6%                |
|           0.4 | rl_zero      | authoritative_bias | 332 | 17.8%                |
|           0.4 | rl_zero      | control            |  74 | 5.4%                 |
|           0.4 | think        | asch_history_5     | 371 | 25.6%                |
|           0.4 | think        | authoritative_bias | 379 | 17.2%                |
|           0.4 | think        | control            | 125 | 0.0%                 |
|           0.4 | think_sft    | asch_history_5     | 386 | 27.7%                |
|           0.4 | think_sft    | authoritative_bias | 379 | 24.5%                |
|           0.4 | think_sft    | control            | 129 | 3.1%                 |
|           0.6 | base         | asch_history_5     | 390 | 60.0%                |
|           0.6 | base         | authoritative_bias | 389 | 62.7%                |
|           0.6 | base         | control            | 210 | 5.7%                 |
|           0.6 | instruct     | asch_history_5     | 375 | 22.9%                |
|           0.6 | instruct     | authoritative_bias | 371 | 35.8%                |
|           0.6 | instruct     | control            | 156 | 5.8%                 |
|           0.6 | instruct_sft | asch_history_5     | 388 | 67.3%                |
|           0.6 | instruct_sft | authoritative_bias | 381 | 60.1%                |
|           0.6 | instruct_sft | control            | 190 | 3.7%                 |
|           0.6 | rl_zero      | asch_history_5     | 374 | 21.4%                |
|           0.6 | rl_zero      | authoritative_bias | 332 | 21.4%                |
|           0.6 | rl_zero      | control            |  67 | 1.5%                 |
|           0.6 | think        | asch_history_5     | 376 | 25.0%                |
|           0.6 | think        | authoritative_bias | 377 | 16.7%                |
|           0.6 | think        | control            | 117 | 0.0%                 |
|           0.6 | think_sft    | asch_history_5     | 377 | 27.1%                |
|           0.6 | think_sft    | authoritative_bias | 382 | 21.2%                |
|           0.6 | think_sft    | control            | 137 | 2.9%                 |
|           0.8 | base         | asch_history_5     | 394 | 48.2%                |
|           0.8 | base         | authoritative_bias | 398 | 45.0%                |
|           0.8 | base         | control            | 393 | 16.8%                |
|           0.8 | instruct     | asch_history_5     | 388 | 37.1%                |
|           0.8 | instruct     | authoritative_bias | 389 | 43.2%                |
|           0.8 | instruct     | control            | 373 | 19.8%                |
|           0.8 | instruct_sft | asch_history_5     | 388 | 60.6%                |
|           0.8 | instruct_sft | authoritative_bias | 390 | 49.5%                |
|           0.8 | instruct_sft | control            | 382 | 15.7%                |
|           0.8 | rl_zero      | asch_history_5     | 400 | 17.5%                |
|           0.8 | rl_zero      | authoritative_bias | 400 | 19.5%                |
|           0.8 | rl_zero      | control            | 396 | 27.0%                |
|           0.8 | think        | asch_history_5     | 385 | 50.1%                |
|           0.8 | think        | authoritative_bias | 384 | 50.5%                |
|           0.8 | think        | control            | 377 | 23.9%                |
|           0.8 | think_sft    | asch_history_5     | 385 | 48.8%                |
|           0.8 | think_sft    | authoritative_bias | 389 | 43.4%                |
|           0.8 | think_sft    | control            | 377 | 25.2%                |
|           1   | base         | asch_history_5     | 393 | 48.6%                |
|           1   | base         | authoritative_bias | 392 | 45.2%                |
|           1   | base         | control            | 389 | 16.2%                |
|           1   | instruct     | asch_history_5     | 388 | 32.5%                |
|           1   | instruct     | authoritative_bias | 388 | 47.2%                |
|           1   | instruct     | control            | 369 | 17.1%                |
|           1   | instruct_sft | asch_history_5     | 388 | 53.6%                |
|           1   | instruct_sft | authoritative_bias | 391 | 49.6%                |
|           1   | instruct_sft | control            | 380 | 14.2%                |
|           1   | rl_zero      | asch_history_5     | 382 | 12.0%                |
|           1   | rl_zero      | authoritative_bias | 370 | 9.7%                 |
|           1   | rl_zero      | control            |  29 | 6.9%                 |
|           1   | think        | asch_history_5     | 385 | 48.6%                |
|           1   | think        | authoritative_bias | 379 | 49.6%                |
|           1   | think        | control            | 371 | 21.6%                |
|           1   | think_sft    | asch_history_5     | 385 | 49.1%                |
|           1   | think_sft    | authoritative_bias | 385 | 53.5%                |
|           1   | think_sft    | control            | 367 | 21.0%                |

### 3.5 Paired temperature deltas (judge `is_correct`)

| condition          |   delta_error_pp |   p_mcnemar |   n_pairs |   b_1to0 |   c_0to1 |
|:-------------------|-----------------:|------------:|----------:|---------:|---------:|
| control            |             9.27 |    1.07e-10 |      1833 |      431 |      261 |
| asch_history_5     |            -4.19 |    0.000479 |      2531 |      400 |      506 |
| authoritative_bias |            -3.53 |    0.00682  |      2406 |      440 |      525 |

