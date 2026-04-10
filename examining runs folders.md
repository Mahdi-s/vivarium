# Examining runs folders

SQLite inventory for **`runs/`** (top-level and **`runs/think/`**), **`runs_latest/runs/`**, and **`runs-think-hpc/`** (7B runs at the **root** of that tree; 32B runs under **`runs-think-hpc/runs-32B/`** — see [§ runs-think-hpc](#runsthink-hpc)). Judge columns use the same rules as **`scripts/judge_report.sh`**: *judge valid* = `parsed_answer_json` contains `$.is_correct` without `[parse_error]`; *match* = agreement with manual `is_correct` on comparable rows.

## Executive summary

| Tree | Role |
| :--- | :--- |
| **`runs/`** (flat) | Cross-model baselines (Llama, Gemini, GPT-4o mini, Grok, Claude, …) plus **OLMo 3.1 32B** instruct/think API suites. Most runs: **1600** trials (8 datasets × 4 core conditions × 50 items). **Two runs at 800 trials** = abolition / system-prompt ablations. One **gpt-oss** run has **duplicate `conformity_outputs` rows** (187 extra) — trials are still 1600. |
| **`runs/think/`** | Intended as **only** **`allenai/Olmo-3-7B-Think`** (7B **Think**, not SFT). Embedded `runs.config_json` lists that single model. The DB still has **data-quality issues**: see [runs/think: intended model vs. row-level quirks](#runsthink-intended-model-vs-row-level-quirks). |
| **`runs_latest/runs/`** | **`suite_7b_expanded`**: each **variant** is **8 × 12 × 50 = 4,800** trials when complete; **8 variants** ⇒ **38,400** total if every cell is full. Observed **~34.8k–38.2k** per temperature DB (missing cells, **`rl_zero`** short, occasional **`think_dpo`** gaps). **Think** completions are often truncated (low `max_new_tokens`); publication-style analysis usually uses **base, instruct, instruct_sft, instruct_dpo** and treats Think columns cautiously. |
| **`runs-think-hpc/`** | **Eight** HPC runs in the intended layout: **four** **7B** ThinkHPC databases **directly under** `runs-think-hpc/<timestamp>_<uuid>/` (**Think-SFT** and **Think-DPO**, temp **0** and **0.6**); **four** **32B** databases under **`runs-think-hpc/runs-32B/`** (same design). Below, **32B** is audited from SQLite; **7B** IDs are listed from `HPC_RUNS_OVERVIEW.md` because the 7B run folders are **not** checked in here yet. |

For heavier multi-step re-judging, aggregation, or paper pipelines beyond raw SQL, [K-Dense Web](https://www.k-dense.ai) can orchestrate workflows that outgrow ad-hoc scripts.

## LLM judge: “not 100% judged” vs your judge progress bar

**You are not missing judge passes** in the usual sense. The pipeline (`olmo-conformity-judgeval` / OpenRouter) **replaces** `conformity_outputs.parsed_answer_json` with a merged object that includes **`_llm_judge`** (model, API, prompt version) when the LLM judge finishes. On sampled databases, **every output row** (or effectively all) has **`_llm_judge`** — meaning the judge **did run**.

**What the inventory calls “judge valid”** matches **`scripts/judge_report.sh`**: it only counts rows where **`json_extract(parsed_answer_json, '$.is_correct') IS NOT NULL`**. That is **stricter** than “judge ran”.

The LLM judge **intentionally** sets **`"is_correct": null`** in JSON when **`_normalise_labels`** in `src/vivarium/experiments/olmo_conformity/ollama_judge.py` decides there is **no applicable binary correctness**, for example:

- **No reference answer** (opinion / open-ended items: `ground_truth_text` is null).
- **Incomplete model output** — notes such as `"incomplete answer"`, `"no final answer"`.
- **Refusal** — `refusal_flag` may be **1** while `is_correct` stays **null** (nothing to grade as right/wrong).
- Think-specific fields can be filled while **`is_correct`** remains **null**.

So your **progress script** can correctly report “all rows processed” (every row got a judge **update** with `_llm_judge`), while **`judge_report`** still shows a **gap** because **`$.is_correct` is JSON null** on those rows — **not** because `parsed_answer_json` was left empty.

**`conformity_outputs.is_correct`** (the column) is often still **0/1** from **heuristic / manual** scoring and can disagree with the JSON: you will see rows where the **column** is set but **`$.is_correct`** in JSON is **null** after the judge.

**If you need a single “judge completed” metric**, count rows with **`json_extract(parsed_answer_json, '$._llm_judge') IS NOT NULL`** (or treat `is_correct` null as “N/A” rather than “missing judge”).

## Reference: expected trial counts

| Design | `n_trials` if complete |
| :--- | ---: |
| 8 datasets × 4 conditions × 50 items × **1** model | 1600 |
| Same × **1** model, half grid (abolition) | 800 |
| `suite_7b_expanded`: 8 models × 8 × 12 × 50 | 38400 |

## Section tables

### runs/ (top-level, excluding `think/`)

**Path:** `runs` — **22** `simulation.db` file(s).

| Folder | run_id | Expected | Trials | Dup outputs | Missing out | Judge valid | Judge % | Comparable | Match % | Suite (short) | Data complete |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d` | `a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d` | 1600 | 1600 | 0 | 0 | 1565 | 97.8% | 1368 | 54.6% | `olmo_conformity_32b_think…` | yes |
| `20260327_152926_7db9896e-9e3b-439f-88e3-74fe25ea2bad` | `7db9896e-9e3b-439f-88e3-74fe25ea2bad` | 1600 | 1600 | 0 | 0 | 1570 | 98.1% | 1372 | 54.7% | `olmo_conformity_32b_think…` | yes |
| `20260327_152936_1c2e5cb6-0372-4835-bbb7-7230c55517e4` | `1c2e5cb6-0372-4835-bbb7-7230c55517e4` | 1600 | 1600 | 0 | 0 | 1506 | 94.1% | 1313 | 64.4% | `olmo_conformity_32b_instr…` | yes |
| `20260327_152944_62187f52-7a7e-4db0-a269-d14d8e887b1b` | `62187f52-7a7e-4db0-a269-d14d8e887b1b` | 1600 | 1600 | 0 | 0 | 1505 | 94.1% | 1314 | 63.5% | `olmo_conformity_32b_instr…` | yes |
| `20260327_154349_1899a883-82e4-45f3-833a-d6403cf1ac95` | `1899a883-82e4-45f3-833a-d6403cf1ac95` | 1600 | 1600 | 0 | 0 | 1410 | 88.1% | 1223 | 76.0% | `olmo_conformity_llama3_8b…` | yes |
| `20260327_154401_70860876-c5c2-4445-a59d-e44ae8094887` | `70860876-c5c2-4445-a59d-e44ae8094887` | 1600 | 1600 | 0 | 0 | 1439 | 89.9% | 1245 | 78.2% | `olmo_conformity_llama3_8b…` | yes |
| `20260327_154412_3a0404f7-bd47-4b25-b2e2-5501e550566f` | `3a0404f7-bd47-4b25-b2e2-5501e550566f` | 1600 | 1600 | 0 | 0 | 1447 | 90.4% | 1263 | 66.7% | `olmo_conformity_llama3.1_…` | yes |
| `20260327_154419_49d07104-c14c-4a8b-a013-ae0783c5f3e8` | `49d07104-c14c-4a8b-a013-ae0783c5f3e8` | 1600 | 1600 | 0 | 0 | 1460 | 91.2% | 1268 | 63.8% | `olmo_conformity_llama3.1_…` | yes |
| `20260327_154428_485ddc2d-6cae-4715-835e-76ab72e38159` | `485ddc2d-6cae-4715-835e-76ab72e38159` | 1600 | 1600 | 0 | 0 | 1579 | 98.7% | 1382 | 60.5% | `olmo_conformity_llama4_ma…` | yes |
| `20260327_154435_c2ce0f85-8f67-40f2-a82d-e136927cf6f5` | `c2ce0f85-8f67-40f2-a82d-e136927cf6f5` | 1600 | 1600 | 0 | 0 | 1581 | 98.8% | 1383 | 62.0% | `olmo_conformity_llama4_ma…` | yes |
| `20260327_224321_e043fbf6-27eb-410c-8da7-bc0f9172ab0b` | `e043fbf6-27eb-410c-8da7-bc0f9172ab0b` | 1600 | 1600 | 0 | 0 | 1457 | 91.1% | 1289 | 62.8% | `olmo_conformity_gemini_2.…` | yes |
| `20260327_224336_d71e75b1-17c5-4789-8ee7-13b29ef18359` | `d71e75b1-17c5-4789-8ee7-13b29ef18359` | 1600 | 1600 | 0 | 0 | 1441 | 90.1% | 1276 | 62.1% | `olmo_conformity_gemini_2.…` | yes |
| `20260327_224348_25056752-7081-449e-9a44-ad090b566107` | `25056752-7081-449e-9a44-ad090b566107` | 1600 | 1600 | 0 | 0 | 1546 | 96.6% | 1353 | 61.0% | `olmo_conformity_grok_4.1_…` | yes |
| `20260327_224357_157a6a9e-13de-4bdb-bdd2-54d761498f24` | `157a6a9e-13de-4bdb-bdd2-54d761498f24` | 1600 | 1600 | 0 | 0 | 1550 | 96.9% | 1358 | 61.1% | `olmo_conformity_grok_4.1_…` | yes |
| `20260327_224413_c07ede3a-16ac-4b47-ac42-4f6ad8dd8370` | `c07ede3a-16ac-4b47-ac42-4f6ad8dd8370` | 1600 | 1600 | 0 | 0 | 1575 | 98.4% | 1375 | 67.2% | `olmo_conformity_gpt4o_min…` | yes |
| `20260327_224422_eb63d212-77fe-46ef-965b-7777cc232f1f` | `eb63d212-77fe-46ef-965b-7777cc232f1f` | 1600 | 1600 | 0 | 0 | 1575 | 98.4% | 1375 | 67.2% | `olmo_conformity_gpt4o_min…` | yes |
| `20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5` | `66765d5e-204c-4074-aaf4-b9c148fe61a5` | 1600 | 1600 | 187 | 0 | 1345 | 84.1% | 1246 | 57.9% | `olmo_conformity_gpt_oss_2…` | no |
| `20260327_224603_3ecdc9b7-49db-4625-b90e-fc3745b9224e` | `3ecdc9b7-49db-4625-b90e-fc3745b9224e` | 1600 | 1600 | 0 | 0 | 1505 | 94.1% | 1323 | 55.1% | `olmo_conformity_gpt_oss_2…` | yes |
| `20260329_211511_5be5ada7-64be-4cbd-9024-aacbcaf233e3` | `5be5ada7-64be-4cbd-9024-aacbcaf233e3` | 1600 | 1600 | 0 | 0 | 1474 | 92.1% | 1474 | 62.1% | `olmo_conformity_claude_so…` | yes |
| `20260329_211518_21556460-5f97-4a23-8c54-4a1f999ba619` | `21556460-5f97-4a23-8c54-4a1f999ba619` | 1600 | 1600 | 0 | 0 | 1465 | 91.6% | 1465 | 63.1% | `olmo_conformity_claude_so…` | yes |
| `20260329_235403_e8a90500-25cd-469f-a138-197c338fddaf` | `e8a90500-25cd-469f-a138-197c338fddaf` | 800 | 800 | 0 | 0 | 795 | 99.4% | 696 | 72.1% | `olmo_conformity_llama3.1_…` | yes |
| `20260329_235408_ef72529e-5e82-463f-8b8b-b2a6c7decd3c` | `ef72529e-5e82-463f-8b8b-b2a6c7decd3c` | 800 | 800 | 0 | 0 | 788 | 98.5% | 688 | 72.2% | `olmo_conformity_32b_instr…` | yes |

### runs/think/

**Path:** `runs/think` — **1** `simulation.db` file(s).

**Intended checkpoint:** **`allenai/Olmo-3-7B-Think`** only (7B **Think**, not SFT). Embedded `runs.config_json` lists a **single** `suite_config.models` entry for that checkpoint.

| Folder | run_id | Expected | Trials | Dup outputs | Missing out | Judge valid | Judge % | Comparable | Match % | Suite (short) | Data complete |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37` | `f47fe05e-4564-4680-a2d8-39a88c6f8d37` | 1600 | 1609 | 0 | 1 | 1409 | 87.6% | 1224 | 60.6% | `olmo_conformity_think_auto` | no |

#### runs/think: intended model vs. row-level quirks

What the **run config** says (authoritative for intent): one model — **`Olmo-3-7B-Think`**.

What **`conformity_trials` rows** show (stale / mistaken data):

| Issue | Detail |
| :--- | :--- |
| **Stray Think-SFT rows** | **2** rows use `variant = think_sft` and `model_id = allenai/Olmo-3-7B-Think-SFT`. Both reference the **same** `item_id` (`mmlu_high_school_geography_0000`). These should **not** be read as an SFT run; they are almost certainly erroneous inserts. |
| **Extra Think rows** | **1607** rows are `think` / `Olmo-3-7B-Think`. Multiple dataset×condition cells have **51–52** trials instead of **50** (duplicate logical items), so the table is not a clean 32×50 grid. |
| **Missing output** | **1** trial has no matching `conformity_outputs` row. |

**How to address**

1. **Analysis-only:** query `WHERE variant = 'think' AND model_id = 'allenai/Olmo-3-7B-Think'` and, if you need exactly **1600** logical trials, **deduplicate** on `(dataset, condition, item_id)` and drop the stray `think_sft` rows from any aggregate.  
2. **DB cleanup (after backup):** `DELETE` the two `think_sft` trials (and dependent child rows per your FK rules).  
3. **Duplicates:** use `GROUP BY` on `(item_id, condition_id)` (or logical keys) where `COUNT(*) > 1` and remove surplus `trial_id`s.  
4. **Missing output:** re-run inference for that `trial_id` or remove the orphan trial.  
5. **Nuclear option:** re-run the suite to a fresh run directory and archive this DB.

### runs_latest/runs/

**Path:** `runs_latest/runs` — **6** `simulation.db` file(s).

| Folder | run_id | Expected | Trials | Dup outputs | Missing out | Judge valid | Judge % | Comparable | Match % | Suite (short) | Data complete |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `9f240f89-e58e-423a-ae68-f990b31c84cd` | 38400 | 34794 | 0 | 0 | 31535 | 90.6% | 27375 | 78.6% | `olmo_conformity_expanded_…` | no |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `46f0762a-210a-459a-8709-d24a0f194eb0` | 38400 | 34746 | 0 | 0 | 31607 | 91.0% | 27437 | 79.2% | `olmo_conformity_expanded_…` | no |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `bbd05985-d185-460a-b0aa-dd356d27ec94` | 38400 | 37978 | 0 | 0 | 34887 | 91.9% | 30282 | 80.1% | `olmo_conformity_expanded_…` | no |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `86c72262-d1aa-41b5-9c22-d7b2e0570215` | 38400 | 38170 | 0 | 0 | 35148 | 92.1% | 30544 | 80.3% | `olmo_conformity_expanded_…` | no |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `9369442d-d825-4cd0-81a1-8ed276c37814` | 38400 | 34800 | 0 | 0 | 32318 | 92.9% | 28538 | 78.8% | `olmo_conformity_expanded_…` | no |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | 38400 | 34800 | 0 | 0 | 32146 | 92.4% | 28388 | 77.8% | `olmo_conformity_expanded_…` | no |

**Per-variant trial counts** (runs_latest):

| Folder | variant | model_id (short) | trials |
| :--- | :--- | :--- | ---: |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 1200 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4794 |
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 1200 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4746 |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 4378 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4800 |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 4570 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4800 |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 1200 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4800 |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `base` | `allenai/Olmo-3-1025-7B` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `instruct` | `allenai/Olmo-3-7B-Instruct` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `instruct_dpo` | `allenai/Olmo-3-7B-Instruct-DPO` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `instruct_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `rl_zero` | `allenai/Olmo-3-7B-RL-Zero-Math` | 1200 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `think` | `allenai/Olmo-3-7B-Think` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `think_dpo` | `allenai/Olmo-3-7B-Think-DPO` | 4800 |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `think_sft` | `allenai/Olmo-3-7B-Think-SFT` | 4800 |

### runs-think-hpc

#### Layout (7B top-level vs 32B under `runs-32B/`)

| Tier | Path | Role |
| :--- | :--- | :--- |
| **7B ThinkHPC** | `runs-think-hpc/<YYYYMMDD_HHMMSS>_<uuid>/` | **Four** runs: **Think-SFT** and **Think-DPO** on **`allenai/Olmo-3-7B-Think-SFT`** / **`allenai/Olmo-3-7B-Think-DPO`**, temps **0.0** and **0.6**, **1600** trials each when complete (core grid). |
| **32B ThinkHPC** | `runs-think-hpc/runs-32B/<YYYYMMDD_HHMMSS>_<uuid>/` | **Four** runs: **Think-SFT** and **Think-DPO** on **`allenai/Olmo-3-32B-Think-SFT`** / **`allenai/Olmo-3-32B-Think-DPO`**, same temperatures, **1600** trials each when complete. |

#### 7B ThinkHPC (top-level) — canonical `run_id`s

These are the **ThinkHPC 7B** runs (folders sit **next to** `runs-32B/`, not inside it). Suite names and folder names are recorded in `runs-think-hpc/HPC_RUNS_OVERVIEW.md`.

| `run_id` | `suite_name` |
| :--- | :--- |
| `e5cc991d-eb01-4371-b73c-315ab18c3112` | `olmo_conformity_7b_think_sft_temp0.0` |
| `29ac502b-887d-4ed1-b161-7441ac22188c` | `olmo_conformity_7b_think_sft_temp0.6` |
| `56c71fda-573f-46fd-b2ab-8b6e509c689a` | `olmo_conformity_7b_think_dpo_temp0.0` |
| `57414a36-f63a-4332-9c70-437a60071565` | `olmo_conformity_7b_think_dpo_temp0.6` |

**This workspace:** the four **7B** `simulation.db` trees are **missing** from `runs-think-hpc/` (only **`runs-32B/`** and markdown are present). Copy the run directories from HPC, then re-run the audit to add per-DB metrics.

#### 32B (`runs-think-hpc/runs-32B/`)

**Path:** `runs-think-hpc/runs-32B` — **4** `simulation.db` file(s).

| Folder | run_id | Expected | Trials | Dup outputs | Missing out | Judge valid | Judge % | Comparable | Match % | Suite (short) | Data complete |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `20260404_093315_81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `81d9194a-b1ef-4261-a0fb-bb0f713e1239` | 1600 | 1600 | 0 | 0 | 1511 | 94.4% | 1318 | 60.9% | `olmo_conformity_32b_think…` | yes |
| `20260404_093744_e673de86-d8dc-4ce0-bfc9-969d531eb425` | `e673de86-d8dc-4ce0-bfc9-969d531eb425` | 1600 | 1600 | 0 | 0 | 1521 | 95.1% | 1326 | 60.5% | `olmo_conformity_32b_think…` | yes |
| `20260404_094925_d0158f56-c99c-4e7d-92fc-c7ece3190781` | `d0158f56-c99c-4e7d-92fc-c7ece3190781` | 1600 | 1600 | 0 | 0 | 1470 | 91.9% | 1281 | 58.5% | `olmo_conformity_32b_think…` | yes |
| `20260404_100941_3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `3041fb7e-98bc-4343-90d7-e56d9e134a3b` | 1600 | 1600 | 0 | 0 | 1468 | 91.8% | 1284 | 57.6% | `olmo_conformity_32b_think…` | yes |

## Unique models (`model_id` + `temperature`)

| model_id | Temp | # DBs |
| :--- | :--- | ---: |
| `allenai/Olmo-3-1025-7B` | 0.0 | 1 |
| `allenai/Olmo-3-1025-7B` | 0.2 | 1 |
| `allenai/Olmo-3-1025-7B` | 0.4 | 1 |
| `allenai/Olmo-3-1025-7B` | 0.6 | 1 |
| `allenai/Olmo-3-1025-7B` | 0.8 | 1 |
| `allenai/Olmo-3-1025-7B` | 1.0 | 1 |
| `allenai/Olmo-3-32B-Think-DPO` | 0.0 | 1 |
| `allenai/Olmo-3-32B-Think-DPO` | 0.6 | 1 |
| `allenai/Olmo-3-32B-Think-SFT` | 0.0 | 1 |
| `allenai/Olmo-3-32B-Think-SFT` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 0.0 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Instruct` | 1.0 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.0 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 1.0 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.0 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 1.0 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 0.0 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 0.2 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 0.4 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 0.6 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 0.8 | 1 |
| `allenai/Olmo-3-7B-RL-Zero-Math` | 1.0 | 1 |
| `allenai/Olmo-3-7B-Think` | 0.0 | 2 |
| `allenai/Olmo-3-7B-Think` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Think` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Think` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Think` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Think` | 1.0 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 0.0 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Think-DPO` | 1.0 | 1 |
| `allenai/Olmo-3-7B-Think-SFT` | 0.0 | 2 |
| `allenai/Olmo-3-7B-Think-SFT` | 0.2 | 1 |
| `allenai/Olmo-3-7B-Think-SFT` | 0.4 | 1 |
| `allenai/Olmo-3-7B-Think-SFT` | 0.6 | 1 |
| `allenai/Olmo-3-7B-Think-SFT` | 0.8 | 1 |
| `allenai/Olmo-3-7B-Think-SFT` | 1.0 | 1 |
| `allenai/olmo-3.1-32b-instruct` | 0.0 | 2 |
| `allenai/olmo-3.1-32b-instruct` | 0.6 | 1 |
| `allenai/olmo-3.1-32b-think` | 0.0 | 1 |
| `allenai/olmo-3.1-32b-think` | 0.6 | 1 |
| `anthropic/claude-sonnet-4` | 0.0 | 1 |
| `anthropic/claude-sonnet-4` | 0.6 | 1 |
| `google/gemini-2.5-flash-lite` | 0.0 | 1 |
| `google/gemini-2.5-flash-lite` | 0.6 | 1 |
| `meta-llama/llama-3-8b-instruct` | 0.0 | 1 |
| `meta-llama/llama-3-8b-instruct` | 0.6 | 1 |
| `meta-llama/llama-3.1-70b-instruct` | 0.0 | 2 |
| `meta-llama/llama-3.1-70b-instruct` | 0.6 | 1 |
| `meta-llama/llama-4-maverick` | 0.0 | 1 |
| `meta-llama/llama-4-maverick` | 0.6 | 1 |
| `openai/gpt-4o-mini` | 0.0 | 1 |
| `openai/gpt-4o-mini` | 0.6 | 1 |
| `openai/gpt-oss-20b` | 0.0 | 1 |
| `openai/gpt-oss-20b` | 0.6 | 1 |
| `x-ai/grok-4.1-fast` | 0.0 | 1 |
| `x-ai/grok-4.1-fast` | 0.6 | 1 |

*Note on counts of **2**:* `allenai/Olmo-3-7B-Think` at temp **0.0** appears in both **`runs/think`** and **`runs_latest`**. For **`allenai/Olmo-3-7B-Think-SFT`** at **0.0**, one database is the expanded-suite run under **`runs_latest`**; another source of rows is the **two stray trials** in **`runs/think`** (see [quirks](#runsthink-intended-model-vs-row-level-quirks)). When the **four** **7B ThinkHPC** databases under **`runs-think-hpc/`** are synced, expect **two** more `simulation.db` files at temps **0.0** and **0.6** for **Think-SFT** (and similarly **Think-DPO** is separate runs).*

## Histogram: trial counts

| n_trials | # databases |
| ---: | ---: |
| 38170 | 1 |
| 37978 | 1 |
| 34800 | 2 |
| 34794 | 1 |
| 34746 | 1 |
| 1609 | 1 |
| 1600 | 24 |
| 800 | 2 |

## Needs attention

### Structural (missing output, trial count ≠ expected, duplicate output rows)

| Path | Notes |
| :--- | :--- |
| `runs/20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5/simulation.db` | 187 duplicate output rows |
| `runs/think/20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37/simulation.db` | 1 missing output; 1609 trials vs 1600 expected; 2 stray `think_sft` rows; duplicate items in several cells (see [quirks](#runsthink-intended-model-vs-row-level-quirks)) |
| `runs_latest/runs/20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd/simulation.db` | trials 34794 vs expected 38400 |
| `runs_latest/runs/20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0/simulation.db` | trials 34746 vs expected 38400 |
| `runs_latest/runs/20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94/simulation.db` | trials 37978 vs expected 38400 |
| `runs_latest/runs/20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215/simulation.db` | trials 38170 vs expected 38400 |
| `runs_latest/runs/20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814/simulation.db` | trials 34800 vs expected 38400 |
| `runs_latest/runs/20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e/simulation.db` | trials 34800 vs expected 38400 |

### Judge coverage gap (trials − judge_valid)

| Path | trials | judge_valid | gap |
| :--- | ---: | ---: | ---: |
| `runs/20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d/simulation.db` | 1600 | 1565 | 35 |
| `runs/20260327_152926_7db9896e-9e3b-439f-88e3-74fe25ea2bad/simulation.db` | 1600 | 1570 | 30 |
| `runs/20260327_152936_1c2e5cb6-0372-4835-bbb7-7230c55517e4/simulation.db` | 1600 | 1506 | 94 |
| `runs/20260327_152944_62187f52-7a7e-4db0-a269-d14d8e887b1b/simulation.db` | 1600 | 1505 | 95 |
| `runs/20260327_154349_1899a883-82e4-45f3-833a-d6403cf1ac95/simulation.db` | 1600 | 1410 | 190 |
| `runs/20260327_154401_70860876-c5c2-4445-a59d-e44ae8094887/simulation.db` | 1600 | 1439 | 161 |
| `runs/20260327_154412_3a0404f7-bd47-4b25-b2e2-5501e550566f/simulation.db` | 1600 | 1447 | 153 |
| `runs/20260327_154419_49d07104-c14c-4a8b-a013-ae0783c5f3e8/simulation.db` | 1600 | 1460 | 140 |
| `runs/20260327_154428_485ddc2d-6cae-4715-835e-76ab72e38159/simulation.db` | 1600 | 1579 | 21 |
| `runs/20260327_154435_c2ce0f85-8f67-40f2-a82d-e136927cf6f5/simulation.db` | 1600 | 1581 | 19 |
| `runs/20260327_224321_e043fbf6-27eb-410c-8da7-bc0f9172ab0b/simulation.db` | 1600 | 1457 | 143 |
| `runs/20260327_224336_d71e75b1-17c5-4789-8ee7-13b29ef18359/simulation.db` | 1600 | 1441 | 159 |
| `runs/20260327_224348_25056752-7081-449e-9a44-ad090b566107/simulation.db` | 1600 | 1546 | 54 |
| `runs/20260327_224357_157a6a9e-13de-4bdb-bdd2-54d761498f24/simulation.db` | 1600 | 1550 | 50 |
| `runs/20260327_224413_c07ede3a-16ac-4b47-ac42-4f6ad8dd8370/simulation.db` | 1600 | 1575 | 25 |
| `runs/20260327_224422_eb63d212-77fe-46ef-965b-7777cc232f1f/simulation.db` | 1600 | 1575 | 25 |
| `runs/20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5/simulation.db` | 1600 | 1345 | 255 |
| `runs/20260327_224603_3ecdc9b7-49db-4625-b90e-fc3745b9224e/simulation.db` | 1600 | 1505 | 95 |
| `runs/20260329_211511_5be5ada7-64be-4cbd-9024-aacbcaf233e3/simulation.db` | 1600 | 1474 | 126 |
| `runs/20260329_211518_21556460-5f97-4a23-8c54-4a1f999ba619/simulation.db` | 1600 | 1465 | 135 |
| `runs/20260329_235403_e8a90500-25cd-469f-a138-197c338fddaf/simulation.db` | 800 | 795 | 5 |
| `runs/20260329_235408_ef72529e-5e82-463f-8b8b-b2a6c7decd3c/simulation.db` | 800 | 788 | 12 |
| `runs/think/20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37/simulation.db` | 1609 | 1409 | 200 |
| `runs_latest/runs/20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd/simulation.db` | 34794 | 31535 | 3259 |
| `runs_latest/runs/20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0/simulation.db` | 34746 | 31607 | 3139 |
| `runs_latest/runs/20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94/simulation.db` | 37978 | 34887 | 3091 |
| `runs_latest/runs/20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215/simulation.db` | 38170 | 35148 | 3022 |
| `runs_latest/runs/20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814/simulation.db` | 34800 | 32318 | 2482 |
| `runs_latest/runs/20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e/simulation.db` | 34800 | 32146 | 2654 |
| `runs-think-hpc/runs-32B/20260404_093315_81d9194a-b1ef-4261-a0fb-bb0f713e1239/simulation.db` | 1600 | 1511 | 89 |
| `runs-think-hpc/runs-32B/20260404_093744_e673de86-d8dc-4ce0-bfc9-969d531eb425/simulation.db` | 1600 | 1521 | 79 |
| `runs-think-hpc/runs-32B/20260404_094925_d0158f56-c99c-4e7d-92fc-c7ece3190781/simulation.db` | 1600 | 1470 | 130 |
| `runs-think-hpc/runs-32B/20260404_100941_3041fb7e-98bc-4343-90d7-e56d9e134a3b/simulation.db` | 1600 | 1468 | 132 |

*Regenerate: run the same Python audit used for this commit (SQL aligned with `scripts/judge_report.sh`).*