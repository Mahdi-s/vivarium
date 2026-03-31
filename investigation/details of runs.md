# Details of Runs

Generated from `simulation.db` files in: `runs/`, `runs/think/`, `runs_latest/runs/`.

**Last updated:** 2026-03-31 — reflects post-cleanup state of `gpt-oss-20b` T=0.0 run and removal of phantom runs from earlier draft.

**Completeness definitions:**
- **Trials** compares `actual_trials` vs `expected_trials` from run config dimensions (`models × conditions × datasets × max_items_per_dataset`).
- **Clean outputs** counts `COUNT(DISTINCT trial_id)` where `raw_text NOT LIKE '<error>%'` — the only reliable measure of usable data.
- **LLM judge response** counts rows where `conformity_outputs.parsed_answer_json` contains `_llm_judge.judge_model`, compared against expected trials.

---

## `runs/` — Cross-Family Study

### Condition Mix

All runs in `runs/` (except ablation runs) use the same 4 conditions:

| Condition Name | Type | Description |
|---|---|---|
| `control` | Baseline | No social pressure; plain question only |
| `asch_zhu_unanimous_confident` | Peer pressure | 5 confederates, unanimous consensus, confident tone |
| `authoritative_bias` | Authority | High-strength user authority claim |
| `authority_trust` | Authority | System-prompt authority claim (trust framing) |

**Ablation runs only** use 2 different conditions (see Ablation section below).

### Design

- 8 datasets × 50 items each = 400 items per condition
- 4 conditions × 1 model × 400 items = **1,600 expected trials per run**
- Two temperatures per model: T=0.0 (greedy, primary) and T=0.6 (supplementary)

### Run Registry — Primary Cross-Family Runs

| Run Folder | Run ID | Model ID | Temp | Trials | Clean Outputs | Status |
|---|---|---|---|---:|---:|---|
| `20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d` | `a34ad9b1` | `allenai/olmo-3.1-32b-think` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_152926_7db9896e-9e3b-439f-88e3-74fe25ea2bad` | `7db9896e` | `allenai/olmo-3.1-32b-think` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_152936_1c2e5cb6-0372-4835-bbb7-7230c55517e4` | `1c2e5cb6` | `allenai/olmo-3.1-32b-instruct` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_152944_62187f52-7a7e-4db0-a269-d14d8e887b1b` | `62187f52` | `allenai/olmo-3.1-32b-instruct` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_154349_1899a883-82e4-45f3-833a-d6403cf1ac95` | `1899a883` | `meta-llama/llama-3-8b-instruct` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_154401_70860876-c5c2-4445-a59d-e44ae8094887` | `70860876` | `meta-llama/llama-3-8b-instruct` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_154412_3a0404f7-bd47-4b25-b2e2-5501e550566f` | `3a0404f7` | `meta-llama/llama-3.1-70b-instruct` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_154419_49d07104-c14c-4a8b-a013-ae0783c5f3e8` | `49d07104` | `meta-llama/llama-3.1-70b-instruct` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_154428_485ddc2d-6cae-4715-835e-76ab72e38159` | `485ddc2d` | `meta-llama/llama-4-maverick` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_154435_c2ce0f85-8f67-40f2-a82d-e136927cf6f5` | `c2ce0f85` | `meta-llama/llama-4-maverick` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_224321_e043fbf6-27eb-410c-8da7-bc0f9172ab0b` | `e043fbf6` | `google/gemini-2.5-flash-lite` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_224336_d71e75b1-17c5-4789-8ee7-13b29ef18359` | `d71e75b1` | `google/gemini-2.5-flash-lite` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_224348_25056752-7081-449e-9a44-ad090b566107` | `25056752` | `x-ai/grok-4.1-fast` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_224357_157a6a9e-13de-4bdb-bdd2-54d761498f24` | `157a6a9e` | `x-ai/grok-4.1-fast` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_224413_c07ede3a-16ac-4b47-ac42-4f6ad8dd8370` | `c07ede3a` | `openai/gpt-4o-mini` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260327_224422_eb63d212-77fe-46ef-965b-7777cc232f1f` | `eb63d212` | `openai/gpt-4o-mini` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5` | `66765d5e` | `openai/gpt-oss-20b` | 0.0 | 1600 | **1413** | ⚠️ Pending re-run (187 rate-limit errors) |
| `20260327_224603_3ecdc9b7-49db-4625-b90e-fc3745b9224e` | `3ecdc9b7` | `openai/gpt-oss-20b` | 0.6 | 1600 | 1600 | ✅ Complete |
| `20260329_211511_5be5ada7-64be-4cbd-9024-aacbcaf233e3` | `5be5ada7` | `anthropic/claude-sonnet-4` | 0.0 | 1600 | 1600 | ✅ Complete |
| `20260329_211518_21556460-5f97-4a23-8c54-4a1f999ba619` | `21556460` | `anthropic/claude-sonnet-4` | 0.6 | 1600 | 1600 | ✅ Complete |

> **Note on `66765d5e` (gpt-oss-20b T=0.0):** This database was corrupted by two concurrent processes writing to the same `simulation.db` (producing 1,145 duplicate outputs) and 681 rate-limit errors from OpenRouter. The DB was cleaned in-place on 2026-03-31: duplicates removed (keeping oldest valid output per trial), error outputs dropped where a valid sibling existed. 1,413 trials have one clean output; 187 trials retain their error output as a placeholder pending re-run via `--resume-auto`. The `--resume-auto` detection logic was also fixed (commit `063d22c` on `vvm_rework`) to count only distinct non-error outputs so the run is correctly identified as incomplete.

### Run Registry — Ghost Run

| Run Folder | Run ID | Model ID | Temp | Trials | Outputs | Status |
|---|---|---|---|---:|---:|---|
| `20260330_172832_621a7698-8e4f-4490-94d1-04f26090e714` | `621a7698` | `anthropic/claude-sonnet-4` | 0.6 | 1600 | 0 | 🔴 Ghost — trials registered, no LLM calls made |

> `621a7698` was created when the async build phase (trial registration) completed but the process was killed before the fan-out phase (LLM calls) began. It is a duplicate of `21556460` (same model, same temp, already complete). It should be deleted or ignored.

### Run Registry — Ablation Study Runs

**Purpose:** These two runs test two additional conditions not present in the main 4-condition suite. They provide an upper-bound baseline (no system prompt = naked Asch) and a null-model baseline (n-gram pattern completion without semantic content).

| Condition Name | Type | Description |
|---|---|---|
| `asch_zhu_naked_unanimous_confident` | Peer pressure (no system prompt) | Same 5-confederate Asch setup as `asch_zhu_unanimous_confident` but with **no system prompt**, testing whether the model's system prompt provides any protection |
| `ngram_sequence_baseline` | Null model | Presents the peer consensus as a sequence-completion task (no adversarial framing) — measures how much of the conformity effect is pure next-token pattern completion |

**Design:** 2 conditions × 400 items × 1 model = **800 expected trials per run**

| Run Folder | Run ID | Model ID | Temp | Conditions | Trials | Clean Outputs | Status |
|---|---|---|---|---|---:|---:|---|
| `20260329_235403_e8a90500-25cd-469f-a138-197c338fddaf` | `e8a90500` | `meta-llama/llama-3.1-70b-instruct` | 0.0 | `asch_zhu_naked_unanimous_confident`, `ngram_sequence_baseline` | 800 | 800 | ✅ Complete |
| `20260329_235408_ef72529e-5e82-463f-8b8b-b2a6c7decd3c` | `ef72529e` | `allenai/olmo-3.1-32b-instruct` | 0.0 | `asch_zhu_naked_unanimous_confident`, `ngram_sequence_baseline` | 800 | 800 | ✅ Complete |

> These runs are **not incomplete**. The 800-trial count is the correct expected total for 2 conditions × 400 items. They are intentional ablation studies.

---

## `runs/think/` — OLMo-7B-Think Exploratory Run

This directory holds a series of attempted runs for `allenai/Olmo-3-7B-Think`. Most attempts failed or were interrupted (see `runs_index.json`). Only one run produced usable data.

### Completed Run

| Run Folder | Run ID | Model ID | Temp | Conditions | Trials | Clean Outputs | Status |
|---|---|---|---|---|---:|---:|---|
| `fK1N5V/20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37` | `f47fe05e` | `allenai/Olmo-3-7B-Think` | 0.0 | core (4 conditions) | 1609 | 1608 | ✅ Complete (+9 overfill) |

> The 9-trial overfill is a known artefact from a mid-run restart. The extra 9 trials are duplicate (model, item, condition) combinations from one dataset and do not affect analysis — they are deduplicated before analysis by taking the first output per unique `(item_id, condition_id)` pair.

### Failed / Incomplete Attempts (no usable data)

The `runs_index.json` in `runs/think/` documents 9 additional run attempts (`suite_think_*.json` configs) from 2026-03-25 through 2026-03-27, all listed as `status: "failed"` or `status: "running"` (stale). None produced a `simulation.db` with meaningful trial data. These are inert artefacts and can be ignored.

---

## `runs_latest/runs/` — OLMo-7B Training Stage Study

This directory contains the within-family study for the OLMo-7B model family across 4 training stages and 6 temperatures. Each database covers **all variants** (base, instruct, instruct_sft, instruct_dpo, think, think_sft, think_dpo, rl_zero_math) at a single temperature.

### Condition Mix (12 conditions)

| Condition Name | Family | Description |
|---|---|---|
| `control` | Baseline | No social pressure |
| `asch_history_5` | Peer (classical) | Free-text Asch format, 5 confederates (historical format) |
| `asch_zhu_unbiased_unanimous_plain` | Peer (structured) | 5 unanimous confederates, plain neutral tone |
| `asch_zhu_unbiased_unanimous_neutral` | Peer (structured) | 5 unanimous confederates, explicitly neutral tone |
| `asch_zhu_unbiased_unanimous_confident` | Peer (structured) | 5 unanimous confederates, confident tone |
| `asch_zhu_unbiased_unanimous_uncertain` | Peer (structured) | 5 unanimous confederates, uncertain/hedged tone |
| `asch_zhu_unbiased_diverse_plain` | Peer (diverse) | Non-unanimous confederates, plain framing |
| `asch_zhu_unbiased_qd` | Mitigation | Question distillation — summarize the consensus before answering |
| `asch_zhu_unbiased_da` | Mitigation | Devil's advocate — one confederate dissents |
| `authoritative_bias` | Authority | High-strength user authority claim |
| `authority_zhu_unbiased_trust` | Authority | Trust-framed authority claim |
| `authority_zhu_unbiased_trust_da` | Authority + Mitigation | Trust-framed authority with devil's advocate counter |

> The 4 conditions used in `runs/` cross-family study (`control`, `asch_zhu_unanimous_confident`, `authoritative_bias`, `authority_trust`) map directly onto 4 of these 12 conditions (`control`, `asch_zhu_unbiased_unanimous_confident`, `authoritative_bias`, `authority_zhu_unbiased_trust`). This is the calibration bridge between the two study arms.

### Design

- 8 model variants × 8 datasets × 50 items × 12 conditions = **38,400 expected trials per temperature**
- 6 temperatures: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

### Run Registry

| Run Folder | Run ID | Temp | Variants | Trials (actual) | Expected | Missing | Status |
|---|---|---|---|---:|---:|---:|---|
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `9f240f89` | 0.0 | base, instruct, instruct_sft, instruct_dpo + think variants | 34,794 | 38,400 | 3,606 | ⚠️ Partial (think variants incomplete) |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `46f0762a` | 0.2 | base, instruct, instruct_sft, instruct_dpo + think variants | 34,746 | 38,400 | 3,654 | ⚠️ Partial (think variants incomplete) |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `bbd05985` | 0.4 | base, instruct, instruct_sft, instruct_dpo + think variants | 37,978 | 38,400 | 422 | ⚠️ Partial (minor gap) |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `86c72262` | 0.6 | base, instruct, instruct_sft, instruct_dpo + think variants | 38,170 | 38,400 | 230 | ⚠️ Partial (minor gap) |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `9369442d` | 0.8 | base, instruct, instruct_sft, instruct_dpo + think variants | 34,800 | 38,400 | 3,600 | ⚠️ Partial (think variants incomplete) |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `9173bfae` | 1.0 | base, instruct, instruct_sft, instruct_dpo + think variants | 34,800 | 38,400 | 3,600 | ⚠️ Partial (think variants incomplete) |

> **All clean outputs match actual trials** (0 errors, 0 duplicates in all 6 `runs_latest` databases). The "missing" counts reflect think-variant trials that were not completed, not data corruption. The primary analysis uses only `base`, `instruct`, `instruct_sft`, and `instruct_dpo` variants, which are fully present in all 6 runs.

---

## Summary

| Study Arm | Directory | Models | Temperatures | Conditions | Total Clean Trials | Status |
|---|---|---|---|---|---:|---|
| Cross-family (primary) | `runs/` | 10 model families | T=0.0, T=0.6 | 4 | 31,613 of 32,000 | 387 pending (gpt-oss-20b T=0.0 re-run) |
| Ablation study | `runs/` | 2 models (llama-3.1-70b, olmo-3.1-32b) | T=0.0 | 2 | 1,600 | ✅ Complete |
| OLMo-7B-Think | `runs/think/` | 1 model (OLMo-7B-Think) | T=0.0 | 4 (core) | 1,608 | ✅ Complete |
| OLMo-7B training stages | `runs_latest/runs/` | 4 primary variants (base/instruct/sft/dpo) | T=0.0–1.0 | 12 | ~208,812 | ✅ Complete (think variants partial, excluded from primary) |
