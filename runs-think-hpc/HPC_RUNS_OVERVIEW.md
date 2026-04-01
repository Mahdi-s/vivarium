# runs-think-hpc overview

This report summarizes what is inside `runs-think-hpc`, using:

- `./scripts/judge_report.sh --inventory runs-think-hpc --show-missing`
- `./scripts/judge_report.sh --config experiments/olmo_conformity/configs/suite_7b_think_sft_temp0p0.json --all runs-think-hpc`
- direct SQLite queries on each `simulation.db`

## 1) What is in the folder

- Total run folders with DBs: **8**
- Models covered: **OLMo-3 Think SFT/DPO, 7B and 32B**
- Temperatures present: **0.0 and 0.6**
- Each run folder has:
  - `simulation.db`
  - `artifacts/figures`, `artifacts/tables` (currently empty)
  - `exports` (currently empty)
  - some partial runs also have active `simulation.db-wal` / `simulation.db-shm`

## 2) Completion status (from `judge_report.sh --inventory`)

| run folder | model | temp | trials done/expected | status |
|---|---|---:|---:|---|
| `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112` | `allenai/Olmo-3-7B-Think-SFT` | 0.00 | 1600/1600 | COMPLETE |
| `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c` | `allenai/Olmo-3-7B-Think-SFT` | 0.60 | 1600/1600 | COMPLETE |
| `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a` | `allenai/Olmo-3-7B-Think-DPO` | 0.00 | 1600/1600 | COMPLETE |
| `20260330_022920_57414a36-f63a-4332-9c70-437a60071565` | `allenai/Olmo-3-7B-Think-DPO` | 0.60 | 1100/1600 | PARTIAL (69%) |
| `20260330_235019_81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `allenai/Olmo-3-32B-Think-SFT` | 0.00 | 438/1600 | PARTIAL (27%) |
| `20260331_002604_e673de86-d8dc-4ce0-bfc9-969d531eb425` | `allenai/Olmo-3-32B-Think-SFT` | 0.60 | 382/1600 | PARTIAL (24%) |
| `20260331_005822_3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `allenai/Olmo-3-32B-Think-DPO` | 0.00 | 339/1600 | PARTIAL (21%) |
| `20260331_014024_d0158f56-c99c-4e7d-92fc-c7ece3190781` | `allenai/Olmo-3-32B-Think-DPO` | 0.60 | 295/1600 | PARTIAL (18%) |

Summary: **3 complete, 5 partial**.

## 3) Config matching verification

I verified against each DB's embedded run metadata (`runs.config_json -> $.suite_config.suite_name`) and cross-checked model + temp + condition structure.

| run folder | embedded suite_name |
|---|---|
| `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112` | `olmo_conformity_7b_think_sft_temp0.0` |
| `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c` | `olmo_conformity_7b_think_sft_temp0.6` |
| `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a` | `olmo_conformity_7b_think_dpo_temp0.0` |
| `20260330_022920_57414a36-f63a-4332-9c70-437a60071565` | `olmo_conformity_7b_think_dpo_temp0.6` |
| `20260330_235019_81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `olmo_conformity_32b_think_sft_temp0.0` |
| `20260331_002604_e673de86-d8dc-4ce0-bfc9-969d531eb425` | `olmo_conformity_32b_think_sft_temp0.6` |
| `20260331_005822_3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `olmo_conformity_32b_think_dpo_temp0.0` |
| `20260331_014024_d0158f56-c99c-4e7d-92fc-c7ece3190781` | `olmo_conformity_32b_think_dpo_temp0.6` |

Conclusion: all runs are consistent with the expected 7B/32B Think SFT/DPO suite family and correct temperature variants.

## 4) Database table verification

For all 8 DBs:

- table count = **25**
- required conformity tables present:
  - `conformity_trials`
  - `conformity_outputs`
  - `conformity_prompts`
  - `conformity_items`
  - `conformity_conditions`
  - `conformity_datasets`

No missing required tables were found.

## 5) Prompt/content verification

### Condition template integrity

Across every run, prompt patterns align exactly with condition semantics:

- `asch_zhu_unanimous_confident`: prompt contains `"participants in an experiment"` for all rows in that condition.
- `authoritative_bias`: prompt begins with `"I'm pretty sure the answer is ..."` for all rows in that condition.
- `authority_trust`: prompt contains `"I trust that the answer must be ..."` for all rows in that condition.
- `control`: none of the above social-pressure trigger phrases.

### Output presence

- Complete runs have output rows for all trials.
- Partial runs are generally missing exactly one output row at current stop points.
- `parsed_answer_text` and `parsed_answer_json` are populated for essentially all available output rows.

Representative extracted rows show plausible condition-dependent behavior (agreement pressure / trust framing reflected in answers), e.g.:

- 7B SFT: control/authority prompts over same math question return concise answers like `18`.
- 7B DPO: prompts and responses are present but often include hesitant short phrases (e.g., `"Wait, maybe..."`) in sampled rows.
- 32B SFT / 32B DPO: condition framing appears in prompts and outputs, including longer textual answers under authoritative/trust prompts.

## 6) Important note about judge fields (`judge_report.sh` pooled output)

`judge_report.sh --all` reports:

- `judge_valid = 0`
- `is_match_rate = n/a`

for this folder, even where runs are complete.

Why: these DBs' `conformity_outputs.parsed_answer_json` currently do **not** include the downstream key `$.is_correct` required by the script's agreement gate logic. The parsed candidate extraction is present, but final judge labeling fields are not yet materialized in these files.

So:

- **Run completion (trials/outputs/cells)**: valid and measurable (inventory).
- **Manual-vs-judge agreement metrics**: not yet computable from these DB snapshots.

## 7) Bottom line

- You have all four requested model variants represented:
  - 7B Think SFT, 7B Think DPO, 32B Think SFT, 32B Think DPO
  - each at temp 0.0 and 0.6
- Folder currently contains:
  - **3 complete runs** (all 7B temp0.0 + 0.6 SFT and 7B DPO temp0.0)
  - **5 partial runs** (7B DPO temp0.6 + all 32B runs in this snapshot)
- DB schemas and prompt templates are consistent with the expected conformity configuration design.
- Agreement/judge scorecards are pending the `is_correct`/judge-label stage in `parsed_answer_json`.
