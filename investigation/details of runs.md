# Details of Runs

Generated from run `simulation.db` files in: `runs/`, `runs/think/`, `runs_latest/runs/`.

Completeness definitions:
- **Trials** compares `actual_trials` vs `expected_trials` from run config dimensions (`models * conditions * datasets * max_items_per_dataset`).
- **LLM judge response** counts rows where `conformity_outputs.parsed_answer_json` contains `_llm_judge.judge_model`, compared against `expected_trials`.

## `runs/`

| Run Folder | Run ID | Suite Name | Matched Config JSON | Expected Trials | Actual Trials | Trial Cells | LLM Judge Responses | LLM Judge Cells |
|---|---|---|---|---:|---:|---|---:|---|
| `20260327_152738_a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d` | `a34ad9b1-abd0-4119-96c6-7b1cd61d7f4d` | `olmo_conformity_32b_think_api_temp0.0` | `suite_32b_think_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_152926_7db9896e-9e3b-439f-88e3-74fe25ea2bad` | `7db9896e-9e3b-439f-88e3-74fe25ea2bad` | `olmo_conformity_32b_think_api_temp0.6` | `suite_32b_think_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_152936_1c2e5cb6-0372-4835-bbb7-7230c55517e4` | `1c2e5cb6-0372-4835-bbb7-7230c55517e4` | `olmo_conformity_32b_instruct_api_temp0.0` | `suite_32b_instruct_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_152944_62187f52-7a7e-4db0-a269-d14d8e887b1b` | `62187f52-7a7e-4db0-a269-d14d8e887b1b` | `olmo_conformity_32b_instruct_api_temp0.6` | `suite_32b_instruct_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154349_1899a883-82e4-45f3-833a-d6403cf1ac95` | `1899a883-82e4-45f3-833a-d6403cf1ac95` | `olmo_conformity_llama3_8b_instruct_api_temp0p0` | `suite_llama3_8b_instruct_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154401_70860876-c5c2-4445-a59d-e44ae8094887` | `70860876-c5c2-4445-a59d-e44ae8094887` | `olmo_conformity_llama3_8b_instruct_api_temp0p6` | `suite_llama3_8b_instruct_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154412_3a0404f7-bd47-4b25-b2e2-5501e550566f` | `3a0404f7-bd47-4b25-b2e2-5501e550566f` | `olmo_conformity_llama3.1_70b_instruct_api_temp0p0` | `suite_llama3.1_70b_instruct_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154419_49d07104-c14c-4a8b-a013-ae0783c5f3e8` | `49d07104-c14c-4a8b-a013-ae0783c5f3e8` | `olmo_conformity_llama3.1_70b_instruct_api_temp0p6` | `suite_llama3.1_70b_instruct_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154428_485ddc2d-6cae-4715-835e-76ab72e38159` | `485ddc2d-6cae-4715-835e-76ab72e38159` | `olmo_conformity_llama4_maverick_api_temp0p0` | `suite_llama4_maverick_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_154435_c2ce0f85-8f67-40f2-a82d-e136927cf6f5` | `c2ce0f85-8f67-40f2-a82d-e136927cf6f5` | `olmo_conformity_llama4_maverick_api_temp0p6` | `suite_llama4_maverick_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224321_e043fbf6-27eb-410c-8da7-bc0f9172ab0b` | `e043fbf6-27eb-410c-8da7-bc0f9172ab0b` | `olmo_conformity_gemini_2.5_flash_lite_api_temp0p0` | `suite_gemini_2.5_flash_lite_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224336_d71e75b1-17c5-4789-8ee7-13b29ef18359` | `d71e75b1-17c5-4789-8ee7-13b29ef18359` | `olmo_conformity_gemini_2.5_flash_lite_api_temp0p6` | `suite_gemini_2.5_flash_lite_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224348_25056752-7081-449e-9a44-ad090b566107` | `25056752-7081-449e-9a44-ad090b566107` | `olmo_conformity_grok_4.1_fast_api_temp0p0` | `suite_grok_4.1_fast_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224357_157a6a9e-13de-4bdb-bdd2-54d761498f24` | `157a6a9e-13de-4bdb-bdd2-54d761498f24` | `olmo_conformity_grok_4.1_fast_api_temp0p6` | `suite_grok_4.1_fast_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224413_c07ede3a-16ac-4b47-ac42-4f6ad8dd8370` | `c07ede3a-16ac-4b47-ac42-4f6ad8dd8370` | `olmo_conformity_gpt4o_mini_api_temp0p0` | `suite_gpt4o_mini_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224422_eb63d212-77fe-46ef-965b-7777cc232f1f` | `eb63d212-77fe-46ef-965b-7777cc232f1f` | `olmo_conformity_gpt4o_mini_api_temp0p6` | `suite_gpt4o_mini_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260327_224552_66765d5e-204c-4074-aaf4-b9c148fe61a5` | `66765d5e-204c-4074-aaf4-b9c148fe61a5` | `olmo_conformity_gpt_oss_20b_api_temp0p0` | `suite_gpt_oss_20b_api_temp0p0.json` | 1600 | 1600 | complete | 390 | missing (1210) |
| `20260327_224603_3ecdc9b7-49db-4625-b90e-fc3745b9224e` | `3ecdc9b7-49db-4625-b90e-fc3745b9224e` | `olmo_conformity_gpt_oss_20b_api_temp0p6` | `suite_gpt_oss_20b_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260329_211511_5be5ada7-64be-4cbd-9024-aacbcaf233e3` | `5be5ada7-64be-4cbd-9024-aacbcaf233e3` | `olmo_conformity_claude_sonnet_4_api_temp0p0` | `suite_claude_sonnet_4_api_temp0p0.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260329_211518_21556460-5f97-4a23-8c54-4a1f999ba619` | `21556460-5f97-4a23-8c54-4a1f999ba619` | `olmo_conformity_claude_sonnet_4_api_temp0p6` | `suite_claude_sonnet_4_api_temp0p6.json` | 1600 | 1600 | complete | 1600 | complete |
| `20260329_235403_e8a90500-25cd-469f-a138-197c338fddaf` | `e8a90500-25cd-469f-a138-197c338fddaf` | `olmo_conformity_llama3.1_70b_instruct_ablations_only_temp0p0` | `suite_llama3.1_70b_instruct_ablations_temp0p0.json` | 800 | 800 | complete | 800 | complete |
| `20260329_235408_ef72529e-5e82-463f-8b8b-b2a6c7decd3c` | `ef72529e-5e82-463f-8b8b-b2a6c7decd3c` | `olmo_conformity_32b_instruct_ablations_only_temp0p0` | `suite_32b_instruct_ablations_temp0p0.json` | 800 | 800 | complete | 800 | complete |
| `20260330_171029_f9233686-adda-40c8-90cc-eb26b0031821` | `f9233686-adda-40c8-90cc-eb26b0031821` | `olmo_conformity_gpt_oss_20b_api_temp0p0` | `suite_gpt_oss_20b_api_temp0p0.json` | 1600 | 483 | missing (1117) | 0 | missing (1600) |
| `20260330_171043_6a8a3b4f-0bf3-4522-9745-e3fe7e68599d` | `6a8a3b4f-0bf3-4522-9745-e3fe7e68599d` | `olmo_conformity_gpt_oss_20b_api_temp0p0` | `suite_gpt_oss_20b_api_temp0p0.json` | 1600 | 1600 | complete | 0 | missing (1600) |
| `20260330_171519_34ff8a84-f6ab-45f3-8e03-3b9e617a9b0b` | `34ff8a84-f6ab-45f3-8e03-3b9e617a9b0b` | `olmo_conformity_gpt_oss_20b_api_temp0p0` | `suite_gpt_oss_20b_api_temp0p0.json` | 1600 | 1600 | complete | 0 | missing (1600) |
| `20260330_171816_d815c364-877a-4253-8ed9-7bf1c143bb2a` | `d815c364-877a-4253-8ed9-7bf1c143bb2a` | `olmo_conformity_gpt_oss_20b_api_temp0p0` | `suite_gpt_oss_20b_api_temp0p0.json` | 1600 | 1600 | complete | 0 | missing (1600) |
| `20260330_172832_621a7698-8e4f-4490-94d1-04f26090e714` | `621a7698-8e4f-4490-94d1-04f26090e714` | `olmo_conformity_claude_sonnet_4_api_temp0p6` | `suite_claude_sonnet_4_api_temp0p6.json` | 1600 | 1600 | complete | 0 | missing (1600) |
| `20260331_001009_1406a11a-7e52-48b5-9c1a-eca2d8833c95` | `1406a11a-7e52-48b5-9c1a-eca2d8833c95` | `olmo_conformity_gpt_oss_20b_api_temp0p0_smoke` | `NO_MATCH in experiments/olmo_conformity/configs/` | 20 | 20 | complete | 0 | missing (20) |

## `runs/think/`

| Run Folder | Run ID | Suite Name | Matched Config JSON | Expected Trials | Actual Trials | Trial Cells | LLM Judge Responses | LLM Judge Cells |
|---|---|---|---|---:|---:|---|---:|---|
| `20260325_010440_f47fe05e-4564-4680-a2d8-39a88c6f8d37` | `f47fe05e-4564-4680-a2d8-39a88c6f8d37` | `olmo_conformity_think_auto` | `NO_MATCH in experiments/olmo_conformity/configs/` | 1600 | 1609 | overfilled (+9) | 1608 | overfilled (+8) |

## `runs_latest/runs/`

| Run Folder | Run ID | Suite Name | Matched Config JSON | Expected Trials | Actual Trials | Trial Cells | LLM Judge Responses | LLM Judge Cells |
|---|---|---|---|---:|---:|---|---:|---|
| `20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd` | `9f240f89-e58e-423a-ae68-f990b31c84cd` | `olmo_conformity_expanded_temp0.0` | `suite_7b_expanded.json (derived name match)` | 38400 | 34794 | missing (3606) | 34794 | missing (3606) |
| `20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0` | `46f0762a-210a-459a-8709-d24a0f194eb0` | `olmo_conformity_expanded_temp0.2` | `suite_7b_expanded.json (derived name match)` | 38400 | 34746 | missing (3654) | 34746 | missing (3654) |
| `20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94` | `bbd05985-d185-460a-b0aa-dd356d27ec94` | `olmo_conformity_expanded_temp0.4` | `suite_7b_expanded.json (derived name match)` | 38400 | 37978 | missing (422) | 37978 | missing (422) |
| `20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215` | `86c72262-d1aa-41b5-9c22-d7b2e0570215` | `olmo_conformity_expanded_temp0.6` | `suite_7b_expanded.json (derived name match)` | 38400 | 38170 | missing (230) | 38170 | missing (230) |
| `20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814` | `9369442d-d825-4cd0-81a1-8ed276c37814` | `olmo_conformity_expanded_temp0.8` | `suite_7b_expanded.json (derived name match)` | 38400 | 34800 | missing (3600) | 34800 | missing (3600) |
| `20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `9173bfae-4e8a-464f-8c9c-7ee91caa8b6e` | `olmo_conformity_expanded_temp1.0` | `suite_7b_expanded.json (derived name match)` | 38400 | 34800 | missing (3600) | 34800 | missing (3600) |

