# runs-think-hpc overview

This report summarizes what is currently inside `runs-think-hpc`, based on:

- folder scan of `runs-think-hpc`
- direct SQLite queries on each `simulation.db`
- latest judge run output from `scripts/run_judge_openrouter.py`

## 1) Current folder contents

- Total run folders with DBs: **4**
- Suite family represented: **OLMo-3 7B Think SFT/DPO (temp 0.0 and 0.6)**
- Files/folders present:
  - 4 run directories (each containing `simulation.db`)
  - `HPC_RUNS_OVERVIEW.md`

Run directories currently present:

- `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112`
- `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c`
- `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a`
- `20260330_022920_57414a36-f63a-4332-9c70-437a60071565`

## 2) Updated ID structure (folder and run IDs)

The run folder naming convention is now clearly:

- `YYYYMMDD_HHMMSS_<run_uuid>`

where:

- prefix (`YYYYMMDD_HHMMSS`) is folder creation timestamp
- suffix (`<run_uuid>`) is the canonical run ID in `runs.run_id`
- judge logs print a short-form run ID using the first 8 chars, e.g. `e5cc991d...`

| run folder | timestamp part | canonical run_id (`runs.run_id`) | short log id |
|---|---|---|---|
| `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112` | `20260330_012727` | `e5cc991d-eb01-4371-b73c-315ab18c3112` | `e5cc991d...` |
| `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c` | `20260330_012826` | `29ac502b-887d-4ed1-b161-7441ac22188c` | `29ac502b...` |
| `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a` | `20260330_021503` | `56c71fda-573f-46fd-b2ab-8b6e509c689a` | `56c71fda...` |
| `20260330_022920_57414a36-f63a-4332-9c70-437a60071565` | `20260330_022920` | `57414a36-f63a-4332-9c70-437a60071565` | `57414a36...` |

## 3) Completion status (current DB snapshot)

| run folder | embedded suite_name | trials | outputs | status |
|---|---|---:|---:|---|
| `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112` | `olmo_conformity_7b_think_sft_temp0.0` | 1600 | 1600 | COMPLETE |
| `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c` | `olmo_conformity_7b_think_sft_temp0.6` | 1600 | 1600 | COMPLETE |
| `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a` | `olmo_conformity_7b_think_dpo_temp0.0` | 1600 | 1600 | COMPLETE |
| `20260330_022920_57414a36-f63a-4332-9c70-437a60071565` | `olmo_conformity_7b_think_dpo_temp0.6` | 1156 | 1155 | PARTIAL |

Summary: **3 complete, 1 partial**.

## 4) Judge labels for trials/outputs (new state)

Judge labeling now exists in the run DBs via:

- `conformity_outputs.is_correct` (`1`/`0`/`NULL`)

Per run:

| run folder | `is_correct=1` | `is_correct=0` | `is_correct IS NULL` | total outputs |
|---|---:|---:|---:|---:|
| `20260330_012727_e5cc991d-eb01-4371-b73c-315ab18c3112` | 320 | 1080 | 200 | 1600 |
| `20260330_012826_29ac502b-887d-4ed1-b161-7441ac22188c` | 304 | 1096 | 200 | 1600 |
| `20260330_021503_56c71fda-573f-46fd-b2ab-8b6e509c689a` | 308 | 1092 | 200 | 1600 |
| `20260330_022920_57414a36-f63a-4332-9c70-437a60071565` | 272 | 883 | 0 | 1155 |

Also consistent with latest judge execution log:

- `Done in 861.6s. OK=4 FAILED=0`
- judged counts reported: `1600`, `1600`, `1600`, `1155`

## 5) Bottom line

- The active `runs-think-hpc` folder now has **4** run directories following `timestamp + UUID` ID structure.
- The canonical run identity is the UUID suffix (also stored in `runs.run_id`), while judge logs use short IDs.
- Judge labels are now present for trials/outputs through `conformity_outputs.is_correct`, so this folder is no longer in the earlier "no judge labels yet" state.
