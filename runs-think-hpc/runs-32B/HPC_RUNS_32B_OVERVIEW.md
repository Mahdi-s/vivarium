# runs-think-hpc/runs-32B overview

This report summarizes what is inside `runs-think-hpc/runs-32B`, based on:

- `./scripts/judge_report.sh --config experiments/olmo_conformity/configs/suite_32b_think_sft_temp0.0.json --inventory runs-think-hpc/runs-32B --show-missing`
- direct SQLite queries on each `simulation.db`

**HPC refresh:** Run folders were replaced with current HPC exports (checkpointed **`simulation.db` only** in this drop—no `-wal` / `-shm` sidecars on disk). Tables below were **re-verified locally on 2026-04-06**.

**Git LFS:** `simulation.db`, and—if present—`simulation.db-wal` / `simulation.db-shm` under `runs-think-hpc/runs-32B/<run_folder>/`, use **Git LFS** (see `.gitattributes`). `.gitignore` exempts these paths from the global `*.db` / `simulation*.db` rules. After clone, run `git lfs pull` if working trees show tiny pointer files instead of SQLite binaries.

Per-run DB sizes (bytes, 2026-04-06): ~13.5M (SFT 0.0), ~15.6M (SFT 0.6), ~13.9M (DPO 0.0), ~11.8M (DPO 0.6).

It parallels the structure of `runs-think-hpc/HPC_RUNS_OVERVIEW.md` (7B-focused), but covers **OLMo-3 32B Think SFT/DPO** only.

## 1) Current folder contents

- Total run folders with DBs: **4**
- Suite family: **OLMo-3 32B Think-SFT** and **Think-DPO** at **temp 0.0** and **0.6**
- Per run you typically see:
  - `simulation.db` (LFS-tracked; WAL/SHM only if copied mid-write from the cluster)
  - `artifacts/` (figures, tables subdirs may be empty)
  - `exports/` (may be empty)

Run directories:

- `20260330_235019_81d9194a-b1ef-4261-a0fb-bb0f713e1239`
- `20260331_002604_e673de86-d8dc-4ce0-bfc9-969d531eb425`
- `20260331_005822_3041fb7e-98bc-4343-90d7-e56d9e134a3b`
- `20260331_014024_d0158f56-c99c-4e7d-92fc-c7ece3190781`

## 2) ID structure (folder and run IDs)

Convention:

- `YYYYMMDD_HHMMSS_<run_uuid>`
- Canonical run id = UUID suffix (stored in `runs.run_id`)
- Short log style: first 8 characters of the UUID, e.g. `81d9194a...`

| run folder | timestamp part | canonical `runs.run_id` | short id |
|---|---|---|---|
| `20260330_235019_81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `20260330_235019` | `81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `81d9194a...` |
| `20260331_002604_e673de86-d8dc-4ce0-bfc9-969d531eb425` | `20260331_002604` | `e673de86-d8dc-4ce0-bfc9-969d531eb425` | `e673de86...` |
| `20260331_005822_3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `20260331_005822` | `3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `3041fb7e...` |
| `20260331_014024_d0158f56-c99c-4e7d-92fc-c7ece3190781` | `20260331_014024` | `d0158f56-c99c-4e7d-92fc-c7ece3190781` | `d0158f56...` |

## 3) Completion status (inventory vs full 8×4×50 grid)

Expected full suite (per `suite_32b_think_*` JSON family): **1600** trials = 8 datasets × 4 conditions × 50 items.

From `./scripts/judge_report.sh --inventory runs-think-hpc/runs-32B --show-missing`:

| run folder | model | temp | trials done / expected | cells ok / total | inventory status |
|---|---|---:|---:|---:|---|
| `20260330_235019_...` | `allenai/Olmo-3-32B-Think-SFT` | 0.00 | 607 / 1600 | 12 / 32 | PARTIAL (~38%) |
| `20260331_002604_...` | `allenai/Olmo-3-32B-Think-SFT` | 0.60 | 983 / 1600 | 16 / 32 | PARTIAL (~61%) |
| `20260331_005822_...` | `allenai/Olmo-3-32B-Think-DPO` | 0.00 | 669 / 1600 | 12 / 32 | PARTIAL (~42%) |
| `20260331_014024_...` | `allenai/Olmo-3-32B-Think-DPO` | 0.60 | 506 / 1600 | 8 / 32 | PARTIAL (~32%) |

Summary: **0 complete, 4 partial** (relative to the full 32-cell design). The inventory output lists missing dataset×condition cells and “short” cells for each run.

## 4) Embedded config + DB snapshot (SQLite)

`runs.config_json -> $.suite_config.suite_name` (and trial rows agree on model/temperature):

| run folder | embedded `suite_name` | trials | outputs | missing output rows |
|---|---|---:|---:|---:|
| `20260330_235019_...` | `olmo_conformity_32b_think_sft_temp0.0` | 607 | 606 | 1 |
| `20260331_002604_...` | `olmo_conformity_32b_think_sft_temp0.6` | 983 | 982 | 1 |
| `20260331_005822_...` | `olmo_conformity_32b_think_dpo_temp0.0` | 669 | 668 | 1 |
| `20260331_014024_...` | `olmo_conformity_32b_think_dpo_temp0.6` | 506 | 505 | 1 |

### Schema check

All four DBs have **25** tables and include the usual conformity tables (`conformity_trials`, `conformity_outputs`, `conformity_prompts`, `conformity_items`, `conformity_conditions`, `conformity_datasets`, etc.).

## 5) Judge labels on `conformity_outputs.is_correct`

Counts are over **existing output rows joined to trials** for the run:

| run folder | `is_correct = 1` | `is_correct = 0` | `is_correct IS NULL` | total outputs |
|---|---:|---:|---:|---:|
| `20260330_235019_...` | 175 | 431 | 0 | 606 |
| `20260331_002604_...` | 223 | 559 | 200 | 982 |
| `20260331_005822_...` | 118 | 550 | 0 | 668 |
| `20260331_014024_...` | 109 | 396 | 0 | 505 |

Notes:

- Three runs have **no** NULL `is_correct` on stored outputs (every output row is labeled 0/1).
- `e673de86...` (32B SFT, temp 0.6) has **200** outputs with `is_correct IS NULL` — consistent with a **partial judge pass** or interrupted labeling on that snapshot.
- None of these DBs had `json_extract(parsed_answer_json, '$.is_correct')` populated in a spot check (column-level judge agreement scripts that require `$.is_correct` in JSON would still see **0** validated rows there).

## 6) Bottom line

- `runs-32B` is a **4-run slice** of the 32B Think fidelity experiment (SFT/DPO × temp 0.0/0.6), all **in progress** toward the full 1600-trial grid.
- **Furthest along** in this folder: 32B SFT temp **0.6** (`983/1600` trials).
- Databases are **structurally sound**; most snapshots are missing **one** output row vs trial count except where many trials simply have not been scheduled yet.
- **Manual-style** judge signal is available via **`conformity_outputs.is_correct`** on most rows; one run has a **200-row** NULL gap worth re-running the judge on if you want a clean ledger.
