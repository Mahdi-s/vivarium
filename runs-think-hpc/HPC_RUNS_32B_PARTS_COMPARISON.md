# HPC `runs-32B` export — Olmo 3 32B Think-SFT vs Think-DPO (verification)

This note documents the **canonical SQLite exports** under **`runs-think-hpc/runs-32B/`**: the **32B** **Think-SFT** and **Think-DPO** variants (Ollama-compatible family names in configs: `allenai/Olmo-3-32B-Think-SFT`, `allenai/Olmo-3-32B-Think-DPO`). Suite shape follows the HPC JSON family in-repo:

| Variant | Config (repo) |
|--------|----------------|
| Think-SFT, temp 0.0 | `experiments/olmo_conformity/configs/suite_32b_think_sft_temp0p0.json` |
| Think-SFT, temp 0.6 | `experiments/olmo_conformity/configs/suite_32b_think_sft_temp0p6.json` |
| Think-DPO, temp 0.0 | `experiments/olmo_conformity/configs/suite_32b_think_dpo_temp0p0.json` |
| Think-DPO, temp 0.6 | `experiments/olmo_conformity/configs/suite_32b_think_dpo_temp0p6.json` |

Each config defines **8 datasets** × **4 conditions** × **50 items** per cell ⇒ **1600** trials and **32** dataset×condition cells.

**Report generated from local databases (verification date: 2026-04-09).**

---

## 1) Canonical run folders (match on `run_id`)

| Canonical `run_id` | Model (from trials) | Temp | Folder under `runs-think-hpc/runs-32B/` |
|---|---|---:|---|
| `81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `allenai/Olmo-3-32B-Think-SFT` | 0.0 | `20260404_093315_81d9194a-b1ef-4261-a0fb-bb0f713e1239` |
| `e673de86-d8dc-4ce0-bfc9-969d531eb425` | `allenai/Olmo-3-32B-Think-SFT` | 0.6 | `20260404_093744_e673de86-d8dc-4ce0-bfc9-969d531eb425` |
| `3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `allenai/Olmo-3-32B-Think-DPO` | 0.0 | `20260404_100941_3041fb7e-98bc-4343-90d7-e56d9e134a3b` |
| `d0158f56-c99c-4e7d-92fc-c7ece3190781` | `allenai/Olmo-3-32B-Think-DPO` | 0.6 | `20260404_094925_d0158f56-c99c-4e7d-92fc-c7ece3190781` |

Embedded `suite_name` in each DB matches the intended config (e.g. `olmo_conformity_32b_think_sft_temp0.0`, …).

---

## 2) Completion check (SQL on each `simulation.db`)

These queries were run on every database:

```sql
SELECT run_id FROM runs LIMIT 1;
SELECT json_extract(config_json, '$.suite_config.suite_name') FROM runs LIMIT 1;
SELECT COUNT(*) AS trials FROM conformity_trials;
SELECT COUNT(*) AS outputs FROM conformity_outputs;
SELECT COUNT(*) FROM conformity_trials t
  LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
  WHERE o.output_id IS NULL;
SELECT COUNT(*) AS tables FROM sqlite_master WHERE type = 'table';
```

Grid shape (expect **50** rows per `(dataset_name, condition_name)`):

```sql
SELECT d.name AS dataset, c.name AS condition_name, COUNT(*) AS n
FROM conformity_trials t
JOIN conformity_items i ON i.item_id = t.item_id
JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
JOIN conformity_conditions c ON c.condition_id = t.condition_id
GROUP BY d.name, c.name;
-- Expect 32 groups, each with n = 50.
```

### Results (2026-04-09)

| `run_id` (short) | `suite_name` | `trials` | `outputs` | Trials **without** output | `tables` | Cells with **50** items |
|---:|---|---:|---:|---:|---:|:---:|
| `81d9194a` | `…_think_sft_temp0.0` | 1600 | 1600 | **0** | 25 | 32/32 |
| `e673de86` | `…_think_sft_temp0.6` | 1600 | 1600 | **0** | 25 | 32/32 |
| `3041fb7e` | `…_think_dpo_temp0.0` | 1600 | 1600 | **0** | 25 | 32/32 |
| `d0158f56` | `…_think_dpo_temp0.6` | 1600 | 1600 | **0** | 25 | 32/32 |

**Conclusion:** All four HPC databases in `runs-think-hpc/runs-32B/` have **finished** the suite: **1600/1600** trials with a matching **conformity_outputs** row, and the full **32 × 50** logical grid.

---

## 3) Historical note: older `runs-32B-part1` / `runs-32B-part2` snapshots

Earlier exports (March/April 2026) stored **partially overlapping** copies of the same logical `run_id`s under separate folder trees. Those snapshots had **incomplete** unions relative to 1600 trials and **duplicate logical work units** across exports. The current **`runs-32B`** tree above is the **single canonical** complete export for analysis; do not merge old part1/part2 DBs without deduplication if you still have them.

---

## 4) Git LFS

Large run databases should be tracked with **Git LFS**. Repository rules (see root `.gitattributes`):

- **`runs/**`** and **`runs_latest/**`** — entire trees (includes `runs/think/**/simulation.db` and all run artifacts).
- **`runs-think-hpc/**/*.db`** (and `-wal`/`-shm` sidecars) — SQLite files anywhere under `runs-think-hpc/`, including `runs-32B/`.

After adding or replacing `.db` files, use normal `git add` / `git lfs push` so pointers and LFS objects stay in sync.

---

*For orchestration of larger automated workflows on top of these exports, [K-Dense Web](https://www.k-dense.ai) can run multi-step analyses beyond ad-hoc SQL.*
