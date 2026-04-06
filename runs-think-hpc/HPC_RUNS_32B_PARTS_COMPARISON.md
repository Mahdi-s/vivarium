# runs-32B part1 vs part2 — ID match, overlap, and suite completeness

This note compares **`runs-think-hpc/runs-32B-part1`** and **`runs-think-hpc/runs-32B-part2`** using SQLite on every `simulation.db`. Suite shape follows the HPC JSON family in-repo:

| Variant | Config (repo) |
|--------|----------------|
| Think-SFT, temp 0.0 | `experiments/olmo_conformity/configs/suite_32b_think_sft_temp0p0.json` |
| Think-SFT, temp 0.6 | `experiments/olmo_conformity/configs/suite_32b_think_sft_temp0p6.json` |
| Think-DPO, temp 0.0 | `experiments/olmo_conformity/configs/suite_32b_think_dpo_temp0p0.json` |
| Think-DPO, temp 0.6 | `experiments/olmo_conformity/configs/suite_32b_think_dpo_temp0p6.json` |

Each config defines **8 datasets** × **4 conditions** × **50 items** per cell ⇒ **1600** trials and **32** dataset×condition cells.

**Report generated from local databases (analysis date: 2026-04-06).**

---

## 1) Folder pairing (match on canonical `run_id`)

Timestamps in folder names differ (March 2026 vs April 2026), but the **UUID suffix is the same** as `runs.run_id` inside each DB. Pairs:

| Canonical `run_id` | Model (from trials) | Temp | part1 folder | part2 folder |
|---|---|---:|---|---|
| `81d9194a-b1ef-4261-a0fb-bb0f713e1239` | `allenai/Olmo-3-32B-Think-SFT` | 0.0 | `runs-32B-part1/20260330_235019_81d9194a-…` | `runs-32B-part2/20260404_093315_81d9194a-…` |
| `e673de86-d8dc-4ce0-bfc9-969d531eb425` | `allenai/Olmo-3-32B-Think-SFT` | 0.6 | `runs-32B-part1/20260331_002604_e673de86-…` | `runs-32B-part2/20260404_093744_e673de86-…` |
| `3041fb7e-98bc-4343-90d7-e56d9e134a3b` | `allenai/Olmo-3-32B-Think-DPO` | 0.0 | `runs-32B-part1/20260331_005822_3041fb7e-…` | `runs-32B-part2/20260404_100941_3041fb7e-…` |
| `d0158f56-c99c-4e7d-92fc-c7ece3190781` | `allenai/Olmo-3-32B-Think-DPO` | 0.6 | `runs-32B-part1/20260331_014024_d0158f56-…` | `runs-32B-part2/20260404_094925_d0158f56-…` |

Embedded `suite_name` in each DB matches the intended config (e.g. `olmo_conformity_32b_think_sft_temp0.0`, …). **Metadata lines up across the two exports.**

---

## 2) Raw SQL row counts (per database)

These queries were run on each `simulation.db`:

```sql
SELECT run_id FROM runs LIMIT 1;
SELECT json_extract(config_json, '$.suite_config.suite_name') FROM runs LIMIT 1;
SELECT COUNT(*) AS trials FROM conformity_trials;
SELECT COUNT(*) AS outputs FROM conformity_outputs;
SELECT COUNT(*) AS tables FROM sqlite_master WHERE type = 'table';
```

| Part | `run_id` (short) | `suite_name` | `trials` | `outputs` | `tables` |
|---:|---|---|---:|---:|---:|
| part1 | `81d9194a` | `…_sft_temp0.0` | 607 | 606 | 25 |
| part2 | `81d9194a` | `…_sft_temp0.0` | 1149 | 1148 | 25 |
| part1 | `e673de86` | `…_sft_temp0.6` | 983 | 982 | 25 |
| part2 | `e673de86` | `…_sft_temp0.6` | 937 | 936 | 25 |
| part1 | `3041fb7e` | `…_dpo_temp0.0` | 669 | 668 | 25 |
| part2 | `3041fb7e` | `…_dpo_temp0.0` | 815 | 814 | 25 |
| part1 | `d0158f56` | `…_dpo_temp0.6` | 506 | 505 | 25 |
| part2 | `d0158f56` | `…_dpo_temp0.6` | 694 | 693 | 25 |

Every snapshot has **one fewer output than trial** (one pending output row per DB).

---

## 3) Do part1 and part2 “complement” each other?

**No.** They are **not** two disjoint halves of the grid.

A **logical trial** is identified here as **`(dataset_name, condition_name, item_id)`**, where:

- `dataset_name` and item come from `conformity_items` → `conformity_datasets`,
- `condition_name` comes from `conformity_conditions` (stable names `control`, `asch_zhu_unanimous_confident`, … — UUIDs in `condition_id` differ between exports).

Query used for comparison:

```sql
SELECT
  d.name AS dataset,
  c.name AS condition_name,
  t.item_id
FROM conformity_trials t
JOIN conformity_items i ON i.item_id = t.item_id
JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
JOIN conformity_conditions c ON c.condition_id = t.condition_id;
```

**`trial_id` strings** have **zero overlap** between part1 and part2 for every paired run (new primary keys on re-export). **Logical keys overlap** on many `(dataset, condition, item_id)` tuples — the same experimental unit appears in **both** DBs with different `trial_id` values.

| `run_id` (short) | part1 logical rows | part2 logical rows | **Intersection** | Only part1 | Only part2 | **Union** |
|---:|---:|---:|---:|---:|---:|---:|
| `81d9194a` | 607 | 1149 | **356** | 251 | 793 | 1400 |
| `e673de86` | 983 | 937 | **537** | 446 | 400 | 1383 |
| `3041fb7e` | 669 | 815 | **69** | 600 | 746 | 1415 |
| `d0158f56` | 506 | 694 | **200** | 306 | 494 | 1000 |

So: **do not merge by concatenating trials** without **deduplicating on `(dataset, condition_name, item_id)`** (and then reconciling outputs/judges if both sides exist).

---

## 4) Completeness vs the 1600-trial grid (union of logical keys)

After taking the **set union** of logical rows from part1 and part2:

| `run_id` (short) | Union trials | Cells present (`dataset`×`condition`) | Cells with **exactly 50** items | Missing cells (of 32) | Other gaps |
|---:|---:|---:|---:|---|---|
| `81d9194a` | 1400 | 28 | 28 | **4** — all `social_conventions_minimal` × 4 conditions | — |
| `e673de86` | 1383 | 28 | 24 | **4** — all `immutable_facts_minimal` × 4 conditions | **TruthfulQA** cells have 45–46 items (not 50) |
| `3041fb7e` | 1415 | 32 | 28 | **0** | **GSM8K** cells have only **3–4** distinct items per condition in the union (far short of 50) |
| `d0158f56` | 1000 | 20 | 20 | **12** — e.g. all of `arc` (×4 conditions) and `immutable_facts_minimal` (×4), all of `mmlu_knowledge` (×4) | — |

**Conclusion:** Even after combining both parts, **no** run reaches a full **1600** / **32×50** clean grid in this snapshot set:

- **SFT 0.0**: entire **`social_conventions_minimal`** stratum missing on the union.
- **SFT 0.6**: **`immutable_facts_minimal`** stratum missing; **TruthfulQA** under-filled.
- **DPO 0.0**: all cells touched but **GSM8K** is critically short in the union (likely sparse / split awkwardly across the two exports).
- **DPO 0.6**: large missing blocks (**12** cells).

---

## 5) Answers in plain language

| Question | Answer |
|----------|--------|
| Do part1 and part2 **match** by id? | **Yes** on **`run_id`** / model / suite name — they are the **same four logical runs**, two export batches. |
| Are they byte-identical DBs? | **No** — trial counts differ; **`trial_id` values differ** across parts. |
| Do they **complement** (disjoint coverage)? | **No** — large **overlap** on the same `(dataset, condition, item_id)` work units. |
| Are they **jointly complete** for the suite JSONs? | **No** — unions are **missing cells** and/or have **short** cells; none is a clean **1600**-trial completion. |

---

## 6) Recommended next steps (operational)

1. **Pick a single canonical export** per `run_id` (usually the **newer** part2 if it strictly supersedes part1 — but here part2 is not a strict superset for every run, so compare per case).
2. For analysis, **deduplicate** on `(dataset, condition_name, item_id)` and prefer one side’s row (newer timestamp / completed output / judge).
3. **Resume HPC jobs** using the resume scripts in-repo (e.g. `job_32b_think_*_resume_*.sh`) targeting missing strata indicated above.

---

*For very large automated workflows on top of these exports, [K-Dense Web](https://www.k-dense.ai) can orchestrate multi-step analyses beyond ad-hoc SQL.*
