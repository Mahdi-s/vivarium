# Judge Label Integrity Investigation Report

**Date:** 2026-03-29 (Final)
**Scope:** Scoped to paper-relevant trials only
**Purpose:** Verify data integrity and prepare for post-hoc analysis

---

## Executive Summary

### Data in Scope

| Source | What's Included | Trials |
|--------|----------------|--------|
| `runs/` | All cross-family models, 4 conditions, 8 datasets, T=0.0 & 0.6 | 27,200 |
| `runs/think/` | OLMo-3-7B-Think, 4 conditions, 8 datasets, T=0.0 | 1,608 |
| `runs_latest/runs/` | OLMo-3-1025-7B: **base, instruct, instruct_sft, instruct_dpo** only; 12 conditions, 8 datasets, 6 temps | 115,200 |
| **Total in scope** | | **144,008** |

**Excluded:** `gpt-oss-20b` T=0.0 (incomplete, 143 trials); `runs_latest/` variants: think, think_sft, think_dpo, rl_zero

### Data Integrity Status: READY FOR ANALYSIS

| Check | Result |
|-------|--------|
| Judge coverage | **100%** (144,008/144,008) |
| Sample validation (250 trials) | **250/250 valid** (0 structural issues) |
| `is_correct` heuristic-judge agreement | **77.4%** |
| `refusal_flag` heuristic-judge agreement | **88.4%** |
| Refusal fix applied | Yes — "unsure", "uncertain" etc. patched in both heuristic and judge |

---

## 1. Scoped Run Inventory

### 1.1 `runs/` — Cross-Family Comparison (17 runs)

All judged by `openai/gpt-oss-20b`. 4 core conditions (`control`, `authoritative_bias`, `authority_trust`, `asch_zhu_unanimous_confident`), 8 datasets, 50 items/cell, 1,600 trials per run.

| Model | Variant | T=0.0 UUID | T=0.6 UUID | Judge `correct=1` rate | Refusal rate | Endorsed rate |
|-------|---------|------------|------------|----------------------|--------------|---------------|
| olmo-3.1-32b-think | think_32b | a34ad9b1 | 7db9896e | 63-64% | 9% | 10% |
| olmo-3.1-32b-instruct | instruct_32b | 1c2e5cb6 | 62187f52 | 48% | 20% | 18% |
| llama-3-8b-instruct | llama3_8b_instruct | 1899a883 | 70860876 | 31% | 30-32% | 13-14% |
| llama-3.1-70b-instruct | llama31_70b_instruct | 3a0404f7 | 49d07104 | 45-49% | 23-25% | 11% |
| llama-4-maverick | llama4_maverick | 485ddc2d | c2ce0f85 | 61-63% | 7% | 15-16% |
| gemini-2.5-flash-lite | gemini_25_flash_lite | e043fbf6 | d71e75b1 | 55-56% | 6-7% | 15% |
| grok-4.1-fast | grok_41_fast | 25056752 | 157a6a9e | 65% | 8% | 8% |
| gpt-4o-mini | gpt4o_mini | c07ede3a | eb63d212 | 55-56% | 3-4% | 20% |
| gpt-oss-20b | gpt_oss_20b | — | 3ecdc9b7 | 59% | 4% | 17% |

**Notable patterns from judge labels:**
- **Highest refusal rate:** llama-3-8b-instruct (30-32%) — this model frequently refuses under pressure
- **Highest endorsement rate:** gpt-4o-mini (20%) — most sycophantic
- **Lowest refusal:** gpt-4o-mini (3-4%) — rarely refuses, tends to comply
- **Highest accuracy:** grok-4.1-fast (65%) — best at resisting wrong answers

### 1.2 `runs/think/` — OLMo 7B Think (1 run)

| UUID | Trials | Judge correct=1 | Refusal | Endorsed |
|------|--------|-----------------|---------|----------|
| f47fe05e | 1,608 | 48% | 14% | 21% |

### 1.3 `runs_latest/runs/` — OLMo 7B Family (6 runs, 4 variants each)

12 conditions, 8 datasets, 50 items/cell, 4,800 trials per variant per temperature = 19,200 per run.

| Temp | UUID | Judge correct=1 | Refusal | Endorsed |
|------|------|-----------------|---------|----------|
| 0.0 | 9f240f89 | 16% | 30% | 24% |
| 0.2 | 46f0762a | 16% | 30% | 24% |
| 0.4 | bbd05985 | 17% | 30% | 24% |
| 0.6 | 86c72262 | 17% | 30% | 23% |
| 0.8 | 9369442d | 18% | 31% | 25% |
| 1.0 | 9173bfae | 19% | 30% | 24% |

**Note:** The low `correct=1` rate (16-19%) is expected — these are the full 12 conditions including heavy social pressure, and the 7B base model is more susceptible to conformity.

---

## 2. Agreement Rates (Heuristic vs Judge)

### 2.1 Overall (Scoped)

| Field | Comparable | Agree | Agreement% |
|-------|-----------|-------|------------|
| `is_correct` | 122,349 | 94,754 | **77.4%** |
| `refusal_flag` | 144,008 | 127,372 | **88.4%** |

### 2.2 By Location

| Location | `is_correct` | `refusal_flag` |
|----------|-------------|----------------|
| `runs/` (gpt-oss-20b judge) | 63.4% | **95.8%** |
| `runs/think/` (gpt-oss-20b judge) | 60.6% | **98.8%** |
| `runs_latest/` (mixed judges) | 80.8% | **86.6%** |

### 2.3 By Model Family

| Model | `is_correct` | `refusal` | Combined | n |
|-------|-------------|-----------|----------|---|
| llama-3-8b-instruct | **77.1%** | **99.4%** | **76.5%** | 2,468 |
| gpt-4o-mini | **67.2%** | **99.2%** | **67.1%** | 2,750 |
| llama-3.1-70b-instruct | 65.2% | 99.2% | 64.9% | 2,531 |
| OLMo-3-1025-7B (4 variants) | 80.8% | 86.6% | — | 98,645 |
| olmo-3.1-32b-instruct | 64.0% | 94.9% | 58.6% | 2,627 |
| gemini-2.5-flash-lite | 62.5% | 92.9% | 61.4% | 2,565 |
| llama-4-maverick | 61.3% | 99.8% | 61.2% | 2,765 |
| grok-4.1-fast | 61.0% | 96.7% | 60.9% | 2,711 |
| OLMo-3-7B-Think | 60.6% | 98.8% | 60.5% | 1,224 |
| olmo-3.1-32b-think | 54.7% | 86.7% | 53.1% | 2,740 |
| gpt-oss-20b (T=0.6 only) | 55.1% | 91.2% | — | 1,323 |

---

## 3. Design Notes for Analysis

### 3.1 `runs/` vs `runs_latest/` — Different Designs

| Dimension | `runs/` (cross-family) | `runs_latest/` (OLMo family) |
|-----------|----------------------|------------------------------|
| Models | 9 diverse model families | 1 model, 4 training stages |
| Conditions | 4 core | 12 full suite |
| Temperatures | 0.0, 0.6 | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| Items/cell | 50 | 50 |
| Purpose | Cross-family conformity comparison | Training stage + temperature effects |

### 3.2 Which Labels to Use

| Label | Source | Use For |
|-------|--------|---------|
| `parsed_answer_json.is_correct` | Judge (LLM) | **Correctness** — authoritative |
| `conformity_outputs.refusal_flag` | Heuristic (post-fix) | **Refusal detection** — authoritative |
| `parsed_answer_json.wrong_answer_endorsed` | Judge (LLM) | **Sycophantic conformity** |
| `parsed_answer_json.refusal_flag` | Judge (post-fix) | Cross-validation for refusal |
| `conformity_outputs.is_correct` | Heuristic | Cross-validation for correctness |

---

## 4. Fixes Applied (Audit Trail)

| Fix | Rows Affected | Method |
|-----|--------------|--------|
| Added 6 refusal phrases to `enhanced_scoring.py` | Code change | `"i am unsure"`, `"i'm unsure"`, `"unsure"`, `"i am not sure"`, `"not confident"`, `"uncertain"` |
| Re-scored heuristic SQL columns | 144,150 | `scripts/rescore_outputs.py` |
| Patched judge `refusal_flag` in `parsed_answer_json` | 25,320 | `investigation/fix_judge_refusal_flags.py` |
| Re-judged `runs/` + `runs/think/` with `gpt-oss-20b` | 28,950 | `scripts/run_judge_openrouter.py --force` |

---

## 5. Post-Hoc Analysis: Ready to Run

### 5.1 OLMo 7B Family Analysis (`runs_latest/`)

The existing pipeline already supports this with judge labels:

```bash
python "Analysis Scripts/expanded_suite_behavioral_breakdown.py" \
  --runs-dir runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json \
  --out-dir Comparing_Experiments/publication_V2 \
  --use-judge-labels \
  --exclude-variants think think_sft think_dpo rl_zero \
  --include-extra-conditions \
  --publication
```

This produces:
- `tables/error_rates_all_conditions.csv` — accuracy per temp/variant/condition/dataset
- `tables/pressure_effects_all_conditions.csv` — delta from control
- `tables/truth_override_all_conditions.csv` — override rates
- `tables/wrong_answer_flip_all_conditions.csv` — flip rates
- `statistical_tests/bootstrap_cis.csv`, `mcnemar_pressure_vs_control.csv`, `cochrans_q_condition_families.csv`
- Publication-quality figures (heatmaps, lollipop plots)

### 5.2 Cross-Family Analysis (`runs/`)

The cross-family runs need either:
- A new metadata JSON pointing to the `runs/` databases, OR
- A new analysis script that discovers runs directly

The existing `expanded_suite_behavioral_breakdown.py` expects a metadata JSON with `{temp: {run_id, run_dir}}` format. For cross-family runs where each model is a separate run (not temperature variant), we'll need a script that iterates per-model rather than per-temperature.

### 5.3 Think Trace Analysis (`runs/think/` + `runs_latest/` think variants)

```bash
python scripts/analyze_think_traces.py \
  --runs-dir runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json \
  --item-set Comparing_Experiments/publication_V2/item_set.csv \
  --out-dir Comparing_Experiments/publication_V2/mechanistic
```

---

## 6. Remaining Issues

| Issue | Impact | Action |
|-------|--------|--------|
| `gpt-oss-20b` T=0.0 incomplete (143 trials) | Excluded from analysis | Complete or accept N/A |
| `runs_latest/` mixed judge models | Potential inconsistency | Optional: re-judge with gpt-oss-20b (~$5) |
| `runs/` needs cross-family analysis script | Can't run existing pipeline directly | Need new script or metadata |

---

## 7. Files Produced

| File | Description |
|------|-------------|
| `INVESTIGATION_REPORT.md` | This report |
| `scoped_analysis_check.py` | Scoped analysis with correct data subset |
| `investigate_v2.py` | Full investigation script |
| `sample_validate_labels.py` | Sample validation (250 trials) |
| `fix_judge_refusal_flags.py` | Surgical judge refusal flag repair |
| `v2_run_summary.csv` | Full inventory |
| `v2_judge_model_agreement.csv` | Per-judge-model agreement |
| `v2_per_condition_agreement.csv` | Per-condition agreement |
| `v2_per_dataset_agreement.csv` | Per-dataset agreement |
| `v2_all_mismatches.csv` | All heuristic/judge disagreements |
