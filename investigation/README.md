# Investigation Folder: Data Integrity & Analysis Audit

**Project:** "How Alignment Shapes Social Conformity" (CoLM 2026)
**Last Updated:** 2026-03-30

This folder contains the complete audit trail for the data integrity investigation, analysis verification, and paper revision process. It is designed to be self-contained — a future collaborator or LLM assistant should be able to reconstruct the entire decision chain from these files.

---

## Reading Order (Recommended)

1. **`INVESTIGATION_REPORT.md`** — The original investigation that identified data integrity issues (missing judge labels, unreliable judge models, missed refusals).

2. **`VERIFICATION_REPORT.md`** — Independent verification of all claims. Documents what holds, what needs hedging, and what was retracted. **Updated post-revision with corrections found during the paper edit.**

3. **`EXPANDED_RESULTS_ANALYSIS.md`** — Table-by-table analysis of all expanded results (cross-family + OLMo family). Includes the corrected McNemar table (post item_id pairing fix).

4. **`ARCHITECTURAL_ANALYSIS.md`** — Cross-referencing conformity results with architectural deep research. Identifies Dense/MoE/Think patterns and alignment strategy correlates. **Note: Some claims here (scale increases conformity, unsupervised > supervised CoT) were retracted in VERIFICATION_REPORT.md.**

5. **`ANALYSIS_DESIGN.md`** — The two-study design rationale: why we keep OLMo and cross-family analyses separate, how the calibration bridge works.

6. **`FINAL_REVISION_LOG.md`** — Every correction applied during the final paper revision, with explanations of why each error occurred and how it was fixed. **Start here if you need to verify that a specific number in the paper is correct.**

7. **`JUDGE_ANALYSIS.md`** — Comprehensive documentation of the 78/22 hybrid labeling pipeline: heuristic parser architecture, GPT-OSS-20B adjudication, label definitions, agreement rates per dataset, and known limitations. Includes judge_report.sh output summaries for all three data directories.

8. **`STATISTICAL_TESTS.md`** — Detailed explanation of every statistical test used in the paper (McNemar's exact binomial, Holm-Bonferroni correction, Wilson score CIs, Haldane-Anscombe OR correction, Cohen's h, Tc metric), including mathematical formulations, assumptions, rationale for choosing each test, and worked examples.

---

## Key Files

### Reports (Markdown)
| File | Purpose |
|------|---------|
| `README.md` | This index |
| `INVESTIGATION_REPORT.md` | Original data integrity investigation |
| `VERIFICATION_REPORT.md` | Independent claim verification |
| `EXPANDED_RESULTS_ANALYSIS.md` | Detailed analysis of all tables |
| `ARCHITECTURAL_ANALYSIS.md` | Architecture × conformity patterns |
| `ANALYSIS_DESIGN.md` | Two-study experimental design rationale |
| `FINAL_REVISION_LOG.md` | All corrections from final paper revision |
| `JUDGE_ANALYSIS.md` | 78/22 hybrid judge pipeline documentation |
| `STATISTICAL_TESTS.md` | Statistical test rationale and formulations |

### Scripts (Python)
| File | Purpose | Idempotent? |
|------|---------|-------------|
| `investigate_all_runs.py` | v1 investigation — scans all DBs for judge coverage | Yes |
| `investigate_v2.py` | v2 investigation — compares heuristic vs judge labels | Yes |
| `sample_validate_labels.py` | Random sample validation of JSON structure | Yes |
| `fix_judge_refusal_flags.py` | Patches judge `refusal_flag` for "unsure"/"uncertain" | **Modifies DBs** |
| `scoped_analysis_check.py` | Scoped analysis with correct data subset | Yes |

### Data (CSV)
| File | Rows | Description |
|------|------|-------------|
| `v2_run_summary.csv` | 25 | Per-run inventory with agreement metrics |
| `v2_judge_model_agreement.csv` | 7 | Per-judge-model agreement breakdown |
| `v2_per_condition_agreement.csv` | 14 | Per-condition heuristic/judge agreement |
| `v2_per_dataset_agreement.csv` | 8 | Per-dataset agreement |
| `v2_all_mismatches.csv` | ~48K | Every heuristic/judge `is_correct` disagreement |
| `v2_qwen_flash_mismatches.csv` | ~5.4K | Qwen/Flash-specific mismatches |
| `per_model_summary.csv` | 11 | Per-model aggregate stats |
| `run_inventory_summary.csv` | 25 | Full run inventory |
| `judge_model_distribution.csv` | 8 | Which judge models were used and how often |
| `missing_judge_labels.csv` | 18 | Runs with missing judge labels (pre-fix) |
| `all_trial_details.csv` | ~244K | Trial-level details (large file) |

---

## Critical Facts for Future Reference

### Label Architecture
- **`conformity_outputs.is_correct`** (SQL column) = heuristic/text-parser label
- **`parsed_answer_json.is_correct`** (JSON) = LLM judge label ← **AUTHORITATIVE for paper**
- **`conformity_outputs.refusal_flag`** (SQL) = heuristic refusal ← **AUTHORITATIVE for paper** (after phrase expansion)
- **`parsed_answer_json.refusal_flag`** (JSON) = judge refusal (surgically patched)
- **`parsed_answer_json._llm_judge`** = metadata (judge model, prompt version)

### Refusal Phrases Added
Added to `src/vivarium/experiments/olmo_conformity/enhanced_scoring.py` line 32-57:
`"i am unsure"`, `"i'm unsure"`, `"unsure"`, `"i am not sure"`, `"not confident"`, `"uncertain"`

### McNemar Pairing
McNemar tests pair by `item_id` (inner join). Items where either control or pressure has null `is_correct` are excluded. This is implemented in `Analysis Scripts/cross_family_behavioral_analysis.py`.

**BUG HISTORY:** The initial script paired by array position (sorted by trial_id), which scrambled items. Fixed by adding `item_id` to the data loader SQL and merging on it. See `FINAL_REVISION_LOG.md` for details.

### Data Directory Summary (as of 2026-03-30)

| Directory | Purpose | DBs | Total Trials | Status |
|-----------|---------|-----|-------------|--------|
| `runs/` | Study 2: cross-family (11 models × T=0.0,0.6) + ablations | 28 | ~44,800 | 27 complete, 1 partial (GPT-OSS-20B T=0.0) |
| `runs/think/` | OLMo-7B-Think standalone run | 1 | 1,609 | Complete |
| `runs_latest/runs/` | Study 1: OLMo-7B within-family (8 variants × 6 temps) | 6 | ~215,288 | All complete |

### Excluded / Incomplete Data
- `gpt-oss-20b` T=0.0 (UUID `66765d5e`): Only 232/1600 outputs — run interrupted. 4 retry attempts (UUIDs starting with `20260330_17...`) all have 0 outputs.
- `runs_latest/` variant `rl_zero`: Only 25% trial coverage at 4 of 6 temperatures. Not used in paper.
- `runs/think/` OLMo-7B-Think-SFT: Only 2 trials (test run). Not usable.

### Analysis Output Locations
- `Comparing_Experiments/expanded_results/olmo_family/` — OLMo 12-condition analysis (judge labels)
- `Comparing_Experiments/expanded_results/cross_family/` — Cross-family 4-condition analysis
- `Comparing_Experiments/expanded_results/bridge/` — Calibrated ranking connecting both studies
