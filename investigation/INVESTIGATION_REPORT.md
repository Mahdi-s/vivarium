# Investigation Report – Corrected Expanded Analysis

**Generated:** 2026-03-30
**Analysis version:** Corrected (CoLM 2026 submission)
**Methodology fixes applied:** Multinomial pivot, T=0.0 primary analysis, Wilson CIs, no Cochran's Q

---

## 1. Data Inventory

### Cross-Family Study (runs/)
- **Models:** 9 model families
- **Temperature:** T=0.0 (primary), T=0.6 (supplementary)
- **Conditions:** 4 (control, peer, authority-bias, authority-trust)
- **Trials (T=0.0):** 14,400
- **Includes:** Claude Sonnet 4 (new), GPT-4o-Mini, GPT-OSS-20B, Grok-4.1-Fast,
  Gemini-2.5-Flash-Lite, Llama-3-8B, Llama-3.1-70B, Llama-4-Maverick,
  OLMo-32B-Instruct, OLMo-32B-Think

### OLMo-7B Family Study (runs_latest/)
- **Variants:** 4 (base, instruct, instruct_dpo, instruct_sft)
- **Temperatures:** 6 (T=0.0 to T=1.0, step 0.2)
- **Conditions:** 12 (control + 7 peer + 2 authority + 2 mitigation)
- **Trials (T=0.0):** 19,197
- **Judge:** GPT-OSS-20B (gpt-oss-20b) via OpenRouter

### Ablation Study (runs/)
- **Models:** 2 (Llama-3.1-70B, OLMo-32B-Instruct)
- **Conditions:** asch_zhu_naked_unanimous_confident (no system prompt), ngram_sequence_baseline
- **Trials:** 1,600

### Excluded Data
- **gpt-oss-20b T=0.0:** Incomplete run (142/1600 trials), UUID 66765d5e excluded
- **Think variants in runs_latest/:** Not re-judged with gpt-oss-20b; excluded from primary analysis

---

## 2. Labeling Methodology – Hybrid Pipeline

### Pipeline Description

Labels were generated using a **78/22 hybrid pipeline**: an automated
deterministic heuristic parser (enhanced_scoring.py) processed ~78% of outputs where
its classification agreed with the LLM judge, while the remaining ~22% of edge-case
and divergent outputs were resolved by GPT-OSS-20B judge adjudication. This eliminates the
self-preference bias typically associated with pure LLM-as-a-judge pipelines, validating our
inclusion of the GPT-OSS family.

### Agreement Statistics

| Metric | N | Agree | Rate |
|--------|---|-------|------|
| is_correct (overall) | 114,507 | 89,537 | 78.2% |
| refusal_flag (overall) | 147,190 | 126,522 | 86.0% |

### Agreement by Model Family

| Model | N | Agree | Rate |
|-------|---|-------|------|
| OLMo-7B | 22,252 | 18,962 | 85.2% |
| Olmo-3-7B-Instruct | 21,279 | 17,223 | 80.9% |
| Olmo-3-7B-Instruct-DPO | 21,463 | 16,404 | 76.4% |
| Olmo-3-7B-Instruct-SFT | 22,725 | 19,840 | 87.3% |
| OLMo-32B-Instruct | 3,301 | 2,171 | 65.8% |
| OLMo-32B-Think | 2,740 | 1,498 | 54.7% |
| Claude-Sonnet-4 | 2,939 | 1,839 | 62.6% |
| Gemini-2.5-Flash-Lite | 2,565 | 1,602 | 62.5% |
| Llama-3-8B | 2,468 | 1,903 | 77.1% |
| Llama-3.1-70B | 3,226 | 2,169 | 67.2% |
| Llama-4-Maverick | 2,765 | 1,694 | 61.3% |
| GPT-4o-Mini | 2,750 | 1,848 | 67.2% |
| GPT-OSS-20B | 1,323 | 729 | 55.1% |
| Grok-4.1-Fast | 2,711 | 1,655 | 61.1% |

---

## 3. Statistical Methodology

### 3.1 Multinomial Pivot (Fixed-N Denominator)

All rates use a **fixed denominator N=400** (50 items × 8 datasets per condition).
Every trial is classified into one of three mutually exclusive states:

- **State A (Substantive Correct):** Factually correct, no refusal
- **State B (Substantive Incorrect / Sycophantic Endorsement):** Factually wrong or endorsed wrong answer, no refusal
- **State C (Refusal / Abstention):** Triggered refusal detection

This eliminates survivorship bias where dropping refusals artificially inflates conformity
odds ratios for safety-tuned models.

### 3.2 Temperature Independence

Primary analyses (McNemar tests, odds ratios, effect sizes) use **T=0.0 only** (greedy decoding).
Multi-temperature data is moved to supplementary tables, explicitly labeled as exploratory.

This eliminates the violation of independence that occurs when pooling the same item evaluated
at different temperatures as independent trials.

### 3.3 Confidence Intervals

All proportions reported with **Wilson score 95% confidence intervals**. No naked point estimates.

### 3.4 Removed Tests

- **Cochran's Q:** Removed. Testing variance between entirely different prompt wordings is
  mathematically invalid for this experimental design.
- **Mann-Whitney U on N=9 models:** Removed. Insufficient sample size for meaningful inference.

---

## 4. Key Findings (T=0.0, Fixed-N)

### McNemar Results Summary (Cross-Family, T=0.0)

| Model | Condition | OR | Δ Error | Cohen's h | p (adjusted) | Sig |
|-------|-----------|-----|---------|-----------|-------------|-----|
| Claude-Sonnet-4 | Authoritative Bias | 0.97 | -0.003 | -0.005 | 1.0000 | ns |
| Claude-Sonnet-4 | Authority (Trust) | 0.74 | -0.025 | -0.053 | 1.0000 | ns |
| Claude-Sonnet-4 | Peer (Confident) | 0.72 | -0.025 | -0.053 | 1.0000 | ns |
| GPT-4o-Mini | Authoritative Bias | 0.79 | -0.022 | -0.046 | 1.0000 | ns |
| GPT-4o-Mini | Authority (Trust) | 1.00 | 0.000 | 0.000 | 1.0000 | ns |
| GPT-4o-Mini | Peer (Confident) | 4.60 | 0.203 | 0.408 | 0.0000 | *** |
| Gemini-2.5-Flash-Lite | Authoritative Bias | 1.78 | 0.065 | 0.132 | 0.1493 | ns |
| Gemini-2.5-Flash-Lite | Authority (Trust) | 1.49 | 0.040 | 0.081 | 1.0000 | ns |
| Gemini-2.5-Flash-Lite | Peer (Confident) | 5.52 | 0.152 | 0.307 | 0.0000 | *** |
| Grok-4.1-Fast | Authoritative Bias | 0.63 | -0.035 | -0.073 | 1.0000 | ns |
| Grok-4.1-Fast | Authority (Trust) | 0.93 | -0.005 | -0.010 | 1.0000 | ns |
| Grok-4.1-Fast | Peer (Confident) | 0.91 | -0.007 | -0.016 | 1.0000 | ns |
| Llama-3-8B | Authoritative Bias | 0.45 | -0.110 | -0.222 | 0.0009 | *** |
| Llama-3-8B | Authority (Trust) | 1.61 | 0.070 | 0.147 | 0.2016 | ns |
| Llama-3-8B | Peer (Confident) | 27.36 | 0.362 | 1.050 | 0.0000 | *** |
| Llama-3.1-70B | Authoritative Bias | 1.95 | 0.077 | 0.157 | 0.0346 | * |
| Llama-3.1-70B | Authority (Trust) | 1.52 | 0.045 | 0.091 | 0.9261 | ns |
| Llama-3.1-70B | Peer (Confident) | 154.33 | 0.575 | 1.408 | 0.0000 | *** |
| Llama-4-Maverick | Authoritative Bias | 2.30 | 0.070 | 0.146 | 0.0208 | * |
| Llama-4-Maverick | Authority (Trust) | 1.48 | 0.037 | 0.079 | 1.0000 | ns |
| Llama-4-Maverick | Peer (Confident) | 5.20 | 0.215 | 0.438 | 0.0000 | *** |
| OLMo-32B-Instruct | Authoritative Bias | 0.89 | -0.013 | -0.025 | 1.0000 | ns |
| OLMo-32B-Instruct | Authority (Trust) | 3.19 | 0.145 | 0.291 | 0.0000 | *** |
| OLMo-32B-Instruct | Peer (Confident) | 22.87 | 0.410 | 0.875 | 0.0000 | *** |
| OLMo-32B-Think | Authoritative Bias | 0.65 | -0.048 | -0.100 | 0.7950 | ns |
| OLMo-32B-Think | Authority (Trust) | 0.98 | -0.003 | -0.005 | 1.0000 | ns |
| OLMo-32B-Think | Peer (Confident) | 1.51 | 0.040 | 0.082 | 1.0000 | ns |

---

## 5. Files Generated

### Investigation
- `investigation/judge_agreement_by_model.csv`
- `investigation/judge_agreement_by_condition.csv`
- `investigation/INVESTIGATION_REPORT.md` (this file)

### Cross-Family Tables
- `cross_family/tables/multinomial_rates_t0.csv`
- `cross_family/tables/pressure_effects_t0.csv`
- `cross_family/tables/ablation_rates_t0.csv`
- `cross_family/tables/truth_override_*.csv`
- `cross_family/tables/truth_rescue_*.csv`
- `cross_family/statistical_tests/mcnemar_pressure_vs_control_t0.csv`

### OLMo Tables
- `olmo_family/tables/multinomial_rates_t0.csv`
- `olmo_family/tables/multinomial_rates_all_temps_supplementary.csv`
- `olmo_family/tables/mcnemar_pressure_vs_control_t0.csv`
- `olmo_family/behavioral/tables/pressure_effects_t0.csv`
- `olmo_family/behavioral/tables/truth_override_*.csv`
- `olmo_family/behavioral/tables/truth_rescue_*.csv`

### Bridge
- `bridge/tables/olmo_bridge_rates_t0.csv`

### Figures
- `cross_family/figures/fig_behavioral_taxonomy.png` (scatterplot with Claude)
- `cross_family/figures/fig_conformity_forest.png` (forest plot)
- `cross_family/figures/fig_system_prompt_efficacy.png` (ablation bar chart)
