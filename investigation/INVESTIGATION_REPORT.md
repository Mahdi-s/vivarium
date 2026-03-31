# Investigation Report – Corrected Expanded Analysis

**Generated:** 2026-03-30
**Analysis version:** Corrected (CoLM 2026 submission)
**Methodology fixes applied:** Multinomial pivot, T=0.0 primary analysis, Wilson CIs, no Cochran's Q

---

## 1. Data Inventory

### Cross-Family Study (`runs/`)

- **Models:** 10 model families
- **Temperature:** T=0.0 (primary, greedy), T=0.6 (supplementary)
- **Conditions:** 4 (control, peer_consensus, authority_bias, authority_trust)
- **Clean Trials (T=0.0):** 15,613 of 16,000 (387 pending gpt-oss-20b re-run)
- **Clean Trials (T=0.6):** 16,000 ✅
- **Includes:**
  - `allenai/olmo-3.1-32b-instruct` — runs `1c2e5cb6` (T=0.0), `62187f52` (T=0.6)
  - `allenai/olmo-3.1-32b-think` — runs `a34ad9b1` (T=0.0), `7db9896e` (T=0.6)
  - `anthropic/claude-sonnet-4` — runs `5be5ada7` (T=0.0), `21556460` (T=0.6)
  - `google/gemini-2.5-flash-lite` — runs `e043fbf6` (T=0.0), `d71e75b1` (T=0.6)
  - `meta-llama/llama-3-8b-instruct` — runs `1899a883` (T=0.0), `70860876` (T=0.6)
  - `meta-llama/llama-3.1-70b-instruct` — runs `3a0404f7` (T=0.0), `49d07104` (T=0.6)
  - `meta-llama/llama-4-maverick` — runs `485ddc2d` (T=0.0), `c2ce0f85` (T=0.6)
  - `openai/gpt-4o-mini` — runs `c07ede3a` (T=0.0), `eb63d212` (T=0.6)
  - `openai/gpt-oss-20b` — run `3ecdc9b7` (T=0.6, complete); run `66765d5e` (T=0.0, 1,413/1,600 clean — 187 rate-limit errors being re-run via `--resume-auto`)
  - `x-ai/grok-4.1-fast` — runs `25056752` (T=0.0), `157a6a9e` (T=0.6)

### Ablation Study (`runs/`)

Tests system-prompt protection and sequential-pattern null model. These are **complete** at 800 trials each (2 conditions × 400 items × 1 model — intentional design, not incomplete runs).

- **Models:** 2 (`meta-llama/llama-3.1-70b-instruct`, `allenai/olmo-3.1-32b-instruct`)
- **Temperature:** T=0.0
- **Conditions:** 2
  - `asch_zhu_naked_unanimous_confident` — same Asch peer setup as the main suite but without a system prompt; tests whether the system prompt provides conformity resistance
  - `ngram_sequence_baseline` — n-gram sequence-completion framing without adversarial social content; isolates pure pattern-completion from genuine social pressure response
- **Runs:** `e8a90500` (llama-3.1-70b), `ef72529e` (olmo-3.1-32b-instruct)
- **Trials:** 800 each ✅

### OLMo-7B-Think Exploratory Run (`runs/think/`)

- **Model:** `allenai/Olmo-3-7B-Think`
- **Run ID:** `f47fe05e`
- **Temperature:** T=0.0
- **Conditions:** 4 (core: control, peer_consensus, authority_bias, authority_trust)
- **Trials:** 1,609 (1,608 clean; +9 overfill from mid-run restart, deduplicated before analysis)
- **Judge:** GPT-OSS-20B via OpenRouter

### OLMo-7B Training Stage Study (`runs_latest/runs/`)

- **Variants (primary):** 4 — base, instruct, instruct_sft, instruct_dpo
- **Variants (excluded from primary):** 4 — think, think_sft, think_dpo, rl_zero_math (incomplete across most temperature runs)
- **Temperatures:** 6 (T=0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
- **Conditions:** 12 (control + 7 peer variants + 2 authority + 2 mitigation)
- **Run IDs:** `9f240f89` (T=0.0), `46f0762a` (T=0.2), `bbd05985` (T=0.4), `86c72262` (T=0.6), `9369442d` (T=0.8), `9173bfae` (T=1.0)
- **Trials (T=0.0, primary variants):** ~17,400 ✅
- **Judge:** GPT-OSS-20B via OpenRouter

### Condition Overlap Between Study Arms

The 4-condition cross-family suite (`runs/`) shares the following conditions with the 12-condition OLMo-7B suite (`runs_latest/runs/`), enabling a calibrated bridge ranking:

| Cross-family condition | Corresponding OLMo-7B condition |
|---|---|
| `control` | `control` |
| `asch_zhu_unanimous_confident` | `asch_zhu_unbiased_unanimous_confident` |
| `authoritative_bias` | `authoritative_bias` |
| `authority_trust` | `authority_zhu_unbiased_trust` |

### Data Notes

- **GPT-OSS-20B T=0.0 (`66765d5e`):** Corrupted by concurrent writes on 2026-03-27 (1,145 duplicate outputs, 681 rate-limit errors). Cleaned 2026-03-31: 1,413 trials have one clean output; 187 remain as rate-limit stubs pending `--resume-auto` re-run. Excluding from primary analysis until re-run completes; T=0.6 results (`3ecdc9b7`) are unaffected and fully usable.
- **Ghost run `621a7698`:** Claude Sonnet 4 T=0.6 with 1,600 registered trials but 0 outputs — killed before fan-out. Duplicate of completed run `21556460`. Ignore.
- **Think variants in `runs_latest/`:** Not re-judged with GPT-OSS-20B; excluded from primary analysis. OLMo-7B-Think uses the separate `f47fe05e` run in `runs/think/`.

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
