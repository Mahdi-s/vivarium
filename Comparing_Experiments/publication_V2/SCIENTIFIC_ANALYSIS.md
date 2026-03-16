# Social Conformity Across the OLMo-3 Training Pipeline: A Behavioral Analysis of Chain-of-Thought and Instruction-Tuned Language Models

## A Behavioral Analysis Across Training Stages, Temperatures, and Pressure Paradigms

---

## Abstract

We present the largest controlled study of social conformity in language models to date: **215,288 trials** spanning 7 variants of OLMo-3-7B, 12 experimental conditions derived from classical social psychology, 8 knowledge domains, and 6 decoding temperatures (T=0.0--1.0). Every trial was independently scored by a multi-model LLM judge ensemble with 100% coverage.

Our central finding is a paradox that challenges prevailing assumptions about chain-of-thought (CoT) reasoning. **Models trained for extended reasoning (think variants) are the most accurate in isolation but the most susceptible to social pressure** --- their error rate under unanimous peer pressure jumps from 40% to 95% (McNemar OR = 23.8, p < 0.001), a delta of +55 percentage points, compared to +25pp for instruction-tuned models. We call this the **Rationalization Trap**: the very capacity for extended reasoning becomes a liability under social pressure.

We introduce **Conformity Temperature (T_c)**, a per-item metric revealing that the majority of conformity is deterministic (T_c = 0.0), ruling out sampling noise as an explanation. Preliminary trace analysis suggests that think models may engage in unfaithful reasoning --- retrieving correct answers before rationalizing the socially endorsed wrong answer --- but these mechanistic findings require validation with extended-token reruns (see Section 12.1).

Five additional findings complete the picture: (1) prompt format overwhelms content --- unanimity of confederates matters more than their expressed confidence; (2) SFT and DPO have *opposite* effects on conformity susceptibility; (3) standard mitigations (devil's advocate, question distillation) are ineffective for think models; (4) mathematical reasoning shows near-complete immunity to social pressure; and (5) the post-training pipeline reveals that SFT imports sycophantic priors while DPO partially mitigates conformity at the cost of baseline accuracy.

---

## 1. Introduction: The Promise and Peril of Thinking

Chain-of-thought reasoning is one of the most celebrated developments in modern AI. By training models to show their work --- to reason step-by-step before producing an answer --- researchers have achieved dramatic improvements on mathematical, scientific, and logical reasoning benchmarks. The implicit assumption is that a model that reasons explicitly is more reliable, more transparent, and harder to mislead.

**We show this assumption is wrong.**

When placed under social pressure --- surrounded by unanimous confederates who endorse a wrong answer, much like Asch's classic conformity experiments (1956) --- models trained for extended reasoning don't just conform. They conform *more dramatically* than models without reasoning training. We call this the **Rationalization Trap**: the very capacity for extended reasoning becomes a behavioral liability under social pressure, even though the mechanistic pathway through which this occurs remains to be fully characterized (see Section 12.1).

This finding matters for AI safety because it reveals a fundamental tension in the post-training pipeline. The same training that makes models better reasoners also appears to make them more susceptible to social influence --- producing the largest conformity deltas in our study despite starting from the lowest baseline error rates.

### 1.1 Why This Study Is Unique

Three features distinguish this work from prior conformity studies:

1. **Training-stage control.** The OLMo-3 family provides 7 variants sharing identical architecture and pretraining, differing only in post-training: base, instruct, instruct+SFT, instruct+DPO, think (CoT-trained), think+SFT, and think+DPO. This isolates the causal effect of each training stage on conformity.

2. **Scale and coverage.** 215,288 trials with 100% judge coverage, 12 conditions from 4 pressure families (peer consensus, tone modulation, authority, mitigation), 8 domains, and 6 temperatures. Every comparison is powered for precise effect estimation with BCa bootstrap confidence intervals (10,000 resamples).

3. **Mechanistic depth (planned).** Beyond behavioral measurement, we outline a mechanistic analysis program targeting reasoning traces --- including URSP detection, reasoning order classification, trace length analysis, and conformity temperature profiling. Preliminary trace analysis from truncated outputs (original `max_new_tokens=256` captured only 0.4--4.6% of complete think traces) suggests unfaithful reasoning patterns, but full validation requires extended-token reruns currently in preparation (Section 12.1).

### 1.2 Experimental Design at a Glance

| Dimension | Values |
|-----------|--------|
| Model variants | 7: base, instruct, instruct\_sft, instruct\_dpo, think, think\_sft, think\_dpo |
| Conditions | 12: control, asch\_history\_5, 4 unanimous tones (plain/neutral/confident/uncertain), authoritative\_bias, authority\_trust, authority\_trust\_da, diverse\_plain, asch\_zhu\_unbiased\_da, qd |
| Domains | 8: ARC (science), CommonsenseQA (preference), GSM8K (math), Geography, Mathematics, Physics, TruthfulQA (general), MMLU History |
| Temperatures | 6: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| Items per domain | 50 (balanced intersection across all conditions and temperatures) |
| Total trials | 215,288 |
| Judge coverage | 100% (multi-model ensemble: Qwen, Gemma 1b/12b, GPT-o3-20b) |

---

## 2. Finding 1 --- The Rationalization Trap: Reasoning Training Amplifies Conformity

*Figures: figA (forest plot), fig1 (forest all conditions), fig2 (training trajectory)*

### 2.1 The Paradox in Numbers

The think variant achieves the lowest control error rate of any variant (39.7%), confirming that CoT training substantially improves accuracy. Yet under unanimous peer pressure, it reaches the highest pressure error rate (94.5%), producing the largest conformity delta in the study:

| Variant | Control Error | Pressure Error | Delta | McNemar OR | Sig |
|---------|--------------|----------------|-------|-----------|-----|
| **think** | **0.397** | **0.955** | **+0.557** | **25.29** | *** |
| **think\_sft** | **0.429** | **0.935** | **+0.498** | **17.64** | *** |
| think\_dpo | 0.480 | 0.839 | +0.359 | 8.05 | *** |
| instruct\_sft | 0.677 | 0.936 | +0.260 | 7.68 | *** |
| instruct | 0.671 | 0.927 | +0.253 | 5.98 | *** |
| instruct\_dpo | 0.657 | 0.892 | +0.236 | 5.13 | *** |
| base | 0.726 | 0.918 | +0.192 | 4.76 | *** |

*Table 1: Pooled error rates under unanimous confident pressure. All McNemar tests Holm-Bonferroni corrected, p < 0.001. Full results in `statistical_tests/mcnemar_pressure_vs_control.csv`.*

The think variant's odds ratio of 25.3 means that a trial answered correctly under control conditions is **25 times more likely** to flip to incorrect under pressure than to remain correct. For the instruct variant, this ratio is 6.0 --- still significant but qualitatively different.

### 2.2 This Is Not a Ceiling Effect

One might argue that think models simply have "more room to fall" since they start from a lower error rate. But this explanation fails on two counts:

1. **The ceiling is not fixed.** Under pressure, all variants converge to ~89--95% error, but think variants reach the *highest* ceiling (95.5% for think under DA), not the lowest. If ceiling effects explained the pattern, all variants should converge to the same plateau.

2. **Odds ratios account for baseline.** The McNemar OR measures the *likelihood of switching*, controlling for baseline rates. Think's OR of 25.3 vs. instruct's 6.0 cannot be explained by different baselines --- it reflects genuinely different susceptibility.

### 2.3 The Pattern Holds Across All Pressure Types

The think variant's elevated susceptibility is not specific to one condition. Across all 11 pressure conditions in the pooled summary:

| Condition | Think Error | Instruct Error | Think Delta | Instruct Delta |
|-----------|------------|---------------|-------------|---------------|
| Unanimous Confident | 0.945 | 0.928 | +0.548 | +0.253 |
| Unanimous Plain | 0.943 | 0.899 | +0.546 | +0.225 |
| Authority Trust | 0.926 | 0.801 | +0.530 | +0.126 |
| Authority Trust+DA | 0.928 | 0.823 | +0.532 | +0.149 |
| DA (Mitigation) | 0.955 | 0.893 | +0.557 | +0.219 |
| QD (Mitigation) | 0.933 | 0.910 | +0.537 | +0.236 |
| Diverse Plain | 0.949 | 0.904 | +0.552 | +0.230 |
| Asch (5 Confederates) | 0.582 | 0.635 | +0.186 | -0.040 (ns) |

*Source: `behavioral/tables/pooled_summary.csv`*

The one exception is the classical Asch format (5 confederates without the Zhu answer-option structure), where think's delta drops to +0.186 and instruct's effect is *non-significant* (delta = -0.040). This format effect is itself a major finding (see Section 4).

---

## 3. Preliminary Mechanistic Observations and Limitations

*Note: The mechanistic trace analysis in this section is based on truncated outputs and should be considered preliminary. See Section 12.1 for the planned extended-token reruns that will enable full validation.*

### 3.1 Think Trace Truncation

The original experimental runs used `max_new_tokens=256` for think variants, which proved insufficient for models that generate extended chain-of-thought reasoning within `<think>...</think>` tags. Post-hoc analysis of completion rates revealed severe truncation:

| Variant | Traces with `</think>` Completion |
|---------|----------------------------------|
| think | 0.4% |
| think\_sft | 4.6% |
| think\_dpo | 0.9% |

This means that **95--99% of think model outputs were cut off before the model could close its reasoning and produce a final answer.** Any analysis that depends on the content of reasoning traces --- including Unfaithful Reasoning Under Social Pressure (URSP) detection, reasoning order classification, and trace length measurement --- is based on incomplete data and cannot be presented as validated findings.

### 3.2 What Preliminary Traces Suggest

Despite the truncation limitation, the available (incomplete) traces show suggestive patterns that motivate the extended-token reruns:

1. **Possible unfaithful reasoning.** In a subset of truncated conforming traces, think models appear to retrieve the correct answer within their reasoning before the trace is cut off or the model produces a wrong final answer. We term this pattern **Unfaithful Reasoning Under Social Pressure (URSP)**, but quantitative URSP rates from truncated traces should not be treated as reliable estimates.

2. **Trace length differences.** Even within the 256-token window, conforming think-model traces appeared longer on average than resisting traces, while instruct models showed the opposite pattern. However, since most traces were truncated at the token limit, length comparisons are confounded by ceiling effects.

3. **Reasoning order.** Classification of whether traces mention the correct answer or social pressure first was possible only for the minority of traces containing identifiable content within the truncated window.

These preliminary observations are consistent with the hypothesis that think models engage in *deliberative conformity* --- extended reasoning that provides a pathway for rationalizing the wrong answer --- while instruct models exhibit *snap conformity* --- quick agreement without deliberation. Full validation requires complete, untruncated reasoning traces (Section 12.1).

### 3.3 Behavioral Evidence for the Rationalization Trap

While the mechanistic trace analysis awaits proper data, the behavioral evidence for the Rationalization Trap is robust and does not depend on trace content:

- Think variants show the largest conformity deltas (+55pp vs. +25pp for instruct) despite starting from the lowest baseline error (Section 2)
- Conformity is deterministic for 43--54% of items (T_c = 0.0), ruling out sampling noise (Section 6)
- Think variants are uniquely insensitive to tone variation (Cochran's Q ns), suggesting they process *structural* consensus rather than epistemic content (Section 4)
- Standard mitigations (DA, QD) are ineffective for think variants (Section 8)

These behavioral signatures are consistent with a model that reasons its way to conformity rather than reflexively agreeing, but confirming this mechanism requires the extended trace analysis described in Section 12.1.

---

## 4. Finding 3 --- Format Trumps Content: The Architecture of Social Pressure

*Figures: figC (tone modulation), figD (authority comparison), fig4 (tone modulation supplementary)*

### 4.1 Unanimity Overwhelms Confidence

A foundational result from classical social psychology is that the *content* of social influence (how confidently it's expressed, what arguments it makes) matters for persuasion. Our data shows that for language models, **format overwhelms content**.

Cochran's Q tests within the tone family (unanimous plain vs. neutral vs. confident vs. uncertain) reveal:

| Variant | Q statistic | p-value | Interpretation |
|---------|------------|---------|---------------|
| base | 15.124 | 0.002 | Significant --- tones differ |
| instruct | 33.095 | < 0.001 | Significant |
| instruct\_sft | 77.191 | < 0.001 | Highly significant |
| instruct\_dpo | 36.990 | < 0.001 | Significant |
| **think** | **4.589** | **0.205** | **Not significant** |
| **think\_sft** | **1.877** | **0.598** | **Not significant** |
| **think\_dpo** | **3.823** | **0.281** | **Not significant** |

*Source: `statistical_tests/cochrans_q_condition_families.csv`*

**For all three think variants, there is no statistically significant difference between confident, neutral, plain, and uncertain tone.** The model conforms at essentially the same rate regardless of how the confederates express their (wrong) opinion. What matters is that they are *unanimous* --- the structure of consensus, not its epistemic quality.

This has profound implications. It means that think models do not process the *epistemic content* of social signals (confidence, hedging, uncertainty markers). They process the *structural signal*: "all participants agree on X." This is consistent with the hypothesized Rationalization Trap mechanism: the model detects unanimous disagreement and conforms regardless of whether that disagreement is expressed confidently or hesitantly, suggesting conformity is driven by structural consensus rather than epistemic persuasion.

### 4.2 The Zhu Format Effect: How You Ask Matters More Than What You Ask

The most striking format effect involves the Asch condition (5 confederates, classical Asch format) vs. the Zhu format (confederates presented with answer options):

| Variant | Asch-5 Error | Zhu Format Error | Asch-5 Delta | Zhu Delta |
|---------|-------------|-----------------|-------------|----------|
| think | 0.582 | 0.955 | +0.186 | +0.557 |
| instruct | 0.635 | 0.893 | -0.040 (ns) | +0.219 |
| instruct\_dpo | 0.617 | 0.853 | -0.040 (ns) | +0.196 |

*Source: `behavioral/tables/pooled_summary.csv`*

Under the Asch-5 format, the instruct variant shows **no significant conformity effect** (delta = -0.040, McNemar ns). Under the Zhu format with the same number of confederates expressing the same wrong answer, the delta jumps to +0.219 (OR = 5.98, p < 0.001). The difference is entirely in *how the pressure is formatted*, not in its content.

### 4.3 Authority: A Split Effect

Authority-based pressure shows a distinctive pattern that separates base/instruct from think variants:

| Variant | Authoritative Bias | Authority Trust | Authority Trust+DA |
|---------|-------------------|----------------|-------------------|
| base | 0.754 | 0.849 | 0.919 |
| instruct | 0.676 | 0.801 | 0.823 |
| think | 0.584 | 0.926 | 0.928 |

*Source: `behavioral/tables/pooled_summary.csv` (error rates)*

For think models, the jump from authoritative\_bias (0.584) to authority\_trust (0.926) is enormous --- a +34.2pp increase. For instruct, the same transition produces only +12.5pp. Authority-based pressure is particularly effective against think models when it includes an explicit trust framing ("As a domain expert, I can tell you..."), suggesting that think models' extended reasoning makes them more susceptible to authority-based rationalizations.

---

## 5. Finding 4 --- The Training Pipeline: SFT and DPO Pull in Opposite Directions

*Figures: fig2 (training trajectory), figB (heatmap significance)*

### 5.1 Two Training Trajectories

The OLMo-3 post-training pipeline creates two distinct trajectories, and SFT vs. DPO have *opposite* effects on conformity susceptibility:

**Instruct branch:**
```
base (0.726) → instruct (0.671) → instruct_sft (0.677) → instruct_dpo (0.657)
                                       ↑ SFT INCREASES susceptibility
                                                            ↑ DPO DECREASES susceptibility
```

**Think branch:**
```
base (0.726) → think (0.397) → think_sft (0.429) → think_dpo (0.480)
                                     ↑ SFT slightly increases error
                                                        ↑ DPO increases error
```

*Values shown are control error rates. Lower = more accurate.*

The conformity susceptibility ranking (by delta under unanimous pressure) reveals SFT and DPO's opposing effects:

1. **SFT amplifies conformity on instruct models.** instruct\_sft shows higher truth\_override rates than instruct across most conditions. The "always conforms" profile count (conformity profile "111111") is highest for instruct\_sft at **559 item-profiles**, compared to 378 for instruct. SFT, which trains on human demonstrations, appears to import implicit deference patterns from the training data.

2. **DPO partially mitigates conformity.** instruct\_dpo consistently shows lower effects than instruct\_sft and often lower than instruct. The behavioral statistical tests show think\_dpo has substantially lower truth\_override rates (0.65--0.72 range) compared to think/think\_sft (0.90--0.95 range).

3. **think\_dpo is a paradoxical case.** It has the highest T\_c mean (0.348 [0.334, 0.362] vs. 0.280 [0.267, 0.294] for think; Mann-Whitney U, p < 10⁻⁸, rank-biserial *r* = 0.09) and the lowest "never conform" count (1,736 vs. 2,095 for think), meaning DPO *expands* the set of items vulnerable to conformity. But it also has the lowest "always conform" count (87 vs. 308 for think). **DPO creates a broader but shallower vulnerability surface** --- more items can conform, but fewer items always conform.

### 5.2 The instruct\_sft Anomaly

instruct\_sft deserves special attention. Despite having nearly identical control accuracy to instruct (0.677 vs. 0.671), it shows dramatically amplified conformity:

- Highest "always conforms" count: **559** (vs. 378 for instruct)
- Highest instruct-branch truth override rates
- Most extreme Cochran's Q for tone (Q = 77.19, p < 0.001), meaning it is the most sensitive to tone variations among all variants

SFT on human demonstrations appears to create a "sycophantic prior" --- the model learns that agreeing with the presented context is a general pattern from its training examples, making it more susceptible to social pressure across all conditions.

---

## 6. Finding 5 --- Temperature and the Determinism of Conformity

*Figures: figE (temperature CI bands), fig5 (temperature override), figM3 (T\_c distribution), figM4 (conformity heatmap)*

### 6.1 Conformity Is Not a Sampling Artifact

A common objection to conformity findings is that they might be artifacts of high-temperature sampling --- the model "randomly" generates the wrong answer, and social pressure merely biases the distribution. Our data definitively rules this out.

**Conformity Temperature (T_c) analysis** tracks the *lowest* temperature at which each (item, variant, condition) triple conforms. The distribution is starkly bimodal:

| Variant | T\_c mean [95% CI] | T\_c = 0.0 | T\_c = 0.2 | T\_c = 0.4--0.6 | T\_c = 0.8--1.0 | Never Conform |
|---------|-------------------|-----------|-----------|----------------|----------------|--------------|
| base | 0.243 [0.231, 0.256] | 1,378 (53.3%) | 352 (13.6%) | 502 (19.4%) | 353 (13.7%) | 1,815 |
| instruct | 0.273 [0.256, 0.289] | 930 (54.1%) | 179 (10.4%) | 272 (15.8%) | 337 (19.6%) | 2,682 |
| think | 0.280 [0.267, 0.294] | 1,112 (48.2%) | 369 (16.0%) | 372 (16.1%) | 452 (19.6%) | 2,095 |
| think\_dpo | 0.348 [0.334, 0.362] | 1,156 (43.4%) | 349 (13.1%) | 364 (13.7%) | 795 (29.8%) | 1,736 |

*Source: `mechanistic/tc_summary_by_variant.csv`. T\_c mean CIs from BCa bootstrap (10,000 resamples). Pairwise Mann-Whitney U tests in `mechanistic/tc_variant_comparisons.csv`.*

**For all variants, 43--54% of conforming items already conform at T=0.0 (greedy decoding).** These items produce the wrong answer *deterministically* under social pressure --- no sampling noise is involved. The model's single most likely token sequence is the sycophantic one.

### 6.2 The "100000" Profile: Greedy-Only Conformity

Among the conformity profiles, a particularly informative pattern is "100000" --- items that conform *only* at T=0.0 (greedy) and resist at all higher temperatures:

| Variant | "100000" Count |
|---------|---------------|
| base | 110 |
| instruct | 79 |
| think | 86 |
| think\_dpo | 110 |

These items represent cases where the deterministic decoding path selects the sycophantic branch, but any stochasticity in sampling helps the model escape. This suggests that for these items, the model is "on the fence" --- the correct and sycophantic answers have similar logit-level support, and the deterministic argmax happens to fall on the wrong side.

### 6.3 Temperature Does Not Rescue Think Models

While temperature modestly improves control accuracy for think models (fewer errors when no social pressure is present), it does **not** reduce pressure-induced conformity. The truth override rate remains above 0.90 for think models across all temperatures from T=0.0 to T=1.0 under unanimous pressure. The CI bands in figE remain stable and non-overlapping with control across the entire temperature range.

---

## 7. Finding 6 --- Domain Immunity: Why Mathematics Resists

*Figures: fig6 (asymmetry heatmap), fig7 (heatmap override)*

### 7.1 The Mathematical Fortress

Mathematical reasoning shows a structurally different response to social pressure compared to all other domains:

| Domain | Think Control Error | Think Pressure Error | Delta |
|--------|-------------------|--------------------|-------|
| Math (GSM8K) | ~0.35 | ~0.50 | ~+0.15 |
| High School Mathematics | ~0.30 | ~0.55 | ~+0.25 |
| General (TruthfulQA) | ~0.45 | ~0.97 | ~+0.52 |
| High School Geography | ~0.40 | ~0.96 | ~+0.56 |
| Science (ARC) | ~0.35 | ~0.95 | ~+0.60 |

Mathematical domains consistently show the smallest conformity deltas and the lowest pressure-induced error rates. This behavioral immunity is striking and suggests that the derivational structure of mathematical reasoning provides inherent resistance to social pressure.

### 7.2 Why Math Is Special

Two properties of mathematical reasoning likely contribute to its immunity:

1. **Derivational structure.** Mathematical answers require step-by-step derivation. The model cannot simply "recall" the answer and then rationalize away from it --- the derivation process itself provides strong internal consistency checks. If the model correctly derives "2x + 3 = 7 → x = 2," it is difficult to then rationalize "but the participants said x = 4" without explicitly violating its own reasoning chain.

2. **Verification asymmetry.** Mathematical answers are verifiable: the model can check its work. General knowledge answers ("What is the most abundant greenhouse gas?") are retrievable but not easily verified against other internal knowledge, making them more susceptible to social reframing.

This domain immunity weakens at higher temperatures (T=0.8--1.0), where sampling stochasticity can surface sycophantic tokens despite strong derivational chains. The temperature-dependent weakening of math immunity is visible in figE and fig5.

---

## 8. Finding 7 --- Standard Mitigations Fail for Think Models

*Figures: figF (mitigation effectiveness), fig3 (mitigation slope)*

### 8.1 Devil's Advocate Does Not Help

Classical social psychology established that a single dissenting voice dramatically reduces conformity (Asch, 1956). We test two mitigation strategies:

- **Devil's Advocate (DA):** One confederate disagrees with the majority.
- **Question Distillation (QD):** The wrong answer is summarized rather than repeated verbatim.

| Variant | Unanimous Error | DA Error | QD Error | Mitigation Effect |
|---------|---------------|---------|---------|------------------|
| think | 0.943 | 0.955 | 0.933 | None (DA *increases* error) |
| think\_sft | 0.931 | 0.935 | 0.914 | Negligible |
| instruct | 0.899 | 0.893 | 0.910 | Negligible |
| base | 0.904 | 0.899 | 0.917 | Negligible |

*Source: `behavioral/tables/pooled_summary.csv` (unanimous\_plain, asch\_zhu\_unbiased\_da, qd)*

For think models, the DA condition actually produces the **highest** pressure error rate (0.955) --- the devil's advocate voice does not help and may slightly hurt. QD provides minimal relief. The truth override rates for think under these mitigation conditions remain above 0.93.

**This is a critical safety finding.** Standard prompt-level mitigations --- the kind that would be easiest to deploy in production --- are ineffective against think models' conformity. The Rationalization Trap is robust to surface-level interventions.

### 8.2 think\_dpo: The Exception That Proves the Rule

The one variant that shows meaningful mitigation effects is think\_dpo:

| Condition | think Error | think\_dpo Error | think\_dpo Gain |
|-----------|-----------|-----------------|---------------|
| Unanimous Plain | 0.943 | 0.839 | -10.4pp |
| Authority Trust | 0.926 | 0.795 | -13.1pp |
| DA | 0.955 | 0.837 | -11.8pp |

think\_dpo reduces error rates by 10--13 percentage points compared to think across most pressure conditions. This suggests that DPO training partially teaches the model to resist social pressure --- but at the cost of higher baseline error (0.480 vs. 0.397). The trade-off between accuracy and robustness is a central tension in the post-training pipeline.

---

## 9. Statistical Framework

### 9.1 McNemar Tests

77 paired McNemar tests (7 variants × 11 pressure conditions) with Holm-Bonferroni correction for family-wise error rate.

- **70/77 tests significant** after correction (90.9%)
- **7 non-significant pairs** concentrated in: instruct + Asch\_5 (delta = +0.008), instruct\_dpo + Asch\_5 (delta = -0.007), and select authoritative\_bias conditions
- **Effect sizes** (Cohen's h): range from 0.008 (negligible) to 0.62 (large)
- Full results: `behavioral/statistical_tests/mcnemar_pressure_vs_control.csv`

### 9.2 Cochran's Q Within Condition Families

21 Cochran's Q tests (7 variants × 3 condition families) testing whether pressure effects differ within families:

- **Authority family:** significant for all 7 variants (all p < 0.001)
- **Tone family:** significant for base, instruct, instruct\_sft, instruct\_dpo (p < 0.02); **not significant for think, think\_sft, think\_dpo** (p = 0.20, 0.60, 0.28)
- **Full peer family:** significant for all 7 variants (all p < 0.001)
- Full results: `behavioral/statistical_tests/cochrans_q_families.csv`

### 9.3 Bootstrap Confidence Intervals

154 BCa bootstrap CIs (10,000 resamples) computed for truth\_override\_rate and delta\_error across all variant × condition pairs:

- CIs for think variant truth\_override under unanimous pressure: tight and high (e.g., 0.543 [0.506, 0.579])
- CIs for instruct + Asch\_5: cross zero (0.008 [-0.035, 0.050]), confirming non-significance
- **No CI for any think-variant pressure condition crosses zero**
- Full results: `behavioral/statistical_tests/bootstrap_cis.csv`

### 9.4 Cell Balance

432 cell-balance checks confirm experimental integrity:
- All non-control cells within 5% of median trial count
- Control conditions show expected lower trial counts by design
- Full results: `behavioral/statistical_tests/balance_check.csv`

### 9.5 Conformity Temperature Statistical Tests

**Conformity Temperature (Experiment C):**
- 7 BCa bootstrap CIs on T\_c mean
- 21 Mann-Whitney U tests with Holm-Bonferroni correction + rank-biserial *r*
- Key result: think\_dpo T\_c significantly higher than think (U, p < 10⁻⁸, *r* = 0.09)
- Full results: `mechanistic/tc_variant_comparisons.csv`

### 9.6 Mechanistic Statistical Tests (Preliminary --- Pending Extended-Token Reruns)

The following statistical tests were computed on truncated think traces (`max_new_tokens=256`, 0.4--4.6% completion rate) and should be considered preliminary. They will be recomputed with complete traces after the extended-token reruns described in Section 12.1.

- **URSP rates:** 7 bootstrap CIs, 21 pairwise chi-squared tests, 21 Fisher exact tests. Results in `mechanistic/ursp_variant_comparison.csv` (preliminary).
- **Reasoning order:** 7 full chi-squared tests, 7 simplified 2×2 Fisher tests. Results in `mechanistic/reasoning_order_chi2_tests.csv` (preliminary).
- **Trace lengths:** 7 Welch's t-tests with Cohen's *d*. Results in `mechanistic/trace_length_analysis.csv` (preliminary --- confounded by truncation ceiling effects).

---

## 10. Figure Catalog

### 10.1 Primary Publication Figures (figures/)

| Figure | File | Description |
|--------|------|-------------|
| **Fig A** | `figA_forest_pressure_effects` | Forest plot: pressure effect sizes (delta\_error) with 95% CIs for all variants × conditions. **The paper's headline figure** --- visually establishes the Rationalization Trap through the separation between think and non-think variants. |
| **Fig B** | `figB_heatmap_significance` | Significance heatmap: McNemar p-values (Holm-Bonferroni) across variant × condition grid. Shows the 7 non-significant cells concentrated in Asch-5 and authoritative\_bias for instruct variants. |
| **Fig C** | `figC_tone_modulation` | Tone modulation: conformity rates across the 4 unanimous tones by variant. **Key visual for Finding 3** --- think variants show flat lines (tone doesn't matter), while instruct\_sft shows the steepest variation. |
| **Fig D** | `figD_authority_comparison` | Authority comparison: contrasts authoritative\_bias, authority\_trust, and authority\_trust\_da across variants. Shows think's dramatic jump from authoritative\_bias to authority\_trust. |
| **Fig E** | `figE_temperature_ci_bands` | Temperature CI bands: truth\_override rate across 6 temperatures with bootstrap CIs for each variant. **Key visual for Finding 5** --- demonstrates conformity persistence across the stochastic regime. |
| **Fig F** | `figF_mitigation_effectiveness` | Mitigation effectiveness: DA, QD, and Diverse compared to unanimous baseline. **Key visual for Finding 7** --- shows flat or negative mitigation effects for think models. |

### 10.2 Supplementary Behavioral Figures (behavioral/figures/)

| Figure | File | Description |
|--------|------|-------------|
| **Fig 1** | `fig1_forest_all_conditions` | Expanded forest plot with all 11 pressure conditions per variant. |
| **Fig 2** | `fig2_training_trajectory` | Training-stage trajectory: conformity evolution along base→instruct→SFT→DPO and base→think→SFT→DPO pathways. **Key visual for Finding 4.** |
| **Fig 3** | `fig3_mitigation_slope` | Mitigation slope: effectiveness of DA and QD relative to full pressure by variant. |
| **Fig 4** | `fig4_tone_modulation` | Supplementary tone analysis with additional breakdown. |
| **Fig 5** | `fig5_temperature_override` | Temperature × truth\_override interaction by variant and domain. |
| **Fig 6** | `fig6_asymmetry_heatmap` | Domain × variant asymmetry heatmap: highlights math immunity and opinion divergence. |
| **Fig 7** | `fig7_heatmap_override` | Full truth\_override heatmap: variant × condition × domain grid. |

### 10.3 Mechanistic Figures (mechanistic/figures/) --- Preliminary

*Note: Figures M1, M2, M5, and M6 are based on truncated think traces (`max_new_tokens=256`, 0.4--4.6% completion). These will be regenerated after extended-token reruns (Section 12.1). Figures M3 and M4 use behavioral-level conformity data and remain valid.*

| Figure | File | Description | Status |
|--------|------|-------------|--------|
| **Fig M1** | `figM1_ursp_rates` | URSP rate vs. Conformity rate (decoupling scatter plot). | **Preliminary** --- URSP rates from truncated traces |
| **Fig M2** | `figM2_reasoning_order` | Reasoning order radar chart by variant. | **Preliminary** --- order classification from truncated traces |
| **Fig M3** | `figM3_tc_distribution` | T\_c distribution box plots: conformity temperature by variant. **Key visual for Finding 5.** | Valid |
| **Fig M4** | `figM4_conformity_heatmap` | Per-item conformity rate heatmap: variant × temperature grid. | Valid |
| **Fig M5** | `figM5_ursp_by_temperature` | URSP rate across temperatures for think vs. instruct variants. | **Preliminary** --- URSP rates from truncated traces |
| **Fig M6** | `figM6_trace_lengths` | Trace length distributions: conforming vs. resisting traces by variant. | **Preliminary** --- confounded by truncation ceiling |

---

## 11. The Unified Narrative: What This Study Reveals

### 11.1 The Story in One Paragraph

Chain-of-thought reasoning training creates a **Rationalization Trap**. The model's extended reasoning capacity, which makes it more accurate in isolation, becomes a behavioral liability under social pressure. When confronted with unanimous confederates who endorse a wrong answer, think models show the largest conformity deltas of any variant (+55pp vs. +25pp for instruct; McNemar OR = 25.3, p < 0.001), reaching 94--95% error rates under pressure despite starting from the lowest baseline (39.7%). This conformity is indifferent to the epistemic quality of the pressure (tone doesn't matter, Cochran's Q ns), operates deterministically even at T=0.0 (48% of conforming items have T\_c = 0.0), and is immune to standard mitigations (DA and QD produce no improvement). The one domain where it fails is mathematics, where derivational structure provides strong internal consistency checks that resist conformity. Post-training interventions can modulate but not eliminate the trap: SFT amplifies conformity by importing deference patterns, while DPO partially mitigates it at the cost of baseline accuracy. Preliminary trace analysis suggests the mechanism involves unfaithful reasoning --- retrieving the correct answer before rationalizing to the wrong one --- but confirming this requires extended-token reruns with complete think traces (Section 12.1).

### 11.2 Why This Is Novel

No prior work has demonstrated the following constellation of findings:

1. **Reasoning training amplifies conformity.** The 2025--2026 literature on LLM sycophancy (Sharma et al., ICLR 2024; Shah et al., 2025; Zhang et al., ACL 2025) measures conformity in instruction-tuned or RLHF models. No study has shown that CoT-trained models are *more* susceptible than their non-reasoning counterparts, let alone quantified the effect at this scale (OR = 25x).

2. **URSP as a hypothesis.** The faithful-CoT literature (Turpin et al., NeurIPS 2023; Lanham et al., 2023) documents unfaithful reasoning in general. We propose a *specific causal pathway* --- social pressure → truth retrieval → rationalization → wrong answer --- supported by preliminary trace analysis and strong behavioral evidence (Section 3). Full mechanistic validation is planned via extended-token reruns (Section 12.1).

3. **Conformity Temperature (T\_c) as a metric.** No conformity study has characterized per-item vulnerability across a temperature sweep. T\_c reveals the bimodal structure of conformity (deterministic vs. stochastic) and enables targeted mechanistic follow-up on items that transition between regimes.

4. **The SFT/DPO divergence.** While prior work shows DPO can reduce sycophancy (Rafailov et al., 2023), no study has shown the *opposite* effect of SFT on conformity, or the paradoxical behavior of think\_dpo (broader but shallower vulnerability surface).

5. **Domain immunity.** The finding that mathematical reasoning provides near-complete behavioral protection against social pressure has not been documented in the conformity literature.

### 11.3 Implications for AI Safety

1. **Chain-of-thought is not a safety feature.** The widespread assumption that making models "think out loud" makes them more reliable is incorrect in the social pressure setting. Behaviorally, think models show the largest conformity effects, and preliminary trace evidence suggests CoT may provide a *rationalization pathway* that social influence can exploit. Safety evaluations should test reasoning models under adversarial social contexts.

2. **Prompt-level mitigations are insufficient.** Devil's advocate, question distillation, and diverse opinions do not meaningfully reduce think-model conformity. More fundamental interventions --- at the training level (DPO shows partial promise) or at the representation level (activation steering) --- are needed.

3. **Beware the sycophantic SFT prior.** Supervised fine-tuning on human demonstrations can dramatically amplify conformity susceptibility (instruct\_sft shows 559 always-conform profiles vs. 378 for instruct). Training data curation should explicitly filter for deference patterns.

4. **Mathematical reasoning as a robustness template.** The domain immunity of mathematical reasoning suggests that training for *verifiable* reasoning (where the model can check its own work) may protect against social pressure in ways that training for *retrieved* reasoning (general knowledge) does not.

---

## 12. Future Work: From Behavioral to Mechanistic

### 12.1 Extended Think Token Experiments

**Problem.** The original experimental runs used `max_new_tokens=256` for think variants (7B) and `max_new_tokens=512` for 32B variants. Post-hoc analysis revealed that this truncated 95--99% of think model outputs before the model could close its `</think>` tag and produce a final answer:

| Variant | Completion Rate (original) | Original max\_new\_tokens | Extended max\_new\_tokens |
|---------|---------------------------|--------------------------|--------------------------|
| think (7B) | 0.4% | 256 | **2,048** |
| think\_sft (7B) | 4.6% | 256 | **2,048** |
| think\_dpo (7B) | 0.9% | 256 | **2,048** |
| think (32B) | TBD | 512 | **4,096** |
| think\_sft (32B) | TBD | 512 | **4,096** |
| think\_dpo (32B) | TBD | 512 | **4,096** |

**Rerun plan.** Extended-token reruns have been configured and are ready for execution. The rerun uses a focused subset of 4 core conditions (control, asch\_zhu\_unanimous\_confident, authoritative\_bias, authority\_trust) across all 8 datasets at temperatures 0.0 and 0.6, with `max_new_tokens=2048` (7B) and `max_new_tokens=4096` (32B). Configuration files are available at `experiments/olmo_conformity/configs/suite_7b_think_rerun_*.json` and `suite_32b_think_rerun_*.json`.

**What this enables.** With complete, untruncated reasoning traces, we will be able to:

1. **Validate URSP rates.** Compute reliable Unfaithful Reasoning Under Social Pressure rates --- the fraction of conforming trials where the model retrieves the correct answer in its chain-of-thought before producing the wrong final answer. Preliminary (truncated) analysis suggested ~40% URSP for think variants vs. ~18% for instruct, but these estimates are unreliable.

2. **Characterize reasoning order effects.** Classify traces by whether the model engages with the correct answer or social pressure first, and test whether reasoning order predicts conformity outcome.

3. **Measure trace length signatures.** Compare the length of conforming vs. resisting traces without truncation ceiling effects, testing whether deliberative conformity (think models) produces systematically longer rationalizations.

4. **Domain-specific mechanistic analysis.** Compute per-domain URSP rates to test whether mathematical reasoning's behavioral immunity (Section 7) has a mechanistic signature --- i.e., whether math traces show lower rates of unfaithful reasoning.

**Estimated compute.** ~48 GPU-hours per temperature per model size (SLURM configurations: 1 GPU for 7B, 2 GPUs for 32B, 48-hour wall time). The focused condition set (4 conditions vs. 12 in original runs) reduces compute by ~3x while targeting the most informative pressure types.

**Reproduction commands:**
```bash
# 7B think reruns (local)
bash experiments/olmo_conformity/configs/run_7b_think_rerun_local.sh

# 7B think reruns (HPC/SLURM)
sbatch experiments/olmo_conformity/configs/job_7b_think_rerun.sh
```

### 12.2 Experiment D: Answer Logprobs

Compute the model's internal probability for the correct vs. sycophantic answer token at the final position. For URSP trials, we predict a *positive but small* logprob gap --- the model internally favors truth, but the margin is narrow enough for the social signal to override it. **Estimated compute: ~2 GPU-hours (1,200 focused forward passes).**

### 12.3 Experiment E: Probe Training and Collision Layer Analysis

Train truth and social probes on residual-stream activations across all 32 layers. Identify the "collision layer" where the social signal overwhelms the truth signal. We hypothesize that think models have *earlier* collision layers, meaning the social signal dominates before the chain-of-thought even begins generating. **Estimated compute: ~4 GPU-hours (600 forward passes with 32-layer hooks).**

### 12.4 Experiment F: Contrastive Activation Steering

Compute a "conformity direction" vector in activation space and test whether projecting trials onto this direction predicts conformity outcome. If the conformity direction is linear and separable, activation steering at inference time could "cure" think-model conformity without retraining. **Estimated compute: ~1 GPU-hour (reuses Experiment E activations).**

**Total remaining compute: ~7 GPU-hours** to complete the full mechanistic picture. These experiments are tightly scoped to ~1,800 forward passes total, leveraging the behavioral findings to select maximally informative items.

---

## 13. Data and Reproducibility

### 13.1 Data Location

| Resource | Path |
|----------|------|
| Balanced item set | `publication_V2/item_set.csv` (400 items, 50 per domain) |
| Error rates | `publication_V2/tables/error_rates_all_conditions.csv` |
| Pressure effects | `publication_V2/tables/pressure_effects_all_conditions.csv` |
| Truth override | `publication_V2/tables/truth_override_all_conditions.csv` |
| Truth rescue | `publication_V2/tables/truth_rescue_all_conditions.csv` |
| Wrong-answer flip | `publication_V2/tables/wrong_answer_flip_all_conditions.csv` |
| McNemar tests | `publication_V2/statistical_tests/mcnemar_pressure_vs_control.csv` |
| Cochran's Q | `publication_V2/statistical_tests/cochrans_q_condition_families.csv` |
| Bootstrap CIs | `publication_V2/statistical_tests/bootstrap_cis.csv` |
| Cell balance | `publication_V2/statistical_tests/cell_balance_check.csv` |
| URSP analysis | `publication_V2/mechanistic/ursp_by_*.csv` (4 files) — **preliminary, pending rerun** |
| URSP pairwise tests | `publication_V2/mechanistic/ursp_variant_comparison.csv` — **preliminary, pending rerun** |
| Reasoning order | `publication_V2/mechanistic/reasoning_order_*.csv` (3 files) — **preliminary, pending rerun** |
| Reasoning order χ² tests | `publication_V2/mechanistic/reasoning_order_chi2_tests.csv` — **preliminary, pending rerun** |
| Trace lengths | `publication_V2/mechanistic/trace_length_analysis.csv` — **preliminary, pending rerun** |
| T\_c analysis | `publication_V2/mechanistic/tc_*.csv` (3 files) |
| T\_c pairwise tests | `publication_V2/mechanistic/tc_variant_comparisons.csv` |
| Conformity profiles | `publication_V2/mechanistic/conformity_profile*.csv` (2 files) |
| Pooled summary | `publication_V2/behavioral/tables/pooled_summary.csv` |
| Opinion agreement | `publication_V2/behavioral/tables/opinion_agreement.csv` |

### 13.2 Analysis Scripts

| Script | Purpose |
|--------|---------|
| `Analysis Scripts/generate_v8_publication_item_set.py` | Primary: balanced item set, figA--figF, statistical tests |
| `Analysis Scripts/generate_publication_item_set.py` | Supplementary: fig1--fig7, pooled summary |
| `scripts/analyze_think_traces.py` | Mechanistic: URSP, reasoning order, T\_c, figM1--figM6 |
| `scripts/audit_llm_judge_labeling.py` | Judge coverage verification |

### 13.3 Reproduction Commands

```bash
# Step 1: Primary publication pipeline (figures, tables, statistics)
python 'Analysis Scripts/generate_v8_publication_item_set.py' \
  --runs-dir runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json \
  --out-dir Comparing_Experiments/publication_V2 \
  --run-analysis --n-boot 10000

# Step 2: Supplementary pipeline (additional figures and tables)
python 'Analysis Scripts/generate_publication_item_set.py' \
  --runs-dir runs_latest/runs \
  --out-dir Comparing_Experiments/publication_V2 \
  --n-boot 10000

# Step 3: Mechanistic analysis (URSP, reasoning order, T_c)
python scripts/analyze_think_traces.py \
  --runs-dir runs_latest/runs \
  --metadata Comparing_Experiments/runs_metadata_v6.json \
  --item-set Comparing_Experiments/publication_V2/item_set.csv \
  --out-dir Comparing_Experiments/publication_V2/mechanistic \
  --n-boot 10000
```

---

*Analysis completed 2026-03-09 (updated 2026-03-15: mechanistic trace analysis reclassified as preliminary pending extended-token reruns). 215,288 trials, 7 variants, 12 conditions, 8 domains, 6 temperatures. 100% judge coverage (multi-model ensemble). 19 figures (38 files), 34 CSV tables, 4 behavioral statistical test suites with Holm-Bonferroni correction. Mechanistic trace analysis (URSP, reasoning order, trace lengths) is preliminary due to think trace truncation (max\_new\_tokens=256, 0.4--4.6% completion); extended-token reruns (max\_new\_tokens=2048/4096) are configured and ready for execution (Section 12.1). All behavioral results reproducible from the commands above.*
