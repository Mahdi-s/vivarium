# Statistical Tests Used in the Conformity Paper

This document provides a comprehensive reference for every statistical test employed in the conformity study, including the rationale for each choice, underlying assumptions, mathematical formulations, and implementation details.

---

## 1. McNemar's Exact Binomial Test

### Purpose

McNemar's test is the primary inferential tool for comparing paired binary outcomes between the control and pressure conditions. For each of the 400 items in the experimental corpus, the model's response is classified as correct (State A) or not correct (States B + C). Because the same items appear under both conditions, the data consist of 400 matched pairs of binary outcomes.

### Why McNemar's Test

The choice of McNemar's test over alternatives follows directly from the study design:

1. **Paired observations.** Each item serves as its own control. The 400 items are tested under both the control condition (no misleading peer or authority information) and the pressure condition (with misleading information). A standard chi-squared test of independence treats observations as unpaired and would discard this within-item structure.

2. **Binary outcome.** Each observation is dichotomous: correct or not correct. This rules out continuous-outcome tests such as the paired $t$-test or Wilcoxon signed-rank test.

3. **Focus on discordant pairs.** McNemar's test conditions exclusively on the pairs that changed between conditions, which is precisely the quantity of interest: how many items flipped from correct to incorrect (or vice versa) under social pressure.

4. **Independence inflation.** An independent-samples test (e.g., two-proportion $z$-test) would treat the 400 control observations and 400 pressure observations as 800 independent data points, artificially doubling the effective sample size and inflating statistical significance.

### Mathematical Formulation

Given $N$ paired items, construct the $2 \times 2$ contingency table of paired outcomes:

|  | Pressure: Correct | Pressure: Not Correct |
|---|---|---|
| **Control: Correct** | $a$ | $b$ |
| **Control: Not Correct** | $c$ | $d$ |

where:

- $a$ = concordant correct (correct under both conditions)
- $b$ = **truth override** (correct under control, incorrect under pressure)
- $c$ = **truth rescue** (incorrect under control, correct under pressure)
- $d$ = concordant incorrect (incorrect under both conditions)

Under the null hypothesis $H_0$: the probability of a discordant pair being of type $b$ equals the probability of it being of type $c$:

$$H_0: P(b) = P(c) = 0.5$$

The test statistic is the number of type-$b$ discordant pairs, $b$, out of the total number of discordant pairs $n = b + c$. Under $H_0$, $b \sim \text{Binomial}(n, 0.5)$.

The two-sided $p$-value is computed as:

$$p = 2 \times \min\left( P(X \leq b), P(X \geq b) \right), \quad X \sim \text{Binomial}(n, 0.5)$$

This is the **exact** binomial form of McNemar's test, preferred over the asymptotic chi-squared approximation $\chi^2 = (b - c)^2 / (b + c)$ because the exact test is valid for any sample size, including cases where $b + c$ is small.

### Worked Example: Llama-3.1-70B (Study 2, Peer Consensus, $T = 0.0$)

From the data:
- $b = 231$ (items that were correct under control but incorrect under pressure)
- $c = 1$ (items that were incorrect under control but correct under pressure)
- $n = b + c = 232$

Under $H_0$: $b \sim \text{Binomial}(232, 0.5)$.

The probability of observing $b \geq 231$ under this null is astronomically small. The two-sided exact $p$-value is effectively $p < 10^{-60}$, far below any conventional significance threshold.

### Handling of Exclusions

- **Pairing**: Items are joined by `item_id` via an inner join between the control and pressure condition datasets. Only items present in both conditions enter the analysis.
- **Refusals in control**: Items where the control condition yields `is_correct = NULL` (typically model refusals) are excluded from the pairing. This reduces the effective $N$ for high-refusal models. For example, Llama-3.1-70B at $T = 0.0$ has only 259 out of 400 items successfully paired due to control-condition refusals.
- **Refusals in pressure**: Under the fixed-$N = 400$ design, refusals are classified as "not correct" (State C) for purposes of the McNemar pairing.

---

## 2. Odds Ratio with Haldane-Anscombe Correction

### Purpose

The odds ratio (OR) quantifies the relative likelihood of truth override versus truth rescue among discordant pairs. It serves as an effect size measure accompanying the McNemar test.

### Definition

The raw odds ratio for discordant pairs is:

$$\text{OR}_{\text{raw}} = \frac{b}{c}$$

However, when either $b = 0$ or $c = 0$, the raw OR is either zero or undefined. Even when both are nonzero, small cell counts produce unstable estimates. The **Haldane-Anscombe correction** adds 0.5 to each cell:

$$\text{OR} = \frac{b + 0.5}{c + 0.5}$$

### Confidence Interval

The 95% confidence interval is constructed on the log scale using the Wald method:

$$\log(\text{OR}) \pm 1.96 \times \sqrt{\frac{1}{b + 0.5} + \frac{1}{c + 0.5}}$$

The interval endpoints are then exponentiated to obtain the CI on the OR scale.

### Worked Example: Llama-3.1-70B

With $b = 231$ and $c = 1$:

$$\text{OR} = \frac{231 + 0.5}{1 + 0.5} = \frac{231.5}{1.5} = 154.33$$

Compare to the raw ratio $231 / 1 = 231$. The correction moderates the estimate, which is important for the confidence interval calculation.

For the 95% CI:

$$\text{SE}[\log(\text{OR})] = \sqrt{\frac{1}{231.5} + \frac{1}{1.5}} = \sqrt{0.00432 + 0.6667} = \sqrt{0.6710} \approx 0.819$$

$$\log(\text{OR}) = \log(154.33) \approx 5.039$$

$$\text{95\% CI (log scale)}: 5.039 \pm 1.96 \times 0.819 = [3.434, 6.644]$$

$$\text{95\% CI (OR scale)}: [e^{3.434}, e^{6.644}] = [30.99, 769.2]$$

The wide interval reflects the near-zero count in the $c$ cell, even after correction.

### Interpretation

- $\text{OR} > 1$: Truth overrides are more common than truth rescues (pressure degrades performance).
- $\text{OR} = 1$: Discordant pairs are equally likely in both directions (no net pressure effect).
- $\text{OR} < 1$: Truth rescues are more common (pressure improves performance, a rare finding).

---

## 3. Holm-Bonferroni Correction for Multiple Comparisons

### Purpose

When multiple McNemar tests are conducted across models, the family-wise error rate (FWER) must be controlled to prevent inflation of Type I errors.

### Why Holm-Bonferroni (Not Plain Bonferroni or Benjamini-Hochberg)

1. **Holm vs. Bonferroni.** The Holm procedure is uniformly more powerful than Bonferroni: it rejects at least as many hypotheses (and often more) while maintaining the same FWER guarantee. Since Holm strictly dominates Bonferroni, there is no reason to prefer the latter.

2. **FWER vs. FDR.** The paper makes definitive binary claims about each model (e.g., "Model X shows significant conformity"). This calls for FWER control, which bounds the probability of making even one false rejection across the family. The Benjamini-Hochberg (BH) procedure controls the false discovery rate (FDR), which bounds the expected proportion of false rejections among all rejections. BH is appropriate when screening many hypotheses and tolerating some false positives; FWER is appropriate when each individual conclusion must be reliable.

### Procedure

Given $k$ hypotheses with raw $p$-values $p_1, p_2, \ldots, p_k$:

1. **Sort** the $p$-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(k)}$.

2. **Adjust** each $p$-value sequentially:

$$p_{\text{adj}}^{(i)} = \max\left( p_{\text{adj}}^{(i-1)},\; (k - i + 1) \times p_{(i)} \right)$$

with $p_{\text{adj}}^{(0)} = 0$.

3. **Reject** $H_{(i)}$ if $p_{\text{adj}}^{(i)} < \alpha$ (typically $\alpha = 0.05$).

The monotonicity enforced by the $\max$ operation ensures that if hypothesis $(i)$ is not rejected, neither is any hypothesis $(j)$ for $j > i$.

### Correction Families in the Paper

The corrections are applied within the following families:

- **Study 2, peer consensus condition (`asch_zhu_unanimous_confident`)**: One McNemar test per model; 11 tests total (10 cross-family model families + OLMo-7B-Think from `runs/think/`). GPT-OSS-20B T=0.0 is provisional pending re-run of 187 error trials.
- **Study 2, authority conditions (`authoritative_bias`, `authority_trust`)**: One McNemar test per model per condition; 11 × 2 = 22 tests corrected together as one family per condition.
- **Study 1 (OLMo within-family analysis)**: 11 tests per experimental variant (one per training condition/checkpoint) across 8 OLMo-7B variants at 6 temperatures.

Each family is corrected independently. Cross-family comparisons (e.g., peer consensus vs. authority) are treated as separate analyses.

### Worked Example

Suppose 11 tests (10 cross-family models + OLMo-7B-Think) yield sorted raw $p$-values; the table below uses a condensed 9-test illustration for brevity:

| Rank $i$ | Raw $p_{(i)}$ | Multiplier $(k - i + 1)$ | Adjusted $p_{\text{adj}}^{(i)}$ |
|---|---|---|---|
| 1 | 0.0001 | 9 | $\max(0, 9 \times 0.0001) = 0.0009$ |
| 2 | 0.0003 | 8 | $\max(0.0009, 8 \times 0.0003) = 0.0024$ |
| 3 | 0.002 | 7 | $\max(0.0024, 7 \times 0.002) = 0.014$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| 9 | 0.08 | 1 | $\max(\ldots, 1 \times 0.08) = 0.08$ |

Hypotheses 1 through 3 are rejected at $\alpha = 0.05$; hypothesis 9 is not.

---

## 4. Wilson Score Confidence Intervals

### Purpose

Wilson score intervals provide confidence intervals for proportions (error rates, endorsement rates, refusal rates) with superior coverage properties compared to the standard Wald (normal approximation) interval.

### Why Wilson (Not Wald)

The Wald interval for a proportion $\hat{p} = x/n$ is:

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

This interval can extend below 0 or above 1 when $\hat{p}$ is near the boundary, which is nonsensical for a probability. It also has poor coverage (actual confidence level below the nominal level) for extreme proportions and small $n$.

Many of the proportions in the conformity study are near 0 or 1. For example, Claude's endorsement rate of approximately 4.8% and certain models' near-zero truth rescue rates make the Wald interval unreliable. The Wilson score interval resolves these problems.

### Formula

For $\hat{p} = x/n$ and $z = z_{\alpha/2}$ (typically $z = 1.96$ for 95% confidence):

$$\text{Wilson CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

### Properties

- The interval is always contained within $[0, 1]$.
- It is centered not at $\hat{p}$ but at a point shrunk toward $0.5$, reflecting the Bayesian intuition that extreme observed proportions are likely more moderate in truth.
- Coverage probability is closer to the nominal level than the Wald interval across all values of $p$ and $n$.

### Worked Example

Suppose a model has an endorsement rate of $\hat{p} = 19/400 = 0.0475$ (i.e., 19 out of 400 items endorsed the incorrect answer under pressure).

With $n = 400$ and $z = 1.96$:

$$\text{Numerator center} = 0.0475 + \frac{1.96^2}{800} = 0.0475 + 0.0048 = 0.0523$$

$$\text{Margin} = 1.96 \sqrt{\frac{0.0475 \times 0.9525}{400} + \frac{1.96^2}{640000}} = 1.96 \sqrt{0.0001131 + 0.000006} = 1.96 \times 0.01091 = 0.02138$$

$$\text{Denominator} = 1 + \frac{3.8416}{400} = 1.0096$$

$$\text{Lower} = \frac{0.0523 - 0.02138}{1.0096} = \frac{0.0309}{1.0096} = 0.0306$$

$$\text{Upper} = \frac{0.0523 + 0.02138}{1.0096} = \frac{0.0737}{1.0096} = 0.0730$$

$$\text{Wilson 95\% CI}: [0.031, 0.073]$$

Compare to the Wald interval: $0.0475 \pm 1.96 \times 0.01064 = [0.027, 0.068]$, which is slightly shifted and could produce a negative lower bound for smaller proportions.

---

## 5. Pressure Effect ($\Delta$)

### Purpose

The pressure effect $\Delta$ is a simple, intuitive measure of the practical magnitude of social pressure on model accuracy.

### Definition

$$\Delta = \text{Error}_{\text{pressure}} - \text{Error}_{\text{control}}$$

where:

$$\text{Error rate} = 1 - \frac{\text{State A count}}{N} = \frac{\text{State B} + \text{State C}}{N}$$

The denominator $N = 400$ is fixed for all models, regardless of how many items were refused.

### Interpretation

- $\Delta = +0.20$ means the model commits 20 percentage points more errors under pressure than under control.
- $\Delta = 0$ indicates no change in overall error rate.
- $\Delta < 0$ would indicate that pressure improves accuracy (rare and typically not observed).

### Relationship to the Odds Ratio

While the OR captures the relative asymmetry among discordant pairs, $\Delta$ captures the absolute change in error rate on the original probability scale. The two measures can diverge substantially:

- When baseline error is very low, even a modest $\Delta$ can correspond to an extreme OR (because few items are available for truth rescue).
- When baseline error is moderate, the same $\Delta$ may correspond to a much smaller OR.

### Limitations

$\Delta$ conflates two distinct behavioral changes: endorsement of the incorrect answer (State B) and refusal (State C). A model that shifts from correct answers to refusals has the same $\Delta$ as one that shifts from correct answers to endorsing falsehoods, but the interpretive implications differ greatly. The three-state decomposition in the paper's figures addresses this by reporting State A, B, and C proportions separately.

---

## 6. Cohen's $h$

### Purpose

Cohen's $h$ is a standardized effect size for comparing two proportions, used in supplementary analyses to facilitate cross-study comparisons on a common scale.

### Formula

$$h = 2 \arcsin\left(\sqrt{p_1}\right) - 2 \arcsin\left(\sqrt{p_2}\right)$$

where $p_1$ and $p_2$ are the two proportions being compared (e.g., error rate under pressure vs. error rate under control).

### Interpretation

| $|h|$ | Effect Size |
|---|---|
| $< 0.20$ | Small |
| $0.20 - 0.80$ | Medium |
| $> 0.80$ | Large |

### Rationale for the Arcsine Transformation

The arcsine square root transformation (also known as the angular transformation) stabilizes the variance of proportions. Unlike the raw difference $p_1 - p_2$, Cohen's $h$ has approximately the same sampling variability regardless of whether the proportions are near 0, near 1, or near 0.5. This makes it a more principled basis for comparing effect magnitudes across models with very different baseline error rates.

---

## 7. Conformity Temperature ($T_c$)

### Purpose

The conformity temperature $T_c$ provides a per-item characterization of the minimum sampling temperature at which a model first exhibits conformity behavior for that item. It connects the observed behavioral phenomenon to the model's output probability distribution.

### Definition

For each item $i$ and a given model:

$$T_c(i) = \min\{T \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\} : \text{model conforms at temperature } T\}$$

where **conformity** is defined as:
- The model gives the **correct** answer under control at temperature $T$, AND
- The model gives an **incorrect** answer under pressure at temperature $T$.

If the model does not conform at any temperature, $T_c(i)$ is undefined (or set to $\infty$).

### Interpretive Framework

- **$T_c = 0.0$ (greedy decoding)**: The conforming response is the highest-probability token under the pressure condition. This represents the strongest form of conformity, where the model's argmax output itself is corrupted by social pressure.

- **$T_c > 0$**: Conformity requires stochastic sampling to manifest. The conforming response is not the greedy output; rather, it is a lower-probability token that is occasionally sampled at higher temperatures. This suggests that the model "knows" the correct answer (it is the argmax) but can be stochastically nudged toward conformity.

- **$T_c = \infty$ (never conforms)**: The model resists pressure at all temperatures tested.

### Statistical Considerations

The conformity temperature analysis is descriptive rather than inferential. No formal hypothesis test is applied to $T_c$ values. Instead, the distribution of $T_c$ across items provides insight into the heterogeneity of conformity susceptibility within a model's item population.

---

## 8. Statistical Independence Considerations

The study's design introduces several dependencies that constrain the valid scope of statistical inference. These are documented here for transparency.

### Within Study 2 (Cross-Family Comparison)

The primary analyses use $T = 0.0$ (greedy decoding) only. At greedy temperature, each item contributes exactly one deterministic observation per model per condition. This ensures that the 400 observations entering each McNemar test are independent (conditional on the items).

Cross-family runs were also collected at $T = 0.6$ (2 temperatures total vs. 6 in Study 1). Table 2 in earlier drafts pooled results across both temperatures, which was flagged by reviewers. The same item at different temperatures does not produce independent observations. The revision restricts the primary cross-model comparison to $T = 0.0$, with $T = 0.6$ reported as a temperature-stability supplementary analysis.

**Note on GPT-OSS-20B T=0.0**: Run `66765d5e` has 1,413 valid outputs (187 pending re-run). All reported $p$-values and effect sizes for this model at $T = 0.0$ are provisional until the re-run completes.

### Within Study 1 (OLMo Within-Family Comparison)

The trajectory comparison (how conformity changes across training stages) examines the same 400 items across multiple checkpoints. When pooling across temperatures (as in Table 2), the same item appears up to 6 times per checkpoint, violating independence. Panel B of the revised figures presents $T = 0.0$-only values to address this concern.

### Across Studies

The bridge design uses the same 400 items in both Study 1 (OLMo within-family) and Study 2 (cross-family). Consequently, OLMo-7B-Instruct's results are not independent across the two studies. Any meta-analytic combination of results across studies must account for this shared item set.

---

## 9. Design Choices and Their Statistical Rationale

### Fixed $N = 400$ Denominator

All error rates use a denominator of $N = 400$, the total number of items in the corpus, regardless of how many items the model refused to answer.

**Rationale**: If refusals were excluded from the denominator, high-refusal models would receive an artificially inflated accuracy rate. A model that refuses 350 items and answers 50 correctly would have $50/50 = 100\%$ accuracy under an exclusion-based denominator, versus $50/400 = 12.5\%$ under the fixed denominator. The fixed denominator prevents this survivorship bias.

**Trade-off**: The fixed denominator conflates endorsement of incorrect answers (State B, a substantive failure) with refusal (State C, a procedural failure). This is why the three-state decomposition is essential for interpretation.

### Three-State Decomposition (A/B/C)

Rather than reporting only binary correct/incorrect, the paper decomposes each observation into:

- **State A**: Correct answer
- **State B**: Incorrect answer matching the pressure-suggested response (endorsement)
- **State C**: Refusal or other non-answer

This decomposition disentangles two fundamentally different "error" modes and is reported in the study's figures to complement the binary McNemar analysis.

### Greedy Decoding ($T = 0.0$) as Primary Analysis

Greedy decoding produces deterministic, reproducible outputs. Each item yields exactly one response per model per condition, eliminating sampling variability as a source of noise. This choice:

1. Ensures perfect reproducibility.
2. Simplifies the independence structure (no within-item, across-sample correlations).
3. Provides a conservative test of conformity: if conformity appears at $T = 0.0$, it reflects the model's highest-probability behavior, not a stochastic artifact.

### Item-Level Pairing

McNemar's test requires paired observations. Pairing by `item_id` ensures that the test captures within-item behavioral change (the same question answered differently under pressure) rather than between-item variation (different questions having different difficulty levels). This is analogous to a within-subjects design in experimental psychology, which is generally more powerful than a between-subjects design because it eliminates individual (here, item-level) differences as a source of variability.

---

## Summary Table

| Test | Purpose | Key Assumption | Output |
|---|---|---|---|
| McNemar's exact binomial | Paired binary comparison (control vs. pressure) | Paired observations by item; binary outcome | Two-sided $p$-value |
| Haldane-Anscombe OR | Effect size for discordant pairs | Same pairing as McNemar | OR with 95% CI |
| Holm-Bonferroni | Multiple comparison correction | Independent tests within family | Adjusted $p$-values |
| Wilson score CI | Confidence interval for proportions | Binomial sampling | 95% CI for $\hat{p}$ |
| Pressure effect $\Delta$ | Absolute change in error rate | Fixed $N = 400$ denominator | Percentage point difference |
| Cohen's $h$ | Standardized effect size for proportions | Arcsine-stable variance | Dimensionless effect size |
| Conformity temperature $T_c$ | Per-item conformity characterization | Six discrete temperatures tested | Descriptive distribution |
