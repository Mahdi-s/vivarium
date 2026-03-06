# Behavioral Report: Conformity and Social Pressure in Language Model Variants

## 1. Introduction

This report summarizes the main behavioral findings from the v7 publication suite, which evaluates multiple model variants under control and social-pressure conditions (Asch-style peer history and authoritative bias) across temperatures 0.0–1.0 and dataset categories (general, knowledge, math, opinion, reasoning, science, truthfulness). The analysis draws on the figures in `behavioral/figures/` and the summary tables in `behavioral/tables/`.

**Variants:** base, instruct, instruct_sft, instruct_dpo, think, think_sft.  
**Conditions:** control, asch_history_5, authoritative_bias.

---

## 2. Summary of Figures and What They Show

| Figure | Content |
|--------|--------|
| `fig1_error_rate_dot_plot.png` | Error rates by variant, condition, and temperature; baseline accuracy and effect of pressure. |
| `fig2_pressure_lollipop.png` | Magnitude of pressure effect (e.g. error-rate increase) by variant and condition. |
| `fig4_topic_analysis.png` | Breakdown of behavior by dataset category (topic). |
| `fig5_truth_override_rescue_dumbbell.png` | Truth override vs truth rescue rates (override = correct→wrong under pressure; rescue = wrong→correct under pressure). |
| `fig6_endorsement_slope.png` | Sensitivity to endorsement strength or similar gradient. |
| `fig9_topic_pressure_heatmap.png` | Heatmap of pressure effect by topic and variant/condition. |
| `fig12_sankey_conformity.png` | Flow of responses across control vs pressure (conformity pathways). |
| `fig_temperature_conformity_curve.png` | Conformity or error rate as a function of sampling temperature. |
| `fig_training_trajectory.png` | Change in conformity-related metrics along a training trajectory (if applicable). |
| `fig_item_difficulty_scatter.png` | Item-level difficulty vs conformity or error rate. |
| `fig_normalized_sensitivity.png` | Normalized sensitivity to pressure (e.g. delta relative to control error rate). |
| `fig_peer_authority_asymmetry.png` | Asymmetry between peer (Asch) and authority pressure effects. |
| `fig_radar_conformity_profiles.png` | Multi-axis profiles of conformity by variant/category. |
| `fig_social_contagion_histogram.png` | Distribution of “social contagion” or wrong-answer spread across items. |
| `factual_control_error_rate_heatmaps.png` | Control error rates by temperature, variant, and category. |
| `factual_pressure_effect_*_heatmaps.png` | Pressure effect (e.g. delta error) for Asch and authority. |
| `factual_truth_override_*_heatmaps.png` | Truth override rates by condition and category. |
| `factual_truth_rescue_*_heatmaps.png` | Truth rescue rates by condition and category. |
| `factual_wrong_answer_flip_*_heatmaps.png` | Wrong-answer flip rate (agreeing with wrong answer under pressure when control did not). |

---

## 3. Key Quantitative Findings from the Tables

### 3.1 Item difficulty and conformity

**Table: `behavioral_stats_summary.csv`**

Across all variants and both pressure conditions, **item difficulty is negatively correlated with conformity**: harder items (higher control accuracy) show less conformity. All Pearson correlations are negative and highly significant (p ≪ 1e−10).

- **Strongest correlations:** think and think_sft (e.g. r ≈ −0.43 to −0.46 for both asch_history_5 and authoritative_bias).
- **instruct** also shows strong negative correlations (r ≈ −0.46 to −0.48).
- **instruct_dpo** and **instruct_sft** are slightly weaker but still robust (r ≈ −0.37 to −0.45).

**Interpretation:** Conformity is consistently **item-dependent**: models conform more on easier items and hold their ground more on harder ones. This supports the view that conformity is modulated by internal confidence or discriminability.

---

### 3.2 Pressure effect size (factual error-rate deltas)

**Table: `factual_pressure_deltas.csv`**

Pressure effect is defined as the increase in error rate from control to pressure (delta_asch, delta_authority).

**Largest Asch (peer) pressure effects (representative):**

- **think, T=0, reasoning:** delta_asch ≈ 0.39; **think_sft, T=0, opinion:** ≈ 0.37; **reasoning:** ≈ 0.36.
- **instruct_sft, T=0:** truthfulness ≈ 0.36, opinion ≈ 0.29, general ≈ 0.16; **T=0.2** truthfulness ≈ 0.37, science ≈ 0.36.
- **think_sft, T=0.4, opinion:** ≈ 0.41; **reasoning:** ≈ 0.36.

**Authority (authoritative_bias) effects** are often similar in sign and sometimes larger than Asch (e.g. think_sft opinion T=0.4 ≈ 0.45; think reasoning T=0 ≈ 0.49).

**Negative deltas (pressure reduces error):** Most prominent for **instruct** and **instruct_dpo** in **truthfulness** and sometimes **science** (e.g. instruct T=0 truthfulness delta_asch ≈ −0.17; T=0.2 ≈ −0.13). So under pressure, these variants sometimes **improve** on truthfulness, consistent with “truth rescue” or resistance to wrong authority/peer on those items.

---

### 3.3 Normalized pressure sensitivity

**Table: `normalized_pressure_sensitivity.csv`**

Relative pressure is defined as (delta / control_error_rate), i.e. sensitivity relative to baseline difficulty.

- **instruct_sft** shows very high relative sensitivity on opinion and truthfulness (e.g. relative_pressure_asch > 0.5, often 0.6–0.87).
- **think / think_sft** show high relative sensitivity on reasoning and opinion (e.g. relative_pressure_authority up to ≈ 0.69–0.79).
- **instruct_dpo** often has low or negative relative pressure on truthfulness and science, again indicating resistance or rescue in those domains.

---

### 3.4 Truth override and truth rescue

**Tables: `truth_override.csv`, `truth_rescue.csv`, `training_trajectory_summary.csv`**

**Truth override** = among items correct in control, the fraction that flip to wrong under pressure.  
**Truth rescue** = among items wrong in control, the fraction that flip to correct under pressure.

**Override (high = strong harmful conformity):**

- **instruct_sft** has the highest overall override under Asch (training_trajectory_summary: ≈ 0.69) and very high category-specific rates (e.g. opinion 1.0, science 0.94 at T=0.4, truthfulness 0.87 at T=0.8).
- **base** is also high (≈ 0.58–0.59 across conditions).
- **think** and **think_sft** are moderate (≈ 0.50–0.54) with **math** override notably lower (e.g. ≈ 0.14–0.27), i.e. more resistance on math under pressure.
- **instruct_dpo** has the lowest overall override (≈ 0.44 Asch) and often low truthfulness override (e.g. 0.18–0.33).

**Rescue (beneficial effect of context):**

- **think** and **think_sft** show substantial rescue in **math** and **science** (e.g. think_sft math 0.5, science 0.58 under Asch).
- **instruct_dpo** shows strong rescue in **science** and **truthfulness** (e.g. science 0.48, truthfulness 0.37–0.54).
- **instruct_sft** often has **zero or very low** rescue on truthfulness (e.g. 0.0 at T=0.4), and generally lower rescue than think/instruct_dpo in factual categories.

So: **instruct_sft** is most prone to harmful override, especially on opinion and truthfulness; **think/think_sft** show more rescue on math/science; **instruct_dpo** is most resistant to override and shows strong rescue on truthfulness and science.

---

### 3.5 Wrong-answer flip rate

**Table: `wrong_answer_flip.csv`**

Wrong-answer flip = P(agree with wrong answer under pressure | control did not agree). It is a ground-truth–aligned factual conformity measure.

- **instruct_sft** and **think** family: high flip rates on **opinion** and **truthfulness** (e.g. instruct_sft opinion 0.80–0.85, truthfulness 0.72–0.82; think truthfulness up to ≈ 0.74).
- **instruct_dpo**: consistently **lowest** flip rates (e.g. opinion/truthfulness often 0.19–0.31 under Asch).
- **math**: think and think_sft have the **lowest** flip rates (≈ 0.15–0.25), in line with stronger math resistance and higher rescue.

---

### 3.6 Opinion (wrong-answer agreement)

**Table: `opinion_agreement.csv`**

On opinion items, “wrong-answer agreement” under pressure is used as a conformity proxy.

- **instruct_sft** under Asch: wrong-answer agreement is very high (0.81–0.87 across temperatures).
- **think** and **think_sft**: 0.58–0.78 under pressure.
- **instruct_dpo**: lowest (0.21–0.36 under Asch).
- Refusal rates under pressure are highest for **think** and **think_sft** (e.g. 0.16–0.31), and elevated for **instruct_sft** (0.19–0.28).

---

### 3.7 Social contagion and item-level spread

**Tables: `social_contagion_spread.csv`, `extreme_contagion_items.csv`**

- **social_contagion_spread** ranks items by how many variant–condition combinations flip to the wrong answer (“spread”). High-spread items (e.g. spread 80–120+) are conformed to across many model/condition combinations; low-spread items (0–6) show little contagion.
- **extreme_contagion_items** lists items with minimal spread (0–6), i.e. items that almost never flip across the suite—useful for identifying robust items and, by contrast, highly contagious ones in the main spread table.

Together with **per_item_difficulty.csv** and **fig_item_difficulty_scatter.png**, this supports that **harder items** (and items with higher control accuracy) are less likely to be flipped under pressure, consistent with the negative item-difficulty–conformity correlation.

---

## 4. Results That Stand Out

1. **Item difficulty consistently predicts conformity**  
   In all variants and both pressure types, higher item difficulty (higher control accuracy) is associated with **less** conformity (negative Pearson r, p ≪ 1e−10). Think/instruct variants show the strongest such correlations.

2. **instruct_sft is the most conformity-prone**  
   Highest truth override (especially on opinion and truthfulness), high wrong-answer agreement on opinion, and high normalized pressure sensitivity. Truth rescue is often absent (e.g. zero on truthfulness in several settings).

3. **instruct_dpo is the most resistant**  
   Lowest truth override and wrong-answer flip rates, and negative pressure deltas on truthfulness/science in several cells—pressure sometimes **reduces** error (rescue or resistance). Strong rescue on science and truthfulness.

4. **Think variants: strong on math, moderate on opinion**  
   Think and think_sft show low math override and low wrong-answer flip on math, and substantial truth rescue on math and science. On opinion and authority they still show sizable pressure effects and high relative sensitivity.

5. **Peer vs authority**  
   Authority (authoritative_bias) often produces effects similar to or larger than Asch (e.g. on opinion and reasoning). **fig_peer_authority_asymmetry** and the delta tables show that the two pressure types are not identical and can differ by variant and category.

6. **Temperature**  
   Error rates and pressure effects vary with temperature; **fig_temperature_conformity_curve** and the rate tables show that conformity is not a simple monotonic function of temperature and can differ by variant and condition.

7. **Topic/category**  
   Opinion and truthfulness show the largest override and flip rates for instruct_sft and think family; math and science show more rescue and resistance, especially for think and instruct_dpo. **fig9_topic_pressure_heatmap** and the category breakdowns in the tables document this clearly.

---

## 5. Conclusion

The v7 behavioral suite shows that **conformity is systematic and variant-dependent**: item difficulty, pressure type (peer vs authority), and dataset category all modulate effects. **instruct_sft** is the most susceptible to harmful conformity (high override, high flip, low rescue), **instruct_dpo** the most robust (low override, negative deltas on truthfulness/science, high rescue), and **think** variants occupy a middle ground with strong math/science rescue and lower math override. These patterns are consistent across the summarized figures and tables and support targeted mitigation (e.g. by variant and domain) and the use of item difficulty and topic in the design of conformity evaluations.
