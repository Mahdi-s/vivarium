# V7 Evidence Audit and Narrative Reconstruction

## Scope and provenance
- Audited artifacts:
  - `Comparing_Experiments/v7_publication/behavioral/figures/*`
  - `Comparing_Experiments/v7_publication/behavioral/tables/*`
- Cross-checked plot semantics against `Analysis Scripts/expanded_suite_behavioral_breakdown.py`.
- Core evidence pool in v7 behavioral tables contains **6 variants**: `base`, `instruct`, `instruct_sft`, `instruct_dpo`, `think`, `think_sft`.

---

## Step 1 — Evidence audit

### A. Figure audit (all files in `behavioral/figures/`)

### Heatmap families

| Figure | What it shows | Key pattern | Nuance/surprise | What it does not show |
|---|---|---|---|---|
| `factual_control_error_rate_heatmaps.png` | Per-variant heatmaps of factual control error; axes: topic (rows) × temperature (cols). | Think/Think-SFT have consistently lower control error than Instruct-family/Base. | Strong topic dependence in baseline error (reasoning/truthfulness often harder than science). | No pressure conditions; no uncertainty/CI; no pooled effect sizes. |
| `factual_pressure_effect_asch_heatmaps.png` | Asch delta error (`Asch - Control`) by topic × temperature per variant. | Most cells are positive; Instruct-SFT has broad, strong positive deltas. | Some cells are negative (pressure improves performance), especially in Instruct/Instruct-DPO science/truthfulness pockets. | No denominator heatmap in this folder; cannot assess low-N instability from this figure alone. |
| `factual_pressure_effect_authority_heatmaps.png` | Authority delta error (`Authority - Control`) by topic × temperature per variant. | Authority often stronger than Asch for Instruct/Instruct-DPO/Think families. | Instruct-SFT shows the reverse (peer stronger than authority). | Same denominator caveat; no inferential test. |
| `factual_truth_override_asch_heatmaps.png` | Truth override rate under Asch (`P(pressure wrong | control correct)`) by topic × temperature per variant. | Instruct-SFT broadly high override. | Topic bands vary strongly; math/science often lower than reasoning/opinion/truthfulness in several variants. | Without paired `n_items` heatmaps here, high/low values can hide denominator differences. |
| `factual_truth_override_authority_heatmaps.png` | Truth override rate under authority by topic × temperature per variant. | Authority override is high for Think family and nontrivial for all variants. | Instruct-DPO remains lower than Instruct-SFT under Asch, but authority closes part of that gap. | Same denominator caveat. |
| `factual_truth_rescue_asch_heatmaps.png` | Truth rescue (`P(pressure correct | control wrong)`) under Asch by topic × temperature per variant. | Think/Think-SFT and Instruct-DPO show notable rescue in some domains. | Rescue and override can co-exist: pressure can both harm and help depending on item subset. | No direct net-effect summary; must combine with base error and override. |
| `factual_truth_rescue_authority_heatmaps.png` | Authority truth rescue by topic × temperature per variant. | Rescue appears in science/truthfulness pockets for Instruct-DPO and Think-family. | Authority is not uniformly harmful at cell level. | Same denominator caveat; cannot infer dominance without paired override context. |
| `factual_wrong_answer_flip_asch_heatmaps.png` | Wrong-answer flip under Asch (`P(pressure endorses wrong | control did not)`) by topic × temperature. | Instruct-SFT and Think-family often high on opinion/truthfulness. | Instruct-DPO frequently lower than peers. | No factual correctness context in this metric; does not measure final correctness directly. |
| `factual_wrong_answer_flip_authority_heatmaps.png` | Wrong-answer flip under authority by topic × temperature. | Authority flips are high across many variants/topics, notably opinion/reasoning. | Instruct/Instruct-DPO authority > peer asymmetry is visible. | Same caveat: conformity signal, not direct error rate. |

### Main publication figures

| Figure | What it shows | Key pattern | Nuance/surprise | What it does not show |
|---|---|---|---|---|
| `fig1_error_rate_dot_plot.png` | Pooled factual error by variant and condition (`control`, `asch_history_5`, `authoritative_bias`), with connectors from control to pressure. | All six variants worsen in pooled factual error under both pressures. | Think-family starts much better at control but loses much of that advantage under pressure. | No per-topic or per-temperature decomposition; no uncertainty intervals. |
| `fig2_pressure_lollipop.png` | Mean pressure deltas by variant (Asch vs authority). | Instruct-SFT has largest Asch delta; Instruct/Instruct-DPO have larger authority deltas. | Makes asymmetry easy to compare at a glance. | Means over topics/temperatures can hide sign reversals in individual cells. |
| `fig4_topic_analysis.png` | Panel (a) control error heatmap (topic × variant), panel (b) Asch deltas by topic. | Reasoning and knowledge/general are pressure-sensitive; science smaller deltas. | Only Asch is shown in panel (b); authority asymmetry not visible in this panel. | Not a full pressure comparison figure; no authority panel. |
| `fig5_truth_override_rescue_dumbbell.png` | Override vs rescue by variant in two panels (Asch and authority). | Instruct-SFT shows large override-rescue gap; Instruct-DPO/Think show comparatively higher rescue. | The "gap" structure highlights harmful vs beneficial pressure effects jointly. | No per-topic breakdown; pooled values can hide domain exceptions. |
| `fig6_endorsement_slope.png` | Panel (a): pooled wrong-answer flip (factual) by pressure type; panel (b): opinion agreement trajectory across conditions. | Strong pressure-induced endorsement rise, especially for Instruct-SFT and Think-family. | Authority-dominant trajectories for Instruct/Instruct-DPO. | Panel (b) is pooled over temperatures; not a temperature slope plot. |
| `fig9_topic_pressure_heatmap.png` | Two heatmaps: topic × variant deltas for Asch and authority. | Reasoning highest average deltas; science lowest; clear peer/authority asymmetry by variant. | Instruct-SFT is peer-dominant; Instruct/Instruct-DPO authority-dominant. | Pooled over temperature; no within-topic temperature curves. |
| `fig12_sankey_conformity.png` | Flow diagrams (by family) from control correctness to pressure correctness. | Large `Correct→Incorrect` flows under pressure, especially Think and Instruct-SFT contexts. | Shows both harmful override and beneficial rescue pathways in one view. | Family-level aggregation (not per-variant); no topic decomposition. |
| `fig_temperature_conformity_curve.png` + `.pdf` | Truth override vs temperature by variant with bootstrap CI, separate panels for Asch/authority. | Temperature matters, but not monotonically across variants. | Mid-temperature peaks for some variants; no single best T across all variants. | Uses truth override only; not full error-rate curves. |
| `fig_training_trajectory.png` + `.pdf` | Stage-path view of pooled truth override (base→instruct/think→SFT/DPO). | Instruct-SFT rises sharply vs Instruct-DPO; Think-SFT remains high relative to base. | Useful for branch-wise narrative (SFT vs DPO divergence). | Observational only; not causal attribution of training method. |
| `fig_peer_authority_asymmetry.png` + `.pdf` | Heatmap of `delta_authority - delta_asch` by topic × variant. | Instruct/Instruct-DPO generally positive (authority stronger); Instruct-SFT negative (peer stronger). | Reveals mechanism asymmetry compactly. | No absolute effect size context unless read with delta figures. |
| `fig_normalized_sensitivity.png` + `.pdf` | Raw delta bars vs normalized sensitivity (`delta / (1 - control_error_rate)`). | Think-family remains highly sensitive even after baseline normalization. | Controls the "room-to-fall" confound from different baseline error rates. | Still pooled; does not replace topic/temperature analysis. |
| `fig_item_difficulty_scatter.png` + `.pdf` | Item control accuracy vs truth override, faceted by variant × pressure; Pearson r annotation. | All reported correlations are negative and significant. | Suggests stronger items (higher control accuracy) resist override. | Pairing logic in the source aggregation merges across temperatures; interpret as exploratory trend, not strict paired-temperature inference. |
| `fig_radar_conformity_profiles.png` + `.pdf` | Six-axis profile per variant (override, flip, opinion-agree under both pressures). | Distinct shape signatures: Instruct-SFT high peer axes; Instruct/Instruct-DPO authority-skewed; Think-family broadly high pressure sensitivity. | Good for rapid profile comparison. | Mixes metrics with different denominators/definitions; not suitable for inferential claims alone. |
| `fig_social_contagion_histogram.png` + `.pdf` | Histogram of "spread" counts from `social_contagion_spread.csv`. | Indicates broad distribution of item susceptibility. | Visually compelling for contagion narrative. | Source computation merges control/pressure without temperature key, so spread is not literal "# variants"; use as exploratory only. |

### B. Table audit (all files in `behavioral/tables/`)

| Table | What it shows | Key pattern | Nuance/surprise | What it does not show |
|---|---|---|---|---|
| `factual_rates_by_temp_variant_condition_category.csv` | Base counts and rates by temp×variant×condition×topic. | Core source for pooled error and cell-level heterogeneity. | Cell counts vary (not exactly equal N). | No direct deltas; must compute from condition contrasts. |
| `factual_pressure_deltas.csv` | Per-cell `delta_asch` and `delta_authority` plus underlying errors. | Most deltas positive; meaningful minority negative (~15.5% each). | Captures where pressure helps. | No uncertainty and no item-level pairing stats. |
| `truth_override.csv` | Conditional override by temp×variant×topic×pressure with denominators. | Instruct-SFT highest Asch override; Instruct/Instruct-DPO authority>peer asymmetry. | Strong topic variation within variant. | No direct rescue in same table. |
| `truth_rescue.csv` | Conditional rescue by temp×variant×topic×pressure. | Think/Think-SFT and Instruct-DPO show substantial rescue pockets. | Pressure can help under control-wrong subsets. | Does not indicate net error direction by itself. |
| `wrong_answer_flip.csv` | Conditional wrong-answer endorsement flips by temp×variant×topic×pressure. | Instruct-DPO lowest Asch flip; Instruct-SFT highest Asch flip. | Authority-heavy flips for Instruct/Instruct-DPO. | Not equivalent to correctness. |
| `opinion_agreement.csv` | Wrong-answer agreement and refusal by temp×variant×condition on opinion subset. | Large pressure-induced agreement increases for all variants. | Think/Think-SFT and Instruct-SFT also have high refusal rates under pressure. | No topic decomposition (single subset). |
| `normalized_pressure_sensitivity.csv` | Raw and normalized deltas per temp×variant×topic. | Sensitivity remains high for Think-family after normalization. | Reduces baseline confound. | Still descriptive; no uncertainty. |
| `training_trajectory_summary.csv` | Weighted pooled truth override by variant and pressure. | SFT vs DPO divergence is visible in both branches. | Compact summary for trajectory plots. | Pooled collapse can hide temperature/topic exceptions. |
| `behavioral_stats_summary.csv` | Pearson item-difficulty correlations (variant×pressure). | All correlations negative and highly significant. | Strongest magnitudes around Instruct/Think families. | Despite filename, no intervention McNemar rows in this v7 export. |
| `per_item_difficulty.csv` | Item-level control accuracy and override rates per variant×pressure. | Basis for scatter-correlation trend. | Large row count gives stable descriptive trend. | Generated with cross-temperature pairing in source merge; not strict within-temperature pairs. |
| `social_contagion_spread.csv` | Item-level spread summary for Asch condition. | Identifies high-spread vs low-spread items. | Useful for ranking susceptible items. | Spread is not literal count of variants due merge-key issue; avoid literal interpretation. |
| `extreme_contagion_items.csv` | Subset of low-spread items from contagion table. | Surfaces resistant items. | Good for case-study selection. | Not a complete view of high-spread tail by itself. |

---

## Step 2 — Audit of pre-revision narrative (paper.tex)

### Well-supported claims
- Social pressure raises pooled factual error across tested variants.
- Instruct-SFT has much larger peer-pressure harm than other variants.
- Instruct/Instruct-DPO show authority-dominant asymmetry.
- Think-family has low control error but large pressure penalties.
- Topic heterogeneity is real (reasoning high, science low).

### Misaligned or overstated claims (pre-revision)
- Scope mismatch: text described 7 variants including Think-DPO, but v7 behavioral evidence includes 6 variants.
- Sample-size mismatch: text reported 50,400 design and 47,661 analyzed; v7 behavioral tables support 43,200 nominal (6-variant scope) and 41,505 analyzed.
- Significance section mismatch: McNemar table in paper was not part of v7 publication figure/table evidence.
- Over-strong generalization risk: phrasing implied uniformly harmful pressure, but ~15.5% of cells have negative deltas.
- "Opinion has no ground truth" framing was inconsistent with the artifact structure used for v7 behavioral tables.

### Gaps (pre-revision)
- Non-monotonic temperature behavior not foregrounded.
- Refusal-rate behavior under pressure (notably Think/Think-SFT, Instruct-SFT) not discussed.
- Additional v7 evidence panels (asymmetry/normalized sensitivity/item difficulty) were underused.

### Flow issues (pre-revision)
- Intro and abstract promised a broader checkpoint scope than the evidence actually used.
- Methods scope and Results evidence source were not tightly coupled.
- Conclusion repeated pooled claims without surfacing heterogeneity constraints.

---

## Step 3 — Reconstructed research story (evidence-grounded)

### One-sentence core finding
Prompt-induced conformity in OLMo-3 is robust in aggregate but mechanistically variant-specific: peer consensus disproportionately harms Instruct-SFT, while authority framing disproportionately harms Instruct/Instruct-DPO, and Think-family models lose much of their baseline accuracy advantage under pressure.

### Supporting findings
1. Pooled factual error rises under both pressure types for all six variants, with strongest peer effect in Instruct-SFT and strongest authority asymmetry in Instruct/Instruct-DPO.
2. Conditional metrics (override/rescue/flip) show distinct conformity mechanisms rather than a single scalar vulnerability.
3. Pressure effects are heterogeneous by topic and non-monotonic in temperature, with meaningful negative-delta pockets.

### Must-acknowledge limitations
- Evidence scope is six variants in the released v7 behavioral artifact.
- Outcomes depend on LLM-judge labels.
- Some supplementary item-level/contagion artifacts are exploratory due aggregation design (temperature pairing caveats).

### Why this matters for ACL/COLM
- Alignment evaluation needs mechanism-aware stress tests (peer vs authority) rather than one aggregate sycophancy score.
- Training-stage comparisons can reveal divergent failure modes (SFT vs DPO) that matter for deployment policy and evaluation design.
- Temperature tuning is not a reliable one-knob mitigation; robust evaluation must report domain- and condition-specific behavior.

---

## Step 4 — Revisions made in `paper/paper.tex`

- Re-scoped narrative to the six variants present in v7 behavioral evidence.
- Updated figure path to v7 publication outputs.
- Rewrote abstract around the true core finding and v7-supported numbers.
- Corrected sample-size reporting to v7 evidence (`41,505` analyzed trials).
- Removed unsupported McNemar results section/table from the main narrative.
- Added explicit temperature non-monotonicity subsection.
- Updated conditional-metric and topic-sensitivity values to match v7 tables.
- Added a Discussion section with evidence-based caveats and interpretation boundaries.

### Figure/table ordering and naming recommendations
1. Promote `fig2_pressure_lollipop.png` to main text immediately after `fig1` (it makes the central asymmetry claim visually explicit).
2. Keep `fig5_truth_override_rescue_dumbbell.png` as the mechanism figure; it directly supports the override-vs-rescue argument.
3. Keep `fig9_topic_pressure_heatmap.png` as the heterogeneity figure; consider adding `fig_peer_authority_asymmetry.png` if space allows.
4. Treat `fig_social_contagion_histogram` and `fig_item_difficulty_scatter` as supplementary/exploratory unless aggregation caveats are corrected.
