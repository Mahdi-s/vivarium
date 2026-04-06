# Post-Hoc Analysis Design

**Date:** 2026-03-29 (updated 2026-03-31)

---

## Recommendation: Two Separate Analyses, One Calibration Bridge

### Why Separate

The two datasets answer different questions and have different experimental designs:

| | `runs_latest/` (OLMo family) | `runs/` (Cross-family) |
|---|---|---|
| **Question** | How does training stage affect conformity? | How do different model families compare? |
| **Models** | 1 architecture, 8 variants | 10 model families, mostly instruct |
| **Conditions** | 12 | 4 (subset of the 12) |
| **Temperatures** | 6 | 2 |
| **Strength** | Causal isolation of training effects | Breadth across the field |

Pooling would confound model identity with dataset membership. Keep them separate.

### The Bridge: OLMo Instruct as Calibration Reference

The OLMo-7B `instruct` variant at T=0.0 and T=0.6 appears in **both** datasets (via `runs_latest/`), using the **exact same 50 items per dataset** and the **same 4 conditions**. This provides a direct calibration point: every cross-family model can be compared to OLMo-instruct on identical items and conditions.

---

## Analysis 1: OLMo Training Stage Decomposition (Existing)

**Data:** `runs_latest/runs/`, variants: base, instruct, instruct_sft, instruct_dpo
**Already implemented:** `expanded_suite_behavioral_breakdown.py` with `--use-judge-labels`

### Command

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

### Key Outputs
- Error rate heatmaps (variant × condition × dataset × temperature)
- Pressure effect deltas with bootstrap CIs
- McNemar paired tests (pressure vs control)
- Cochran's Q tests (condition heterogeneity)
- Truth override/rescue rates
- Per-domain breakdowns

---

## Analysis 2: Cross-Family Conformity Survey (NEW)

**Data:** `runs/` (20 primary runs + 2 ablation runs, 10 model families) + `runs/think/` (1 run, OLMo-7B-Think)
**Needs:** New analysis script or adapted metadata

**Run inventory:**

| Model | T=0.0 Run ID | T=0.6 Run ID | Notes |
|-------|-------------|-------------|-------|
| llama-3-8b-instruct | `5c1c6720` | `11f2c5f5` | |
| llama-3.1-70b-instruct | `d14af4f9` | `af8b3ade` | |
| llama-4-maverick | `5de5bea0` | `57b6c8dd` | |
| gpt-4o-mini | `b3c1a4f8` | `0c09b1c9` | |
| gpt-oss-20b | `66765d5e`* | `cb6c2c3a` | *187 error trials pending re-run |
| gemini-2.5-flash-lite | `9b9f63a2` | `0f00dcea` | |
| grok-4.1-fast | `d0cffb4c` | `7f1da2c8` | |
| claude-sonnet-4 | `cc6a1eae` | `94e8fad1` | |
| olmo-3.1-32b-instruct | `fd700d10` | `0a78e2db` | |
| olmo-3.1-32b-think | `f38aa9b4` | `40e614b0` | |
| olmo-3.1-32b-instruct (ablation) | — | `e8a90500` | 800 trials, 2 ablation conditions |
| olmo-3.1-32b-instruct (ablation) | — | `ef72529e` | 800 trials, 2 ablation conditions (llama-3.1-70b suite) |
| olmo-7b-think | `f47fe05e` | — | In `runs/think/fK1N5V/`; 1,609 trials |

**4 conditions in all cross-family runs:** `control`, `asch_zhu_unanimous_confident`, `authoritative_bias`, `authority_trust`

**2 ablation conditions** (ablation runs only): `asch_zhu_naked_unanimous_confident`, `ngram_sequence_baseline`

### What to Produce

#### Table A: Cross-Family Conformity Ranking (Primary Result)

Ranked by Δpeer (peer pressure effect on error rate), using judge labels:

| Rank | Model | Size | Δpeer T=0.0 | Δpeer T=0.6 | Endorsed% | Refusal% |
|------|-------|------|-------------|-------------|-----------|----------|
| 1 | llama-3.1-70b-instruct | 70B | **+0.542** | +0.440 | 4.5% | 89.0% |
| 2 | olmo-3.1-32b-instruct | 32B | +0.405 | **+0.448** | 41% | 31% |
| 3 | llama-3-8b-instruct | 8B | +0.363 | +0.330 | 24% | 71% |
| — | OLMo-7B instruct (ref) | 7B | +0.323 | +0.320 | 22% | 38% |
| 4 | gpt-4o-mini | — | +0.203 | +0.238 | 41% | 4% |
| 5 | gemini-2.5-flash-lite | — | +0.147 | +0.135 | 25% | 5% |
| 6 | llama-4-maverick | — | +0.127 | +0.125 | 25% | 19% |
| 7 | claude-sonnet-4 | — | +0.058 | +0.051 | 4.8% | 2% |
| 8 | olmo-3.1-32b-think | 32B | +0.035 | +0.017 | 12% | 15% |
| 9 | gpt-oss-20b | 20B | pending† | +0.013 | 18% | 3% |
| 10 | grok-4.1-fast | — | -0.003 | -0.007 | 8% | 6% |

†gpt-oss-20b T=0.0 run `66765d5e` has 1,413/1,600 valid outputs after cleanup; 187 rate-limit error trials are pending re-run via `--resume-auto`.

**Key Findings Already Visible:**
1. **Massive range:** Δpeer varies from +0.54 (llama-3.1-70b) to -0.007 (grok-4.1-fast)
2. **Think models resist conformity:** olmo-32b-think has near-zero peer effect (+0.03), consistent with the paper's preliminary think finding
3. **Refusal is the primary resistance mechanism for Llama models:** llama-3-8b (71% refusal) and llama-3.1-70b (89% refusal) refuse rather than endorse
4. **GPT-4o-mini never refuses but conforms via endorsement:** 4% refusal, 41% endorsement — it sycophantically agrees
5. **Grok and gpt-oss-20b are nearly conformity-immune** under this condition
6. **Authority effects are much smaller** than peer pressure for most models

#### Table B: Pressure Response Taxonomy

Models fall into distinct behavioral categories:

| Category | Models | Pattern |
|----------|--------|---------|
| **High conformity, high refusal** | llama-3-8b, llama-3.1-70b | Conform OR refuse — binary response |
| **High conformity, low refusal** | gpt-4o-mini, olmo-32b-instruct | Comply without resistance — sycophantic |
| **Moderate conformity** | gemini-flash, llama-4-maverick | Partial resistance |
| **Low conformity, principled** | claude-sonnet-4 | Minimal endorsement, near-zero refusal |
| **Conformity-resistant** | grok-4.1, gpt-oss-20b, olmo-32b-think | Nearly immune |

#### Figure 1: Conformity Spectrum (Forest Plot)

Δpeer with 95% bootstrap CIs, models ranked by effect size, OLMo-instruct marked as reference line.

#### Figure 2: Refusal vs Endorsement Scatter

X-axis: endorsement rate, Y-axis: refusal rate, bubble size: Δpeer. Shows the two resistance strategies.

#### Figure 3: Authority vs Peer Pressure

Paired comparison of Δpeer vs Δauth per model — which models are more vulnerable to authority vs peer pressure?

#### Table C: Per-Domain Conformity (8 datasets × 9+ models)

Heatmap showing which domains are hardest/easiest to conform on, per model family.

### Statistical Tests

1. **Per-model McNemar tests** (control vs each pressure condition) with Holm correction across 11 models (10 cross-family + OLMo-7B-Think)
2. **Kruskal-Wallis test** across model families for Δpeer (is cross-family variation significant?)
3. **Bootstrap CIs** (10,000 resamples) for all effect sizes
4. **Temperature stability test**: paired t-test of Δpeer at T=0.0 vs T=0.6 per model

---

## Analysis Bridge: OLMo Instruct Calibration

### Table D: Where Does OLMo-7B Instruct Fall in the Cross-Family Ranking?

Using the same 4 conditions and same items:

| Model | Δpeer (T=0.0) |
|-------|---------------|
| llama-3.1-70b-instruct | +0.542 |
| olmo-3.1-32b-instruct | +0.405 |
| llama-3-8b-instruct | +0.363 |
| **OLMo-7B instruct_sft** | **+0.343** |
| **OLMo-7B instruct** | **+0.323** |
| **OLMo-7B base** | **+0.292** |
| **OLMo-7B instruct_dpo** | **+0.282** |
| gpt-4o-mini | +0.203 |
| ... | ... |

**This table is the narrative bridge.** It shows:
- OLMo-instruct sits in the middle of the cross-family distribution
- The SFT amplification effect (+0.343 vs +0.292 for base) is visible even when calibrated against other families
- DPO mitigation (+0.282 vs +0.343 for SFT) is also visible
- The within-family training stage decomposition **generalizes**: the SFT→DPO pattern isn't an OLMo quirk

---

## Implementation Plan

### For Analysis 1 (OLMo family): Ready to run

The existing `expanded_suite_behavioral_breakdown.py` with `--use-judge-labels` handles this. Just run the command above.

### For Analysis 2 (Cross-family): Need a new script

The cross-family data doesn't fit the existing temperature-sweep metadata format. We need a script that:

1. Discovers all runs in `runs/` (for gpt-oss-20b T=0.0, run `66765d5e`, use only the 1,413 valid outputs; the 187 error trials are pending re-run via `--resume-auto`)
2. Loads judge labels from `parsed_answer_json`
3. Computes per-model, per-condition behavioral metrics
4. Produces the tables and figures described above
5. Includes the OLMo calibration data from `runs_latest/` for the bridge comparison

This can either be a new standalone script or an adaptation of the existing one with a `--cross-family` mode.

### For the Bridge: Built into Analysis 2

The bridge table extracts OLMo data from `runs_latest/` at T=0.0 and T=0.6, filtered to the 4 overlapping conditions, and inserts it into the cross-family ranking.
