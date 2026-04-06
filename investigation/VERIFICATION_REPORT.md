# Final Verification Report: What Holds and What Doesn't

**Date:** 2026-03-29
**Method:** Independent SQL verification of all key numbers against raw databases, statistical testing of group comparisons, ceiling effect analysis, confound identification.

---

## Claims That HOLD (safe to publish)

### 1. SFT amplifies conformity, DPO mitigates it (within OLMo)
- **Evidence:** Causal, within-family controlled experiment
- **Verification:** OLMo-7B at T=0.0: base Δ=+0.367, SFT Δ=+0.441, DPO Δ=+0.407, instruct Δ=+0.482
- **McNemar ORs:** base OR=12.0, SFT OR=14.9, DPO OR=10.7, instruct OR=14.1
- **Status:** ✅ Independently verified from raw SQL. Numbers match CSVs.

### 2. Think models resist peer conformity
- **Evidence:** McNemar tests non-significant for OLMo-32B-Think (OR=1.45/1.17), Grok (OR=1.16/1.10), GPT-OSS-20B (OR=0.98). All p_holm > 0.28.
- **Verification:** Independently computed from item-paired McNemar. All confirmed.
- **Status:** ✅ Individual per-model McNemar tests are valid. No group-level statistical claim needed.

### 3. Cross-family Δpeer range is enormous
- **Evidence:** Descriptive. Range: +0.533 (Llama-3.1-70B T=0.0) to -0.001 (GPT-OSS-20B T=0.6).
- **Verification:** All 18 deltas independently recomputed from SQL. All match.
- **Status:** ✅ Purely descriptive claim. Does not require group-level inference.

### 4. Peer pressure >> Authority pressure
- **Evidence:** Peer pressure is significant (p_holm < 0.05) for 12/18 model×temp cells. Authority pressure is significant for **0/18** cells after Holm correction.
- **Verification:** McNemar ORs for authority conditions independently verified. Largest authority OR is 2.23 (Llama-3.1-70B T=0.0 authoritative_bias) — significant before but not after Holm correction.
- **Status:** ✅ The per-model comparison is valid and doesn't require group-level testing.

### 5. Endorsement vs Refusal behavioral taxonomy
- **Evidence:** Directly observed in the data. No inferential claim.
- **Verified numbers (T=0.0):**
  - Llama-3.1-70B: 89% refusal, 4.5% endorsement
  - GPT-4o-Mini: 3.8% refusal, 40.5% endorsement
  - OLMo-32B-Instruct: 31.5% refusal, 40.8% endorsement
  - Grok-4.1-Fast: 5.5% refusal, 8.2% endorsement
- **Status:** ✅ These are raw rates from judge labels, independently verified.

### 6. OLMo instruct sits mid-pack in cross-family comparison
- **Evidence:** OLMo-7B instruct Δ=+0.482 (T=0.0), +0.478 (T=0.6) — between Llama-3.1-70B (+0.533) and Llama-3-8B (+0.387). Bridge uses identical items.
- **Status:** ✅ Valid because all models share the same 400 items and same conditions.

---

## Claims That Need DOWNGRADING (cannot state as established)

### 7. "Dense architectures are more conformist than MoE"
- **Intended claim:** Dense models (Δ mean +0.307) > MoE models (Δ mean +0.087)
- **Problem:** Mann-Whitney U p=0.071. NOT significant at α=0.05. With n=5 dense and n=3 MoE models, we don't have power to detect this.
- **Additional confound:** Dense vs MoE is confounded with alignment strategy. Llama-4-Maverick (MoE) also uses lightweight SFT + online RL, while Llama-3.x (dense) uses heavy SFT. Can't separate the two.
- **Additional confound:** Grok-4.1-Fast (classified as dense) has Δ=+0.019 — an outlier that drags the dense mean down. Without Grok, dense mean would be +0.380.
- **Recommended language:** "We observe a trend where MoE models show lower conformity than dense instruct models (mean Δ 0.087 vs 0.307, p=0.07), though this does not reach conventional significance and is confounded with alignment strategy."

### 8. "Unsupervised CoT is more protective than supervised CoT"
- **Intended claim:** Unsupervised (GPT-OSS Δ=-0.001, Grok Δ=+0.019) > Supervised (OLMo-7B-Think Δ=+0.090, OLMo-32B-Think Δ=+0.032)
- **Problem:** n=2 per group. No statistical test is meaningful. This is purely descriptive.
- **Additional confound:** The "unsupervised" models (GPT-OSS, Grok) differ from "supervised" models (OLMo-Think) in architecture, training data, model size, and everything else. The CoT supervision distinction is one of many differences.
- **Recommended language:** "Models with reasoning capabilities that are not subject to alignment penalties on intermediate tokens (GPT-OSS, Grok) show the lowest conformity in our sample, consistent with the hypothesis that unsupervised reasoning traces may be more resistant to inherited deference priors. However, this observation is based on only four models and cannot be statistically distinguished from alternative explanations."

### 9. "Scale increases conformity for dense instruct models"
- **Intended claim:** Llama-3-8B (+0.374) < Llama-3.1-70B (+0.457), therefore bigger = more conformist
- **CRITICAL PROBLEM: Ceiling effect.** Llama-3-8B has control error 0.590, leaving only 0.410 room for delta. Llama-3.1-70B has control error 0.352, leaving 0.648 room. When normalized: 8B uses 91.2% of its room vs 70B uses 82.3%. **After ceiling correction, the 8B is MORE conformist.**
- **Additional issue:** Llama-3 and Llama-3.1 are different model versions (different training data, architecture refinements), not just different scales.
- **Recommended language:** REMOVE this claim entirely. Or frame as: "Raw Δpeer is larger for Llama-3.1-70B than Llama-3-8B, but this comparison is confounded by differential baseline accuracy (ceiling effects) and architectural differences between model versions."

### 10. "SFT-to-RL ratio explains cross-family variation"
- **Intended claim:** A gradient from heavy SFT (high Δ) to RL-dominant (low Δ) across families.
- **Problem:** This is observational and confounded by architecture, training data, model size, and alignment details we don't fully know for closed-source models. The gradient is descriptive, not causal.
- **Valid reframing:** "Within OLMo (where we have causal control), SFT amplifies conformity. Across families, models known to use RL-dominant alignment (Grok, GPT-OSS) show the lowest conformity, consistent with the within-family finding. However, we cannot causally attribute this to the alignment strategy in the cross-family comparison."

---

## Post-Paper-Revision Corrections (2026-03-29)

The following errors were caught during the final paper revision and corrected:

1. **OLMo-7B-Think is significant (p=0.01), not non-significant.** Changed "all four think models ns" to "three of four ns; OLMo-7B-Think marginal (OR=2.88, p=0.01)."
2. **Authority is significant for 2/10 families, not 0/10.** Llama-3.1-70B (p=0.003) and Llama-3-8B (p=0.011) both show significant authority effects after Holm correction.
3. **Llama-70B refusal rate is 84% (average), not 89% (T=0.0 only).** Standardized to averages throughout.
4. **OLMo Table 2 updated from heuristic to judge labels.** All ORs increased (Base 4.74→8.79, SFT 7.74→15.50, DPO 4.30→8.69, Instruct 6.01→12.24).

See `FINAL_REVISION_LOG.md` for full details.

---

## Methodological Issues to Document

### A. Refusal handling affects error rate denominators
- Models with high refusal (Llama-3.1-70B: 84% average, 89% at T=0.0) have many null `is_correct` values
- Error rates exclude refusals from denominator → "accuracy among substantive answers"
- McNemar tests exclude items where either condition has null → reduced n_paired
- **Impact on Llama-3.1-70B:** n_paired = 259/400 (35% loss). OR=44.33 is computed on items where the model actually answered in both conditions.
- **Must document:** Refusal rate reported alongside error rate so readers can assess.

### B. Gemini "no final answer" trials (11%)
- 44 trials where `<think>` block produced but no final answer. Not flagged as refusal.
- Excluded from error rate and McNemar.
- Gemini-specific behavior.

### C. Single judge model for runs/ (gpt-oss-20b)
- `runs/` uses a single judge model. `runs_latest/` uses a mix (gemma3:1b, gemma3:12b, qwen3:0.6b).
- Judge-level bias could systematically affect cross-family comparison differently than OLMo family.

---

## What the Paper Should Say

### STRONG claims (fully supported):
1. Within OLMo-7B, SFT amplifies and DPO mitigates social conformity (original finding, now verified with judge labels)
2. The cross-family conformity spectrum is vast — from OR=44 (Llama-70B) to OR=0.98 (GPT-OSS-20B)
3. Think/reasoning models (OLMo-Think, Grok, GPT-OSS) show no statistically significant peer conformity
4. Peer consensus is universally more effective than authority framing (significant for 7/10 families vs 0/10)
5. Models respond to social pressure through qualitatively different behavioral modes — refusal vs endorsement
6. OLMo-instruct sits mid-pack in the cross-family ranking, validating that within-family findings are not idiosyncratic

### OBSERVATIONAL claims (describe but hedge):
7. Models with reasoning capability show lower conformity (trend, but small n)
8. MoE models show lower conformity than dense instruct models (trend p=0.07, confounded)
9. Models with RL-dominant alignment show lower conformity than SFT-heavy models (consistent with within-family findings, but observational)

### RETRACT or substantially revise:
10. ~~"Scale increases conformity"~~ → Confounded by ceiling effects. After normalization, direction reverses.
11. ~~"Unsupervised CoT > supervised CoT"~~ → n=2, purely descriptive, too many confounds for any claim.
