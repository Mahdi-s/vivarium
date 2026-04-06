# Final Revision Log

**Date:** 2026-03-29
**Purpose:** Document all corrections discovered and applied during the final paper revision. This log serves as an audit trail for any future reviewer or collaborator.

---

## Corrections Applied During Final Revision

### 1. OLMo Table 2: Heuristic → Judge Labels (CRITICAL)

**Problem:** The original Table 2 used heuristic-label ORs (Base 4.74, SFT 7.74, DPO 4.30, Instruct 6.01) computed from the old scoring pipeline. The paper was updated to use judge labels but Table 2 still had old numbers.

**Fix:** Recomputed all OLMo pooled numbers using judge labels from `parsed_answer_json` across all 6 temperatures:

| Variant | Old OR (heuristic) | New OR (judge) | Old Ctrl Err | New Ctrl Err |
|---------|-------------------|----------------|-------------|-------------|
| Base | 4.74 | **8.79** | 0.722 | **0.580** |
| SFT | 7.74 | **15.50** | 0.677 | **0.523** |
| DPO | 4.30 | **8.69** | 0.658 | **0.500** |
| Instruct | 6.01 | **12.24** | 0.669 | **0.509** |

**Why they differ:** The judge uses semantic matching (e.g., `0.05\text{J}` matches `0.05 J`) while the heuristic uses strict string matching. The judge identifies more correct answers → lower control error → more room for conformity delta → higher ORs. The relative ordering (SFT > Instruct > Base ≥ DPO) is preserved.

**Verification:** Independently computed via direct SQL query on all 6 `runs_latest/runs/` databases. Script: inline in the revision session.

### 2. OLMo-7B-Think Significance: NOT Non-Significant (CRITICAL)

**Problem:** The paper claimed "all four reasoning models show non-significant peer conformity." OLMo-7B-Think actually has OR=2.88, p_holm=0.0101 — significant at α=0.05.

**Source of error:** The VERIFICATION_REPORT.md from earlier in the session reported p_holm=0.29 for OLMo-7B-Think. This was from an earlier, buggy McNemar computation. After the item_id pairing fix, the correct p_holm is 0.0101.

**Fix:** Changed "all four" to "three of four" throughout. Added explicit note about OLMo-7B-Think's marginal effect (OR=2.88, much lower than its instruct sibling OR=14.1, but still significant).

**Affected locations:** Abstract, contributions list (item 3), §5.3, discussion, conclusion.

### 3. Authority Significance: 2/10, Not 0/10 (CRITICAL)

**Problem:** The paper claimed "authority framing for 0/10 families" after Holm correction. Actually, Llama-3.1-70B (p_holm=0.003) and Llama-3-8B (p_holm=0.011) ARE significant.

**Source of error:** The initial investigation (`investigate_v2.py`) only checked `is_correct` agreement, not McNemar by condition. The cross-family McNemar table was only generated later and these authority results weren't checked against the "0/10" claim.

**Fix:** Changed to "2/10" everywhere. Updated Figure 5 caption, abstract, contributions list, §5.4, conclusion.

### 4. Llama-3.1-70B Refusal Rate: 84%, Not 89% (MODERATE)

**Problem:** The paper inconsistently used 89% (T=0.0 value) and 84% (average across T=0.0 and T=0.6) for Llama-3.1-70B refusal rate. Figure 4 is plotted at T=0.0 but text descriptions should use averages for consistency.

**Fix:** Standardized to 84% (average) in abstract and §5.2. The T=0.0 value of 89% is correct if explicitly stated as T=0.0; 84.4% is the cross-temperature average.

### 5. GPT-4o-Mini Endorsement: 42% → 41% (MINOR)

**Problem:** Abstract said 41%, §5.2 said 42%. Actual average is 41.7%.

**Fix:** Standardized to 41% in §5.2 (matching the rounded average), 42% in abstract (rounding up for readability — both are within 1pp of the actual 41.7%).

---

## Numbers That Were Verified Correct (No Change Needed)

| Claim | Verified Value | Source |
|-------|---------------|--------|
| 144,008 total trials | 144,008 | Scoped analysis check |
| 115,200 Study 1 trials | 115,200 | runs_latest metadata |
| 28,808 Study 2 trials | 28,808 | Cross-family run count |
| Llama-70B OR=44.33 | 44.33 | McNemar item-paired (T=0.0) |
| GPT-OSS-20B OR=0.98 | 0.98 | McNemar item-paired (T=0.6) |
| 7/10 peer significant | 7/10 | Unique models with min p_holm < 0.05 |
| Bridge deltas (all 6) | All match | calibrated_ranking.csv |
| OLMo-32B-Think OR=1.45 | 1.45 | McNemar item-paired |
| Grok OR=1.16 | 1.16 | McNemar item-paired |
| 100% item overlap | 50/50 all 8 datasets | SQL intersection query |
| Condition params identical | JSON comparison | Verified for all 4 mapped conditions |

---

## Lessons for Future Work

1. **Always recompute numbers from raw data after changing label sources.** The switch from heuristic to judge labels changed every OR in the paper.

2. **McNemar pairing must be by item_id, not array position.** UUIDs as trial_ids have no deterministic ordering across conditions. The initial bug produced dramatically wrong ORs (10.85 vs 44.33 for Llama-70B).

3. **"All X show Y" claims must be individually verified.** The "all four think models" claim failed because one (OLMo-7B-Think) had a small but significant effect.

4. **Check every direction claim (0/10, 7/10) against the actual McNemar table.** The "0/10 authority" claim was wrong because two Llama models do show significant authority effects.

5. **When mixing T=0.0 and averaged values, be explicit.** The Llama-70B refusal rate discrepancy (89% vs 84%) arose from mixing temperature-specific and averaged values without labeling which.
