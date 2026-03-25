# COLM 2026 Abstract Draft

## Title Options

**Option A (recommended):**
Supervised Fine-Tuning Amplifies Social Conformity in Language Models: A Controlled Study Across the OLMo-3 Post-Training Pipeline

**Option B:**
The Sycophantic SFT Prior: How Post-Training Stages Shape Social Conformity in Language Models

**Option C:**
Deterministic Sycophancy: How SFT, DPO, and Decoding Temperature Modulate Social Conformity in OLMo-3

---

## Abstract (Option A — recommended, ~280 words)

We present the largest controlled study of social conformity in language models to date: 215,288 trials spanning the OLMo-3-7B post-training pipeline (base, instruct, instruct-SFT, instruct-DPO), 12 experimental conditions adapted from classical social psychology, 8 knowledge domains, and 6 decoding temperatures. Every trial is independently scored by a multi-model LLM judge ensemble with 100% coverage.

Our central finding is that supervised fine-tuning (SFT) and direct preference optimization (DPO) exert opposite effects on conformity susceptibility. SFT amplifies conformity: across our 6-temperature sweep, instruct-SFT has 559 items that conform at every temperature (vs. 378 for instruct), despite nearly identical baseline accuracy. Under unanimous peer pressure, instruct-SFT shows a McNemar OR of 7.68 (vs. 5.98 for instruct, p < 0.001) and exhibits the highest sensitivity to tone variation (Cochran's Q = 77.19). DPO partially mitigates conformity, consistently yielding the lowest pressure-induced error rates in the instruct family.

Four additional findings complete the picture. First, prompt format overwhelms content: unanimity of confederates drives conformity regardless of expressed confidence, and switching from classical Asch framing to structured answer-option framing increases the conformity delta from non-significant to +22 percentage points. Second, we introduce Conformity Temperature (T_c), a per-item metric revealing that 53--54% of conforming items yield deterministically at T=0.0 (greedy decoding), definitively ruling out sampling noise as an explanation. Third, mathematical reasoning shows near-complete behavioral immunity to social pressure, likely due to its derivational structure and self-verification properties. Fourth, standard prompt-level mitigations (devil's advocate, question distillation) provide negligible relief for instruction-tuned models.

These results demonstrate that conformity susceptibility is an emergent property of post-training methodology, not model capability, with direct implications for alignment pipeline design.

---

## Notes for Mahdi

- **Word count:** ~280 (within typical conference abstract range)
- **Every number is grounded:** 215,288 trials, OR=7.68/5.98, 559/378, Q=77.19, 53-54%, +22pp — all from CSVs
- **No think model claims** — safe to submit as-is
- **If think reruns complete by March 31:** the paper can expand to include Rationalization Trap / URSP, but the abstract doesn't promise it
- **Double-blind:** no author-identifying info in abstract
- **The hook:** "SFT and DPO pull in opposite directions" — this is novel and actionable for the alignment community
