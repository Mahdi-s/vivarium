# COLM 2026 Abstract — Final

## Title

Supervised Fine-Tuning Amplifies Social Conformity in Language Models: A Controlled Study Across the OLMo-3 Post-Training Pipeline

---

## Abstract

We present a large-scale controlled study of social conformity in language models: 215,288 trials spanning the OLMo-3-7B post-training pipeline (base, instruct, instruct-SFT, instruct-DPO), 12 experimental conditions adapted from classical social psychology, 8 knowledge domains, and 6 decoding temperatures. Every trial is independently scored by a multi-model LLM judge ensemble with 100% coverage.

Our central finding is that supervised fine-tuning (SFT) and direct preference optimization (DPO) exert opposite effects on conformity susceptibility. SFT amplifies conformity: across a 6-temperature sweep, instruct-SFT has 559 items that conform at every temperature (vs. 378 for instruct), despite nearly identical baseline accuracy. Under unanimous peer pressure, instruct-SFT shows a McNemar OR of 7.68 (vs. 5.98 for instruct, p < 0.001) and exhibits the highest sensitivity to tone variation (Cochran's Q = 77.19). DPO partially mitigates conformity, consistently yielding the lowest pressure-induced error rates in the instruct family.

Four additional findings complete the picture. First, prompt format overwhelms content: unanimity of confederates drives conformity regardless of expressed confidence, and switching from classical Asch framing to structured answer-option framing increases the conformity delta from non-significant to +22 percentage points. Second, we introduce Conformity Temperature (T_c), a per-item metric revealing that 53--54% of conforming items yield deterministically at T=0.0 (greedy decoding), definitively ruling out sampling noise as an explanation. Third, mathematical reasoning shows near-complete behavioral immunity to social pressure. Fourth, standard prompt-level mitigations (devil's advocate, question distillation) provide negligible relief.

We are extending this study to the three OLMo-3 chain-of-thought (think) variants at both 7B and 32B scales to test whether extended reasoning training---which produces substantially longer output traces---amplifies or attenuates the conformity patterns observed in instruction-tuned models. Preliminary results from these extended-token reruns will be incorporated into the full paper.
