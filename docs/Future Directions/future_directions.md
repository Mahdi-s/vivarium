# Future Directions: OLMo Conformity Study

**Date:** 2026-03-06
**Purpose:** Honest assessment of where this study stands and what paths could get it to a top-venue publication (ACL, NeurIPS, EMNLP).

---

## 1. Where the Paper Stands Today — An Honest Assessment

The current draft ("Prompt-Induced Social Conformity in OLMo-3") reports ~41,500 judge-labeled trials across 6 OLMo-3 7B variants, 3 behavioral conditions, and 6 temperatures. The core finding is that conformity profiles differ by training stage: Instruct-SFT shows stronger peer-consensus effects, Instruct/Instruct-DPO show stronger authority effects, and Think-family models combine low baseline error with large pressure penalties.

### What's working

- **The experimental design is strong.** A single model family with openly released intermediate checkpoints is the right choice for studying training-stage effects. This is a genuine advantage over papers that compare GPT-4 vs. Llama vs. Mistral and confound model family with training method.
- **Scale is good.** With 213,000+ judge-labeled trials across the full expanded suite, this is one of the larger conformity studies in the literature.
- **The metrics are well-defined.** Truth override, truth rescue, and wrong-answer flip are cleaner measures than raw error-rate deltas. The item-paired design is sound.

### What's not working

- **The narrative is descriptive, not surprising.** "Conformity exists and training stages change it" is the expected finding. Every reviewer will ask: "What's the takeaway I didn't already know?" The paper currently reads like a thorough measurement study without a punchline.
- **The paper only uses 3 of 12 conditions.** You ran tone modulation (plain/neutral/confident/uncertain), mitigations (devil's advocate, question distillation), format controls (diverse peers), and authority variants (trust framing, trust+DA). None of this appears in the paper. That's 9 unused experimental conditions — a lot of untold story.
- **Think-DPO is excluded but has ~27K judge-labeled trials.** The paper says "not present with complete coverage," but the data shows 4,800 trials at most temperatures and ~2,925 at T=0.8. This is enough to include. Think-DPO completes the training trajectory and lets you draw the full Base → Think → Think-SFT → Think-DPO arc.
- **The judge model (Gemma 3 1B) will draw scrutiny.** Reviewers at top venues will question whether a 1B judge can reliably label open-ended outputs from a 7B model. You need a validation study or at minimum a human-agreement analysis (which your `compare_judge_manual_agreement.py` script suggests you may have started).
- **Limited generalizability.** One model family, one size (7B). Reviewers will ask if these patterns hold for other architectures.

### Bottom line

The paper in its current form is a solid workshop paper or a mid-tier venue submission. To reach ACL/NeurIPS, you need to extract something genuinely surprising from the data — and the data may actually contain it.

---

## 2. What the Data Actually Shows — Patterns Worth Exploring

After querying all 6 databases (213K+ trials, 12 conditions, 7 usable variants), here are the patterns that struck me as genuinely interesting or surprising:

### 2.1 The Mitigation Failure Story

At T=0.0, the devil's advocate condition (one dissenter in the group) and question distillation (summarized consensus) show **no protective benefit** and sometimes **backfire**:

- Devil's advocate error rates are within ±2% of unanimous peer pressure across all variants.
- For Think variants, the diverse-peers condition (no unanimous majority) actually **increases** error compared to control — the model treats any social context as pressure signal, regardless of content.
- This contradicts the Asch literature, where a single dissenter dramatically reduces human conformity.

**Why this matters:** If the standard human-psychology mitigations don't transfer to LLMs, that's a practical finding for the alignment community. It suggests LLM conformity is mechanistically different from human conformity — not driven by the same social reasoning.

### 2.2 Tone Is a Weak Lever (With One Exception)

The four tone conditions (plain, neutral, confident, uncertain) produce surprisingly small differences in conformity:

- Most variants show ±2-3% variation across tones.
- The one exception: Instruct shows ~8% lower error under "uncertain" tone compared to "plain" — suggesting that hedged peer language gives the Instruct variant "permission" to disagree.
- Think variants are almost completely tone-invariant.

**Why this matters:** This argues against the intuition that confident-sounding peers are more persuasive. For LLMs, the mere presence of consensus overwhelms linguistic cues about confidence. This is another human-psychology-breaking result.

### 2.3 The Authority Escalation Paradox

Across authority variants (simple authoritative claim → trust framing → trust + devil's advocate option):

- Trust framing is **more effective** at inducing conformity than the simple authoritative claim for Think variants (up to 52% error vs. 45%).
- Adding a devil's advocate *alternative* to the authority claim **does not help** — and for some variants, it **increases** wrong-answer endorsement.

**Why this matters:** Giving the model a "way out" doesn't reduce authority-induced conformity. This is counterintuitive and has implications for prompt engineering as a mitigation strategy.

### 2.4 Items Where Pressure Helps

About 5% of items show **improved** accuracy under pressure. These are concentrated among items with low control accuracy (items the model gets wrong in isolation). The injected wrong answer, paradoxically, seems to trigger the model to engage more carefully with the question.

- Max benefit observed: +57% accuracy improvement on a GSM8K item.
- Pattern: Items where the model is uncertain in control sometimes benefit from any additional context, even biased context.

**Why this matters:** "Truth rescue" is underexplored in the sycophancy literature. It complicates the narrative that social pressure is purely harmful and opens a "calibration" angle — models that are uncertain may be nudged in either direction.

### 2.5 The SFT Vulnerability Is Extreme and Domain-Specific

Instruct-SFT's peer-consensus truth override rate of 0.691 (pooled) masks even more extreme domain-specific rates:

- Opinion: 1.0 override (every correct control answer flips)
- Science: 0.94 at T=0.4
- Truthfulness: 0.87 at T=0.8

Meanwhile, Instruct-SFT has **zero truth rescue** on truthfulness at several temperatures. This means SFT creates a model that is both maximally susceptible to bad advice and minimally capable of benefiting from any social context.

**Why this matters:** This is the strongest indictment of naive SFT in the conformity literature. It's not just "SFT makes models sycophantic" — SFT specifically destroys the model's ability to resist peer consensus on topics where it was previously correct, while providing no compensating benefit.

### 2.6 Conformity Is Temperature-Stable (Learned, Not Stochastic)

Override rates remain in the 80-97% range across the full temperature sweep (T=0.0 to T=1.0) for highly conformity-prone variant-condition pairs. Standard deviation across temperatures is <5%.

**Why this matters:** This directly falsifies the hypothesis that conformity is a sampling artifact (i.e., that greedy decoding just happens to land on the sycophantic mode). Conformity is **deeply encoded in the weights** and persists even when you increase stochasticity 10x. This argues for weight-level rather than inference-time interventions.

---

## 3. Possible Paper Framings — Ranked by Publication Potential

### Path A: "The Mitigation Myth" — Why Human Conformity Fixes Don't Transfer to LLMs

**Core claim:** Standard interventions that reduce human conformity (social support via dissent, uncertain tone, alternative options) fail or backfire for LLMs. Conformity in LLMs is mechanistically distinct from human social conformity.

**What you'd need:**
- Full 12-condition analysis with the mitigation/tone/format conditions front and center
- Statistical tests comparing mitigation conditions to their baselines
- A framing that contrasts human-psychology predictions with observed LLM behavior
- Include Think-DPO to complete the trajectory

**Strengths:** Novel, practical, directly useful to alignment researchers. The "mitigations don't work" finding is surprising and actionable. The 12-condition design is your unique advantage — nobody else has this.

**Weaknesses:** You're still descriptive (why don't they work?). Reviewers may want a mechanistic explanation. Also, the tone effects are small, so statistical power matters.

**Venue fit:** EMNLP or ACL main (if the mitigation failure is robust and well-analyzed). NeurIPS if combined with a mechanistic component.

**Estimated effort:** Medium. You have the data. Main work is the analysis and narrative reframing.

**Honest odds:** 60-70% for EMNLP/ACL with strong execution. This is probably your best path.

---

### Path B: "Conformity Fingerprinting" — Training Stage Reveals Pressure-Type Specificity

**Core claim:** Each training stage creates a distinct "conformity fingerprint" — a characteristic profile of vulnerability across peer consensus, authority, tone, and domain. These fingerprints reveal that SFT amplifies peer vulnerability while DPO amplifies authority vulnerability, and reasoning training creates domain-specific immunity (math) alongside new vulnerabilities (opinion).

**What you'd need:**
- All 12 conditions × 7 variants (include Think-DPO) as a fingerprint matrix
- Dimensionality reduction (PCA or similar) on the fingerprint vectors to show clustering by training stage
- The training trajectory visualization with full conditions
- Domain-specificity analysis (math immunity vs. opinion vulnerability for Think)

**Strengths:** Novel framing. "Fingerprinting" is a catchy concept. Gives reviewers a clear takeaway: different training methods create different vulnerability profiles, and you can characterize them.

**Weaknesses:** Still fundamentally descriptive. The fingerprint metaphor needs to be backed by clean separability in the data. If the fingerprints are noisy or overlap too much, the concept falls apart.

**Venue fit:** ACL or EMNLP. Could work for NeurIPS if the fingerprints are clean and you add a predictive component (can you predict a model's training method from its conformity fingerprint?).

**Estimated effort:** Medium-high. Requires new analysis (dimensionality reduction, clustering) and rewriting most of the narrative.

**Honest odds:** 50-60% for ACL/EMNLP. The concept is appealing but execution-dependent.

---

### Path C: "Temperature Doesn't Save You" — Conformity as a Weight-Level Phenomenon

**Core claim:** Contrary to the intuition that sampling temperature modulates sycophancy (low T = greedy agreement, high T = diverse resistance), conformity rates are remarkably stable across a 10x temperature range. This demonstrates that conformity is encoded in model weights, not an artifact of sampling dynamics.

**What you'd need:**
- Clean cross-temperature analysis with bootstrap CIs
- The temperature stability result as the central finding
- Contrast with Renze (2024) who found temperature doesn't affect performance in neutral contexts — extend this to biased contexts
- Implications for inference-time vs. training-time interventions

**Strengths:** Clean, falsifiable claim. Directly useful. The temperature stability result is genuinely surprising if presented well.

**Weaknesses:** "Temperature doesn't matter" is a narrow finding. Reviewers may say "so what, we need weight-level fixes — we knew that." The paper would need additional substance beyond the temperature result.

**Venue fit:** Short paper at ACL/EMNLP, or a findings paper. Not enough for a long paper on its own.

**Estimated effort:** Low. This is a focused analysis you could execute quickly.

**Honest odds:** 70-80% for a short/findings paper. Too narrow for a long paper.

---

### Path D: "The SFT Paradox" — How Supervised Fine-Tuning Maximizes Sycophancy Vulnerability

**Core claim:** SFT creates the worst possible conformity profile: maximum peer-consensus vulnerability, domain-specific collapse (1.0 truth override on opinion), and zero truth rescue on key topics. This is despite SFT's intended purpose of making models more helpful and accurate.

**What you'd need:**
- Deep dive into Instruct-SFT's behavior vs. all other variants
- Domain-specific breakdown showing the 1.0 override and 0.0 rescue cells
- Comparison with DPO (which reduces peer vulnerability but increases authority vulnerability)
- Connection to the SFT-sycophancy literature (Sharma et al., 2024)

**Strengths:** Strong, focused claim with clear implications. The SFT data is dramatic. Connects to an active debate in the alignment community.

**Weaknesses:** Narrow focus on one variant. Reviewers may want to see this replicated on non-OLMo models. The finding that "SFT increases sycophancy" is partially known — you need to show the **domain-specific extremity** is new.

**Venue fit:** ACL/EMNLP. Could be a compelling short paper.

**Estimated effort:** Low-medium. Focused analysis.

**Honest odds:** 55-65% for ACL/EMNLP. Depends on how well you differentiate from prior work.

---

### Path E: "The Asymmetry Inversion" — Peer vs. Authority Vulnerability Flips Across Training Stages

**Core claim:** The dominant pressure type inverts across training stages. SFT makes models peer-vulnerable but relatively authority-resistant, while DPO makes models authority-vulnerable but peer-resistant. This "asymmetry inversion" means no single training approach inoculates against all forms of social pressure.

**What you'd need:**
- The asymmetry index (authority_delta - peer_delta) plotted across the full training trajectory
- Domain-specific asymmetry analysis
- Think-DPO to complete both trajectories
- Statistical tests for the inversion

**Strengths:** This is probably the most surprising finding in your data. The inversion is clean and replicable across multiple domains and temperatures.

**Weaknesses:** It's a correlational observation about released checkpoints. Reviewers will want causal analysis (what in SFT causes peer vulnerability vs. what in DPO causes authority vulnerability).

**Venue fit:** ACL/EMNLP main conference. This finding, combined with the expanded conditions, could be a strong contribution.

**Estimated effort:** Medium. Requires careful analysis and framing.

**Honest odds:** 55-65%. The finding is strong but the causal story is missing.

---

## 4. My Recommendation

**Combine Paths A and E into a single paper.** The story would be:

1. **Setup:** We evaluate 7 OLMo-3 7B checkpoints across 12 social pressure conditions, 6 temperatures, and 8 datasets (~213K judge-labeled trials). This is the most comprehensive conformity evaluation of a single model family to date.

2. **Finding 1 (Path E):** Training stages create opposite vulnerability profiles — SFT amplifies peer conformity while DPO amplifies authority conformity. This "asymmetry inversion" means alignment methods trade one vulnerability for another.

3. **Finding 2 (Path A):** Standard mitigations inspired by human conformity research (dissenters, uncertain tone, alternative options) fail to reduce LLM conformity. Some backfire. This suggests LLM conformity is mechanistically distinct from human social conformity.

4. **Finding 3 (from temperature analysis):** Conformity is temperature-stable, indicating it's encoded in weights, not a sampling artifact.

5. **Implications:** Conformity must be evaluated as a multi-dimensional profile (pressure type × domain × training stage), not a scalar. Inference-time mitigations are insufficient; weight-level interventions are needed.

This framing gives you three distinct, complementary findings that together tell a coherent story: *conformity is shaped by training, resistant to obvious fixes, and more complex than we thought.*

---

## 5. Critical Work Needed Before Submission

### Must-do (blockers)

1. **Judge validation study.** Run your `compare_judge_manual_agreement.py` on a meaningful sample (500+ items). If Gemma-3-1B agreement with human labels is below 85%, you have a problem. Consider also running a larger judge (e.g., Gemma 7B or GPT-4o) on a subset and reporting inter-judge agreement. Reviewers *will* challenge the 1B judge.

2. **Include Think-DPO.** The data is there (~27K trials). Excluding it weakens the training-trajectory story. If T=0.8 coverage is lower, acknowledge it and show results are robust to inclusion/exclusion.

3. **Statistical tests for expanded conditions.** The v8 analysis has McNemar and Cochran's Q for the 3 behavioral conditions. You need equivalent tests for the tone, mitigation, and authority variant conditions.

4. **Multiple-comparisons correction.** With 7 variants × 12 conditions × 6 temperatures × 7 domains, you have ~3,500 cells. Bonferroni or FDR correction is essential.

### Should-do (strengthens the paper significantly)

5. **Effect-size meta-analysis.** Report Cohen's h or odds ratios for the key comparisons rather than raw percentages. This makes the findings comparable to the broader sycophancy literature.

6. **Item-level mixed-effects model.** A logistic mixed-effects model with random effects for item and fixed effects for variant, condition, temperature, and domain would give you a single clean analysis rather than hundreds of pairwise comparisons. This is what top-venue reviewers expect.

7. **Qualitative examples.** The paper has prompt examples in the appendix but no analysis of *what models actually say* when they conform. A small qualitative analysis of truth-override cases (how does the model justify its flip?) would add richness.

8. **Power analysis.** Report whether your cell sizes (350-400 items per cell for the 3 core conditions, potentially smaller for expanded conditions) give you adequate power for the effects you're detecting.

### Nice-to-have (if time allows)

9. **Cross-model replication.** Run even a subset of conditions on one other model family (Llama-3, Qwen-2.5) at a single temperature to show the asymmetry pattern isn't OLMo-specific.

10. **Mechanistic analysis.** Your codebase has probe/intervention infrastructure (empty tables in the DB suggest this was planned). Even a preliminary logit-lens or probe analysis showing *where* in the model the conformity decision happens would elevate this from measurement to understanding.

---

## 6. What Probably Won't Work

Being honest about dead ends:

- **"We have lots of data" is not a contribution.** 213K trials is impressive, but scale alone doesn't get you into top venues. The contribution must be in the findings, not the count.

- **The temperature story alone is too narrow** for a full paper. It's a supporting finding, not a central one.

- **Trying to claim causal effects from released checkpoints** will get rejected. You don't have access to intermediate training data or reward models. Stick to correlational language ("SFT is associated with...").

- **The opinion/social-conventions subset** is interesting but small (50 items) and lacks ground truth. Don't hang your main claims on it.

- **Mechanistic interpretability** without actually doing the probing/intervention work. Your DB has empty probe tables. Don't promise mechanistic insights you haven't computed.

---

## 7. Timeline Estimate

| Task | Effort | Priority |
|------|--------|----------|
| Judge validation study | 1-2 weeks | Blocker |
| Include Think-DPO in analysis | 2-3 days | Blocker |
| Statistical tests for expanded conditions | 1 week | Blocker |
| Rewrite paper narrative (A+E framing) | 2-3 weeks | High |
| Mixed-effects model | 1 week | High |
| Multiple-comparisons correction | 2-3 days | High |
| Qualitative case analysis | 3-5 days | Medium |
| Cross-model replication (if pursued) | 2-4 weeks | Nice-to-have |
| Mechanistic probing (if pursued) | 3-6 weeks | Nice-to-have |

**Realistic timeline to a submission-ready draft:** 6-8 weeks if focused on A+E framing with must-do items. 12-16 weeks if adding cross-model replication or mechanistic analysis.

---

## 8. Venue Recommendations

| Venue | Deadline (typical) | Fit | Notes |
|-------|-------------------|-----|-------|
| EMNLP 2026 | ~June 2026 | Strong | Best fit for empirical NLP work with alignment implications |
| ACL 2026 | ~Feb 2026 (passed) | Strong | If ARR submission cycle allows |
| NeurIPS 2026 | ~May 2026 | Medium | Needs mechanistic component or formal model to be competitive |
| NAACL 2026 | Check | Good | Backup if EMNLP timing doesn't work |
| ICLR 2027 | ~Oct 2026 | Medium | Needs more ML depth (probing, interventions) |

**My recommendation:** Target EMNLP 2026 with the A+E framing. It's the best match for an empirical study of LLM social behavior with practical alignment implications.

---

## 9. Summary

You have a large, well-designed dataset with several genuinely interesting patterns. The current paper undersells the data by focusing on 3 conditions when you have 12, excluding a usable checkpoint, and framing the findings descriptively. The most promising path combines the asymmetry inversion finding (training stages trade peer for authority vulnerability) with the mitigation failure finding (human-psychology fixes don't transfer), supported by the temperature stability result. This gives you three complementary findings that together make a contribution: conformity is training-shaped, mitigation-resistant, and more structurally complex than the scalar "sycophancy score" the field currently uses.

The main risk is the judge validation — if the 1B judge labels are noisy, the whole analysis is undermined. Address that first. Everything else is analysis and writing.
