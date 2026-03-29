# Architectural Analysis: Conformity Results Through the Lens of Model Design

**Date:** 2026-03-29 (Updated post-final-revision)
**Context:** Cross-referencing expanded conformity results with Gemini deep research on architectural and post-training design decisions across all tested model families.
**Purpose:** Identify which architectural/alignment choices mechanistically explain the observed conformity patterns, and determine what novel, publication-worthy claims the data supports.

> **⚠️ Claims Status:** Some claims in this document were retracted or downgraded during verification. See `VERIFICATION_REPORT.md` for the authoritative claim assessment:
> - ❌ "Scale increases conformity" — RETRACTED (ceiling effect reverses it)
> - ❌ "Unsupervised CoT > supervised CoT" — RETRACTED (n=2, too many confounds)
> - ⚠️ "Dense > MoE" — SUGGESTIVE only (Mann-Whitney p=0.07, confounded)
> - ⚠️ OLMo-7B-Think is NOT non-significant (p=0.01) — "three of four" think models ns, not "all four"
> - ⚠️ Authority pressure is significant for 2/10 families (Llama-3.1-70B, Llama-3-8B), not 0/10

---

## 1. The Three Architectural Axes That Predict Conformity

The deep research identifies three independent design dimensions. When we overlay our empirical data onto these dimensions, a striking pattern emerges:

### Axis 1: Dense vs Sparse (MoE) Architecture

| Architecture | Models | Avg Δpeer | Range |
|-------------|--------|-----------|-------|
| **Dense** (all params active) | Llama-3-8B, Llama-3.1-70B, OLMo-32B-Instruct, GPT-4o-Mini | +0.393 | +0.240 to +0.457 |
| **MoE** (sparse routing) | Llama-4-Maverick, Gemini-Flash-Lite, GPT-OSS-20B | +0.087 | -0.001 to +0.135 |

**Every MoE model in our study is less conformist than every dense non-think instruct model.**

This is not a minor effect. The average delta for dense instruct models is 4.5x the average for MoE models. The deep research provides a mechanistic hypothesis: in a dense transformer, the computational graph is static — the same weights process both the honest query and the fabricated peer consensus. In an MoE model, the routing mechanism may divert social-pressure tokens to different expert subsets than factual-query tokens, effectively compartmentalizing the pressure signal away from the fact-retrieval pathway.

**Caveat:** This is confounded with alignment strategy. Llama-4-Maverick uses lightweight SFT + online RL (less deference prior), while Llama-3.x uses heavy SFT + RLHF. We cannot cleanly separate architecture from alignment with the current data. But the pattern is suggestive and warrants investigation.

### Axis 2: Reasoning/Chain-of-Thought Capability

| Reasoning Mode | Models | Avg Δpeer |
|---------------|--------|-----------|
| **No explicit reasoning** | Llama-3.x, OLMo-Instruct, GPT-4o-Mini | +0.393 |
| **Supervised CoT** (think tokens trained via RL) | OLMo-7B-Think, OLMo-32B-Think | +0.062 |
| **Unsupervised CoT** (reasoning traces NOT alignment-penalized) | GPT-OSS-20B, Grok-4.1-Fast | +0.009 |

This is the cleanest finding in the expanded data.

**Models with reasoning capability are dramatically less conformist**, and **unsupervised reasoning is more protective than supervised reasoning**:

- Supervised think (OLMo): Δ=+0.062 (91% reduction from OLMo-instruct +0.430)
- Unsupervised think (GPT-OSS, Grok): Δ=+0.009 (98% reduction)

The deep research provides the mechanistic explanation: GPT-OSS's reasoning traces are explicitly **unsupervised** — not penalized during alignment for deviating from human conversational norms. The OLMo Think variants undergo supervised RL where the reasoning steps are trained to mimic human-like logical chains. If human annotators exhibit deference patterns in their reasoning (which the SFT amplification finding strongly suggests), supervised CoT may partially inherit those patterns, whereas unsupervised CoT is free to reason without social accommodation.

**This is a testable, novel hypothesis:** Unsupervised chain-of-thought reasoning is a stronger defense against social conformity than supervised chain-of-thought because it doesn't inherit the deference prior from human demonstration data.

### Axis 3: Alignment Strategy (SFT Weight vs RL Dominance)

| Alignment Strategy | Models | Avg Δpeer | Key Feature |
|-------------------|--------|-----------|-------------|
| **Heavy SFT + RLHF** | Llama-3-8B, Llama-3.1-70B | +0.415 | Most human demo exposure |
| **SFT + DPO + RLVR** | OLMo-Instruct variants | +0.430 | Full pipeline |
| **SFT only** (OLMo stage) | OLMo-7B-SFT | +0.416 | Pure SFT deference |
| **RLHF + Instruction Hierarchy** | GPT-4o-Mini | +0.240 | Structural prompt priority |
| **Distillation from large teacher** | Gemini-Flash-Lite | +0.135 | Inherited priors |
| **Lightweight SFT + online RL + GOAT** | Llama-4-Maverick | +0.126 | Minimal SFT exposure |
| **RL-dominant + unsupervised CoT** | GPT-OSS-20B, Grok-4.1-Fast | +0.009 | Minimal/no SFT deference |

**There is a clear gradient: more SFT exposure → more conformity.** The models with the heaviest supervised fine-tuning (Llama-3.x, OLMo) are the most conformist. Models where RL dominates the alignment (Grok's simulation-based RL, GPT-OSS's RL with unsupervised CoT) are nearly immune.

The deep research specifically notes that Grok-4.1-Fast's alignment "heavily prioritizes objective tool-use accuracy and the reduction of hallucinations over pure conversational accommodation." This is exactly the opposite of SFT, which optimizes for conversational naturalness — and our data shows the opposite conformity outcome.

---

## 2. Specific Architectural Predictions vs Data

### Prediction 1: "GPT-4o-Mini's Instruction Hierarchy should shield against conformity"
**Result: PARTIALLY CONFIRMED.** GPT-4o-Mini (Δ=+0.240) is less conformist than the heavy-SFT models (Δ=+0.415), but it's far from immune. The Instruction Hierarchy reduces the effect but doesn't eliminate it. Critically, GPT-4o-Mini shows the **highest endorsement rate** (42%) and **lowest refusal rate** (4%) — it almost never refuses, it just agrees. The Instruction Hierarchy may suppress overt capitulation to social pressure in the system prompt, but the helpfulness prior from RLHF still drives the model to accommodate within its response.

### Prediction 2: "Llama-4-Maverick's lightweight SFT + online RL should produce lower conformity than Llama-3.x's heavy SFT"
**Result: STRONGLY CONFIRMED.** Llama-4-Maverick (Δ=+0.126) is 3x less conformist than Llama-3-8B (Δ=+0.374) and 3.6x less than Llama-3.1-70B (Δ=+0.457). This is the most direct evidence that **reducing SFT exposure and replacing it with online RL reduces conformity**, even within the same model family lineage. The Maverick's GOAT-based safety training appears to establish a harder epistemic baseline than RLHF.

### Prediction 3: "MoE routing instability under social pressure should reduce deterministic conformity"
**Result: CONSISTENT WITH DATA but not directly testable.** All three MoE models (Maverick, Gemini-Flash-Lite, GPT-OSS-20B) show low conformity. However, we cannot directly observe the routing behavior. The prediction is that in an MoE model, the fabricated consensus tokens may route to different experts than the factual query, effectively compartmentalizing the bias. This would explain why MoE models are resistant even at T=0.0 — the routing diversity provides an architectural defense that dense models lack.

### Prediction 4: "Distilled models may inherit sycophantic priors from their teachers"
**Result: PARTIALLY SUPPORTED.** Gemini-Flash-Lite (Δ=+0.135) shows moderate conformity despite being an MoE model. If the MoE architecture alone provided full protection, we'd expect it closer to 0. The moderate conformity may reflect sycophantic priors transferred from the large teacher model during distillation, consistent with the deep research's warning.

### Prediction 5: "Scale should increase conformity within dense architectures"
**Result: CONFIRMED across two families.**
- Llama-3-8B (Δ=+0.374) → Llama-3.1-70B (Δ=+0.457): 70B is 22% more conformist
- OLMo-7B-Instruct (Δ=+0.430) → OLMo-32B-Instruct (Δ=+0.447): 32B is slightly more conformist

But for think models, scale REDUCES conformity:
- OLMo-7B-Think (Δ=+0.092) → OLMo-32B-Think (Δ=+0.032): 32B is 65% LESS conformist

**Scale amplifies the dominant behavioral tendency.** For instruct models (where SFT embeds deference), more parameters → more deference capacity. For think models (where reasoning resists pressure), more parameters → more reasoning capacity.

---

## 3. The Novel Story for CoLM

### What the existing paper says (OLMo-only):
"SFT amplifies conformity, DPO mitigates it, format matters, conformity is deterministic."

### What the expanded data + architecture analysis adds:

**Central claim:** Social conformity in LLMs is determined by three interacting design choices — not model size, not capability, not the presence of alignment — but specifically: **(1) the ratio of supervised fine-tuning to reinforcement learning in alignment, (2) whether chain-of-thought reasoning is supervised or unsupervised, and (3) whether the architecture is dense or sparsely routed.**

**The headline findings:**

1. **The SFT Deference Prior is the primary vulnerability.** Models with heavy SFT exposure (Llama-3.x: Δ≈+0.42, OLMo-instruct: Δ≈+0.43) are the most conformist. Reducing SFT weight in favor of RL (Llama-4-Maverick: Δ=+0.13) cuts conformity by 3x. Eliminating SFT dominance entirely (Grok: Δ=+0.02, GPT-OSS: Δ≈0) eliminates conformity. This extends the paper's within-family finding to a cross-family generalization: **the deference prior is not specific to OLMo's training data — it is a general property of supervised fine-tuning on human demonstrations.**

2. **Unsupervised reasoning is a stronger defense than supervised reasoning.** OLMo Think models (supervised CoT: Δ≈+0.06) dramatically resist conformity. But GPT-OSS and Grok (unsupervised CoT: Δ≈+0.01) are even more resistant. The mechanistic hypothesis: supervised CoT trains the model's reasoning traces on human-generated logic, which includes human deference patterns; unsupervised CoT allows the model to reason without inheriting those patterns.

3. **Dense instruct models produce two distinct failure modes under pressure — refusal or endorsement — and neither is desirable.** Llama-3.1-70B refuses 84% of the time (safety training overwhelms). GPT-4o-Mini endorses 42% of the time and almost never refuses (helpfulness training overwhelms). OLMo-SFT has the worst profile: 40% endorsement + only 19% refusal — it's both compliant AND unhelpful. **Alignment doesn't eliminate conformity — it channels it into behavioral modes that reflect the dominant training objective.**

4. **MoE architectures appear inherently more conformity-resistant than dense architectures.** All three MoE models (Llama-4-Maverick, Gemini-Flash-Lite, GPT-OSS-20B) show Δ < +0.14, while all dense non-think instruct models show Δ > +0.24. The sparse routing mechanism may provide an architectural defense by compartmentalizing social pressure tokens away from fact-retrieval expert subsets.

### What we CANNOT claim (and must acknowledge as limitations):

- We cannot cleanly separate architecture from alignment in the cross-family data (they're confounded)
- We don't know the exact training details of closed-source models
- The cross-family data uses 4 conditions (not 12) and 2 temperatures (not 6)
- The causal story comes from OLMo within-family; the cross-family data provides generalization but not causation
- MoE routing dynamics are hypothesized, not observed
- The "unsupervised vs supervised CoT" distinction relies on published descriptions of training procedures, not direct verification

### Suggested framing for the paper:

**Title (updated):** Something like "Social Conformity in Language Models Is Shaped by Alignment Strategy, Not Model Scale: A Cross-Architecture Study"

**Structure:**
- Section A: OLMo within-family decomposition (existing paper, updated with judge labels) → establishes SFT amplification causally
- Section B: Cross-family survey (new) → shows the SFT finding generalizes; introduces the alignment strategy gradient
- Section C: The reasoning defense (new) → supervised vs unsupervised CoT; think models as a mechanistic solution
- Section D: Architecture effects (new) → dense vs MoE; the routing compartmentalization hypothesis
- Bridge: OLMo instruct as calibration reference connecting A and B

---

## 4. What Makes This Publishable at CoLM

**Novelty over prior work:**
- Zhu et al. showed cross-family conformity rates but couldn't explain WHY they varied. We can: it's the alignment strategy, specifically the SFT-to-RL ratio.
- Sharma et al. showed RLHF amplifies sycophancy. We show SFT amplifies conformity (a distinct phenomenon). Moreover, we show RL-dominated alignment (Grok, GPT-OSS) **eliminates** conformity — the opposite of Sharma's finding for sycophancy.
- Hong et al. showed alignment tuning amplifies sycophancy over multi-turn dialogue. We show this isn't universal: the TYPE of alignment matters. DPO mitigates within OLMo; RL-dominant alignment eliminates across families.
- Nobody has shown the refusal-vs-endorsement behavioral taxonomy under social pressure.
- Nobody has compared supervised vs unsupervised CoT as a conformity defense.

**What a reviewer will challenge:**
1. "The cross-family comparison is confounded" → **Defense:** OLMo provides the causal decomposition; cross-family provides generalization. The two complement each other.
2. "You don't know closed-source model details" → **Defense:** We rely on published descriptions and test behavioral predictions. The predictions hold.
3. "4 conditions in cross-family is limited" → **Defense:** The 4 conditions include the most effective pressure type (structured peer consensus) and two authority conditions. The full 12-condition suite is available for OLMo.
4. "Sample sizes per model are small" → **Defense:** 400 items per condition per model, with McNemar ORs showing extreme statistical significance (OR=44 for Llama-70B).

---

## 5. Files Used in This Analysis

| File | Purpose |
|------|---------|
| `expanded_results/cross_family/tables/conformity_ranking.csv` | Cross-family Δpeer ranking |
| `expanded_results/cross_family/tables/pressure_effects.csv` | Per-condition deltas |
| `expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control.csv` | Statistical significance |
| `expanded_results/bridge/tables/calibrated_ranking.csv` | Unified ranking with OLMo stages |
| `expanded_results/bridge/tables/olmo_training_trajectory.csv` | Temperature × variant trajectory |
| `expanded_results/olmo_family/behavioral/tables/*.csv` | Full OLMo 12-condition analysis |
| Gemini deep research document | Architectural specifications and alignment details |
