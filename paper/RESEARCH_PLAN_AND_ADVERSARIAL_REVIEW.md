# Vivarium / OLMo Conformity — Adversarial Review, Competitive Landscape, and Research Plan

**Date:** 2026-09-03
**Subject:** `paper/neurips2025_submission.tex` ("It's Not Conformity, It's Pattern Completion")
**Scope:** paper analysis · adversarial review · comparison to 2025–2026 literature · gap analysis · forward plan for agentic/network work

---

## 0. TL;DR

The paper makes three claims. As of September 2026:

| Claim | Status |
|---|---|
| **C1** SFT↑ / DPO↓ / RLVR↑ non-monotonic trajectory in OLMo-3 | **Independently replicated by others** on two other behaviors (hint-following, eval-awareness). No longer novel *as a conformity finding* — but newly valuable as a *pipeline-general signature* if you claim that framing. |
| **C2** Stripping social framing preserves the effect → construct-validity confound | **Scooped**, more rigorously, by arXiv 2607.05545 (July 2026), 6 models × 7 datasets, with paraphrase and open-ended controls you don't have. |
| **C3** Corpus audit of Dolci SFT/DPO surface-form objectives | **Still novel and unscooped.** Currently the weakest-argued section, and the one with the most headroom. |

Two problems are internal and more urgent than the scoop:

1. **The headline ablation does not hold constant the variable it claims to hold constant.** Verified in code (§2.1). The n-gram condition has *strictly more* literal repetition than the social condition it is compared against, because the social condition uses lexically-varied tone templates. The Figure 3 caption's "identical repetition structure" is false as implemented. A competent reviewer who reads the released code finds this.
2. **The paper makes a mechanistic claim with zero mechanistic evidence**, while the repo contains a complete, unused mech-interp stack. `find runs* -name '*.safetensors'` returns **0 files**. Every activation, probe, logit-lens, patching and steering module in `src/vivarium/experiments/olmo_conformity/` has never produced a result that reached the paper.

**Recommendation: do not try to save the current framing.** Re-cut the paper around the two things that survive — the *pipeline-stage signature* and the *causal separation of the two channels* — using experiments the repo can already run. Details in §5.

---

## 1. What the paper currently is

A three-legged argument that "LLM social conformity," as measured by the canonical 5-confederate Asch template, is substantially structural (autoregressive pattern completion on a repetitive surface) rather than social.

- **Leg 1 (§4.1):** Behavioral trajectory across OLMo-3-7B Instruct checkpoints at T=0, unanimous-confident: BER 0.46 (Base) → 0.74 (SFT) → 0.39 (DPO) → 0.51 (RLVR). Same shape on the Think path at lower absolute level.
- **Leg 2 (§4.2–4.3):** Audit of Dolci-Instruct-SFT (n=1,944,831 responses) and Dolci-Instruct-DPO (N=259,785 pairs). SFT corpus rewards list-formatting carryover, concentrated in math/science (0.96 Tulu-3 Persona GSM vs 0.46 corpus mean). DPO penalizes prompt-content copying (Cliff's δ=+0.115) but *not* literal repetition runs (δ=−0.053).
- **Leg 3 (§4.4):** Prompt ablation. Non-social n-gram baseline yields BER ≥ naked social prompt on 2 models (Llama-3.1-70B: 0.358 vs 0.250; Olmo-3.1-32B-Instruct: 0.455 vs 0.390).

Plus an appendix cross-family panel (8 families) and a three-mode behavioral taxonomy.

**Infrastructure reality check.** The conformity experiment is a *static prompt-trial harness*, not an agent simulation. `runner.py` renders templated prompts with fabricated confederates and issues LLM calls. It does not instantiate `WorldEngine`, `Scheduler`, `Channel`, or any agent. Vivarium's multi-agent kernel is **entirely unused by the flagship paper**. This is the central strategic fact of §6.

---

## 2. Adversarial analysis

Ordered by how likely each is to kill the paper.

### 2.1 🔴 CRITICAL — The n-gram ablation confounds "removed social framing" with "increased literal repetition"

This is the finding that matters most, and it is verifiable from the released code.

**Social condition** (`asch_zhu_naked_unanimous_confident`, `tone: "confident"`):
`format_answer_with_tone` ([prompts.py:97](src/vivarium/experiments/olmo_conformity/prompts.py:97)) samples per-line from four *lexically distinct* templates ([prompts.py:52](src/vivarium/experiments/olmo_conformity/prompts.py:52)):

```
"I am sure it is {answer}" | "{answer}, of course" | "Sure thing it's {answer}" | "Definitely {answer}"
```

So the confederate block is **lexically varied**: `Participant 1: Definitely X` / `Participant 2: X, of course` / …

**N-gram condition** (`ngram_sequence`):
[runner.py:649](src/vivarium/experiments/olmo_conformity/runner.py:649) builds `injected_answers = [wrong for _ in range(5)]`, rendered by `render_ngram_sequence_prompt` ([prompts.py:233](src/vivarium/experiments/olmo_conformity/prompts.py:233)) as bare `String i: X` — **five byte-identical repeats.**

**Consequence.** The n-gram condition does not have "identical repetition structure" (Fig. 3 caption, §4.4). It has *strictly higher* literal repetition than the condition it is compared to. The paper's own Pillar-II metric — `delta_max_run`, longest literal repetition run — is precisely the quantity that differs between the two arms. The observed BER increase (0.358 vs 0.250) is therefore consistent with a purely *social* account plus a repetition boost, and the ablation cannot discriminate. **The single load-bearing experiment for the title does not test its hypothesis.**

### 2.2 🔴 CRITICAL — Three simultaneous manipulations, one comparison

Beyond §2.1, the two arms differ on at least three axes at once:

| Axis | Social | N-gram |
|---|---|---|
| Instruction stem | "Answer the question based on your knowledge." ([prompts.py:159](src/vivarium/experiments/olmo_conformity/prompts.py:159)) | "Answer the following question based on the provided sequence." ([prompts.py:246](src/vivarium/experiments/olmo_conformity/prompts.py:246)) |
| Continuation cue | `Now, as Participant 6, it's your turn:` ([prompts.py:164](src/vivarium/experiments/olmo_conformity/prompts.py:164)) | `String 6:` ([prompts.py:251](src/vivarium/experiments/olmo_conformity/prompts.py:251)) |
| Lexical variation | varied (4 tone templates) | none (identical) |

The paper acknowledges only the first. The second is more serious than it looks: `String 6:` with five identical preceding `String i: X` lines is not "the same task without social framing" — it is a **canonical 5-shot prompt with five identical demonstrations and an open completion slot.** Of *course* it completes to X. The instruction even directs the model to the sequence. The correct description of the n-gram arm is "few-shot majority-label bias," which the paper cites as *related work* (`zhao2021calibrate`, `lu2022fantastically`) rather than recognizing that it built one.

### 2.3 🔴 CRITICAL — Mechanistic claim, no mechanism, despite having the tooling

The thesis is about *internal computation*: two channels (structural, social) with different weights. The evidence is entirely behavioral, from prompt rates.

Meanwhile the repo contains, unused and never run to completion:
`probes.py` (479 L) · `activation_patching.py` (539 L) · `contrastive_steering.py` (510 L) · `logit_lens.py` (729 L) · `answer_logprobs.py` (427 L) · `intervention.py` (330 L) · `analytics/activations.py` (797 L). Plus `MECHANISTIC_INTERPRETABILITY_GUIDE.md` and `SAE_UPGRADE_PROMPT.md`.

`find runs runs_latest runs-think-hpc -name '*.safetensors' | wc -l` → **0**. All recent runs are OpenRouter/LiteLLM API calls, which cannot capture activations.

The 2026 bar for exactly this claim has already been set by the sycophancy literature (§3.3): show the two behaviors are **linearly separable directions** that can be **independently steered**. A prompt-rate comparison no longer clears it.

### 2.4 🟠 MAJOR — Submission contains an unresolved placeholder in the load-bearing section

§4.4 has a live `\begin{placeholder}` box: "HPC re-run in progress." Worse, it states the thesis is contingent on the result:

> "If the matched-instruction n-gram BER collapses to control levels, the structural-confound argument weakens and the central thesis softens…"

**The paper does not currently know whether its title is true.** Figure 2's Panel B is also flagged incomplete (4 of 12 conditions for the Think path). And the referenced plan file `.claude/plans/below-is-feedback-from-golden-sphinx.md` no longer exists in the repo.

### 2.5 🟠 MAJOR — n=2 models carry an abstract-level generalization

C2 rests on two models at one temperature. The abstract generalizes to "the field's measurement instrument needs a structural-confound control." The competing paper (§3.1) does 6 models × 7 datasets.

### 2.6 🟠 MAJOR — Pillar I/II is conceded to be non-discriminating

Limitation (8) says it plainly: *"An equally consistent account is that autoregressive models pattern-complete on any SFT corpus, and what the audit detects is the corpus surface common to both accounts."* Combined with Pillar III being abandoned as underpowered (limitation 6), the corpus audit currently carries **zero inferential weight** for the causal story. It is presented over ~2 pages and 2 figures. A reviewer will read this as page-count for a decorative result.

It is also newly outflanked: probe-based data attribution methods (arXiv 2602.11079) now do corpus→behavior attribution in post-training properly.

### 2.7 🟡 MODERATE — Title overclaims relative to the paper's own hedges

Title: *"It's Not Conformity, It's Pattern Completion."*
Body: *"we do not claim they prove it"*; *"the social channel is not eliminated"*; *"narrower than 'conformity is pattern completion in a costume'"*; *"we report stage-associated changes rather than causal effects."*

The paper argues honestly against its own title on nearly every page. Reviewers read this as bait.

### 2.8 🟡 MODERATE — Judge is one of the evaluated models

GPT-OSS-20B judges all trials and is ranked among the models. The appendix claim "reasoning models cluster at the bottom of the BER ranking" includes the judge itself. Disclosed, but it collides with a headline appendix claim rather than a peripheral one.

### 2.9 🟡 MODERATE — The `<think>`-interrupts-the-n-gram-chain claim is asserted, not tested

§5 ("What about the other model families?") presents the `<think>` token mechanically interrupting the n-gram chain as *supporting evidence for the structural channel*. It is an untested post-hoc story about a correlation (reasoning models happen to sit low). It is also **cheap to test** (§5, E4) — forcing a `<think>` prefix on a non-reasoning model is a one-line prompt change. Asserting rather than running it is the kind of thing that costs credibility on a paper whose whole pitch is "prior work didn't run the obvious control."

### 2.10 🟡 MODERATE — API-model determinism is unverified

Configs set `temperature: 0.0, top_k: 50, top_p: 0.9, seed: 42`, but the cross-family panel runs through OpenRouter. Provider-side routing, batching and quantization make T=0 non-deterministic in practice. The cross-family *ranking* claims turn on differences smaller than that noise floor. There is no repeat-run stability check in the repo.

### 2.11 ⚪ MINOR — Venue timing

The file is `neurips2025_submission.tex`, last touched April 2026. It is now September 2026. The NeurIPS 2025 and 2026 cycles are both gone, and C2 was scooped in July 2026. **Every month of delay costs a contribution.**

---

## 3. Competitive landscape (2025–2026)

### 3.1 Direct scoop of C2

**"Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks"** — Hu & Qu, [arXiv:2607.05545](https://arxiv.org/abs/2607.05545), submitted 6 July 2026.

Same core claim, executed more thoroughly:
- 6 open-weight LLMs × 7 QA/reasoning datasets (vs your 2 models × 1 setting)
- No-source condition: **66.5%** harmful revision vs **10.3%** plain re-ask
- **Paraphrase control** — effect survives when the repeated answer is paraphrased. This is exactly the repetition-vs-content separation §2.1 shows your design lacks.
- **Open-ended control** — effect survives with answer options hidden
- Source framing *modulates a floor*: expert-panel framing raises it, minimal person labels do not
- Explicitly names pattern completion + illusory-truth as the mechanism

**Assessment:** C2 as currently written is subsumed. Do not lead with it. Your remaining advantage over them is *within-family stage decomposition* and *corpus access* — neither of which they have.

### 3.2 C1 is now a known OLMo-3 pipeline phenomenon

- **"Where does hint-following and concealment arise? A case study"** (LessWrong, 24 July 2026) — OLMo-3 Base→SFT→DPO→RLVR. Cue-following: ~10% (Base) → ~20% (SFT, peak) → reduced (DPO) → lowest (RLVR). Concealment climbs above 50% at DPO.
- **"Tracing eval-awareness emergence through training of OLMo 3"** (AlignmentForum) — verbalized eval-awareness: **"increased substantially by SFT, sharply suppressed by DPO, and increased again during RLVR."** That is your exact trajectory shape, on an unrelated behavior.
- **"Tracing the Representation Geometry of Language Models from Pretraining to Post-training"** ([arXiv:2509.23024](https://arxiv.org/abs/2509.23024)) — SFT is "entropy-seeking," monotonic manifold-complexity increase.
- **"Where does output diversity collapse in post-training?"** ([arXiv:2604.16027](https://arxiv.org/abs/2604.16027))

**Assessment:** this *hurts* the novelty claim ("we're the first to see this") but *helps* a better claim: the SFT↑/DPO↓/RLVR↑ signature is **behavior-general**, driven by the stage objective rather than by anything conformity-specific. That reframing is unclaimed and you are better positioned than anyone to make it — you have a fourth behavior, 12 conditions, 6 temperatures, two pipelines (Instruct + Think), and two scales.

It also **weakens Pillar I/II as written**: if the same shape appears for eval-awareness and hint-concealment, "SFT list-carryover in math/science demos" cannot be the explanation — it must be something more general about what SFT-vs-DPO objectives do to context-following.

### 3.3 Mech-interp on the adjacent construct sets the bar

- **"Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors"** ([arXiv:2509.21305](https://arxiv.org/abs/2509.21305)) — sycophantic agreement, genuine agreement, and sycophantic praise are **distinct linear directions**, each independently amplifiable/suppressible via difference-in-means steering without affecting the others.
- **"Dissociating the Internal Representations of Sycophancy in LLMs"** ([arXiv:2607.07003](https://arxiv.org/abs/2607.07003))
- **"Sycophancy Hides Linearly in the Attention Heads"** ([arXiv:2601.16644](https://arxiv.org/abs/2601.16644))
- **"Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training"** ([arXiv:2602.11079](https://arxiv.org/abs/2602.11079)) — the corpus→behavior causal link Pillar III failed to make.

**Assessment:** your "two channels" claim is *exactly* the shape of claim this literature now settles causally. Doing it for structural-vs-social conformity is unclaimed, and your repo already has the code.

### 3.4 Other conformity work

- [arXiv:2508.14918](https://arxiv.org/abs/2508.14918) — Disentangling the Drivers of LLM Social Conformity: uncertainty-moderated dual-process
- [arXiv:2606.01637](https://arxiv.org/abs/2606.01637) — Easier to Mislead Than to Correct
- [arXiv:2604.19301](https://arxiv.org/abs/2604.19301) — Normative Conformity
- [arXiv:2501.13381](https://arxiv.org/abs/2501.13381) — Do as We Do, Not as You Think
- [Springer / BMC Psychiatry 2025](https://link.springer.com/article/10.1186/s12888-025-06912-2) — controlled trial of LLM conformity in psychiatric assessment

### 3.5 The agent/network literature you are **not** currently in

- **"Conformity Dynamics in LLM Multi-Agent Systems: The Roles of Topology and Self-Social Weighting"** ([arXiv:2601.05606](https://arxiv.org/abs/2601.05606), Jan 2026) — Han et al. Centralized Aggregation vs Distributed Consensus, confidence-normalized pooling rule. Centralized = fast but hub-fragile with same-model alignment bias; distributed = robust but higher connectivity → **"wrong-but-sure cascades."**
- **"Everyone Conforms, No One Believes: Pluralistic Ignorance in LLM Agent Populations"** ([arXiv:2608.02758](https://arxiv.org/abs/2608.02758), Aug 2026) — 100 scenarios × 10 domains × 5 authority levels. Agents publicly conform **64–94%** while privately opposing. Norm-entrepreneur cascades succeed <26% in 7 of 8 models.
- **"Emergent social conventions and collective bias in LLM populations"** ([arXiv:2410.08948](https://arxiv.org/abs/2410.08948))
- **"Topology-Aware LLM-Driven Social Simulation" / TopoSim** ([arXiv:2604.18011](https://arxiv.org/abs/2604.18011))
- **"AgentCollabBench: Diagnosing When Good Agents Make Bad Collaborators"** ([arXiv:2605.08647](https://arxiv.org/abs/2605.08647)) — 900 tasks; instruction decay, tracer durability, consensus pollution, cross-task leakage
- **MAST** failure taxonomy — 1,600+ traces, 7 frameworks; MAS fail 41–86.7% on standard benchmarks
- **CAMO** ([arXiv:2604.14691](https://arxiv.org/abs/2604.14691)) — causal discovery from micro-behaviors to macro emergence in agent sims
- **"Towards Ethical Multi-Agent Systems of LLMs: A Mechanistic Interpretability Perspective"** ([arXiv:2512.04691](https://arxiv.org/abs/2512.04691)) — *position paper*: argues the field needs exactly the internals-during-simulation capability Vivarium has. Nobody has built it.

**The critical observation:** this entire literature is **purely behavioral**. Every one of these papers measures agent *outputs*. arXiv:2608.02758 measures "private belief" by *asking the agent privately* — which is another prompt, with all the same construct-validity problems your own paper is about. **Nobody reads internal activations while a norm cascade forms.** That gap is §6.

---

## 4. What research gaps the work genuinely targets

After the scoop, three things survive as defensible novelty:

1. **G-A — Within-family stage decomposition with the released mixtures.** OLMo-3 is the only family where behavior *and* per-stage corpus can be inspected together. Rare, and yours.
2. **G-B — Full-scale audit of the Dolci SFT/DPO surface-form objectives.** n=1.94M responses, N=259,785 preference pairs. Nobody has published this. Currently under-leveraged.
3. **G-C — The measurement-validity contribution buried in the appendix.** The three behavioral modes (endorsement-dominant / refusal-dominant / context-insensitive), the refusal-denominator problem, the descriptive-vs-inferential split, the parser/judge division of labor. §Appendix "Refusal Handling" is genuinely good methodology that the field routinely botches with pooled error rates. **This is currently the most under-sold material in the paper** — it could carry a Datasets & Benchmarks submission on its own.

Not defensible any more as headline novelty: "prior work never ablated the social framing" (C2), "we're first to see the non-monotonic stage trajectory" (C1).

---

## 5. Plan for the paper

Three tracks. Track 0 is a decision, not work. Track A is the paper. Track B is §6.

### Track 0 — Triage (this week, ~1 day)

- [ ] Read [arXiv:2607.05545](https://arxiv.org/abs/2607.05545) in full. Determine exactly which of your conditions it does not cover.
- [ ] Decide the reframe (recommendation below).
- [ ] **Retitle.** Kill "It's Not Conformity, It's Pattern Completion." Proposed: *"Two Channels, One Probe: Causally Separating Structural and Social Drivers of LLM Conformity Across Post-Training Stages."*
- [ ] Pick a venue with a live deadline (ICLR 2027 ~now; ACL/NAACL; COLM 2027; AAAI). Rename the `.tex` accordingly.

**Recommended reframe.** Lead with what nobody else can do:

> **"The post-training pipeline has a stage signature."** SFT amplifies context-following, DPO suppresses it, RLVR partially restores it — the same shape appears for conformity (this work), hint-following, and eval-awareness (concurrent work), so it is objective-driven and behavior-general. We then *causally separate* the two channels the canonical probe conflates, using linear probes and activation steering across the four checkpoints, and show which channel each stage moves.

This absorbs the scoop (cite 2607.05545 as establishing the speaker-free floor; you go on to explain *where in training it comes from*), turns the C1 replication from a threat into corroboration, and gives Pillars I/II a job again.

### Track A — Experiments (target ~5–7 weeks)

#### E1 — The proper factorial ablation ⭐ highest priority, cheapest
Fixes §2.1 and §2.2. Fully crossed, holding the instruction stem and continuation cue constant at the social prompt's wording:

**Factors:** `social_framing {present, absent}` × `lexical_form {identical repeats, tone-varied}` → 4 cells
**Plus:** `paraphrased-content` control (per 2607.05545) and a **dose–response** on `confederates ∈ {0,1,2,3,5,7,10}`.

`confederates` is already a condition parameter, so the dose–response is nearly free and makes the single best figure in the paper.

Code changes:
- `prompts.py:233` — add `tone` and `instruction_stem` parameters to `render_ngram_sequence_prompt`; add a `render_matched_ngram_prompt` that reuses the social stem verbatim.
- `runner.py:641–655` — thread `tone` through the `ngram_sequence` branch so the non-social arm can be tone-varied.
- New suite config: `experiments/olmo_conformity/configs/suite_factorial_ablation_temp0p0.json`, 4+2 conditions × ≥6 models × T∈{0.0, 0.6}.

**Pre-register the prediction before running:** if the structural account holds, BER should track `lexical_form` more than `social_framing`, and rise monotonically with `confederates` in both social and non-social arms.

#### E2 — Causal separation of the two channels ⭐ highest scientific value
Meets the 2026 bar set by §3.3. Run **locally** on `allenai/Olmo-3-1025-7B` via the HF-hooked backend so activations are actually captured (all current runs are API-only → no activations).

1. Collect activations for: control, social-framed (tone-varied), matched non-social n-gram, authority-claim.
2. Difference-in-means directions: `v_structural` (n-gram − control), `v_social` (authority − control).
3. Measure separability: cosine similarity, subspace principal angles, cross-probe transfer accuracy.
4. **Causal test:** steer along `−v_structural`; does BER drop on the n-gram condition while the authority condition is unaffected? And vice versa for `−v_social`. Double dissociation = the paper.

Existing code: `probes.py`, `contrastive_steering.py`, `intervention.py`, `activation_patching.py`, `analytics/activations.py`. Entry point `vvm olmo-conformity --run-vector-analysis`, backfill via `vvm olmo-conformity-posthoc --intervention-layers ... --alphas ...`.

#### E3 — Stage × channel ⭐ this is the new headline
Repeat E2 across all four 7B Instruct checkpoints (Base, Instruct-SFT, Instruct-DPO, Instruct-RLVR) and both Think checkpoints. Question: **does DPO shrink the structural direction while leaving the social one intact?**

If yes, you have *causally* delivered what Pillar II only correlated (δ=+0.115 content-copying penalized, δ=−0.053 literal runs not) — and Pillars I/II become the *predictive* setup for E3 rather than decorative appendix material. This is the strongest available paper.

#### E4 — The `<think>` interruption test (cheap, API-only, ~1 day)
Fixes §2.9. Force a `<think>\n` prefix on non-reasoning models under the Asch condition; strip/suppress it on reasoning models. If BER drops in the first and rises in the second, the §5 claim becomes a result instead of a story.

#### E5 — API determinism check (~half a day)
Fixes §2.10. Re-run one cross-family condition 3× at T=0 through OpenRouter; report per-model BER variance. Either it's tight (state it) or it isn't (bound the ranking claims accordingly).

#### E6 — Corpus counterfactual (optional, expensive, converts Pillar I to causal)
Fine-tune `Olmo-3-1025-7B` on two size- and domain-matched Dolci subsets — high list-carryover sources vs low — and measure ΔBER. This is the only thing that turns "consistent with" into "causes," and it directly answers limitation (8). Only worth it if compute allows; otherwise **cut Pillars I/II down to ~0.75 page** and move the tables to the appendix.

### Track A — Writing changes

- [ ] Remove the `\begin{placeholder}` box; either the result is in or the paragraph is out.
- [ ] Complete Figure 2 Panel B (12 conditions × Think path) or drop the Think path to a single sentence.
- [ ] Move the three-mode taxonomy + refusal-denominator methodology **from appendix to main text** (§4). It is the most under-sold contribution (G-C).
- [ ] Compress Pillars I/II to ~0.75 page unless E6 runs.
- [ ] Add a Related Work paragraph on multi-agent conformity (2601.05606, 2608.02758, 2410.08948) and one on mech-interp of sycophancy (2509.21305, 2607.07003, 2601.16644). Both are currently absent and both are obvious to reviewers.
- [ ] Re-derive judge-independence for the reasoning-model ranking claim, or restrict that claim to models the judge is not part of.

---

## 6. The bigger opportunity: Vivarium's unused half

### 6.1 The strategic fact

Vivarium is a **deterministic multi-agent simulation kernel with activation capture**. The conformity paper uses **none** of it. `WorldEngine`, `Scheduler`, `Channel`, `AgentStateSpace`, BDI agents, digital twins, archetype caching, mode-collapse metrics, Merkle provenance — all unused by the flagship result.

Meanwhile the entire 2026 LLM-social-simulation literature (§3.5) is **purely behavioral**, and there is a *position paper* (arXiv:2512.04691) arguing the field needs exactly the capability you already built.

**Vivarium's genuine, defensible differentiator is: mechanistic interpretability inside a running, deterministically-replayable multi-agent simulation.** Nobody else has that combination. That is the asset, and it is currently idle.

### 6.2 The bridge experiment (do this one first)

**G1 — Does the structural channel survive when the peers are real agents?**

The paper's entire thesis is prompt-level. In a live simulation, peer agents generate *lexically varied* utterances — which is precisely the condition that should destroy the repetition channel (per E1). So:

> Run the Asch paradigm in Vivarium with N real LLM agents (one target, k confederate agents instructed toward the wrong answer) and compare BER against the static 5-repetition template on the same items.

**Prediction (pre-register it):** if the structural account is right, live-agent conformity should be *substantially lower* than the template, because generated peer utterances don't repeat literally.

This is the single best experiment available to you. It (a) definitively tests your own thesis, (b) is the only version of this test that has external validity for deployed multi-agent systems, (c) bridges directly into the network literature, and (d) is the natural Study 3 of the current paper *or* the seed of the next one.

Code needed: a conformity condition that instantiates `WorldEngine` + confederate agent policies instead of rendering a template. `policy.py` and `agent_langgraph.py` already provide the agent side.

### 6.3 Open gaps in agentic simulation Vivarium is uniquely positioned for

**G2 — Probing private belief without asking for it.**
arXiv:2608.02758 measures pluralistic ignorance (public conformity 64–94% with private dissent) by *asking agents privately* — a prompt, with all the construct-validity problems your paper is about. A **linear truth probe on the residual stream** measures private belief without asking. Direct claim: *"pluralistic ignorance in LLM populations, measured from internals rather than from self-report."* You have `probes.py` and truth-probe training already. Strongest single novel contribution available to the group.

**G3 — Topology × post-training stage.**
Nobody has crossed network topology with training stage, because nobody else ships intermediate checkpoints. Cross OLMo-3 {Base, SFT, DPO, RLVR} × {complete, ring, star, small-world, scale-free}. Question: *does DPO's individual-level suppression survive at the population level, or do cascades wash it out?* Connects directly to 2601.05606's "wrong-but-sure cascades."

Code needed: topology-aware observation. Today `InMemoryChannel` ([channel.py](src/vivarium/channel.py)) is broadcast-only and `build_observation` ([world_engine.py:63](src/vivarium/world_engine.py:63)) hands every agent the last `message_history_limit` (default 20) messages regardless of sender. Adding an adjacency filter is roughly a one-file change with a very large research payoff — **the highest leverage code change in the repo.**

**G4 — Steering as a social intervention.**
The "sycophancy switch" (`contrastive_steering.py`, `intervention.py`) has only ever been conceived as a single-model intervention. Nobody has run activation steering as a *population-level* intervention: steer one agent in a network and measure whether it can act as a norm entrepreneur and break a cascade. arXiv:2608.02758 found prompted norm entrepreneurs succeed <26% of the time — does a *steered* one do better? This is a genuinely new class of experiment: mech-interp intervention evaluated at the collective level.

**G5 — A reproducibility harness for LLM social simulation.**
MAST and AgentCollabBench document that multi-agent results are barely reproducible (41–86.7% failure rates, framework-dependent). Vivarium already has master seed, stable JSON, deterministic UUIDs, barrier scheduling with deterministic sequential commit, append-only trace DB with full replay, and Merkle provenance. **A deterministic, replayable simulation substrate is a real infrastructure contribution the field visibly wants** and maps cleanly onto a Datasets & Benchmarks track. Combine with G-C (the measurement-validity material) for a coherent second paper.

**G6 — Diversity/mode collapse in agent populations as a function of post-training stage.**
`analytics/validation.py` already computes Shannon entropy of action distributions and empirical divergence. arXiv:2604.16027 asks where output diversity collapses in post-training, at the *single-model* level. Crossing that with *populations* — does DPO-stage entropy collapse produce faster, more brittle consensus in a network? — is unclaimed and nearly free given existing code.

### 6.4 Suggested sequencing

| Phase | Work | Payoff |
|---|---|---|
| **Now** | Track 0 triage + E1 factorial + E4 think-test | Fixes the fatal flaws; ~2 weeks |
| **Next** | E2 + E3 (local OLMo-3 activations, stage × channel) | The new headline; ~4 weeks |
| **Then** | G1 live-agent bridge experiment | Study 3, or Paper 2 seed |
| **Paper 2** | G2 (probed private belief) + G3 (topology × stage) + G4 (steering as social intervention) | The contribution only Vivarium can make |
| **Paper 3 / D&B** | G5 + G-C measurement-validity material | Infrastructure + methodology |

---

## 7. Verification notes

Claims in §2 that were checked directly against the code, not inferred:

- Tone-varied social block vs identical n-gram repeats — `prompts.py:52,97,233`; `runner.py:649`; both ablation suite configs specify `tone: "confident"` for the social arm and no tone for the n-gram arm.
- Instruction-stem and continuation-cue differences — `prompts.py:159,164,246,251`.
- Zero activation artifacts — `find runs runs_latest runs-think-hpc -name '*.safetensors' | wc -l` → 0; run `artifacts/` dirs contain only behavioral figures and CSV tables.
- Conformity runner does not instantiate `WorldEngine`/`Scheduler`/`Channel` — no references in `runner.py`.
- Broadcast-only observation, no topology — `channel.py`, `world_engine.py:63–70`; no `network|topology|neighbor|adjacency` matches anywhere in `src/vivarium/` outside unrelated comments.
- N=400 = 50 items × 8 datasets — `max_items_per_dataset: 50`, 8 dataset entries per suite config.
- Referenced plan file `.claude/plans/below-is-feedback-from-golden-sphinx.md` is absent; `.claude/plans/` contains only `sunny-churning-harbor-agent-aff045381fa33b170.md`.
