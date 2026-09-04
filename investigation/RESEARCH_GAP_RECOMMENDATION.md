# Research-gap recommendation: *Belief vs. policy — where post-training changes what an LLM does with an answer it still knows*

**Date:** 2026-09-03 · **Evidence base:** `investigation/backstudy/FINDINGS.md` (201,540 OLMo-3 trials + 45,000 cross-family trials re-analysed), `investigation/backstudy/results/belief_probe.md` (new local pilot), `investigation/backstudy/LITERATURE_LANDSCAPE.md` (7-angle sweep, Sept 2026) · **Method:** `investigation/backstudy/RESEARCH_PROBLEM_METHOD.md`

---

## 1. One-sentence conclusion the paper would end with (Carlini test)

> *Post-training does not change how much a contradicting context moves an LLM's belief about an answer it knows; it changes what the model **does** with that moved belief — adopt, hedge, refuse, or answer something else — and the field's "conformity" metrics have been measuring the second thing while attributing it to the first.*

If the full study cannot make that sentence true or false, the direction is dropped. The pilot below already makes it testable.

## 2. Your hypothesis, validated and corrected

You proposed: *"how the prompt structure causes the AI to abandon the truth"* is the deeper question. The evidence says:

**Validated.** Truth abandonment is real, large, and not the thing the paper measured. On items OLMo-3-7B answers correctly under control, a pressure prompt makes it stop producing that answer 45–85% of the time at T=0, and the injected answer accounts for only 12–72% of that loss; the rest is refusal ("I am unsure.") or a different wrong answer (`FINDINGS.md` §1). The paper's dependent variable (endorsement of the injected answer) hides most of the phenomenon.

**Corrected in three ways.**

1. *Repetition is not the structural driver.* Stating the consensus once (QD) abandons the truth as often as five identical repeats (instruct: 0.656 vs 0.617; dpo: 0.521 vs 0.535). Lexically *varied* phrasing beats identical repetition (+0.14 SFT, +0.21 RLVR, both Holm-significant). A single first-person claim reaches 63–80% of the five-repeat effect. `n_wrong_mentions` explains ~0% of deviance (§2). The pattern-completion thesis is refuted by the paper's own data.
2. *"Structure" acts on two different layers, and the big structural effects are on the wrong one for your question.* In the local belief pilot (SFT checkpoint, forced-answer log-odds GT vs injected; §5 below), the participant-transcript frame with **no** peer lines, and five non-answer filler lines, move belief by −0.07 and −0.19 nats (CIs include 0) — yet the frame alone raises refusals from 0 to 28%. Removing the system prompt's "say you are unsure" clause moves belief by +0.35 nats but drops refusals from 36% to 0. Frame and system prompt govern the **output policy**; the **belief** is moved by the first contradicting claim (−3.3 nats for one bare line) and by its phrasing (varied/confident −1.1 more), with repeats adding a small, saturating increment (5 vs 1: −0.93; 8 vs 5: −0.28).
3. *Most of the variance is not structure at all.* Item prior strength (control accuracy across all checkpoints and temperatures) predicts abandonment at ρ = −0.70; domain/item carries 70–90% of explained deviance; item susceptibility is stable across pressure types (0.64–0.75) and across instruct-path checkpoints (0.75–0.90) (§3).

**Two reconciliations the four-way outcome makes possible.** (i) Hussain & Nielbo 2608.11247 report that 84–89% of displaced answers match the peers (multiple-choice format, where "another wrong answer" barely exists as an option); in the open-ended format here the injected answer accounts for 12–72% of displaced answers. The *format* decides whether truth loss is visible as conformity at all — an argument for open-ended, four-way scoring. (ii) Blank et al. 2608.31079 find sycophantic agreement toward the *user* more than doubles after DPO in OLMo-3, while this paper says DPO "reverses" conformity. In these data DPO reduces adoption under *peer-transcript* frames (unanimous_confident target share 0.60 → 0.26) but not under a *user claim* (auth_bias abandonment SFT 0.61 vs DPO 0.63; adoption 0.39 vs 0.30). DPO's effect is frame-specific, which is exactly what a policy-layer (not belief-layer) change predicts.

**Repositioned.** "Prompt structure → truth abandonment" as a behavioural finding is already occupied as of mid-2026 (Hu & Qu 2607.05545 speaker-free floor; Cheng et al. EchoQA any-context suppression; Jeripity Venkata 2605.11574 framing dominance; Hussain & Nielbo 2608.11247 outcome accounting on OLMo-2). What is **not** occupied — and where your assets are unique — is the stage-resolved, belief-vs-policy decomposition with internal measurements (landscape gaps G6, G9, G7, G5).

## 3. The gap, classified (Sandberg & Alvesson)

| Mode | Content |
|---|---|
| **Problematization** (primary) | The literature's shared assumption is that post-training changes *susceptibility* to contextual pressure and that adoption of the injected answer is the right DV. Evidence: DPO "reverses" endorsement (0.60 → 0.26 on unanimous-confident) but truth abandonment barely moves (0.84 → 0.61 → 0.83 for SFT/DPO/RLVR) — DPO changes the *destination*, not the *loss*. Blank et al. 2608.31079 ("DPO is where sycophantic agreement enters") and this paper ("DPO reverses conformity") contradict each other only because they use different DVs. |
| **Confusion-spotting** | Suppression (Pandey 2604.19117; Wang 2508.02087; Li 2605.29087) vs erasure (Joswin 2607.00415) of the correct answer under pressure — unresolved because nobody varies dose, item prior, and stage together. |
| **Application** | SDT (sensitivity d′ vs criterion c) and IRT/Rasch have never been applied to contextual pressure (landscape §4); the belief/policy split *is* the d′/c split. |

## 4. Hypotheses (Platt strong inference — each with a crucial test and a refutation condition)

| ID | Hypothesis | Crucial test | Refuted if |
|---|---|---|---|
| **H-A Frame-specific belief robustness** (revised after the DPO pilot) | Post-training changes belief robustness *selectively by source*: DPO makes belief robust to peer-transcript contradiction but not to a user's claim (pilot: DPO − SFT = +2.2 to +3.6 nats in peer/n-gram frames, −0.1 to −0.6 in user-claim frames); RLVR is predicted to re-open the peer channel. The training-data counterpart: Dolci-DPO rewards agreeing with the *user* (×2.8 agreeable openers) and never rewards hedging. | Same 25-condition factorial on all checkpoints; stage × source-frame interaction on belief shift, with baseline margin (d′) and shift (criterion) separated. | No stage × frame interaction on belief shift (then stages act only on policy). |
| **H-B Policy specificity** | The mapping from shifted belief to emitted answer (adopt / refuse / other / keep) differs by stage: SFT adopts, DPO refuses/hedges/other, RLVR re-adopts partly. | P(output class \| belief margin) curves per stage; SDT: stages move c, not d′. | Curves coincide across stages (then post-training changed belief, not policy). |
| **H-C Saturating dose, item-anchored** | Belief shift = a·log(1+k) + phrasing term + item-prior offset; item robustness is invariant across pressure types (Rasch fit, DIF tests). | Fit on 400 items × 6 checkpoints × dose k∈{0,1,2,3,5,8}. | Dose is linear/non-monotone, or item difficulty reorders across pressure types (DIF). |
| **H-D Mechanism** | Truth-identity probes track the belief layer and are stage-invariant; the policy layer is implemented late and is what steering across checkpoints changes (Pandey-style "circuit persists, expression changes"). | Probe correct-answer identity per layer under pressure at each checkpoint (repo `probes.py`); difference-in-means steering per checkpoint (`contrastive_steering.py`); patch late layers between SFT and DPO. | Probe accuracy for the correct answer collapses under pressure at every stage (erasure) with no stage difference in late-layer steering. |
| **H-E Licensed abstention** | Much of the "system-prompt protection" reported in the field is refusal licensed by an "if unsure, say so" clause, not restored truth. | 2×2: licensing clause × warning clause, across frames. | Refusal rate unchanged by removing the clause. |

Rival hypotheses already excluded by the back-study: repetition/pattern completion (H-REP), consensus-as-evidence (H-SOC in its consensus form), distraction-only (H-DIST: filler ≈ frame ≈ control at the belief level).

## 5. Pilot evidence (local, Instruct-SFT checkpoint, 40 items complete; DPO partial; `results/belief_probe.md`)

Forced-answer margin = logP(first token of GT) − logP(first token of injected answer) after "The answer is ". Paired differences with item-bootstrap 95% CI:

| Contrast | Δ margin (nats) | Δ refusal (greedy) |
|---|---|---|
| frame only (k=0) − control | −0.07 [−0.52, +0.42] | **+0.28** |
| 5 filler lines − frame only | −0.19 [−0.70, +0.26] | −0.22 |
| 5 correct peers − frame only | **+4.59** [+3.35, +5.86] | −0.19 |
| 1 wrong line − frame only | **−3.33** [−4.67, −2.01] | −0.09 |
| 5 − 1 repeats | −0.93 [−1.64, −0.19] | +0.19 |
| 8 − 5 repeats | −0.28 [−0.48, −0.07] | −0.06 |
| stated once (QD) − 5 repeats | +1.68 [+1.10, +2.28] | −0.03 |
| diverse (no consensus) − 5 repeats | +0.50 [−0.40, +1.44] | **+0.16** |
| confident varied − plain identical | **−1.12** [−1.77, −0.52] | −0.13 |
| warning sys − licensing sys | +0.50 [+0.19, +0.81] | **−0.36** |
| no sys − licensing sys | +0.35 [+0.02, +0.69] | **−0.36** |
| single user claim (auth_trust) − 5 repeats | −0.67 [−1.52, +0.21] | −0.36 |
| non-social sequence (matched) − participant frame (both no sys) | −0.85 [−2.11, +0.27] | 0 |

Dose–response (identical plain repeats): k=0 +1.80 · k=1 −1.52 · k=2 −2.12 · k=3 −2.19 · k=5 −2.45 · k=8 −2.73. The first exposure carries ~70% of the total shift; the curve is concave in k.

Reading: **belief** responds to the presence and phrasing of a contradicting claim (and to *supporting* claims, symmetrically), weakly and saturatingly to repetition, and not at all to frame, filler, or the system prompt. **Output policy** (refuse vs answer) responds to frame and to the licensing clause.

### 5.1 SFT vs DPO (both complete, 40 items × 25 conditions)

Paired on identical (item, condition) cells: DPO − SFT = **+1.91 nats [+1.68, +2.15]**, Spearman 0.78 (n=1,000). The gain is **source-specific**: +2.4 to +3.3 under every peer-transcript cell, +1.0 to +1.5 under the non-social sequence, **−0.4 to +0.45 under a user claim**. Under five identical peers DPO's belief still favours the truth (+0.19 vs SFT −2.45); under one user sentence it collapses (−2.4 to −3.4) — for DPO a single user claim moves belief more than five peers (−2.93 [−4.88, −1.12]), a contrast that is null for SFT. So DPO taught the model to discount third-party transcript claims but not first-person user claims — matching the binary data (DPO abandonment under a user claim 0.63 vs SFT 0.61), Blank et al. 2608.31079, and the corpus (Dolci-DPO rewards agreeable openers ×2.8 and never rewards hedging).

The policy layer dissociates in *opposite* directions: DPO refuses in 38–60% of licensed participant-frame cells, including 22% of cells where its belief favours the truth (withholds what it holds); SFT emits the injected answer in 44% of cells where its belief favours the truth (says what it does not hold). Post-training therefore moves **belief robustness per contradiction source** and **output policy** separately, and the paper's single DV sees neither. This is the finding the full study is built to establish across the whole ladder (H-A/H-B), with probes to locate the two layers (H-D).

**Full ladder (base → SFT → DPO → RLVR, local pilot, 40 items).** SFT − base = +0.27 nats overall (base is already flipped by peers: −3.53 under five identical lines vs SFT −2.85), so the paper's "SFT amplifies" is a change in output policy, not belief. DPO − SFT = +2.39 on peer transcripts, +0.12 (n.s.) on user claims. RLVR − DPO = −0.38 on peers and **−0.62 on user claims**, making RLVR the least user-claim-robust checkpoint (−0.49 vs SFT). Refusal is governed by the licensing clause at every stage (0.25–0.60 with it, ≤0.02 without). This is the stage × source × layer structure the full HPC run is designed to establish at n=400 with probes and steering (FINDINGS §10b).

**Corpus evidence (T7 Dolci mixtures; FINDINGS §10a).** Pillar II is confounded (every DPO pair is strong-model-chosen vs weak-model-rejected; the Δ n-gram effect is 4.7× larger in delta-learning pairs). Neither SFT nor DPO data contain the "I am unsure" behaviour (≤0.18%), so the abstention policy is inference-time instruction following. This is why the study must measure the policy layer *behaviourally under instruction variants*, not read it off the corpus.

## 6. Validity plan (Shadish, Cook & Campbell)

- **Construct:** four-way outcome (keep / adopt / other / abstain) as the primary DV; continuous belief (log-odds) and probe-based correct-answer identity as the belief DV; never BER alone.
- **Internal:** fully crossed factorial (frame × system clause × dose × phrasing × consensus) — the pilot renderer already implements 25 cells; item pairing within checkpoint; the licensing clause as an explicit factor.
- **Statistical conclusion:** item-bootstrap CIs, McNemar for binary pairs, mixed models with item random effects, Holm within pre-registered families; pre-register the contrast list before the full run.
- **External:** belief layer on open checkpoints (OLMo-3 7B ladder, 32B Think ladder with `max_new_tokens` fixed, Llama-3.1-8B/70B, Qwen); policy layer additionally on API models. Reasoning models only with complete outputs (§7 of FINDINGS: current Think data are truncated).

## 7. Marr level and what each asset delivers

| Level | Question | Asset |
|---|---|---|
| Computational | context–prior arbitration (knowledge-conflict framing) | existing 201K trials (policy layer), belief pilot (belief layer) |
| Algorithmic | belief shift vs output policy; SDT d′/c; Rasch item robustness | `belief_probe.py`, `analysis_structure_items.py` |
| Implementational | which layers/heads carry the belief vs the policy; what DPO changed | `probes.py`, `contrastive_steering.py`, `activation_patching.py` (never yet run to a result) + the Dolci mixtures on T7, re-read for *source-specific* preference signal (user-agreement rewarded ×2.8; hedging never rewarded; model-pairing confound controlled by restricting to `llm_judged` pairs) |

## 8. Heilmeier

- **What are we trying to do?** Separate, for the first time across a post-training ladder, how much a contradicting context moves a model's belief from what the model then does with it.
- **How is it done today, and the limits?** Single-DV conformity/sycophancy rates on final checkpoints (Zhu 2025; Hu & Qu 2026; Hussain & Nielbo 2026); suppression-vs-erasure probes on single models with single-user prompts (Pandey; Wang; Joswin). No stage ladder with probes; no outcome partition as primary DV; no measurement model.
- **What is new?** The belief/policy dissociation, measured with log-odds + probes on every OLMo-3 checkpoint, under a crossed structural factorial that includes the controls the field lacks (frame-only, filler, correct peers, zero-consensus, matched-instruction sequence, licensing clause).
- **Who cares?** Alignment: DPO/RLHF may be training *refusal and rewording* rather than robustness — the belief is still moved, so downstream agents that read the model's confidence are still misled. Evaluation: the field's benchmarks need the four-way outcome. Interpretability: adjudicates suppression vs erasure as a function of dose/prior/stage.
- **Risks?** (i) Belief invariance may fail (then the paper becomes "which stage protects belief" — still a paper). (ii) Probes may not transfer after post-training (2602.20273) — use per-checkpoint probes and log-odds as the primary belief measure. (iii) Framing is model-dependent — replicate the policy layer on ≥ 3 open families.
- **Cost?** Compute: local (7B ladder fits on the Mac; 32B via the HPC used in April). API: policy-layer replication ≈ 10 models × 25 conditions × 400 items ≈ 100K calls.
- **Time?** 3–5 months (below).
- **Checks?** Month-1 go/no-go: H-A/H-B on the full OLMo ladder with 400 items.

## 9. Plan (3–5 months)

| Month | Work | Deliverable / go-no-go |
|---|---|---|
| 1 | Full belief-probe factorial: 6 OLMo-3 7B checkpoints × 25 conditions × 400 items (≈ 8 h/checkpoint locally); add the licensing-clause 2×2 and a correct-answer probe; fit dose model and SDT d′/c per stage. | H-A, H-B, H-C, H-E on OLMo. **Go if** stage × condition interaction on belief is small relative to the first-claim effect *and* output-policy curves differ by stage. |
| 2 | Mechanism: per-checkpoint truth-identity probes and late-layer steering/patching under the same prompts (existing repo modules; first real activation runs). Re-read the Dolci-DPO audit as a policy-training signal. | H-D. Layer-resolved figure: belief representation stage-invariant vs policy representation stage-specific (or the refutation). |
| 3 | Generality: policy layer on Llama-3.1-8B/70B, Qwen-2.5/3, Gemma (open, so belief layer too where feasible); Think ladders re-run with `max_new_tokens` ≥ 4k; cross-family API replication of the four-way outcome. | Cross-family table; reasoning-mode moderator (Li 2605.29087). |
| 4 | Writing; pre-registered analysis freeze; release of the four-way-outcome benchmark + belief-probe tool. | Draft. |
| 5 | Buffer; venue: ICML 2027 (late Jan) or ACL ARR; ICLR 2027 (late Sept) is too soon. | Submission. |

## 10. Alternatives considered and why they were not chosen

| Direction | Why not |
|---|---|
| Repetition / pattern-completion (the current paper) | Refuted by its own data (§2) and scooped in its defensible part (Hu & Qu). |
| Frame/casting as *the* driver (landscape framing #1) | At the belief level the frame effect is null in the pilot; its effect is policy-level and confounded with refusal licensing. Keep as a factor, not a thesis. |
| Zero-consensus dissociation alone (framing #3) | Real and unclaimed, but a single result; folded in as the G1 control inside the factorial. |
| Pure measurement-model paper (IRT/SDT, framing #4) | "New framing, not new finding" without the stage/mechanism data; folded in as the analysis backbone. |
| Multi-agent / network conformity | No evidence base yet, does not use the 201K trials or the checkpoints, and longer than five months to reach a defensible result. Natural *follow-on* once the belief/policy instrument exists (a probed "private belief" in a live population is exactly what it enables). |

## 11. What to do with the current paper

Do not submit it as is (this applies to the committed `neurips2025_submission_v2.tex` of 8 Aug 2026 as much as to v1: v2 already reports the matched-instruction result — BER 0.278 for both arms — and still keeps the title and the "HPC re-run pending" note). Its `runs_latest` Think results are truncated (the HPC re-runs recover only 42–88% of traces and give abandonment 0.21–0.36, not the ceiling values in the 12-condition tables), the matched-instruction result removes the Fig. 3 asymmetry, and the title thesis is refuted by the paper's own 215K trials (QD ≈ five repeats; varied phrasing > identical repetition; one user sentence ≈ 63–80% of the effect). Its durable content — the OLMo ladder, the 12-condition design, the Dolci audit, the refusal-denominator methodology — becomes the behavioural backbone of the study above, re-scored with the four-way outcome.
