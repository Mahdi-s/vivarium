# Method: how the research gap is being identified and justified

This document fixes, *before* the evidence is in, the frameworks used to (a) locate a gap, (b) turn it into
falsifiable hypotheses, (c) decide what counts as a crucial experiment, and (d) judge whether the result is
a paper. Every decision in the final recommendation cites one of these.

## 1. Locating the gap — Sandberg & Alvesson (2011) and Alvesson & Sandberg (2011)

Sandberg & Alvesson classify how research questions are generated from literature:

| Mode | What it does | Typical weakness |
|---|---|---|
| **Gap-spotting: confusion** | Competing explanations for the same phenomenon exist | Cheap; often resolved by a control the field skipped |
| **Gap-spotting: neglect** | An area/measure/population is under-researched | "Nobody did X" is not a reason X matters |
| **Gap-spotting: application** | Extend an established theory/method to a new domain | Novelty comes from the theory, not the finding |
| **Problematization** | Challenge an *assumption* shared by the literature | Highest impact; requires evidence that the assumption fails |

Applied here: the conformity literature shares the assumption that "the model adopted the *peers'* answer"
(the dependent variable is *endorsement of the injected answer*). The back-study tests whether that assumption
holds (does the model abandon the truth when there is *no* majority to adopt?). If it fails, the gap is a
**problematization**, not a neglect gap — the field's dependent variable is the wrong construct.
The knowledge-conflict and conformity literatures also explain the same behaviour with different vocabularies
(**confusion gap**), and measurement theories from psychophysics/IRT have not been applied (**application gap**).
A candidate direction that rests *only* on application or neglect is rejected as insufficient for a paper.

## 2. Turning the gap into hypotheses — Platt (1964), *Strong Inference*

1. Devise **alternative hypotheses** (not one favoured story).
2. Devise a **crucial experiment** whose possible outcomes exclude one or more alternatives.
3. Run it cleanly.
4. Recycle with sub-hypotheses.

The rival hypotheses for "why does the model stop producing an answer it knows":

| ID | Hypothesis | What must be true | Excluded by |
|---|---|---|---|
| H-REP | **Repetition / pattern completion**: literal repeats of the wrong token sequence drive endorsement | Effect scales with repeat count; identical > varied; a single mention is weak | k-dose-response flat; QD (stated once) ≈ 5 repeats; asch_history (5 identical) ≪ participant frame |
| H-SOC | **Social/informational influence**: the model treats peer answers as evidence | Consensus > no-consensus; diverse peers should not cause abandonment; tone/confidence matters | Diverse ≈ unanimous; tone effects small |
| H-FRAME | **Frame / role casting**: casting the model as a transcript participant switches it from "assistant answering" to "document continuation" (simulator/role-play theory) | Frame alone (k=0) or filler lines already degrade truth; frame effect > repetition effect; frame × system-prompt interaction | k=0 and filler ≈ control |
| H-DIST | **Distraction**: any added text degrades accuracy (irrelevant-context effect) | Filler lines degrade as much as wrong answers | Filler ≈ control while wrong-answer lines degrade |
| H-INSTR | **Instruction hierarchy**: the system prompt's warning, not the user-turn structure, decides | Warning system prompt removes most of the effect | Warning has small effect relative to frame |

The pilot factorial (`tools/belief_probe.py`) is designed so that each row above has at least one cell whose
outcome excludes it. The existing 215K-trial dataset is re-analysed the same way (`FINDINGS.md`).

## 3. Threats to validity — Shadish, Cook & Campbell (2002)

- **Construct validity**: is "conformity" (endorsement of the injected answer) the right operationalisation of
  the behaviour? The 4-state outcome (correct / target-wrong / other-wrong / refusal) tests this directly.
- **Internal validity**: existing conditions confound system prompt with frame (asch_history vs. participant);
  the pilot crosses them.
- **Statistical conclusion validity**: pairing by item within variant × temperature; bootstrap over items;
  Holm correction for pre-specified contrast families.
- **External validity**: one model family for stages; cross-family for the behavioural pattern only.

## 4. Levels of explanation — Marr (1982)

- *Computational*: what function does the behaviour serve (context-vs-prior arbitration)?
- *Algorithmic*: what quantity moves (forced-answer log-odds; belief flip vs. output suppression)?
- *Implementational*: which representations/heads (probes, steering) — deferred to the mechanistic phase.
The pilot lives at the algorithmic level: it measures the *continuous* belief (log-odds GT vs wrong) rather
than only the binary output, which is what separates *suppression* (belief intact, output changed) from
*overwrite* (belief itself moved).

## 5. Programme justification — Heilmeier Catechism

Answered in the final recommendation: what are we trying to do; how is it done today and what are the limits;
what is new and why will it succeed; who cares; risks; cost; time; mid-term and final checks.

## 6. Decision rule for "is it a paper" (adapted from Carlini's conclusion-first test and Wobbrock & Kientz contribution types)

State the one-sentence conclusion the paper would end with *before* running the study. If the pilot cannot
produce data that would make that sentence true or false, the direction is dropped. Contribution type must be
**empirical + methodological** (new construct/measure with evidence), not "survey" or "opinion".
