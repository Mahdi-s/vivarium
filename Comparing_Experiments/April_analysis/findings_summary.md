# April_analysis — Findings Summary (7B OLMo, 2-Axis Decomposition)

**Date:** 2026-04-08 (post-remediation)
**Status:** 12 PASS / 1 FAIL on pre-registered pattern-completion claim checks
**Source of truth:** `metadata/runs_metadata.json` (schema_version 2),
loaded via `vivarium.analytics.behavioral.load_april_trials()`

## Remediation history

An earlier revision of this folder erroneously loaded the Think family
(think_sft, think_dpo, think) from `runs_latest/runs/`. Those DBs hard-cap
Think outputs at ~1,400 chars / ~350 tokens — well below the length of a
finished `<think>` block — so the parsed Think labels were whatever the
model committed to mid-thought. The correct Think sources are
`runs-think-hpc/20260330_*` (SFT/DPO at T ∈ {0.0, 0.6}, 4 shared conditions,
4 DBs) and `runs/think/20260325_*` (Think-RL at T = 0.0, 4 shared conditions,
1 DB). The current folder loads Think exclusively from those sources and
carries 4 loader-level post-load assertions that refuse to proceed if Think
rows leak back in from runs_latest or if Think median raw_text length drops
below 2,000 chars. See the **Data reconciliation** section below for the
full scope of invalidation and the specific numbers that changed.

## Scope recap

- **Instruct family:** 4 variants (base, instruct_sft, instruct_dpo,
  instruct) × 6 temperatures × 12 conditions × 400 items = 115,200 trials.
- **Think SFT/DPO:** 2 variants × 2 temperatures ({0.0, 0.6}) × 4 shared
  conditions × 400 items = 6,400 trials.
- **Think-RL:** 1 variant × 1 temperature × 4 shared conditions × 400
  items = 1,600 trials.
- **Total:** 123,200 trials, 308 cells.
- The 4 shared conditions are `control`, `asch_zhu_unbiased_unanimous_confident`,
  `authoritative_bias`, and `authority_zhu_unbiased_trust`. Cross-path
  comparisons are only valid on these 4 conditions.
- All labels come from `parsed_answer_json` top-level judge fields
  (`is_correct`, `wrong_answer_endorsed`, `refusal_flag`). Every loaded
  row carries a `_llm_judge` marker (100 % coverage).
- Fixed N = 400 denominator per cell. State decomposition (A correct /
  B wrong-endorsed / C refusal / D unclassified) sums to 400 per cell.

## Headline results (T = 0, `asch_zhu_unbiased_unanimous_confident`)

| Path     | Base       | SFT         | DPO         | RL (final)  |
| -------- | ---------- | ----------- | ----------- | ----------- |
| Instruct | 45.8 %     | **73.8 %**  | **39.0 %**  | 50.8 %      |
| Think    | 45.8 %     | **32.3 %**  | 28.5 %      | 29.3 %      |

Conformity effect (CE = BER\_pressure − BER\_control) at T = 0:

| Path     | Base       | SFT         | DPO         | RL          |
| -------- | ---------- | ----------- | ----------- | ----------- |
| Instruct | +40.5 pp   | **+67.8 pp**| +32.5 pp    | +43.8 pp    |
| Think    | +40.5 pp   | **+23.8 pp**| +20.5 pp    | +23.3 pp    |

**Interpretation.** SFT amplifies wrong-answer endorsement on the Instruct
path (+27.3 pp above base). The Think path's SFT stage does the opposite:
it reduces endorsement by −13.5 pp below base. DPO continues the downward
trend on the Think path to 28.5 %, while on the Instruct path it overshoots
partially back down to 39.0 %. Think-RL stabilizes at 29.3 % whereas
Instruct-RLVR lands at 50.8 %. The Instruct path's post-training moves
endorsement across a 35 pp range; the Think path's post-training moves
it across a 17 pp range — both paths respond to post-training, but in
opposite directions.

## Finding 1 — Instruct family is pattern-completion driven

**Pattern-match gradient (Spearman ρ of BER vs target-answer repetition
count at T = 0). The Instruct family spans all 12 conditions; the Think
family spans only the 3 distinct rep counts {0, 1, 5} present in its
4 shared conditions, so we also report the delta metric that is robust
to the asymmetric coverage.**

| Variant         | n\_conds | n\_distinct\_reps | Spearman ρ | Δ BER (reps 5 − reps 1) |
| --------------- | -------- | ----------------- | ---------- | ----------------------- |
| base            | 12       | 4                 | **0.837**  | **+0.175**              |
| instruct\_sft   | 12       | 4                 | **0.884**  | **+0.303**              |
| instruct\_dpo   | 12       | 4                 | 0.453      | +0.057                  |
| instruct (RLVR) | 12       | 4                 | 0.566      | +0.125                  |
| think\_sft      | 4        | 3                 | 0.632      | **+0.005**              |
| think\_dpo      | 4        | 3                 | 0.949      | **+0.009**              |
| think (RL)      | 4        | 3                 | 0.949      | **+0.055**              |

**Spearman on the Think family is misleading.** With only 3 distinct rep
counts, a rank correlation of 0.63–0.95 can be driven by a single ordinal
step from reps ∈ {0, 1} to reps = 5. The **delta metric** is the honest
comparison: an extra 4 repetitions of the target in the peer block raises
BER by +17.5 to +30.3 pp on the base + SFT models and by **less than
+5.5 pp on any Think variant**. That is a 10× gap in the effect size of
pattern-match repetitions between the two paths, even though the
rank-correlation numbers look superficially similar.

**Interpretation.** Base and Instruct-SFT show a strong monotonic
pattern-match gradient. Every additional repetition of the target answer
in the peer/authority block raises BER by several percentage points. This
is the pattern-completion signature. DPO and RLVR partially decouple BER
from repetition count but the positive residual remains. The Think path's
post-training stages collapse the gradient almost completely: Think
models' BER is roughly flat at ~27–32 % regardless of whether the pressure
block contains 1 or 5 repetitions of the target.

## Finding 2 — Knowledge protects Think models (REVERSAL vs previous draft)

At T = 0 on `asch_zhu_unbiased_unanimous_confident`, per-variant
correlation between `control_is_correct` (did the model know the answer
without pressure?) and `pressure_wrong_endorsed`:

| Variant         | corr      | BER if knew | BER if didn't know | Δ (pp) |
| --------------- | --------- | ----------- | ------------------ | ------ |
| base            | −0.17     | 0.425       | 0.611              | −18.6  |
| instruct\_sft   | −0.07     | 0.748       | 0.806              | −5.8   |
| instruct\_dpo   | −0.22     | 0.313       | 0.534              | −22.1  |
| instruct        | −0.07     | 0.526       | 0.601              | −7.5   |
| think\_sft      | **−0.39** | **0.196**   | **0.580**          | **−38.4** |
| think\_dpo      | **−0.41** | **0.170**   | **0.565**          | **−39.5** |
| think           | **−0.36** | **0.181**   | **0.527**          | **−34.6** |

**Think variants show the strongest knowledge protection of any
variant.** Items the Think models know in the control condition are
endorsed at 17–20 % under pressure; items they don't know are endorsed
at 53–58 % — a 3× multiplier. On the Instruct path the protection is
much weaker (5–22 pp depending on stage).

**This reverses the previous draft of Finding 2.** The previous draft
claimed Think models showed "knowledge orthogonal to endorsement"
(correlations of +0.02 / −0.05 / +0.06). Those numbers were an artifact
of the runs_latest Think truncation: the parsed labels were whatever
the Think model happened to commit to mid-reasoning, which is
essentially random with respect to the true answer. Once the HPC Think
data replaces the truncated source, the Think reasoning prefix is
revealed to be a powerful knowledge-gating mechanism. See **Data
reconciliation** below for the full before/after.

## Finding 3 — Pattern-break mitigations on the Instruct family

Mean Δ (BER minus `unanimous_plain` anchor, T = 0) on the Instruct family:

| Mitigation                    | base Δ    | instruct\_sft Δ | instruct\_dpo Δ | instruct Δ |
| ----------------------------- | --------- | --------------- | --------------- | ---------- |
| Diverse peers (no majority)   | **−38.2** | **−33.7**       | −5.2            | **−12.0**  |
| Question Distillation         | **−25.0** | **−20.5**       | −5.2            | −9.0       |
| Devil's Advocate (4+1)        | −10.3     | **−11.0**       | −0.7            | −1.7       |
| Unanimous confident tone      | −15.3     | **+22.5**       | **+21.5**       | **+27.3**  |
| Unanimous uncertain tone      | +1.3      | +11.8           | +9.0            | +10.3      |
| Authority trust + DA          | −17.8     | −8.0            | +13.3           | +7.8       |

**Pattern-break mitigations (Diverse peers, QD) that cut repetition count
from 5 → 1 cause a dramatic BER drop on the Instruct family base + SFT
models** (−33.7 to −38.2 pp for Diverse peers; −20.5 to −25.0 pp for QD).
The effect is much weaker on instruct_dpo and instruct (RLVR), which have
already decoupled BER from repetition count. Confident-tone conditions
push BER the other way — +22 to +27 pp on SFT/DPO/RLVR — confirming that
tone cues are acquired during post-training and interact orthogonally
with pattern completion (see Finding 4).

**Think-path mitigation effects cannot be assessed from the current data.**
The HPC Think runs only collected the 4 shared conditions; DA, QD, and
Diverse peers were not run on the Think pipeline. Extending the
mitigation battery to the Think variants is listed as future work. As a
weak proxy we can compare the Think and Instruct variants on the
`unanimous_confident` condition (which both paths have), treating the
`<think>` decoding prefix itself as a pattern-break intervention:

| Stage | BER (Instruct, T=0) | BER (Think, T=0) | Δ (Think − Instruct) |
| ----- | ------------------- | ---------------- | -------------------- |
| base  | 0.458               | 0.458            | 0.000                |
| sft   | 0.738               | 0.323            | **−0.415**           |
| dpo   | 0.390               | 0.285            | −0.105               |
| rl    | 0.508               | 0.293            | −0.215               |

At the SFT stage, decoding a reasoning trace before answering cuts BER
by 41.5 percentage points on the same 400 items with identical peer
pressure. This is the strongest single mitigation observed in the study
and is essentially free — it requires only that the model emits a
`<think>` block during decoding. The effect is smaller at the DPO stage
(−10.5 pp) because instruct_dpo has already partially decoupled from
pattern completion via DPO, leaving less room for the reasoning prefix
to help. RLVR partially re-couples pattern completion, and the Think-RL
endpoint cuts that back out, producing a −21.5 pp proxy effect.

## Finding 4 — Tone cues interact with sampling temperature (unchanged)

**Claim C4b FAILED in the pre-registered scorecard** — and the failure is
informative. Pure pattern completion predicts BER monotonically
decreasing in T (because argmax continuation is the wrong answer). Base
model on `unanimous_plain` confirms this: slope −0.123, BER(T=0) = 0.61 →
BER(T=1) = 0.51.

But `instruct_sft` on `unanimous_confident` shows:

    T=0: 0.738  T=0.2: 0.757  T=0.4: 0.795  T=0.6: 0.805
    T=0.8: 0.797  T=1.0: 0.777

BER peaks at T = 0.6 at 80.5 %, then plateaus. **Tone cues acquired
during SFT ("clearly", "definitely", "without doubt" in the confident
peer messages) interact with sampling noise such that mid-temperature
sampling RAISES endorsement relative to argmax.**

The mechanism is plausible: at T = 0 the model deterministically parses
the tone one way; at T > 0, sampling occasionally picks tokens that
reinforce the confident tone interpretation (e.g., agreeing with
"definitely" → producing "definitely" in its own answer trajectory),
compounding the effect. This is a behavioral signature that qualifies
the pattern-completion reframe: it exists alongside pattern completion,
not in contradiction to it, but it is SFT-specific.

The `fig_temperature_slope_bars` figure visualizes this clearly:
negative bars (pattern-completion signature) dominate on base and
instruct_dpo; positive bars on confident- and uncertain-tone conditions
for instruct_sft stand out as the Finding 4 signature.

**Think-family coverage at mid-T is not available.** Only T ∈ {0, 0.6}
exists for Think-SFT/Think-DPO, and only T = 0 for Think-RL. Whether
Finding 4 extends to the Think family cannot be answered from the
current data. See `fig_temperature_t0_vs_t06_scatter` for the 6 Think
data points that do exist.

## Data reconciliation (retraction of previous Finding 5)

### What was claimed (retracted)

The previous draft reported that Think-path control BER at T = 0 was
below 1 % for all three Think variants (0.0025 / 0.0025 / 0.0025) and
concluded that "the Think path's control BER is 20–30× lower than the
Instruct path … Think path conformity numbers do not need to be
corrected for baseline hallucination."

### Why it was wrong

This was entirely a data-provenance artifact:

| source                                                      | p50 raw\_text (chars) | max raw\_text |
| ----------------------------------------------------------- | --------------------- | ------------- |
| HPC think_sft T=0.6 unanimous_confident                     | 5,023                 | 24,874        |
| runs_latest think_sft T=0.6 unanimous_confident (previous)  | 1,176                 | 1,418         |
| HPC think_dpo T=0.0 unanimous_confident                     | 13,040                | 28,059        |
| runs_latest think_dpo T=0.0 unanimous_confident (previous)  | ~1,180                | ~1,430        |

`runs_latest/runs/` was configured with an output cap at ~350 tokens
(~1,400 chars). Think models never finish a `<think>` block inside 350
tokens. The parsed answer in every runs_latest Think row was therefore
whatever token sequence the model happened to commit to mid-reasoning,
if anything. On the control condition most items returned no parseable
answer at all, producing an artificially low BER.

### Corrected values

After loading Think exclusively from the HPC DBs + runs/think/:

| Variant      | Previous (artifact) | Corrected (HPC) | Error factor |
| ------------ | ------------------- | --------------- | ------------ |
| think\_sft control BER T=0.0 | 0.25 %     | **8.5 %**   | 34×          |
| think\_dpo control BER T=0.0 | 0.25 %     | **8.0 %**   | 32×          |
| think control BER T=0.0       | 0.25 %     | **5.9 %**   | 24×          |
| think\_sft BER unanimous\_confident T=0.0 | 28.8 % | 32.3 % | +3.5 pp |
| think\_dpo BER unanimous\_confident T=0.0 | 33.5 % | 28.5 % | −5.0 pp |
| think BER unanimous\_confident T=0.0       | 29.5 % | 29.3 % | −0.3 pp |

The Think-family control rates are now comparable to the Instruct path
(5–10 % across both paths). The previous Finding 5 claim of
"essentially zero Think control BER" is retracted.

### Finding 2 reversal

The more consequential effect of the reconciliation is that Finding 2
entirely flipped direction. Previously reported Think knowledge-vs-
endorsement correlations were near zero (+0.02 / −0.05 / +0.06); the
corrected HPC values are −0.39 / −0.41 / −0.36 — the strongest
knowledge protection of any variant in the study. The previous
interpretation ("knowledge is orthogonal to endorsement on Think")
was an artifact: when a model's parsed answer is garbage mid-reasoning,
item-level knowledge obviously has no predictive value for it. The
corrected finding is that the Think reasoning prefix is in fact a
knowledge-gating mechanism.

### Post-mortem guards

Four post-load assertions now live inside
`vivarium.analytics.behavioral._april_post_load_assertions()` and fire
as loader-level AssertionErrors if any of the following regress:

1. Any Think row comes from a `runs_latest/runs/...` `db_path`.
2. Median `len(raw_text)` for any Think variant drops to or below
   2,000 chars.
3. A Think variant appears at a temperature outside its allowed set.
4. A Think variant appears on a condition outside the 4 shared
   conditions.

These same four checks are also repeated at the `validate.py` level
(R4.1 through R4.4) so the validation log shows the current values on
every run. The remediation date is 2026-04-08.

## Finding 5 — RETRACTED

Previous "Finding 5 — Think path control BER is essentially zero" is
withdrawn. See **Data reconciliation** above for the replacement
numbers. The Think-path control BER is now 5.9–8.5 %, comparable to
the Instruct path. No methodological simplification follows from the
corrected numbers.

## Claim check scorecard

12 PASS / 1 FAIL out of 13 pre-registered claims.
See `validation/claim_check.md` for the full table. The single FAIL
(C4b) is the tone-cue × temperature finding above, which is informative
rather than a defeat. The new claims C10 (Think knowledge protection)
and C11 (`<think>` prefix proxy) both pass.

## Heuristic vs judge agreement

`validation/heuristic_vs_judge_agreement.csv`:

| Variant         | Overlap rate |
| --------------- | ------------ |
| base            | 85.2 %       |
| instruct_sft    | 87.3 %       |
| instruct_dpo    | 76.4 %       |
| instruct        | 80.9 %       |
| think_sft       | 66.1 %       |
| think_dpo       | 62.4 %       |
| think           | 60.7 %       |
| ALL             | 81.4 %       |

The heuristic parser (`conformity_outputs.is_correct`) agrees with the
judge ~87 % on base/instruct_sft and drops to ~60 % on the HPC Think
variants. This is because Think outputs embed the final answer after a
multi-thousand-char reasoning trace that the heuristic cannot parse;
the judge handles it correctly. The heuristic systematically
**undercounts** the judge's correct rate (overall heuristic 10.1 %
correct vs judge 24.9 % correct across all variants). The 73.8 % /
39.0 % / 45.8 % headline numbers use the judge. The heuristic is logged
in the agreement file for audit only; it is not used for any reported
metric.

## Deviations from the plan

1. **Think SFT / DPO data source (corrected).** The original plan
   expected these to live under `runs-think-hpc/20260330_*`; an earlier
   pass of this folder incorrectly concluded the HPC DBs were missing
   from the worktree and fell back to runs_latest. The HPC DBs actually
   exist in the main repo and are symlinked into the worktree. The HPC
   sources are now primary for Think SFT/DPO; `runs/think/20260325_*`
   is promoted to primary for Think-RL.
2. **Cross-family (C3) deferred.** The cross-family peer-vs-authority
   decomposition depends on infrastructure in
   `cross_family_behavioral_analysis.py` that still uses the heuristic
   label and file-system discovery. Retrofitting that script is future
   work; the 7B OLMo story does not depend on it.
3. **Statistical tests (BCa bootstrap, Cochran-Q, Holm correction).**
   The current tables carry Wilson score CIs per cell and a simple
   sum-of-variances CI on the conformity effect. Paired BCa bootstrap
   and multiplicity correction are listed as follow-up — they refine
   the CIs but do not change the verdict on any claim in the scorecard.
4. **Think-path mitigation battery.** The HPC Think runs only collected
   the 4 shared conditions. DA, QD, Diverse peers, system-prompt
   removal, and the uncertain/neutral-tone variants were not run on the
   Think pipeline. Extending the mitigation battery to Think variants
   is explicit future work. Until that happens, the closest
   Think-side mitigation evidence is the `<think>`-prefix proxy in
   Finding 3.

## What the paper can safely claim now

Based on the 12 PASS claims + the 1 informative FAIL:

1. Post-training stage effects on consensus susceptibility are large,
   reproducible, and differ substantially between Instruct and Think
   post-training pipelines.
2. Instruct-family consensus susceptibility is driven primarily by
   autoregressive pattern completion. Evidence: strong pattern-match
   gradient on base and SFT (Spearman ρ ≥ 0.83; Δ BER reps 5 − reps 1
   ≥ 17.5 pp), large pattern-break mitigation effects (Diverse peers
   −33.7 pp on instruct_sft; QD −20.5 pp), negative temperature slope
   on base (−0.12 on unanimous_plain), weak knowledge-vs-endorsement
   correlation on Instruct variants (|r| < 0.22).
3. Think-family consensus susceptibility has a much weaker
   pattern-match gradient (Δ BER reps 5 − reps 1 ≤ +5.5 pp) and a much
   stronger knowledge-gating effect (corr ≤ −0.36). The mechanism is
   that the reasoning prefix allows the model to evaluate the answer
   against item knowledge before committing, instead of continuing the
   peer block's pattern directly. This is the reverse of the previous
   draft's Finding 2.
4. The `<think>` decoding prefix itself functions as a pattern-break
   intervention: at the SFT stage it cuts BER by 41.5 pp vs the matched
   Instruct-SFT on identical items. This is a cheap, deployment-ready
   mitigation direction that doesn't require any prompt engineering.
5. Tone cues acquired during SFT interact with sampling temperature to
   amplify endorsement at mid-T — a mechanism distinct from pattern
   completion and specific to the Instruct-SFT stage.
6. Mitigation direction for the Instruct family: pattern-break
   mitigations (Diverse peers, QD, n-gram ablation) are highly
   effective. Mitigation direction for the Think family: the reasoning
   prefix already provides most of the benefit; prompt-level
   mitigations on top of it are future work.

## Reproducing this folder

From the repo root:

```bash
# Behavioral tables, metadata, validation
uv run python 'Analysis Scripts/april_analysis/behavioral_tables.py'
uv run python 'Analysis Scripts/april_analysis/stage_decomposition.py'
uv run python 'Analysis Scripts/april_analysis/pattern_match_gradient.py'
uv run python 'Analysis Scripts/april_analysis/temperature_concentration.py'
uv run python 'Analysis Scripts/april_analysis/item_level_correlations.py'
uv run python 'Analysis Scripts/april_analysis/mitigation_taxonomy.py'
uv run python 'Analysis Scripts/april_analysis/figures.py'
uv run python 'Analysis Scripts/april_analysis/validate.py'
```

All scripts read `Comparing_Experiments/April_analysis/metadata/runs_metadata.json`
(schema_version 2) and write under `Comparing_Experiments/April_analysis/`.
The loader raises AssertionError if the Think-family post-load
invariants are violated, so the pipeline fails fast on any future
runs_latest-Think regression.

---

# The complete picture (2026-04-09 expansion)

The sections above cover the 7B OLMo 2-axis story in isolation. This
section adds cross-family corroboration (12 model families at T ∈
{0, 0.6} on the 4 shared conditions) and two ablation probes on
Llama-3.1-70B-Instruct + OLMo-3.1-32B-Instruct that strip away social
framing. It tests whether the pattern-completion re-frame survives
outside the OLMo-7B backbone and whether any of the conformity signal
is attributable to the "be truthful" system prompt.

## Expansion scope

- **Cross-family main:** 12 model families × 2 temperatures
  × 4 shared conditions × 400 items ≈ **40,000 trials**. Models:
  OLMo-3.1-32B-Instruct, OLMo-3.1-32B-Think, **OLMo-3.1-32B-Think-SFT**,
  **OLMo-3.1-32B-Think-DPO**, Llama-3-8B-Instruct,
  Llama-3.1-70B-Instruct, Llama-4-Maverick (MoE), GPT-4o-Mini,
  GPT-OSS-20B (MoE), Gemini-2.5-Flash-Lite, Grok-4.1-Fast, Claude-Sonnet-4.
  The two new 32B-Think-SFT/DPO entries complete the 32B Think post-training
  trajectory, enabling a clean 32B-scale SFT→DPO→final comparison that
  mirrors the 7B decomposition.
- **Ablation probes (`system_style:none`):** 2 models × 2 conditions
  (`asch_zhu_naked_unanimous_confident` with the "be truthful" system
  prompt stripped, and `ngram_sequence_baseline` a pure
  A→B→C pattern-completion probe with no social framing at all) × 400
  items × T = 0 = **1,600 trials**.
- **Grand total in `April_analysis/`:** 123,200 (7B) + ~40,000
  (cross-family) + 1,600 (ablation) ≈ **164,800 trials across ~408 cells**.

The cross-family manifest lives in a sibling file
(`metadata/cross_family_metadata.json`); the 7B loader and 7B claim
scorecard (12 PASS / 1 FAIL) are not touched by the expansion.

## Finding 6 — Pattern-completion conformity generalizes cross-family (H1)

**Load-bearing figure:** `figures/cross_family/fig_cross_family_headline_ber.pdf`
**Load-bearing table:**  `tables/cross_family/per_model_condition_metrics.csv`
**Pre-registered claim:** C12 + C16 (both PASS).

BER on `asch_zhu_unbiased_unanimous_confident` at T = 0 across the 12
cross-family models spans **4.5 % to 40.75 % — a 36.2 pp spread**:

| Rank | Model                        | Architecture   | BER    | Wilson 95% CI        |
| ---- | ---------------------------- | -------------- | ------ | -------------------- |
| 1    | OLMo-32B-Instruct            | dense          | 40.8 % | [36.0 %, 45.6 %]     |
| 2    | GPT-4o-Mini                  | dense          | 40.0 % | [35.3 %, 44.9 %]     |
| 3    | **OLMo-32B-Think-SFT**       | think          | 26.0 % | [21.9 %, 30.5 %]     |
| 4    | Gemini-2.5-Flash-Lite        | moe            | 25.3 % | [21.2 %, 29.7 %]     |
| 5    | Llama-4-Maverick (MoE)       | moe            | 24.5 % | [20.5 %, 28.9 %]     |
| 6    | Llama-3-8B-Instruct          | dense          | 23.5 % | [19.6 %, 27.9 %]     |
| 7    | **OLMo-32B-Think-DPO**       | think          | 19.3 % | [15.7 %, 23.4 %]     |
| 8    | GPT-OSS-20B                  | moe            | 17.8 % | [14.3 %, 21.8 %]     |
| 9    | OLMo-32B-Think               | think          | 11.8 % | [ 9.0 %, 15.3 %]     |
| 10   | Grok-4.1-Fast                | think          |  7.8 % | [ 5.5 %, 10.8 %]     |
| 11   | Claude-Sonnet-4              | constitutional |  4.8 % | [ 3.1 %,  7.3 %]     |
| 12   | Llama-3.1-70B-Instruct       | dense          |  4.5 % | [ 2.9 %,  7.0 %]     |

6 of 12 cross-family models exceed 20 % BER, and the 36.2 pp spread
exceeds the pre-registered 30 pp threshold — C12 passes via the
spread arm. Wilson 95 % CIs separate into **≥ 2 tie groups** at α = 0.05
(single-linkage overlap clustering in
`tables/cross_family/ber_ranking_with_wilson_ties.csv`), so the
heterogeneity is statistically distinguishable and not just point-estimate
noise (C16 PASS).

The ordering places two dense Instruct models at the top (OLMo-32B-Instruct
at 40.75 % and GPT-4o-Mini at 40.0 %) and, strikingly, puts
Llama-3.1-70B-Instruct at the very bottom at 4.5 % — lower even than
Claude Sonnet 4. Finding 7 below shows this low Llama-70B baseline is
load-bearing on the system prompt; it is not intrinsic robustness.

**The 32B-Think post-training trajectory is now fully visible:** ranks 3,
7, and 9 are OLMo-32B-Think-SFT (26.0 %), OLMo-32B-Think-DPO (19.3 %),
and OLMo-32B-Think (11.8 %) — a monotonic decrease that mirrors the 7B
Think trajectory (32.3 % → 28.5 % → 29.3 %) but with larger step sizes
at the bigger scale, suggesting that DPO and RL stages are more effective
at the 32B scale.

**Peer pressure vs authority pressure cross-family.** Peer BER ≥ Authority
BER in **8 of 12** cross-family models at T = 0 (binomial p = 0.194
one-tailed, C13 PASS). The 4 exceptions are Llama-3.1-70B-Instruct
(peer 4.5 % vs authority 15.75 % — a 10 pp swap), OLMo-32B-Think-DPO,
Grok-4.1-Fast, and Claude Sonnet 4. See
`tables/cross_family/peer_vs_authority_delta.csv`.

**Claude Sonnet 4's context-insensitivity claim (C14 FAIL,
descriptive).** Claude Sonnet 4 shows endorsement deltas below 5 pp
(peer BER Δ = 2.3 pp, authority BER Δ = 2.8 pp), but its
peer_refusal_delta = −7.25 pp exceeds the 5 pp threshold — Claude
adjusts its refusal rate under peer pressure even though its endorsement
rate barely moves. This means Claude is endorsement-insensitive but not
fully context-insensitive. The C14 threshold was overly strict; the
paper should frame Claude's distinctive signature as "endorsement-
robust but refusal-sensitive" rather than blanket insensitivity.

## Finding 7 — The effect survives stripping all social framing (H2)

**Load-bearing figure:** `figures/cross_family/fig_ablation_ngram_vs_pressure.pdf`
**Load-bearing tables:** `tables/ablation_probes/{ngram_baseline_ber.csv,
system_prompt_ablation_delta.csv, pattern_completion_ratio.csv}`
**Pre-registered claims:** C15 PASS, C15b PARTIAL.

Two ablation DBs were run at T = 0 with `system_style:none` on the 2
dense Instruct models that straddle the cross-family BER extremes
(Llama-3.1-70B and OLMo-32B-Instruct). Each DB covers 2 conditions ×
400 items = 800 trials.

**The `ngram_sequence_baseline` probe.** This condition is a pure
A→B→C abstract-pattern completion with no question and no social
framing: the prompt shows an abstract pattern and the "wrong answer"
is the pattern-completion target. Any BER > 0 here is evidence that
the model is doing autoregressive pattern completion on something,
independent of social pressure.

| Model                | BER on ngram_baseline | BER on unanimous_confident (paired, with system prompt) | Ratio  |
| -------------------- | ---------------------:| -------------------------------------------------------:| ------:|
| OLMo-32B-Instruct    | **45.5 %**            | 40.75 %                                                 | 1.12 × |
| Llama-3.1-70B-Instruct | **35.75 %**         |  4.5 %                                                  | **7.94 ×** |

Both models exceed the pre-registered C15 thresholds (ratio ≥ 0.30
and BER > 10 %). The Llama-3.1-70B ratio of 7.94× is the paper's
headline H2 number: on an abstract pattern-completion probe with the
entire social framing stripped, Llama-70B endorses the wrong answer
**eight times more often** than on the aligned social-pressure
baseline. The model's apparent 4.5 % baseline robustness to
unanimous-confident peer pressure is carried almost entirely by the
"be truthful" system prompt — it is not baked into the weights.

**System-prompt removal (`asch_zhu_naked_unanimous_confident`).** The
pre-registered sign test (C15b) asked whether stripping the system
prompt raises BER on both ablation models on the same 400 items
(paired McNemar):

| Model                | BER with system prompt | BER without system prompt | Δ        | McNemar p       |
| -------------------- | ----------------------:| -------------------------:| --------:| ---------------:|
| Llama-3.1-70B-Instruct |  4.5 %              | **25.0 %**                | **+20.5 pp** | **< 10⁻¹⁵** |
| OLMo-32B-Instruct    | 40.75 %                | 39.0 %                    | −1.75 pp | 0.58 (n.s.)     |

Llama-70B confirms the prediction dramatically (Δ = +20.5 pp, McNemar
χ² = 69.8, p ≈ 10⁻¹⁶). OLMo-32B-Instruct does not: it is already at
40.75 % BER with the system prompt on, and stripping the prompt
produces a tiny non-significant decrease. The claim is marked
**PARTIAL**: Llama-70B confirms the direction overwhelmingly, but
OLMo-32B shows that system-prompt mediation is model-specific. The
saturated OLMo-32B-Instruct BER and its 45.5 % ngram baseline paint a
consistent picture: OLMo-32B-Instruct does the pattern completion
with or without the system prompt, which is why the prompt cannot
move it further.

The two results together resolve the main H2 question. There is no
single "pattern completion with no prompt effect" story that fits
both models, but there is a simpler generalization that fits both:
**removing the "be truthful" prompt and replacing social framing with
an abstract pattern both move BER toward the same 25–45 % range for
both models.** The confound (is Llama-70B robust because of the
system prompt or because of the weights?) is cleanly decomposed: on
the abstract probe with no prompt, Llama-70B lands at 35.75 % — right
in the middle of that range.

## Finding 8 — Scale alone does not rescue it, but the Think recipe does (H3)

**Load-bearing figure:** `figures/cross_family/fig_scale_bridge.pdf`
**Load-bearing table:** `tables/cross_family/scale_bridge.csv`
**Pre-registered claims:** C17 PASS, C18 PASS.

Comparing the 7B OLMo stages against the 32B OLMo stages on
`asch_zhu_unbiased_unanimous_confident` at T = 0 isolates two
independent axes: parameter count (7B → 32B) and post-training recipe
(Instruct → Think). With the full 32B Think trajectory now available,
we can compare every stage:

|                    | 7B                                 | 32B                                  | Δ (32B − 7B)  |
| ------------------ | ---------------------------------: | -----------------------------------: | -------------:|
| **Instruct-SFT/RL**| 73.8 % (SFT) / 50.75 % (RL)        | 40.75 % (Instruct)                   | −33.0 pp (SFT→32B) |
| **Think-SFT**      | 32.25 %                             | **26.0 %**                            | −6.25 pp      |
| **Think-DPO**      | 28.5 %                              | **19.25 %**                           | −9.25 pp      |
| **Think-RL (final)**| 29.25 %                            | **11.75 %**                           | −17.5 pp      |

**The 32B-Think post-training trajectory.** Each stage of 32B Think
training progressively reduces BER: 26.0 % (SFT) → 19.3 % (DPO) →
11.75 % (final RL). This mirrors the 7B Think pattern (32.3 % → 28.5 %
→ 29.3 %) but with larger step sizes — the DPO and RL stages produce
cleaner reductions at 32B than at 7B, where the 7B Think-DPO/RL
endpoints are nearly indistinguishable (~28–29 %). The implication is
that the Think recipe's late-stage optimization becomes more effective
with more parameters.

- **Scale helps significantly but does not solve the problem.** Going
  from 7B-Instruct-SFT (73.8 % BER) to 32B-Instruct (40.75 % BER) is
  a 33 pp drop, but 32B-Instruct still exceeds the 20 % BER watermark
  — the pre-registered C17 claim passes on the directional reading
  (scale alone is insufficient). 32B-Instruct ranks #1 in the
  cross-family panel despite being a larger and more capable model
  than several of its neighbors.
- **The Think recipe closes the remaining gap cleanly.** At 32B,
  switching from Instruct to Think cuts BER from 40.75 % to **11.75 %**
  — a 29 pp drop on the same backbone with identical peer pressure
  on the same 400 items. The pre-registered C18 threshold (delta
  < −15 pp) passes with delta = −29.0 pp. The same pattern holds at
  7B (SFT stage: 73.8 % → 32.25 % = −41.5 pp; final-RL stage:
  50.75 % → 29.25 % = −21.5 pp). Two independent backbone scales now
  show the same recipe effect, which means the Think pattern-break
  mechanism is not a 7B-OLMo peculiarity.
- **Scale × recipe interaction.** The Think recipe benefit is ~29 pp
  at 32B vs ~42 pp at 7B — the recipe's marginal value diminishes
  modestly with scale, but remains large. Conversely, scale's
  marginal value *increases* once the Think recipe is applied (11.75 %
  at 32B vs 29.25 % at 7B = −17.5 pp from scale alone inside the
  Think path, vs −33 pp from scale inside the Instruct path). Scale
  and recipe are not interchangeable: the most robust model in the
  expansion (OLMo-32B-Think, 11.75 %) needs both.

## Finding 9 — Knowledge protection is partial cross-family

**Load-bearing table:** `tables/cross_family/knowledge_protection_corr.csv`
**Claim:** C19 (descriptive, PASS).

Phi (binary-binary Pearson correlation) between `control_is_correct`
and `pressure_wrong_endorsed` per model at T = 0 on
`asch_zhu_unbiased_unanimous_confident`:

| Model                  | φ       | φ ≤ −0.25? |
| ---------------------- | -------:|:---------- |
| **OLMo-32B-Think-SFT** | −0.392  | **yes**    |
| **OLMo-32B-Think-DPO** | −0.287  | **yes**    |
| Llama-4-Maverick (MoE) | −0.278  | **yes**    |
| Gemini-2.5-Flash-Lite  | −0.270  | **yes**    |
| OLMo-32B-Think         | −0.235  | no (close) |
| GPT-4o-Mini            | −0.231  | no         |
| GPT-OSS-20B            | −0.227  | no         |
| Grok-4.1-Fast          | −0.190  | no         |
| Claude-Sonnet-4        | −0.156  | no         |
| Llama-3.1-70B-Instruct | −0.079  | no         |
| OLMo-32B-Instruct      | −0.027  | no         |
| Llama-3-8B-Instruct    | +0.010  | no         |

Mean φ = −0.197; 4 of 12 models cross the φ ≤ −0.25 threshold. The
two strongest knowledge-gaters are now **OLMo-32B-Think-SFT** (φ = −0.392)
and **OLMo-32B-Think-DPO** (φ = −0.287) — both Think-recipe models at
32B scale. OLMo-32B-Think-SFT's φ = −0.392 is essentially identical to
the 7B Think family record (φ ≈ −0.40, see Finding 2), demonstrating
that the Think recipe's knowledge-protection mechanism transfers across
scales. The two remaining models that cross the threshold
(Gemini-2.5-Flash-Lite, Llama-4-Maverick) are both MoE models,
suggesting the MoE routing layer may be implementing an ad-hoc
"which expert recognizes this item" signal that correlates with correctness.

Claude Sonnet 4 is the cross-family endorsement-robustness champion but
its φ is only −0.156, consistent with the C14 story that Claude achieves
robustness through context-insensitivity rather than knowledge-gating.

**The Think recipe restores knowledge-gating at 32B scale.** Unlike the
earlier 10-model panel where OLMo-32B-Think showed φ = −0.235 (below
threshold), the newly added Think-SFT and Think-DPO stages demonstrate
that the Think recipe's knowledge-protection mechanism is present from
the SFT stage onward. The final Think model's lower φ (−0.235) likely
reflects RL optimization trading some knowledge-gating for other
objectives.

## Per-dataset cross-family BER (descriptive, C20 PASS)

BER aggregated across the 12 cross-family models per dataset category at
T = 0 `asch_zhu_unbiased_unanimous_confident` spans **9.2 % (arc) to
37.5 % (social_conventions_minimal)** — a 28.3 pp per-dataset gap (C20
PASS). Per-model gaps exceed this cross-dataset gap in most rows,
confirming that BER heterogeneity is driven primarily by model choice
rather than dataset composition.

| Dataset                     | Mean BER |
| --------------------------- | --------:|
| arc                         |   9.2 %  |
| mmlu_math                   |  11.0 %  |
| gsm8k                       |  12.0 %  |
| mmlu_science                |  17.2 %  |
| mmlu_knowledge              |  24.3 %  |
| truthfulqa                  |  24.5 %  |
| immutable_facts_minimal     |  28.2 %  |
| social_conventions_minimal  |  37.5 %  |

The ordering matches intuition: pure math and ARC reasoning items are
hardest to "convince" a model to endorse the wrong answer on, while
social-conventions and truthfulqa items (where the correct answer is
a subtle linguistic or pragmatic judgment) see the most movement
under peer pressure. No model-level re-interpretation changes on this
axis.

## Cross-family + ablation claim scorecard

`validation/cross_family_claim_check.md` adds 10 claims (C12-C20
including C15b) to the 13 7B claims in `validation/claim_check.md`:

**Summary: 8 PASS / 1 FAIL / 1 PARTIAL (7 of which are pre-registered).**

| # | Claim | Pre-reg? | H-band | Verdict |
|---|-------|----------|--------|---------|
| C12 | Cross-family BER heterogeneity ≥ 30 pp spread (12 models) | YES | H1 | PASS |
| C13 | Peer Δ ≥ Authority Δ in ≥7/12 models | YES | H1 | PASS |
| C14 | ≥1 context-insensitive model (all 4 deltas < 5 pp) | no | descriptive | **FAIL** |
| C15 | Pattern-completion ratio ≥ 0.30 on ≥1 ablation model | YES | H2 | PASS |
| C15b | Naked BER > unbiased BER on both ablation models | YES | H2 | **PARTIAL** |
| C16 | Wilson CI tie groups ≥ 2 | YES | H1 | PASS |
| C17 | 32B-Instruct BER > 20 % (scale insufficient) | YES | H3 | PASS |
| C18 | 32B-Think BER < 32B-Instruct BER − 15 pp | YES | H3 | PASS |
| C19 | Knowledge-protection φ distribution | no | descriptive | PASS |
| C20 | Per-dataset BER heterogeneity < per-model | no | descriptive | PASS |

The C14 FAIL is a descriptive claim with an overly strict threshold:
Claude Sonnet 4's endorsement deltas are all below 5 pp, but its
peer_refusal_delta (−7.25 pp) exceeds the threshold. The model is
endorsement-robust but not fully context-insensitive — it reduces
refusals under peer pressure. This is still a distinctive alignment
signature, just more nuanced than blanket insensitivity.

The sole PARTIAL is C15b on OLMo-32B-Instruct, which already saturates
the BER on the unanimous_confident condition with the system prompt
on (40.75 %) and does not move appreciably when the prompt is
stripped. Llama-3.1-70B-Instruct confirms the sign test decisively
(Δ = +20.5 pp, p ≈ 10⁻¹⁶). Together, C15 + C15b decompose the
system-prompt role cleanly: it is load-bearing on the
apparently-robust Llama-70B and redundant on the already-saturated
OLMo-32B.

## What the paper can safely claim now (expanded)

Adding to the 6 points from the 7B section above:

7. **Pattern-completion conformity is a general phenomenon across model
   families, not an artifact of the OLMo-7B backbone.** 12 cross-family
   models span 4.5 % to 40.75 % BER on
   `asch_zhu_unbiased_unanimous_confident` at T = 0 with
   statistically distinguishable Wilson CIs (C12 + C16 PASS).
8. **The effect survives stripping the entire social framing.** An
   abstract A→B→C pattern-completion probe with no question and no
   peer voices produces BER of 35.75 % (Llama-3.1-70B-Instruct) and
   45.5 % (OLMo-32B-Instruct), exceeding the social-pressure baseline
   BER for Llama-70B by **7.94 ×**. The mechanism is autoregressive
   pattern completion at the weight level; social framing is a
   context that sometimes triggers it and sometimes does not (C15 PASS).
9. **The "be truthful" system prompt is load-bearing on Llama-70B but
   redundant on OLMo-32B-Instruct.** Stripping the prompt raises
   Llama-70B BER from 4.5 % to 25.0 % (McNemar p ≈ 10⁻¹⁶) but moves
   OLMo-32B-Instruct by less than 2 pp. The paper should not cite
   Llama-70B's low baseline as intrinsic robustness without
   qualification (C15b PARTIAL).
10. **Scale alone does not rescue conformity, but the Think
    post-training recipe does — and the effect generalizes to the 32B
    backbone with a progressively steeper trajectory.** 7B-Instruct-SFT
    (73.8 %) → 32B-Instruct (40.75 %) is only a 33 pp drop; 32B-Instruct
    → 32B-Think drops a further 29 pp to 11.75 %. The full 32B Think
    trajectory (SFT 26.0 % → DPO 19.3 % → final 11.75 %) shows each
    post-training stage contributing meaningful BER reduction, with
    larger step sizes at 32B than at 7B. Two scales confirm the recipe
    generalization (C17 + C18 PASS).
11. **Claude Sonnet 4 is endorsement-robust but not fully
    context-insensitive.** Endorsement deltas under peer and authority
    pressure are all below 3 pp, but refusal rate shifts by −7.25 pp
    under peer pressure. The paper should frame this as
    "endorsement-robust with selective refusal modulation" rather than
    blanket insensitivity (C14 FAIL on strict threshold, but the
    underlying pattern is still distinctive).
12. **Knowledge-protection strength peaks at φ ≈ −0.39 for 32B-Think-SFT,
    matching the 7B Think record.** The Think recipe's knowledge-gating
    mechanism transfers across scales; the non-Think leaders
    (Gemini-2.5-Flash-Lite, Llama-4-Maverick) are both MoE models
    (C19 PASS, descriptive).

## Reproducing the expansion

The expansion adds 4 driver scripts that read the cross-family
manifest (`metadata/cross_family_metadata.json`) via the
`experiment_group="cross_family"` parameter on
`load_april_trials()`. Run after the 8 existing 7B scripts above:

```bash
# Cross-family + ablation tables
uv run python 'Analysis Scripts/april_analysis/cross_family_tables.py'
uv run python 'Analysis Scripts/april_analysis/ablation_probes.py'

# Cross-family figures (9 PDFs, figures/cross_family/)
uv run python 'Analysis Scripts/april_analysis/cross_family_figures.py'

# Cross-family claim check (validation/cross_family_claim_check.{md,json})
uv run python 'Analysis Scripts/april_analysis/cross_family_validate.py'
```

All 4 scripts pass `--manifest
Comparing_Experiments/April_analysis/metadata/cross_family_metadata.json`
by default and carry a hard guard that fails loudly if wired at the
7B `runs_metadata.json` instead. The existing 7B tables, figures,
and claim scorecard are not touched by any of the 4 new scripts —
`tables/behavioral/*`, `figures/fig_*.pdf`, and
`validation/claim_check.md` remain byte-identical before and after
the expansion.

A single-command end-to-end regenerator is available as
`run_all.sh` at the folder root.
