# Mechanistic Thermodynamics of Truth (Olmo-3): A Full Conformity + Interpretability Experiment Protocol

This document is meant to be "tough-reviewer ready": it specifies a complete experiment matrix (not just a single run), the mechanistic questions, the ablations, what to run, what artifacts to expect, what analyses to perform, and what is (and is not) supported by the current repo.

It is explicitly aligned with the existing paper draft in `paper/paper.tex` (temperature-dependent conformity across the Olmo-3 family) and extends it with mechanistic interpretability and causal validation using this codebase.

---

## 0) What We Are Trying to Prove (Paper-Grade Contribution)

### 0.1 Core claim (behavioral + mechanistic)
Temperature-dependent conformity is not "just randomness." It emerges from a competition between:
- a **truth-supporting internal signal** (call it the truth direction), and
- a **social pressure / agreement internal signal** (call it the social direction),
whose relative dominance varies across:
- **training stage** (base vs instruct vs SFT vs think vs RL-Zero), and
- **prompted social pressure** (Asch-style confederates vs authoritative user),
and whose *behavioral expression* is modulated by **decoding temperature** (the "thermodynamics" story).

### 0.2 What makes this publishable (vs another benchmark paper)
We do not stop at error-rate curves. We will add:
- **Layerwise mechanistic signatures** (truth vs social projections, collision layers) that predict conformity.
- **Causal tests** (activation steering interventions) that can flip behavior in controlled ways.
- **Reviewer-friendly ablations** that rule out confounds (probe leakage, dual-stack risk, prompt artifacts, sampling variance).

### 0.3 Primary deliverables for the final paper
1. Behavioral: temperature-dependent conformity curves across the Olmo-3 family (already in spirit in `paper/paper.tex`, but we will re-run with strict ablations and seed replicates).
2. Mechanistic: truth-vs-social signal trajectories across layers and temperatures, and their relationship to behavioral conformity.
3. Causal: intervention results demonstrating that targeted activation edits reduce conformity on pressure trials without broadly destroying control performance.

---

## 1) What the Repo Actually Supports (Evidence)

The repo already contains an end-to-end conformity + interpretability stack under `src/aam/experiments/olmo_conformity/`:
- `runner.py`: suite runner (datasets x conditions x models) + optional activation capture.
- `vector_analysis.py` + `probes.py`: capture probe datasets, train per-layer logistic probes, compute projections.
- `logit_lens.py`: unembed captured vectors into top-k tokens (best-effort), plus think-token parsing.
- `intervention.py`: activation steering via forward hooks (social-vector subtraction) and evaluation.
- `analysis.py` + `src/aam/analytics/*`: plots/tables and derived metrics.
- `aam.analytics.reporting.ScientificReportGenerator`: scientific validity checks (eg dual-stack risk).

CLI entrypoints:
- Defined in `src/aam/run.py`
- Runnable as `vvm ...` (if installed) or `python -m aam ...` (recommended for reproducibility on HPC).

### 1.1 Important constraint (scientific validity + current CLI shape)
Many of the existing single-run analysis commands implicitly assume:
- a single `model_id` per run directory, and
- a single probe artifact per run.

However, the provided suite configs (eg `experiments/olmo_conformity/configs/suite_expanded_temp0.0.json`) include **multiple variants per run**.

To avoid scientific validity issues and tool mismatches, this protocol uses:
- **One run per (variant, temperature, seed, suite)** for all mechanistic analyses (probes, interventions, logit lens).

We can still run multi-variant "overview" runs for quick behavioral exploration, but they are not sufficient for a mechanistic, causal package unless we add variant-aware tooling.

### 1.2 Suite configs we will use (and why)

We will treat suite configs as experimental "protocol definitions". The repo already contains good templates:

Behavioral-only, validity-focused (recommended for sanity + some ablations):
- `experiments/olmo_conformity/configs/suite_fixed_behavioral.json`
  - Datasets: immutable facts only
  - Conditions: control + asch_history_5 + authoritative_bias
  - Notes: explicitly removes probe-capture conditions and includes scientific validity notes.

Paper-aligned minimal suite (matches the structure referenced in `paper/paper.tex`):
- `experiments/olmo_conformity/configs/suite_complete_temp0.json`
- `experiments/olmo_conformity/configs/suite_complete_temp0.5.json`
- `experiments/olmo_conformity/configs/suite_complete_temp1.json`
  - Datasets: immutable facts (20) + social conventions (20)
  - Conditions: includes probe-capture entries, but the paper analysis should focus on behavioral conditions only.
  - Recommendation: for efficient compute, generate a "behavioral-only" copy that removes probe-capture conditions.

Expanded domain coverage (benchmarks + opinions + immutable facts):
- `experiments/olmo_conformity/configs/suite_expanded_temp0.0.json` ... `suite_expanded_temp1.0.json`
  - Datasets: 8 categories
  - Conditions: behavioral only (control + asch + authority)
  - Temperature sweep: 0.0..1.0 step 0.2

Expected trial counts (variant-only runs):
- suite_complete behavioral-only: 2 datasets * 20 items * 3 conditions = 120 behavioral trials per run.
- suite_expanded: 8 datasets * 30 items * 3 conditions = 720 behavioral trials per run.
Add-on trials from interpretability workflow per run:
- vector-analysis probe capture adds ~150 trials (100 truth + 50 social) per run (per variant).

---

## 2) Literature We Build On (Recent, High-Quality, and Mechanistic)

This protocol links each major design choice to a recent body of work.

### 2.1 Sycophancy and conformity as an alignment failure
- Sharma et al., "Towards Understanding Sycophancy in Language Models" (ICLR 2024): defines sycophancy as systematic and measurable; motivates condition design and measurement framing.  
  https://openreview.net/forum?id=TVY6GoRDzG

### 2.2 Mechanistic explanations of "truth vs social" conflict (directly relevant)
- Shah et al., "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models" (arXiv 2025): motivates mechanistic signatures for sycophancy; our suite adds temperature sweeps, training-stage comparisons, and causal interventions in one unified protocol.  
  https://arxiv.org/abs/2508.02087
- Kim et al., "Sycophancy is Not One Thing: Causal Separation of 'Social Sycophancy' and 'User Sycophancy'" (arXiv 2025): motivates separating social-context conformity (Asch-like) from user-assertion agreement (authoritative bias).  
  https://arxiv.org/abs/2505.13995

### 2.3 Interventions / activation steering as causal validation
- Stolfo et al., "Improving Instruction Following in Language Models through Activation Steering" (ICLR 2025): validates activation steering as a causal tool; we apply the same logic to pressure-induced conformity as a safety-relevant behavior.  
  https://openreview.net/forum?id=Fc9l3L6tK1

### 2.4 Truthfulness control and sycophancy mitigation (context and comparisons)
- Wang et al., "TruthFlow: Controllable Truthfulness for Large Language Models" (ICML 2025): motivates explicit truth control objectives; we provide an activation-level (rather than parameter-level) mechanistic approach.  
  https://proceedings.mlr.press/v267/wang25a.html
- Panickssery et al., "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning" (ICML 2024): motivates targeted interventions and careful ablations; our activation-level intervention battery provides a complementary causal story.  
  https://proceedings.mlr.press/v235/panickssery24a.html

### 2.5 Interpretability validity requirements (why the ablations matter)
- Goldowsky-Dill et al., "Causal Abstraction for Mechanistic Interpretability" (JMLR 2025): motivates causal/counterfactual tests and strong validity criteria; our protocol treats interventions and negative controls as first-class.  
  https://jmlr.org/papers/v26/23-1377.html

### 2.6 Why OLMo-3 is the correct "glass box" family for this protocol
- AI2 OLMo-3 release and checkpoint collection: provides a family of checkpoints spanning training stages. This matches the motivation in `paper/paper.tex`.  
  https://allenai.org/blog/olmo3  
  https://huggingface.co/collections/allenai/olmo-3-models-675c246e6e4e85206067a84f

---

## 3) Research Questions (RQs) and Hypotheses (Hs)

### RQ1 (training stages)
How do training stages (base vs instruct vs SFT vs think vs RL-Zero) change susceptibility to social pressure, across domains and temperatures?

### RQ2 (temperature as "thermodynamics")
How does temperature reshape conformity curves, and can the non-linearity in `paper/paper.tex` be explained mechanistically (via internal signal competition)?

### RQ3 (mechanistic signature)
Is there a stable, layerwise signature of truth vs social competition that predicts conformity and its temperature dependence?

### RQ4 (causal)
If we subtract a learned social direction at the right layers, do we causally reduce conformity (increase truthfulness) without broadly breaking performance?

Primary hypotheses:
- H1: Pressure conditions increase the social projection (SVP) and decrease the truth projection (TVP) relative to control on factual tasks.
- H2: There exists a collision layer (first layer where SVP > TVP) whose depth depends on (variant, temperature, pressure type).
- H3: Interventions at or near the collision layer increase flip-to-truth rates more than interventions far from it.
- H4: The SVP - TVP gap predicts trial-level conformity, and helps explain temperature-dependent reversals in specific variants (eg RL-Zero in `paper/paper.tex`).

---

## 4) Dataset Inventory (Repo-Saved Benchmark Samples)

Counts below reflect the current repo contents.

### 4.1 Behavioral datasets (wrong-answer variants, used for pressure manipulations)
- `experiments/olmo_conformity/datasets/immutable_facts/minimal_items_wrong.jsonl` (20 items, factual, has ground truth)
- `experiments/olmo_conformity/datasets/social_conventions/minimal_items_wrong.jsonl` (20 items, opinion/preference, no ground truth)
- `experiments/olmo_conformity/datasets/math/gsm8k_items_wrong.jsonl` (50 items, factual)
- `experiments/olmo_conformity/datasets/reasoning/arc_items_wrong.jsonl` (50 items, factual)
- `experiments/olmo_conformity/datasets/science/mmlu_science_items_wrong.jsonl` (50 items, factual)
- `experiments/olmo_conformity/datasets/knowledge/mmlu_knowledge_items_wrong.jsonl` (50 items, factual)
- `experiments/olmo_conformity/datasets/truthfulness/truthfulqa_items_wrong.jsonl` (50 items, factual-ish; wrong answer is a misconception)

### 4.2 Probe training datasets (used only for probe training, not for behavioral eval)
- `experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl` (100 labeled items)
- `experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl` (50 labeled items)

---

## 5) Experimental Factors and the Full Matrix

### 5.1 Model variants (Olmo-3 family; already in suite configs)
Variant -> model_id:
- base -> `allenai/Olmo-3-1025-7B`
- instruct -> `allenai/Olmo-3-7B-Instruct`
- instruct_sft -> `allenai/Olmo-3-7B-Instruct-SFT`
- think -> `allenai/Olmo-3-7B-Think`
- think_sft -> `allenai/Olmo-3-7B-Think-SFT`
- rl_zero -> `allenai/Olmo-3-7B-RL-Zero-Math`

### 5.2 Temperature grid (primary and expanded)
Primary (paper-aligned):
- T in {0.0, 0.5, 1.0}

Expanded sweep (already present in configs + scripts):
- T in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
Configs: `experiments/olmo_conformity/configs/suite_expanded_temp{T}.json`

### 5.3 Pressure conditions (primary + ablations)
Primary conditions (paper-aligned):
- control
- synthetic_asch_history (default: confederates=5, confidence=high)
- authoritative_bias (default: user_claim_strength=high)

Pressure-strength ablations (supported today by condition params in `src/aam/experiments/olmo_conformity/runner.py`):
- Asch confederates: {1, 3, 5, 7, 9}
- Asch confidence: {low, high}
- Authoritative claim strength: {low, high}

Prompt artifacts for reproducibility:
- `experiments/olmo_conformity/prompts/synthetic_asch/asch_system.txt`
- `experiments/olmo_conformity/prompts/synthetic_asch/asch_user_template.txt`
- `experiments/olmo_conformity/prompts/controls/control_system.txt`

### 5.4 Randomness / seeds (tough-reviewer requirement)
For T > 0, we must quantify sampling variance.

Replicate each (variant, temperature, suite) at:
- Seeds = {41, 42, 43, 44, 45}

Rationale: the runner passes a fixed seed into each generation; each run is deterministic given (model, temperature, seed), so replicates are clean and reproducible.

### 5.5 Activation capture plan (what we capture, where it lands, and why)
For mechanistic analyses we must capture activations during trial generation.

Default capture (sufficient for probes + logit-lens + interventions in this repo):
- Layers: all 32 decoder layers (0..31 for Olmo-3)
- Component: `resid_post` (stored as `hook_resid_post` in the DB index)
- Token position: -1 (last token of the most recent forward pass; see `CaptureContext.record_activation`)
- Dtype: float16

Artifacts (per run directory):
- Raw shards: `runs/<...>/activations/step_<time_step>.safetensors`
- DB index: `activation_metadata` table inside `runs/<...>/simulation.db`

Storage expectations (order-of-magnitude):
- One float16 vector per (layer, component) per trial. If the model dimension is ~4096 and we capture 32 layers, this is on the order of a few hundred KB per trial.
- For expanded runs (720 trials) this is typically in the low-GB range per (variant, temp, seed) run.

Why resid_post:
- It is a standard, stable hook point for representation analysis and aligns with existing probe and intervention code paths.

Optional deep-capture (only for a small subset of runs; increases storage and complexity):
- Add `resid_pre` to capture both pre and post residual stream representations.
- Add attention components if needed (HF gateway supports `attn.hook_q`, `attn.hook_k`, `attn.hook_v`, `attn.hook_result`, `attn.hook_pattern` when requested via CaptureContext hook names).

---

## 6) Metrics (Behavioral, Mechanistic, Causal) With Definitions

### 6.1 Behavioral metrics on factual tasks (ground truth available)
Define the factual subset as items where `ground_truth_text IS NOT NULL`.

Per (variant, temperature, dataset_category, condition):
- Accuracy = mean(is_correct)
- ErrorRate = 1 - Accuracy

Social Pressure Effect (SPE) on factual tasks:
- SPE_Asch = ErrorRate_Asch - ErrorRate_Control
- SPE_Authority = ErrorRate_Authority - ErrorRate_Control

Truth-Override Rate (TOR) (sycophancy rate; reviewer-friendly):
- Among items where control is correct, TOR is the fraction that become incorrect under a pressure condition.

Repo support:
- `src/aam/analytics/behavioral.py` already computes sycophancy-like rates on factual trials.

### 6.2 Behavioral metrics on opinion tasks (no ground truth)
Opinion subset: items where `ground_truth_text IS NULL` (eg social_conventions).

Metric we need (currently TODO in repo analytics):
- Pressure Agreement Rate (PAR): fraction of trials whose parsed answer matches the confederate-suggested answer (the item's `wrong_answer`).

Repo status:
- The dataset stores `wrong_answer`.
- `src/aam/analytics/behavioral.py` has `pressure_agreement_rate = []  # TODO`.
Action item: implement PAR before final paper submission. (Not currently implemented.)

### 6.3 Mechanistic metrics (probes and projections)
We operationalize two internal signals as linear probes trained on separate labeled corpora:
- Truth probe: trained on `truth_probe_train.jsonl`.
- Social probe: trained on `social_probe_train.jsonl`.

For each layer L we learn (w_L, b_L) and compute:
- Projection_L(x) = dot(w_L, x_L) + b_L

Define:
- TVP_L = mean(Projection_L) for truth probe
- SVP_L = mean(Projection_L) for social probe
- Gap_L = SVP_L - TVP_L

Collision layer per trial:
- collision(trial) = min { L : SVP_L(trial) > TVP_L(trial) } (if any)

Repo support:
- Training and projections: `src/aam/experiments/olmo_conformity/probes.py`
- Workflow: `src/aam/experiments/olmo_conformity/vector_analysis.py`
- Plots: `src/aam/analytics/probes.py`

### 6.4 Causal metrics (interventions)
We run activation steering using the learned social direction:
- At selected layers L in S, replace hidden state h with h - alpha * v_social_L

Metrics:
- FlipToTruth rate: fraction of pressure trials that go from incorrect -> correct after intervention.
- DeltaAccuracy = Accuracy_after - Accuracy_before.

Negative controls (tough-reviewer requirement):
- Random-vector control: subtract a random unit vector of the same dimension (requires a small code extension; not currently in repo).
- Layer controls: apply at early layers and late layers far from collision to confirm localization.

### 6.5 Temperature-dependent mechanistic metric (new; requires small analysis extension)
To connect "thermodynamics" directly to mechanism, we want a token-level quantity:
- LogitGap = logit(correct_answer_token) - logit(wrong_answer_token)

Under temperature sampling:
- p_wrong ~= softmax([logit_correct/T, logit_wrong/T])[wrong]

Repo status:
- `src/aam/experiments/olmo_conformity/logit_lens.py` can unembed captured residual vectors into token distributions and store top-k.
- It does not yet compute explicit logits for (correct, wrong) answer tokens.
Action item: add a targeted logit-gap extractor for single-token answers (or first-token approximations) to support the mechanistic temperature model.

### 6.6 Statistical analysis plan (reviewer-ready, minimal researcher degrees of freedom)
We will treat this as a "pre-registered style" analysis to prevent cherry-picking.

Unit of analysis:
- A single trial (one item under one condition for one (variant, temperature, seed)).

Primary behavioral model (factual tasks only):
- Outcome: `is_correct` (Bernoulli)
- Fixed effects: condition, temperature, variant, and key interactions (temperature x condition, condition x variant, temperature x variant).
- Random effects: random intercept for `item_id` (and optionally for `seed` if we pool seeds into one fit).
- Report: odds ratios + 95% CIs, plus simple effect sizes (eg Cohen's h on aggregated proportions) for interpretability.

Temperature modeling:
- For the primary paper story: treat temperature as categorical (T=0, T=0.5, T=1) to match `paper/paper.tex`.
- For the expanded sweep: treat temperature as continuous and fit either:
  - a linear + quadratic model, or
  - a piecewise model with a knot at 0.5 (motivated by the non-linearity described in the paper draft).

Opinion-task model (if/when PAR is implemented):
- Outcome: agreement with confederate suggestion (Bernoulli)
- Same fixed/random effect structure, but interpret as "social alignment" rather than truth suppression.

Mechanistic models:
- TVP/SVP/Gaps: linear mixed models with random intercepts for item_id (and optionally trial_id), fixed effects as above.
- Collision layer: treat as an ordered outcome or a time-to-event (first layer where SVP > TVP); report shifts by condition/temperature.

Multiple comparisons:
- Predefine a small set of primary comparisons (eg RL-Zero vs Instruct under Asch at T=1; control vs Asch within each variant at T endpoints).
- Use Benjamini-Hochberg FDR control across exploratory comparisons (eg across all variants x temperatures x datasets) and clearly label exploratory vs confirmatory results.

---

## 7) Threats to Validity and How We Address Them (Reviewer Checklist)

### 7.1 Answer leakage (invalid pressure manipulation)
Mitigation: runner fails fast if `wrong_answer` is missing or equals `ground_truth_text` (see `_get_wrong_answer` in `src/aam/experiments/olmo_conformity/runner.py`).

### 7.2 Probe leakage / cross-model leakage
Mitigation:
- Train probes only on the probe datasets, not on behavioral trials.
- Use single-variant runs for mechanistic analysis so probes are never applied cross-model.

### 7.3 Dual-stack risk (different weights used for different stages)
Mitigation:
- Prefer a single inference stack per run (HF gateway) for both capture and intervention.
- Use `ScientificReportGenerator` outputs as a pre-submission validity gate.

### 7.4 Sampling variance mistaken for temperature effect
Mitigation:
- 5-seed replicates for each T > 0.
- Report confidence intervals and include seed as a random effect in statistical models.

### 7.5 Prompt artifacts / "Please answer correctly" confound
Mitigation:
- Keep prompts fixed in the main study (to match the paper draft).
- Add a prompt-ablation suite that removes the "Please answer correctly" line (requires prompt-file variants; not currently in repo).

### 7.6 Ablation checklist (what a tough reviewer will ask for)
The goal of this checklist is to ensure we can answer "is this effect robust, specific, and mechanistically grounded?"

Behavioral robustness:
- A1: Seed replicates at each T > 0 (already mandated).
- A2: Dataset-category breakdown (general, math, science, knowledge, truthfulness, reasoning, opinion).
- A3: Pressure-strength sweep (confederates count and confidence; authoritative strength).

Prompt robustness:
- A4: Remove / modify the "Please answer the question with the correct answer." instruction (requires prompt variants).
- A5: Swap order of confederate statements; include mixed-majority conditions (eg 3 wrong, 2 correct) (requires new condition type).

Mechanistic robustness:
- A6: Token-position ablation for probes (eg -1 vs first generated token) (requires capture logic change; currently capture is last forward pass).
- A7: Component ablation (resid_pre vs resid_post).
- A8: Cross-temperature probe reuse: train probes at T=0 and apply at T=1 (should still be meaningful if representations are stable).

Causal robustness:
- A9: Intervention layer sweep (early vs mid vs late) and alpha dose-response curves.
- A10: Random-vector control intervention (requires small code extension).
- A11: Cross-variant vector transfer (apply a social vector learned on one variant to another) as a negative control for representation-space mismatch.

We will implement A1-A3 and A9 immediately. Others are staged once initial results indicate where the signal is strongest.

---

## 8) Full Run Protocol (What to Run for the Complete Picture)

We will run three tiers: sanity, paper-replication, full ablation suite.

### 8.1 Prereqs (local or HPC)
Install deps:
```bash
uv sync --extra cognitive --extra interpretability --extra analysis
```

Make CLI available:
```bash
export PYTHONPATH=src
```

HPC paths (optional):
- `experiments/olmo_conformity/configs/paths.json` contains scratch paths.
- You can override with env vars:
  - `AAM_MODELS_DIR=/path/to/hf_cache`
  - `AAM_RUNS_DIR=/path/to/runs`

### 8.2 Sanity gate (must pass before expensive sweeps)
Goal: verify end-to-end capture -> probes -> projections -> plots works on one small run.

1) Create a minimal variant-only sanity suite config (single model, few items).

This avoids multi-variant confusion and makes the probe/projection pipeline strictly well-defined.

```bash
python - <<'PY'
import json, pathlib
src = pathlib.Path("experiments/olmo_conformity/configs/suite_fixed_behavioral.json")
dst = pathlib.Path("experiments/olmo_conformity/configs/generated/sanity_base_t0_seed42.json")
dst.parent.mkdir(parents=True, exist_ok=True)
cfg = json.loads(src.read_text())
cfg["suite_name"] = "sanity_base_t0_seed42"
cfg["models"] = [m for m in cfg["models"] if m["variant"] == "base"]
cfg["run"]["temperature"] = 0.0
cfg["run"]["seed"] = 42
cfg["run"]["max_items_per_dataset"] = 5
dst.write_text(json.dumps(cfg, indent=2))
print(f"Wrote {dst}")
PY
```

2) Run the suite (activation capture enabled):
```bash
python -m aam olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/generated/sanity_base_t0_seed42.json \
  --runs-dir runs \
  --capture-activations \
  --capture-components resid_post \
  --capture-dtype float16
```

3) Run vector analysis (truth + social probes + projections):
```bash
python -m aam olmo-conformity-vector-analysis \
  --run-id <RUN_ID> \
  --db runs/<timestamp>_<RUN_ID>/simulation.db \
  --model-id <MODEL_ID_FROM_DB> \
  --truth-probe-dataset experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl \
  --social-probe-dataset experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --component hook_resid_post \
  --token-position -1 \
  --dtype float16 \
  --artifacts-dir runs/<timestamp>_<RUN_ID>/artifacts
```

4) Generate the report figures/tables:
```bash
python -m aam olmo-conformity-report \
  --run-id <RUN_ID> \
  --db runs/<timestamp>_<RUN_ID>/simulation.db \
  --run-dir runs/<timestamp>_<RUN_ID>
```

Only once this passes do we proceed to the full suites.

### 8.3 Paper replication (behavioral core; then mechanistic extension)
This replicates the structure used in `paper/paper.tex`:
- Variants: all 6
- Temperatures: 0.0, 0.5, 1.0
- Datasets: minimal (immutable_facts_minimal + social_conventions_minimal)
- Conditions used for analysis: control, asch_history_5, authoritative_bias

Use the existing suite configs as templates:
- `experiments/olmo_conformity/configs/suite_complete_temp0.json`
- `experiments/olmo_conformity/configs/suite_complete_temp0.5.json`
- `experiments/olmo_conformity/configs/suite_complete_temp1.json`

For mechanistic work, generate variant-only copies (single model per config). Example generator for one temp config:
```bash
python - <<'PY'
import json, pathlib
src = pathlib.Path("experiments/olmo_conformity/configs/suite_complete_temp0.json")
dst_dir = pathlib.Path("experiments/olmo_conformity/configs/generated")
dst_dir.mkdir(parents=True, exist_ok=True)
cfg = json.loads(src.read_text())
models = cfg["models"]
for m in models:
    out = dict(cfg)
    out["suite_name"] = f"{cfg['suite_name']}__{m['variant']}"
    out["models"] = [m]
    (dst_dir / f"{out['suite_name']}.json").write_text(json.dumps(out, indent=2))
print(f"Wrote {len(models)} configs to {dst_dir}")
PY
```

Bulk generator (all three paper temperatures, all variants, and seed replicates; also removes probe-capture conditions):
```bash
python - <<'PY'
import json, pathlib

SUITES = [
  "experiments/olmo_conformity/configs/suite_complete_temp0.json",
  "experiments/olmo_conformity/configs/suite_complete_temp0.5.json",
  "experiments/olmo_conformity/configs/suite_complete_temp1.json",
]
BEHAVIORAL_CONDS = {"control", "asch_history_5", "authoritative_bias"}
SEEDS_T0 = [42]
SEEDS_TPOS = [41, 42, 43, 44, 45]

dst_dir = pathlib.Path("experiments/olmo_conformity/configs/generated/paper_complete")
dst_dir.mkdir(parents=True, exist_ok=True)

for src_path in SUITES:
  src = pathlib.Path(src_path)
  cfg = json.loads(src.read_text())

  # Keep only behavioral conditions (probe capture is handled by vector analysis).
  cfg["conditions"] = [c for c in cfg.get("conditions", []) if c.get("name") in BEHAVIORAL_CONDS]

  temp = float(cfg.get("run", {}).get("temperature", 0.0))
  seeds = SEEDS_T0 if temp == 0.0 else SEEDS_TPOS

  for m in cfg["models"]:
    for seed in seeds:
      out = dict(cfg)
      out["suite_name"] = f"{cfg['suite_name']}__{m['variant']}__seed{seed}"
      out["models"] = [m]
      out["run"] = dict(cfg.get("run", {}))
      out["run"]["seed"] = int(seed)
      (dst_dir / f"{out['suite_name']}.json").write_text(json.dumps(out, indent=2))

print(f"Wrote configs to {dst_dir}")
PY
```

Run matrix recommendation:
- For T=0.0: run 1 seed (deterministic)
- For T in {0.5, 1.0}: run 5 seeds (41..45)

Per run, execute:
1) `python -m aam olmo-conformity ... --capture-activations ...`
2) `python -m aam olmo-conformity-vector-analysis ...`
3) `python -m aam olmo-conformity-report ...`

Automation tip (recommended for running many configs reproducibly):

Run a single config and immediately follow with vector analysis + report (bash pattern):
```bash
cfg="experiments/olmo_conformity/configs/generated/paper_complete/<SUITE_NAME>.json"
out="$(python -m aam olmo-conformity --suite-config "$cfg" --runs-dir runs --capture-activations --capture-components resid_post --capture-dtype float16)"
run_dir="$(echo "$out" | sed -n 's/^run_dir=//p')"
run_id="$(echo "$out" | sed -n 's/^run_id=//p')"
db_path="$(echo "$out" | sed -n 's/^db=//p')"

python -m aam olmo-conformity-vector-analysis \
  --run-id "$run_id" \
  --db "$db_path" \
  --model-id "$(python - <<'PY'\nimport sqlite3,sys\nconn=sqlite3.connect(sys.argv[1])\nrun_id=sys.argv[2]\nprint(conn.execute('SELECT DISTINCT model_id FROM conformity_trials WHERE run_id=? LIMIT 1',(run_id,)).fetchone()[0])\nPY "$db_path" "$run_id")" \
  --truth-probe-dataset experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl \
  --social-probe-dataset experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --component hook_resid_post \
  --token-position -1 \
  --dtype float16 \
  --artifacts-dir "$run_dir/artifacts"

python -m aam olmo-conformity-report \
  --run-id "$run_id" \
  --db "$db_path" \
  --run-dir "$run_dir"
```

Optional orchestrator (useful for small suites only):
- `python -m aam olmo-conformity-full` can run trials -> probes -> reports end-to-end.
- For large suites, prefer manual control so interventions/logit-lens are only run on targeted batteries.

### 8.4 Full expanded suite (domain coverage + dense temperature sweep)
This is the "complete picture" suite reviewers will ask for:
- Temperatures: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- Datasets: expanded (8 categories, max_items_per_dataset=30)
- Conditions: control, asch_history_5, authoritative_bias
- Variants: all 6, but run as variant-only configs for mechanistic validity

The repo already has a temperature sweep script:
- `experiments/olmo_conformity/configs/run_expanded_experiments.py`

Bulk generator (expanded sweep temperatures, all variants, seed replicates):
```bash
python - <<'PY'
import json, pathlib, re

cfg_dir = pathlib.Path("experiments/olmo_conformity/configs")
src_paths = sorted(cfg_dir.glob("suite_expanded_temp*.json"))

def _temp_from_name(p: pathlib.Path) -> float:
  m = re.search(r"temp([0-9.]+)\\.json$", p.name)
  return float(m.group(1)) if m else 0.0

SEEDS_T0 = [42]
SEEDS_TPOS = [41, 42, 43, 44, 45]

dst_dir = pathlib.Path("experiments/olmo_conformity/configs/generated/expanded_sweep")
dst_dir.mkdir(parents=True, exist_ok=True)

for src in src_paths:
  cfg = json.loads(src.read_text())
  temp = float(cfg.get("run", {}).get("temperature", _temp_from_name(src)))
  seeds = SEEDS_T0 if temp == 0.0 else SEEDS_TPOS

  for m in cfg["models"]:
    for seed in seeds:
      out = dict(cfg)
      out["suite_name"] = f"{cfg['suite_name']}__{m['variant']}__seed{seed}"
      out["models"] = [m]
      out["run"] = dict(cfg.get("run", {}))
      out["run"]["seed"] = int(seed)
      (dst_dir / f"{out['suite_name']}.json").write_text(json.dumps(out, indent=2))

print(f"Wrote configs to {dst_dir}")
PY
```

That script currently runs multi-variant configs. For this protocol, either:
- extend the script to iterate variants and use variant-only configs (recommended), or
- run a manual loop per generated config.

Minimum required outputs per run:
1) trials + activation capture
2) probes + projections (vector analysis)
3) report figures/tables

### 8.5 Interventions and logit lens (targeted, because expensive)
Interventions require re-generating text and are the most compute-intensive part. We do not run them for every (variant, T, suite) combination. Instead, we define two targeted intervention batteries.

Battery I (paper-aligned, minimal datasets, full temperature sweep):
- Dataset subset: immutable_facts_minimal only (20 items)
- Conditions: pressure-only (asch_history_5 + authoritative_bias)
- Temperatures: 0.0, 0.5, 1.0
- Variants: all 6
- Layers: use collision-layer results to choose a small band (eg 12..20)
- Alphas: {0.5, 1.0, 2.0}

Battery II (stress test, expanded factual benchmarks, endpoints only):
- Dataset subset: {truthfulqa, gsm8k, arc} (repo samples)
- Temperatures: endpoints only {0.0, 1.0}
- Variants: {base, instruct, rl_zero}
- Same alpha/layer sweep

Getting the required identifiers (model_id, probe_id, probe_path):

1) model_id for a variant-only run:
```bash
python - <<'PY'
import sqlite3, sys
db = sys.argv[1]
run_id = sys.argv[2]
conn = sqlite3.connect(db)
cur = conn.execute("SELECT DISTINCT model_id FROM conformity_trials WHERE run_id = ?;", (run_id,))
rows = [r[0] for r in cur.fetchall()]
print("model_ids:", rows)
PY runs/<timestamp>_<RUN_ID>/simulation.db <RUN_ID>
```

2) latest social probe for that run (after running vector analysis):
```bash
python - <<'PY'
import sqlite3, sys
db = sys.argv[1]
run_id = sys.argv[2]
conn = sqlite3.connect(db)
cur = conn.execute(
  "SELECT probe_id, artifact_path FROM conformity_probes WHERE run_id = ? AND probe_kind = 'social' ORDER BY created_at DESC LIMIT 1;",
  (run_id,),
)
row = cur.fetchone()
print("social_probe_id:", row[0] if row else None)
print("social_probe_path:", row[1] if row else None)
PY runs/<timestamp>_<RUN_ID>/simulation.db <RUN_ID>
```

Intervention command template:
```bash
python -m aam olmo-conformity-intervene \
  --run-id <RUN_ID> \
  --db runs/<timestamp>_<RUN_ID>/simulation.db \
  --model-id <MODEL_ID> \
  --probe-path runs/<timestamp>_<RUN_ID>/artifacts/social_probe.safetensors \
  --social-probe-id <SOCIAL_PROBE_ID> \
  --layers 15,16,17,18,19,20 \
  --alpha 0.5,1.0,2.0 \
  --max-new-tokens 64
```

Logit-lens command template (trial-level top-k by layer):
```bash
python -m aam olmo-conformity-logit-lens \
  --run-id <RUN_ID> \
  --db runs/<timestamp>_<RUN_ID>/simulation.db \
  --model-id <MODEL_ID> \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --k 20
```

### 8.6 Per-run paper figures/tables (regeneration and auditability)
The repo includes standalone "paper figure" regeneration scripts under `Analysis Scripts/`.

For any single run directory (after trials + probes + interventions are populated), run:
```bash
python "Analysis Scripts/generate_all_figures.py" --run-dir runs/<timestamp>_<RUN_ID>
python "Analysis Scripts/generate_all_tables.py" --run-dir runs/<timestamp>_<RUN_ID>
```

Outputs (by convention):
- figures: `runs/<...>/artifacts/figures/`
- tables/logs: `runs/<...>/artifacts/tables/` and `runs/<...>/artifacts/logs/`

### 8.7 Cross-temperature composites (T=0, T=0.5, T=1)
To generate composite, cross-temperature figures across the three paper temperatures, use:
- `Analysis Scripts/generate_cross_temperature_figures.py`

This script currently has a `RUNS = {...}` dictionary mapping {0.0, 0.5, 1.0} to run directories and run IDs. For your experiment, update that mapping to the three runs you want to compare, then run:
```bash
python "Analysis Scripts/generate_cross_temperature_figures.py" --out-dir cross_temp_analysis
```

Expected precondition: the selected runs already contain:
- behavioral trial outputs for the three behavioral conditions
- probe projections in `conformity_probe_projections`
- intervention results in `conformity_intervention_results` (if you want the intervention composite)

Note: the repo already contains an example `cross_temp_analysis/` output directory, which is useful as a reference for expected artifacts.

---

## 9) What We Expect to Learn (The "Compelling Story" Outcomes)

### Outcome A: training-stage fingerprint
Each Olmo variant yields a distinct behavioral + mechanistic signature:
- instruct vs SFT vs RL-Zero differ not only in error rates, but in where and how SVP competes with TVP across layers.

### Outcome B: temperature curve explained mechanistically
The non-linear temperature behavior in `paper/paper.tex` is explained by:
- changes in the (truth vs wrong) logit gap under pressure (mechanistic),
- amplified into sampling differences by temperature (thermodynamic).

### Outcome C: causal leverage
If interventions reduce conformity specifically under pressure without destroying control accuracy, we have strong evidence that:
- the learned social direction is causally implicated in conformity behavior,
- and the mechanistic picture is not just correlational.

---

## 10) Implementation Gaps (Explicitly Marked)

These are the key missing pieces for a full tough-reviewer package:
- Pressure Agreement Rate (opinion tasks) is TODO in `src/aam/analytics/behavioral.py`.
- The logit-gap temperature model requires a small analysis extension (not currently in repo).
- Random-vector intervention control is not implemented (small extension).
- Variant-aware posthoc analysis utilities are limited; this protocol avoids the issue by using variant-only runs.

These should be treated as pre-submission checklist items after initial results are in.
