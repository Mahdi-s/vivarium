# SAE Upgrade Prompt: From “Turn Layer” (Linear Probes) → “Feature Dominance Layer” (SAEs)

**Theme:** Validating the “Turn Layer” claim by moving beyond linear probes, and strengthening the mechanistic story to be NeurIPS/ICLR‑ready.  
**Primary deliverable:** a revamped **Mechanistic** section for `paper/paper.tex` that replaces “Turn Layer” with **Feature Dominance Layer** and grounds it in **Sparse Autoencoder (SAE)** features.

---

## 0) Context (what exists today in this repo)

We currently operationalize a “truth vs social” tug‑of‑war using **per‑layer linear probes** on `resid_post` (last token) and define:

- **TVP(ℓ)**: truth‑probe logit at layer ℓ  
- **SVP(ℓ)**: social‑probe logit at layer ℓ  
- **Turn Layer**: first layer where `SVP > TVP`

Key touchpoints:
- Paper section to replace/upgrade: `paper/paper.tex` (Mechanistic subsection + metric definition).
- Mechanistic guide: `paper/MECHANISTIC_INTERPRETABILITY_GUIDE.md`.
- Probe pipeline: `src/aam/experiments/olmo_conformity/probes.py`.
- Captured activations (expanded sweep): `runs-hpc-full/probe/runs/*/activations/*.safetensors` + `simulation.db`.

Why reviewers will push back:
- Linear probes can be **prefix detectors** (esp. social framing).
- Two probes have **scale/calibration mismatch**; crossing heuristics can be brittle.
- Current story is largely **correlational** without feature‑level specificity or causal tests.

---

## 1) Objective (SAE upgrade)

Replace linear probes with **Sparse Autoencoders** (prefer SAELens) trained on OLMo‑3 residual streams to discover *actual features* rather than probe directions.

Core mechanistic hypothesis to validate/upgrade:
- Under Asch/authority prompts, **social compliance features** become active and compete with **truth features**.
- A depth‑localized transition exists where **social features dominate**, predicting (and ideally causally influencing) **behavioral flips**.

Concrete experiment requirements:
1) Train SAEs on OLMo‑3 residual streams (initially `resid_post`; consider prompt‑end capture as an upgrade).
2) Identify **social compliance latents**: activate in Asch pressure prompts, not in control prompts (and ideally generalize to new pressure templates).
3) Verify **distinctness** from truth latents (low overlap / low correlation / different logit effects).
4) Quantify prediction: correlate “social latent magnitude” with **flip probability** (e.g., truth override or wrong‑answer flip).
5) Write the revamped paper section, renaming the key statistic to **Feature Dominance Layer**.

---

## 2) Definitions we will use in the paper (proposed)

### 2.1 Social latents vs truth latents
We will define sets of SAE latents per layer:
- **Social set** `S_ℓ`: latents with high selectivity for pressure prompts (Asch / Authority) vs control.
- **Truth set** `T_ℓ`: latents with high selectivity for truthfulness / correctness (truth‑probe dataset, or control‑correct vs control‑incorrect trials).

Selectivity can be operationalized via (choose 1–2 and report robustness):
- standardized mean difference (Cohen’s *d*)
- AUROC for pressure vs control classification using single latent
- mutual information estimates / entropy reduction

### 2.2 Feature Dominance Score and Feature Dominance Layer
For each trial *i* and layer ℓ:
- Let `a_{i,ℓ} ∈ R^k` be SAE latent activations (post‑ReLU / post‑threshold, consistent with SAE training).
- Define **social mass** `M^soc_{i,ℓ} = Σ_{j∈S_ℓ} |a_{i,ℓ,j}|`
- Define **truth mass** `M^tru_{i,ℓ} = Σ_{j∈T_ℓ} |a_{i,ℓ,j}|`
- Define **dominance** `D_{i,ℓ} = M^soc_{i,ℓ} − M^tru_{i,ℓ}`

Then define:
- **Feature Dominance Layer (FDL)** for trial *i*: `min{ ℓ : D_{i,ℓ} > 0 }`

Alternative (possibly stronger) variant if we can compute token‑specific logit effects:
- Replace mass with **logit contributions** toward (wrong answer tokens) vs (correct answer tokens).

---

## 3) Candidate solution paths (pick one as the “mainline”)

Below are *distinct end‑to‑end approaches* with an explicit probability distribution over “this yields a reviewer‑resistant mechanistic section on our timeline”.

### Path A — “Fast retrofit” SAE on **existing captured vectors** (lowest disruption)
**Idea:** Train SAEs offline on the already‑captured per‑trial vectors (likely one vector per trial per layer), then do selectivity + flip‑prediction.

- **Pros:** no new inference runs; fastest iteration; integrates cleanly with existing DB + safetensors.
- **Cons:** dataset may be too small / too biased (single vector per trial); latents risk encoding *prompt string artifacts*; weaker generalization story.
- **Probability of success:** **0.22**

### Path B — “Prompt‑end capture + delta‑SAE” (best signal‑to‑noise for “social pressure”)
**Idea:** Modify capture to record activations at **end of prompt (pre‑generation)** for both Control and Asch, then train SAE on **deltas**:  
`Δa_{i,ℓ} = a^{asch}_{i,ℓ} − a^{control}_{i,ℓ}` (matched by item/variant/temp/seed).

- **Pros:** isolates “what pressure adds” vs general content; reduces trivial string‑detector confounds; makes “social compliance feature” definition sharper; aligns naturally with Feature Dominance.
- **Cons:** requires re‑running activation capture (or updating capture hooks); extra compute; careful matching logic.
- **Probability of success:** **0.34**

### Path C — “SAE + causal feature intervention” (strongest mechanistic claim)
**Idea:** Do Path B (or A) **plus** causal tests: ablate/steer selected social latents at candidate layers and measure changes in flip rates under pressure with minimal control degradation.

- **Pros:** converts correlational signatures into causal evidence; strongest story for top‑tier reviewers.
- **Cons:** hardest engineering; requires tight controls (no general “damage”); expensive to run across variants/temps.
- **Probability of success:** **0.30**

### Path D — “Hybrid: SAE features as primary, probes as sanity baseline” (risk‑management)
**Idea:** Paper foregrounds SAE results; keep linear probes as a *supplementary* continuity check (FDL correlates with old Turn Layer), to de‑risk interpretability disagreement.

- **Pros:** reduces risk of “SAE didn’t work” or “features are too messy”; easier narrative continuity.
- **Cons:** not a clean “replacement”; some reviewers may still focus on probe weaknesses.
- **Probability of success:** **0.14**

**Remaining probability mass (0.00–0.10, optional):** reserved for “none of the above works without major new data/compute” scenarios; if you want explicit accounting, allocate **0.10** to an “unexpected blocker” bucket and renormalize.

---

## 4) Shared evaluation criteria (what decides the winner)

We will pick the path that best satisfies:

1) **Selectivity:** social latents activate in pressure prompts and not control (robust across items/temps; not just a prefix detector).
2) **Distinctness:** social vs truth latent sets are meaningfully different (low correlation / low overlap; different token/logit associations).
3) **Predictiveness:** social latent magnitude (or dominance score) predicts flips (AUC / logistic‑reg OR; cross‑validated; item‑held‑out).
4) **Generalization:** effects persist across at least:
   - both pressure mechanisms (Asch and Authority), or
   - multiple Asch prompt styles (confederate count, confidence, wording), or
   - multiple seeds at `T>0`.
5) **(Optional but high value) Causality:** feature ablation/steering changes flip rates without collapsing base accuracy.

---

## 5) Concrete work plan (what we will implement once a path is chosen)

### Step 1 — Data plumbing (per variant, per layer)
- Build a dataset loader that reads `simulation.db` + safetensors shards and yields `(trial_id, variant, condition, layer, resid_vector, labels...)`.
- Ensure we can match trials across conditions by `(item_id, variant, temperature, seed)` for delta computations.

### Step 2 — SAE training (SAELens preferred; fallback is a minimal PyTorch SAE)
- Train SAEs per **(variant, layer)** for a prioritized layer range:
  - start with layers 0–12 (where current Turn Layers live), then expand if needed.
- Save SAE checkpoints + config + dataset hash into run artifacts for reproducibility.

### Step 3 — Latent discovery: “social compliance” and “truth”
- Social latents: rank by pressure selectivity (Asch vs control), then intersect with flip‑predictive latents.
- Truth latents: rank by truth selectivity (truth dataset) **or** by control correctness selectivity.
- Negative controls:
  - “format latents” (e.g., `Prior users answered:`) should not survive cross‑template tests.
  - check latents on synthetic controls that keep strings but remove social meaning.

### Step 4 — Feature Dominance Layer computation
- Compute `D_{i,ℓ}` and FDL per trial; aggregate into heatmaps analogous to the existing Turn Layer figure.
- Replace collision heatmaps with **latent dominance heatmaps** (social mass − truth mass).

### Step 5 — Behavioral linkage
- Model: `flip ~ social_magnitude` (and covariates: temperature, variant, condition, item fixed effects if feasible).
- Report: effect sizes, AUC, calibration curves; stress test across held‑out items and across seeds.

### Step 6 — (If Path C) causal interventions
- At candidate layers: clamp/ablate selected social latents (or subtract decoder contribution) and rerun pressure trials.
- Controls: random latent ablation of matched sparsity; “truth latent” ablation as contrast; measure control accuracy drift.

### Step 7 — Paper update
Update:
- `paper/paper.tex`: rename and redefine mechanistic metric(s), update figures and captions, revise limitations.
- `paper/MECHANISTIC_INTERPRETABILITY_GUIDE.md`: new narrative for SAE‑based features.
- Supplement: include probe results only as continuity/sanity if we choose Path D.

---

## 6) Expected paper figures/tables (SAE version)

Minimum set (main paper):
1) **FDL heatmap** across variant × temperature × pressure mechanism (replaces Turn Layer heatmap).
2) **Dominance heatmaps** (social mass − truth mass) at a representative temperature (replaces SVP−TVP collision heatmaps).
3) **Flip prediction plot**: social latent magnitude vs flip probability (with confidence intervals / calibration).

High value additions (supplement / optional main):
- Latent “dictionary cards”: top activating prompts; top token/logit associations; interpretability sanity checks.
- Generalization table across prompt variants/seeds.
- (If causal) ablation/steering bar charts: Δflip, Δcontrol accuracy.

---

## 7) Decision request (what I need from you)

Pick **one mainline path** (A/B/C/D) and one “backup” path.

If compute is constrained, pick:
- Mainline: **B** (delta‑SAE)  
- Backup: **A** (fast retrofit)

If reviewers are the primary constraint, pick:
- Mainline: **C** (causal SAE)  
- Backup: **B**

