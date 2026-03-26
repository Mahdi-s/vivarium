# Olmo-3 Activation Extraction Issue: Technical Analysis

## Executive Summary

The Olmo-3 conformity experiment successfully runs behavioral trials and captures activations, but **cannot complete the interpretability analysis pipeline** (probes, logit-lens, interventions) due to an architectural mismatch: the main experiment uses a custom HuggingFace-based gateway that supports Olmo-3, while the analysis stages require TransformerLens, which does not support Olmo-3.

---

## Current State

### What Works ✅

1. **Behavioral Trials** (`aam olmo-conformity`)
   - Successfully runs 60 trials (20 items × 3 conditions)
   - Generates model outputs for each trial
   - Evaluates correctness and refusal flags
   - **Gateway**: `HuggingFaceHookedGateway` (custom implementation)

2. **Activation Capture**
   - Successfully captures activations from all 32 layers
   - Stores activations as `.safetensors` files (60 files, one per trial)
   - Records metadata in `activation_metadata` table (1920 rows = 60 trials × 32 layers)
   - **Hook naming convention**: `blocks.{L}.hook_resid_post` (compatible with TransformerLens-style names)

3. **Basic Analytics**
   - Behavioral metrics (accuracy by condition)
   - Output parsing and correctness evaluation
   - Activation metadata tracking

### What's Missing ❌

The following interpretability stages **cannot run** because they require TransformerLens:

1. **Probes** (`conformity_probes` table: 0 rows)
   - Truth probes: Linear classifiers trained on labeled data to detect truthfulness in activations
   - Social probes: Linear classifiers to detect social conformity signals
   - **Required for**: Truth Vector Projection (TVP) and Social Vector Projection (SVP) metrics

2. **Probe Projections** (`conformity_probe_projections` table: 0 rows)
   - Scalar projections of trial activations onto probe weight vectors
   - **Required for**: Layer-by-layer analysis of truth vs. social signals

3. **Interventions** (`conformity_interventions` + `conformity_intervention_results` tables: 0 rows)
   - Activation steering: Subtracting social vectors to "steer" model toward truth
   - **Required for**: Causal analysis of whether social signals can be removed

4. **Logit Lens** (`conformity_logit_lens` table: 0 rows)
   - Top-k token predictions at each layer
   - **Required for**: Understanding how representations evolve through layers

5. **Think Tokens** (`conformity_think_tokens` table: 0 rows)
   - Parsing of `<think>...</think>` reasoning blocks (if model uses them)
   - **Required for**: Analysis of internal reasoning processes

6. **JudgeEval Scores** (Present but **fake defaults**)
   - All 60 outputs have identical scores: `{"conformity":0.5,"truthfulness":0.5}`
   - These are fallback values, not real judge evaluations
   - **Cause**: JudgeEval tracer failed (missing API keys), scorers returned defaults

---

## Root Cause: Architectural Mismatch

### Two Gateway Implementations

The codebase has **two separate gateway implementations** for model inference:

#### 1. `HuggingFaceHookedGateway` (Works with Olmo-3) ✅

**Location**: `src/aam/llm_gateway.py` lines 667-986

**Purpose**: Custom gateway for architectures not supported by TransformerLens (explicitly designed for Olmo-3)

**Key Features**:
- Uses `transformers.AutoModelForCausalLM` directly (HuggingFace)
- Registers custom forward hooks on `model.model.layers[L]`
- Emits hook names compatible with TransformerLens: `blocks.{L}.hook_resid_post`
- Supports activation capture via `CaptureContext`
- **Used by**: Main experiment runner (`olmo-conformity`)

**Why It Works**:
- HuggingFace Transformers library has native support for Olmo-3
- No model weight conversion required
- Direct access to model internals via PyTorch hooks

#### 2. `TransformerLensGateway` (Does NOT work with Olmo-3) ❌

**Location**: `src/aam/llm_gateway.py` lines 441-586

**Purpose**: Standard gateway using TransformerLens library

**Key Features**:
- Uses `transformer_lens.HookedTransformer`
- Requires model to be in TransformerLens's official model registry
- Provides standardized hook interface and utilities
- **Used by**: Probe training, logit-lens, interventions

**Why It Fails**:
- TransformerLens maintains a curated list of supported models (`OFFICIAL_MODEL_NAMES`)
- Olmo-3 is **not** in this list (verified: no models with "olmo" in name)
- TransformerLens requires specific weight conversion/loading logic per architecture
- Attempting to load unsupported models raises `ValueError` or `KeyError`

### Code Evidence

**Probe Pipeline** (`src/aam/experiments/olmo_conformity/probes.py:134`):
```python
gateway = TransformerLensGateway(model_id=str(capture.model_id), capture_context=cap_ctx)
```

**Logit Lens** (`src/aam/experiments/olmo_conformity/logit_lens.py`):
- Also uses `TransformerLensGateway` or `HookedTransformer` directly

**Interventions** (`src/aam/experiments/olmo_conformity/intervention.py`):
- Uses `TransformerLensGateway` for activation steering

**Main Runner** (`src/aam/experiments/olmo_conformity/runner.py:332`):
```python
print(f"\nUsing local hooked HF gateway for Olmo model: {model_id}")
# Uses HuggingFaceHookedGateway
```

---

## What We Need

### Functional Requirements

To complete the interpretability pipeline, we need:

1. **Probe Training**
   - Load captured activations from safetensors files
   - Train logistic regression probes on labeled truth/social data
   - Save probe weights (per layer) to safetensors
   - Store probe metadata in `conformity_probes` table

2. **Probe Projections**
   - For each trial, load activations and probe weights
   - Compute dot product: `activation · probe_weight + bias`
   - Store scalar projection values in `conformity_probe_projections` table
   - Enables TVP/SVP analysis by layer

3. **Interventions**
   - Load probe weights (social vector)
   - For each trial, generate output with and without intervention
   - Intervention: Subtract social vector from activations at specified layers
   - Compare correctness before/after intervention
   - Store results in `conformity_intervention_results` table

4. **Logit Lens**
   - At each layer, compute unembedding: `activations @ unembedding_matrix`
   - Extract top-k token predictions
   - Store in `conformity_logit_lens` table

5. **Think Token Parsing** (if applicable)
   - Parse `<think>...</think>` blocks from outputs
   - Extract reasoning tokens
   - Store in `conformity_think_tokens` table

### Technical Requirements

All of these operations require:

1. **Model Access**
   - Ability to load model weights
   - Access to unembedding matrix (for logit lens)
   - Ability to modify activations mid-forward (for interventions)

2. **Activation Access**
   - Load safetensors files created during capture
   - Align activations to trials via `activation_metadata` table
   - Match hook names: `blocks.{L}.hook_resid_post`

3. **Hook Compatibility**
   - The hook naming convention is already compatible (`blocks.{L}.hook_resid_post`)
   - Activations are stored in the same format
   - **The blocker is model loading, not activation format**

---

## Why This Is The Issue Right Now

### The Problem Chain

1. **Main experiment works** → Uses `HuggingFaceHookedGateway` → Olmo-3 supported by HuggingFace
2. **Analysis pipeline fails** → Uses `TransformerLensGateway` → Olmo-3 NOT supported by TransformerLens
3. **Result**: Activations are captured but cannot be analyzed

### Why TransformerLens Doesn't Support Olmo-3

TransformerLens is a research library that:
- Maintains a curated list of supported architectures
- Requires manual integration for each new model family
- Needs architecture-specific weight loading/conversion logic
- Olmo-3 is relatively new (released 2024) and hasn't been added yet

**Verification**:
```python
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
olmo_models = [m for m in OFFICIAL_MODEL_NAMES if 'olmo' in m.lower()]
# Result: [] (empty list)
```

### Why The Split Architecture Exists

The codebase was designed with a **hybrid approach**:
- Use TransformerLens for models it supports (better utilities, standardized interface)
- Use custom HuggingFace gateway for unsupported models (like Olmo-3)
- Both gateways emit compatible hook names for interoperability

**However**, the analysis pipeline was not updated to support the HuggingFace gateway path.

---

## Possible Forward Paths

### Option 1: Modify Analysis Pipeline to Use HuggingFace Gateway ⚡ (Recommended)

**Effort**: Medium (2-3 days)

**Changes Required**:

1. **Probes** (`src/aam/experiments/olmo_conformity/probes.py`):
   - Replace `TransformerLensGateway` with `HuggingFaceHookedGateway`
   - Ensure hook names match: `blocks.{L}.hook_resid_post`
   - Verify activation loading from safetensors works

2. **Logit Lens** (`src/aam/experiments/olmo_conformity/logit_lens.py`):
   - Load model via HuggingFace: `AutoModelForCausalLM.from_pretrained()`
   - Access unembedding: `model.lm_head` or `model.embed_tokens.weight.T`
   - Compute logits: `activations @ unembedding.T`
   - Extract top-k tokens

3. **Interventions** (`src/aam/experiments/olmo_conformity/intervention.py`):
   - Use `HuggingFaceHookedGateway` for generation
   - Implement activation patching: Modify activations in forward hooks
   - Compare outputs before/after patching

**Pros**:
- Works with existing activation captures
- No need to re-run experiments
- Maintains compatibility with both gateways

**Cons**:
- Requires careful testing of activation alignment
- Need to verify unembedding matrix access for logit lens
- Activation patching logic needs to be implemented

**Implementation Notes**:
- The hook names are already compatible (`blocks.{L}.hook_resid_post`)
- Activation format is the same (safetensors)
- Main work is replacing model loading and ensuring tensor operations match

---

### Option 2: Switch to TransformerLens-Supported Model

**Effort**: Low (1-2 hours)

**Models Available**:
- `gpt2-xl` (1.5B parameters)
- `facebook/opt-6.7b` (6.7B parameters)
- `pythia-6.9b` (6.9B parameters)
- `llama-2-7b-hf` (if available)

**Changes Required**:
- Update `suite_small.json`: Change `model_id` to supported model
- Re-run entire experiment: `aam olmo-conformity-full ...`
- All analysis stages will work out-of-the-box

**Pros**:
- Zero code changes
- Full interpretability pipeline works immediately
- Well-tested code paths

**Cons**:
- Lose Olmo-3 specific results
- Need to re-run all 60 trials
- Model behavior may differ from Olmo-3

---

### Option 3: Add Olmo-3 Support to TransformerLens (Upstream)

**Effort**: High (1-2 weeks)

**Process**:
1. Fork TransformerLens repository
2. Add Olmo-3 architecture definition
3. Implement weight loading/conversion logic
4. Test and submit PR to upstream
5. Wait for merge/release (or use fork)

**Pros**:
- Benefits entire community
- Long-term solution
- Maintains standardized interface

**Cons**:
- Significant engineering effort
- Requires deep understanding of TransformerLens internals
- May not be accepted upstream (maintainer discretion)
- Timeline uncertain

---

### Option 4: Hybrid Approach - Use HuggingFace for Analysis, TransformerLens for Supported Models

**Effort**: Medium-High (3-5 days)

**Strategy**:
- Modify analysis pipeline to detect model support
- If TransformerLens supports model → use `TransformerLensGateway`
- If not → use `HuggingFaceHookedGateway`
- Abstract differences behind a common interface

**Pros**:
- Works with any model (current and future)
- Maintains backward compatibility
- Best of both worlds

**Cons**:
- More complex codebase
- Need to handle two code paths
- More testing required

---

## Recommended Path Forward

**Option 1: Modify Analysis Pipeline to Use HuggingFace Gateway**

This is the most pragmatic solution because:

1. **Existing data is preserved**: All 60 trials and activations are already captured
2. **Minimal disruption**: Only analysis code needs changes, not experiment runner
3. **Quick turnaround**: Can be implemented in 2-3 days
4. **Proven approach**: The main experiment already uses this gateway successfully

### Implementation Checklist

- [ ] Update `probes.py` to use `HuggingFaceHookedGateway`
- [ ] Verify probe training works with HuggingFace model
- [ ] Update `logit_lens.py` to access unembedding matrix from HuggingFace model
- [ ] Implement activation patching in `intervention.py` using HuggingFace hooks
- [ ] Test end-to-end: probes → projections → interventions
- [ ] Verify TVP/SVP metrics compute correctly
- [ ] Update notebook to display new metrics

### Key Technical Challenges

1. **Unembedding Matrix Access**:
   - HuggingFace models: `model.lm_head.weight` or `model.embed_tokens.weight.T`
   - Need to verify Olmo-3's exact structure

2. **Activation Patching**:
   - Modify activations during forward pass
   - HuggingFace hooks: `layer.register_forward_hook()` with output modification
   - Ensure patched activations propagate correctly

3. **Tensor Alignment**:
   - Verify safetensors loading matches expected shapes
   - Ensure layer indices align between capture and analysis

---

## Current Workaround

Until the analysis pipeline is fixed, you can still:

1. **View Behavioral Metrics**:
   - Accuracy by condition
   - Correctness distribution
   - Refusal rates

2. **Inspect Activations Manually**:
   - Load safetensors files directly
   - Analyze activation patterns with custom scripts
   - Visualize activations (PCA, t-SNE, etc.)

3. **Run JudgeEval Backfill** (if Ollama is available):
   ```bash
   aam olmo-conformity-judgeval \
     --run-id 80e4b6ee-624d-4891-b616-175dc7ca4bce \
     --db ./runs/20251216_214409_80e4b6ee-624d-4891-b616-175dc7ca4bce/simulation.db \
     --force
   ```

---

## Conclusion

The issue is **not** with activation capture (which works perfectly), but with the **analysis pipeline's dependency on TransformerLens**, which doesn't support Olmo-3. The solution is to modify the analysis code to use the same `HuggingFaceHookedGateway` that successfully runs the main experiment. This is a straightforward engineering task that will unlock the full interpretability pipeline for Olmo-3 experiments.

---

## Appendix: Database Schema Reference

### Tables Populated ✅
- `conformity_trials`: 60 rows
- `conformity_outputs`: 60 rows
- `activation_metadata`: 1920 rows (60 trials × 32 layers)
- `conformity_prompts`: 60 rows
- `conformity_trial_steps`: 60 rows

### Tables Empty ❌ (Require Analysis Pipeline)
- `conformity_probes`: 0 rows
- `conformity_probe_projections`: 0 rows
- `conformity_interventions`: 0 rows
- `conformity_intervention_results`: 0 rows
- `conformity_logit_lens`: 0 rows
- `conformity_think_tokens`: 0 rows

### Tables With Fake Data ⚠️
- `conformity_outputs.parsed_answer_json`: All values are `{"conformity":0.5,"truthfulness":0.5}` (defaults, not real scores)
