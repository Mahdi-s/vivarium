# Engineer Handoff: Vivarium — OLMo Conformity + Interpretability

This document trains a new engineer to *own* the OLMo Conformity experiment codepath end-to-end: how runs are configured, how prompts are rendered, where model calls happen, what artifacts are produced, how activations/probes/interventions work, and where the sharp edges are.

It is written from the perspective of the person who designed the system: it includes not just “what it does”, but *why the architecture looks like this* and what assumptions are baked in.

## 0) TL;DR (Mental Model)

- The repo is two things living together:
  1) **Vivarium**: a deterministic multi-agent simulation kernel (Phases 1–4) with trace logging, replay, optional LLM cognition, and optional activation capture.
  2) **OLMo Conformity experiment**: a *non-simulation* experiment suite that reuses the same persistence + interpretability primitives (SQLite + safetensors + hook naming + provenance) to run a paper-grade conformity protocol.
- The experiment is configured by **suite JSON** files under `experiments/olmo_conformity/configs/`.
- The core experiment runner writes everything into:
  - `runs/<timestamp>_<run_id>/simulation.db` (SQLite), and optionally
  - `runs/<timestamp>_<run_id>/activations/step_*.safetensors` (activation shards).
- For **OLMo-3**, we do *not* rely on TransformerLens weight conversion. We instead use a custom **HuggingFaceHookedGateway** that:
  - loads OLMo weights with `transformers`,
  - registers forward hooks on `model.model.layers[*]`, and
  - emits **TransformerLens-style hook names** (`blocks.{L}.hook_resid_post`) so downstream tooling stays unified.

## 1) Repo Map (What Lives Where)

### 1.1 Primary docs you should read first

- `README.md`: broad platform overview + CLI entrypoints.
- `experiments/olmo_conformity/EXPERIMENT_SCOPE.md`: plain-language scope + definitions (sycophancy, Asch, wrong_answer, “turn layer”, etc.).
- `PROMPT_CATALOG.md`: exhaustive prompt families, rendered examples, and prompt-building logic mapping.
- `INTERPRETABILITY_CONFORMITY_EXPERIMENT.md`: “tough reviewer ready” protocol and what artifacts should exist.
- `Olmo inspection.md`: historical note about a now-mostly-resolved architecture mismatch (TL vs HF); still useful context.

### 1.2 Code directories

- Core package: `src/vivarium/`
  - `run.py`: **CLI** entrypoint (`vvm`).
  - `persistence.py`: SQLite schema + insert/upsert helpers (**the single source of truth for data model**).
  - `llm_gateway.py`: model access layer (LiteLLM, TransformerLens, HF-hooked OLMo, mock).
  - `interpretability.py`: activation capture + safetensors sharding + activation_metadata indexing + Merkle provenance.
  - `analytics/`: analysis/plotting modules that consume `simulation.db`.
  - Simulation kernel (phases 1–4): `world_engine.py`, `scheduler.py`, `policy.py`, `agent_langgraph.py`, `channel.py`, etc.
- Experiment implementation: `src/vivarium/experiments/olmo_conformity/`
  - `runner.py`: **behavioral trial suite** runner (datasets × conditions × models).
  - `prompts.py`: prompt rendering utilities (Zhu tones, confederate blocks, authority claims).
  - `probes.py`: capture probe datasets, train per-layer logistic probes, compute projections.
  - `vector_analysis.py`: truth-vs-social vector workflow (TVP/SVP, “turn layers”, plots).
  - `intervention.py`: activation steering (social-vector subtraction) using HF hooks.
  - `logit_lens.py`: “best-effort” logit lens using captured vectors + HF unembedding; optional TL-based think-token analysis.
  - `answer_logprobs.py`: post-hoc answer preference analysis via cached forward passes (correct vs wrong answer).
  - `ollama_judge.py`: post-hoc judge that scores transcripts via Ollama (OpenAI-compatible API).
  - `resume.py`: recovery utilities for older runs (e.g., re-capture activations if overwritten in legacy runs).
- Experiment configs and assets: `experiments/olmo_conformity/`
  - `configs/suite_*.json`: suite definitions (datasets/conditions/models/run params).
  - `configs/paths.json`: HPC paths (models_dir, runs_dir, etc.).
  - `datasets/*/*.jsonl`: curated question sets (factual + opinion + probe training candidates).
  - `prompts/*/*.txt`: prompt templates (control system prompt, Asch system prompt, etc.).
- “Orchestrate a bunch of runs” scripts: `experiments/olmo_conformity/configs/*.py` and `scripts/*.py`.

### 1.3 Libraries used (and where)

This is a “map” of dependencies → the codepaths that actually use them.

**Always-on / base dependencies (imported widely)**
- `sqlite3` (stdlib): persistence (`src/vivarium/persistence.py`) and many scripts.
- `argparse` (stdlib): CLI (`src/vivarium/run.py`) and most scripts.
- `pydantic`: suite/config validation (`src/vivarium/experiment_config.py`, `src/vivarium/experiments/olmo_conformity/suite_config.py`).
- `numpy`, `pandas`, `matplotlib`, `scipy`: analytics + plots (`src/vivarium/analytics/*`, `src/vivarium/experiments/olmo_conformity/analysis.py`).

**Model execution / local inference**
- `torch`: required for local inference + activation handling (`src/vivarium/llm_gateway.py`, `src/vivarium/interpretability.py`, interventions, answer logprobs).
- `transformers`: HF model loading and generation (`HuggingFaceHookedGateway`), plus model download tooling.

**Interpretability extras**
- `safetensors`: activation shard writing/reading + probe weight artifacts (`CaptureContext`, probes, logit lens).
- `transformer_lens`: HookedTransformer path for supported models (`TransformerLensGateway`) and optional think-token logit lens.

**Remote model execution**
- `litellm`: unified remote provider access (`LiteLLMGateway`).

**Cognitive layer (simulation phases)**
- `langgraph`: orchestration for agent “think” graphs (`src/vivarium/agent_langgraph.py`).
- `json-repair`: utility used by the cognitive layer (where installed).

**Judge / evaluation**
- `httpx`: Ollama judge client (`src/vivarium/experiments/olmo_conformity/ollama_judge.py`).
- `judgeval`: optional inline judge scoring during the runner (`src/vivarium/experiments/olmo_conformity/judgeval_scorers.py`).

**Memory system (not central to conformity)**
- `chromadb`, `sentence-transformers`: vector memory (`src/vivarium/memory.py`).

### 1.4 `src/vivarium/` quick reference (core modules)

This is the “platform” backbone. The conformity experiment reuses pieces of it (DB, gateways, capture), but does not use the full simulation kernel.

- `src/vivarium/run.py`: CLI, wires subcommands to modules.
- `src/vivarium/persistence.py`: schemas + insert/upsert helpers; if you change “what we store”, you change it here.
- `src/vivarium/llm_gateway.py`: inference gateways (remote + local); also houses `select_local_gateway()`.
- `src/vivarium/interpretability.py`: activation capture, sparse capture controls, safetensors sharding, metadata indexing, Merkle provenance.
- `src/vivarium/output_parsing.py`: output normalization + “output quality” classifier (prompt leakage, degenerate repetition, etc.) used for metadata.
- `src/vivarium/provenance.py`: Merkle accumulator primitives used by capture.
- `src/vivarium/world_engine.py`, `src/vivarium/scheduler.py`: deterministic simulation kernel + barrier scheduler (Phases 1–4).
- `src/vivarium/agent_langgraph.py`: cognitive agent policy abstraction that calls gateways and parses tool calls / JSON actions.
- `src/vivarium/tools.py`: tool definitions (what agents can do in the simulation).
- `src/vivarium/llama_cpp.py`, `src/vivarium/model_discovery.py`: local GGUF discovery + llama.cpp server integration (not used in OLMo conformity, except as an alternative inference backend).
- `src/vivarium/export.py`: Parquet export utilities.

## 2) The OLMo Conformity Experiment: Data Model + Artifact Layout

### 2.1 Run directory layout

When you run a suite, you get:

```
runs/<timestamp>_<run_id>/
  simulation.db
  activations/                        # optional (only when capture is enabled)
    step_000001.safetensors
    step_000002.safetensors
    ...
  artifacts/
    figures/                          # generated plots
    tables/                           # csv exports, probe safetensors, etc.
  exports/                            # reserved for larger exports (varies by pipeline)
```

Note: unlike `vvm experiment` (Phase 4), the conformity runner does not currently write a `run_metadata.json` file to disk; instead it stores run config state in `runs.config_json` inside the SQLite DB.

### 2.2 SQLite schemas (core + conformity)

All data is persisted in a single SQLite DB (`simulation.db`) via `TraceDb` (`src/vivarium/persistence.py`):

**Core (platform) tables**
- `runs`: one row per run_id with a JSON config snapshot.
- `trace`: step-by-step action log (used by simulation phases; typically unused by conformity runner).
- `messages`: shared message feed (simulation phases).
- `activation_metadata`: index of activation tensors stored in safetensors shards.
- `merkle_log`: per-(step, agent) provenance records (prompt_hash + activation_hash → Merkle root).

**Conformity experiment tables**
- `conformity_datasets`: dataset registry (name/version/path/sha256).
- `conformity_items`: each question/item; for factual items `ground_truth_text` is populated; `source_json` stores `wrong_answer`.
- `conformity_conditions`: condition registry (name + params_json).
- `conformity_trials`: one *trial* = (run_id, model_id, variant, item_id, condition_id, seed, temperature).
- `conformity_prompts`: rendered prompt parts per trial (system/user/history) + a deterministic hash.
- `conformity_prompt_metadata`: structured prompt-render decisions (tone, consensus, DA/QD, alt answers, etc.).
- `conformity_trial_metadata`: generation config + model config + gateway class.
- `conformity_trial_steps`: maps trial_id → `(time_step, agent_id)` for activation alignment.
- `conformity_outputs`: model completion + parsed fields + correctness/refusal + token_usage_json.
- `conformity_probes`: registry for trained probes (artifact_path to safetensors).
- `conformity_probe_projections`: scalar projection values per (trial, probe, layer).
- `conformity_think_tokens`: optional token traces for `<think>...</think>` (coarse fallback).
- `conformity_logit_lens`: per-layer top-k token predictions (stored as JSON).
- `conformity_answer_logprobs`: post-hoc candidate logprob comparisons.
- `conformity_interventions` / `conformity_intervention_results`: activation steering definitions + before/after results.

### 2.3 How activations are aligned to trials

We unify *all* activation-dependent workflows around one invariant:

> Every captured activation shard corresponds to a deterministic `(run_id, time_step, agent_id)` tuple.

For conformity runs:
- `conformity_trial_steps` stores `(trial_id → time_step, agent_id)`.
- `activation_metadata` stores `(run_id, time_step, agent_id, layer_index, component, shard_file_path, tensor_key, …)`.
- The safetensors shard contains tensors keyed by `"{agent_id}.blocks.{L}.hook_resid_post"` (and other hook names if configured).

Downstream code always follows the same lookup pattern:
1) resolve `(run_id, time_step, agent_id)` for a trial via `conformity_trial_steps`,
2) query `activation_metadata` for the layer/component,
3) `safetensors.load_file(shard_file_path)[tensor_key]`.

## 3) Entry Points: How the Code is Run

### 3.1 CLI entrypoint (`vvm` / `aam`)

The console scripts are defined in `pyproject.toml`:
- `aam = "aam.run:main"`
- `vvm = "aam.run:main"`

So everything routes through:
- `src/vivarium/run.py:main()`

Relevant subcommands for the OLMo experiment:

- `vvm olmo-conformity`: behavioral suite runner (writes DB; optional activation capture).
- `vvm olmo-conformity-full`: orchestration wrapper (trials → probes → interventions → reports).
- `vvm olmo-conformity-vector-analysis`: truth/social probes + projections + turn-layer analysis.
- `vvm olmo-conformity-posthoc`: backfill missing analyses (logit lens, answer logprobs, interventions, report).
- `vvm olmo-conformity-judgeval`: post-hoc Ollama judge scoring into `conformity_outputs.parsed_answer_json`.
- `vvm olmo-conformity-report`: regenerate figures/tables from DB.
- (Legacy-ish) `vvm olmo-conformity-probe`, `vvm olmo-conformity-logit-lens`, `vvm olmo-conformity-intervene`, `vvm olmo-conformity-resume`.

### 3.2 Suite config flow

Suite configs are JSON validated by Pydantic:
- schema: `src/vivarium/experiments/olmo_conformity/suite_config.py`
- loader: `src/vivarium/experiments/olmo_conformity/io.py:load_suite_config()`

Key suite fields:
- `datasets[]`: name/version/path (+ notes)
- `conditions[]`: name + params (type, tone, consensus, confederates, distillation, devils_advocate, etc.)
- `models[]`: variant + model_id
- `run`: seed, temperature, top_k, top_p, max_items_per_dataset
- optional `paths_config`: points at `configs/paths.json` for HPC paths

### 3.3 Behavioral runner flow (`run_suite`)

Behavioral execution is in:
- `src/vivarium/experiments/olmo_conformity/runner.py:run_suite()`

High-level pipeline:
1) resolve paths (`paths.json` + env vars like `AAM_HF_CACHE`, `AAM_RUNS_DIR`)
2) create `runs/<timestamp>_<run_id>/` and open `simulation.db`
3) initialize schemas: `TraceDb.init_schema()` + `TraceDb.init_conformity_schema()`
4) register datasets/items in DB (store `wrong_answer` in `conformity_items.source_json`)
5) register conditions in DB
6) optionally set up `CaptureContext` (writes safetensors + activation_metadata)
7) for each model in suite:
   - choose gateway (mock vs API vs local OLMo vs TL)
   - for each item × condition:
     - insert trial row + trial metadata
     - render prompt + insert prompt row + prompt metadata
     - assign `(time_step, agent_id)` for activation alignment (`conformity_trial_steps`)
     - **call `gateway.chat()`** (this is the actual model invocation)
     - parse, score, and insert output row
     - if capture enabled: commit activations and flush shard

## 4) Where We Call Models (Exact Places + Parameters)

Everything that “runs a model” goes through a `gateway.chat(...)` call (or cached forward passes for logprobs).

### 4.1 Behavioral trials

File: `src/vivarium/experiments/olmo_conformity/runner.py`

Inside the item×condition loop, we call:
- `resp = gateway.chat(model=..., messages=..., temperature=..., top_k=..., top_p=..., seed=...)`

Where:
- `temperature`, `top_k`, `top_p`, `seed` come from suite config (`cfg["run"]`).
- `model` is:
  - the HF model id (for local HF gateway),
  - or an API model name (for Ollama/OpenAI-compatible endpoints; OLMo IDs are normalized via `get_ollama_model_name()`).

### 4.2 Probe dataset capture (truth/social training sets)

File: `src/vivarium/experiments/olmo_conformity/probes.py:capture_probe_dataset_to_db()`

Calls:
- `gateway = select_local_gateway(model_id_or_path=..., capture_context=cap_ctx, max_new_tokens=...)`
- `resp = gateway.chat(... temperature=..., seed=42)`

Notes:
- probe capture fixes the RNG seed to reduce variance even when temperature > 0.
- it *offsets time_step* to avoid overwriting shards from the behavioral run:
  - `base_ts = MAX(conformity_trial_steps.time_step) + 1`
  - `time_step = base_ts + i`

### 4.3 Interventions (activation steering)

File: `src/vivarium/experiments/olmo_conformity/intervention.py:run_intervention_sweep()`

Calls:
- `gateway = HuggingFaceHookedGateway(...)`
- baseline: `gateway.chat(...)`
- intervention: register hooks via `gateway.register_intervention_hook(...)`, then `gateway.chat(...)`, then remove handles.

### 4.4 Logit lens

File: `src/vivarium/experiments/olmo_conformity/logit_lens.py`

Two modes:
- OLMo-compatible “offline” logit lens:
  - loads captured residual vectors from safetensors,
  - loads unembedding matrix via `HuggingFaceHookedGateway.get_unembedding_matrix()`,
  - computes `logits = unembed @ resid`, then stores top-k in DB.
- optional TL-based multi-token analysis for `<think>` content:
  - `HookedTransformer.from_pretrained(model_id)` + `run_with_cache()`
  - **This is not guaranteed to work for OLMo**; treat as optional / experimental.

### 4.5 Answer logprobs (correct vs conforming candidate preference)

File: `src/vivarium/experiments/olmo_conformity/answer_logprobs.py:compute_and_store_answer_logprobs_for_model()`

This does *not* call `generate`; instead it:
- builds context token ids for a prompt prefix,
- runs `model(input_ids=..., use_cache=True)` once,
- then walks candidate token sequences reusing `past_key_values`,
- stores logprob_sum/logprob_mean in `conformity_answer_logprobs`.

## 5) Model Gateways and How OLMo is Loaded

All model access is abstracted behind gateway classes in `src/vivarium/llm_gateway.py`.

### 5.1 LiteLLMGateway (remote APIs)

Class: `LiteLLMGateway`
- wraps `litellm` and emits OpenAI-ish responses (`choices[0].message.content`).
- supports:
  - `temperature`, `top_k`, `top_p`, `seed` (best-effort; provider-dependent),
  - tool calling when the provider supports it (not used in conformity runner).
- includes `RateLimiter` that estimates tokens and enforces RPM/TPM and max concurrency.

Use this when:
- you want to run via Ollama/OpenAI-compatible endpoints (`--api-base`),
- or external providers (OpenAI/Anthropic/etc) via LiteLLM.

Critical caveat:
- **Activation capture does not work through remote APIs**. If you run with `--api-base`, you can still log behavior, but you will not get activations.

### 5.2 TransformerLensGateway (local TL models)

Class: `TransformerLensGateway`
- loads a `transformer_lens.HookedTransformer` and uses `.generate(...)`.
- integrates with `CaptureContext` via `with self._model.hooks(fwd_hooks=...)`.

Relationship to OLMo:
- TransformerLens has not historically provided “official” OLMo-3 support (weight conversion/hook dict).
- The gateway includes a *best-effort bridge*:
  - load HF `AutoModelForCausalLM` + tokenizer, then call `HookedTransformer.from_pretrained_no_processing("gpt2", hf_model=..., tokenizer=...)`.
  - This is hacky and should be treated as **experimental** unless validated for OLMo.
- Practically: we do **not** rely on TL for OLMo; we use HF-hooked gateway for OLMo.

### 5.3 HuggingFaceHookedGateway (the OLMo workhorse)

Class: `HuggingFaceHookedGateway`
- loads:
  - `AutoConfig.from_pretrained(..., trust_remote_code=True)`
  - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
  - `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, dtype=...)`
- device selection: CUDA → MPS → CPU (overridable via `VVM_DEVICE`).
- dtype selection default: fp16 on CUDA/MPS, fp32 on CPU (override via `dtype`).
- prompt formatting:
  - prefers tokenizer chat templates (`tokenizer.apply_chat_template(...)`) when present (instruct/think checkpoints),
  - falls back to a simple “SYSTEM/USER/ASSISTANT” formatting for base checkpoints.

Activation capture:
- if `capture_context` is provided, it registers PyTorch hooks on `model.model.layers[i]`
- it emits **TransformerLens-style hook names**:
  - `blocks.{i}.hook_resid_post`
  - (optionally) `blocks.{i}.hook_resid_pre`, `blocks.{i}.hook_mlp_out`, and several attn projections if requested.
- when the runner calls `cap_ctx.on_action_decided(...)`, the capture context commits buffered activations, flushes `step_*.safetensors`, and inserts `activation_metadata` rows.

Why this exists:
- it gives us “TransformerLens-like” introspection for architectures TL doesn’t support, without fragmenting the rest of the pipeline.

### 5.4 select_local_gateway(): the “don’t shoot yourself” helper

Function: `src/vivarium/llm_gateway.py:select_local_gateway()`
- If `"olmo"` in model_id/path → returns `HuggingFaceHookedGateway`.
- Else tries TL first, then falls back to HF.
- Has a `scientific_mode` option that rejects GGUF models when capture is enabled (prevents dual-stack validity threats).

## 6) Activation Capture: What Exactly is Captured (and What It Means)

### 6.1 CaptureContext mechanics

File: `src/vivarium/interpretability.py`

Key objects:
- `CaptureConfig(layers, components, trigger_actions, token_position, ...)`
- `CaptureContext(output_dir, config, dtype, trace_db)`

How it works:
1) before inference, call `cap_ctx.begin_inference()` → clears `_pending`.
2) each hook calls `cap_ctx.record_activation(hook_name=..., activations=...)`:
   - slices token position `config.token_position` (default `-1`),
   - moves to CPU,
   - stores **one vector per hook name** (batch index 0) in `_pending`.
3) after the caller decides to “keep” the activations, call `cap_ctx.on_action_decided(...)`:
   - if action_name matches `trigger_actions`, `_pending` is committed to a per-step buffer.
4) call `cap_ctx.flush_step(time_step=...)`:
   - writes a safetensors shard (`step_<time_step>.safetensors`),
   - inserts activation rows into `activation_metadata`,
   - optionally writes Merkle provenance into `merkle_log`.

### 6.2 Hook name expansion and the resid_post mapping

In configs we often specify `components=["resid_post"]` (human-friendly).

`CaptureContext` expands that to TL hookpoints:
- `resid_post` → `hook_resid_post`
- which becomes fully qualified: `blocks.{L}.hook_resid_post`.

In `activation_metadata.component`, we store:
- `hook_resid_post` (not `resid_post`).

This is why probe/logit-lens defaults typically use `component="hook_resid_post"`.

### 6.3 A subtle but important scientific detail: “token_position=-1”

Default capture uses `token_position=-1`, described in multiple docs as “last token”.

What that *actually* means depends on how generation runs:
- In cached generation, the model may be called repeatedly with a sequence length of 1 (just the new token).
- Because `CaptureContext` stores **only the latest activation per hook name** (it overwrites `_pending[hook_name]` each time), the final stored vector is typically the activation at the **final forward pass of generation**, not necessarily the activation at the prompt boundary.

This is often fine if you want a single “end-of-completion” state per layer, but it is *not the same* as “the residual stream right before the first answer token”.

If you want “prompt boundary” activations, you likely need to:
- capture only on the *first* forward call, or
- run with `max_new_tokens=1` for capture, or
- change `CaptureContext` to buffer activations per generation step/token position.

(This is one of the key places where you can accidentally tell the wrong mechanistic story if you assume the wrong alignment.)

## 7) Prompt System: Conditions, Determinism, and Metadata

### 7.1 Where prompt logic lives

Behavioral prompt construction is in:
- `src/vivarium/experiments/olmo_conformity/runner.py:_build_prompt_for_condition()`

Helper utilities live in:
- `src/vivarium/experiments/olmo_conformity/prompts.py`
- `experiments/olmo_conformity/prompts/**.txt` (templates)

### 7.2 Condition “types” you’ll see in suite configs

Suite conditions have a user-facing `name` and a JSON `params` object. The runner maps `params["type"]` to prompt families, including:
- `control`
- `synthetic_asch_history` (classic “prior users answered …”; uses `wrong_answer`)
- `asch_peer_pressure` (Zhu-style dialogue; supports tone/consensus, diverse/QD/DA)
- `authoritative_bias` (existing authority claim templates)
- `authority_claim` (Zhu-style “I trust the answer must be …”; supports DA)

Every pressure condition must have a valid injected wrong answer:
- `runner.py:_get_wrong_answer()` fails fast if `wrong_answer` is missing or equals ground truth.

### 7.3 Deterministic prompt variation

We have two reproducibility primitives:
- `stable_int_seed(...)` (sha256-derived) for deterministic “random” choices in rendering (tone template selection, distractor selection for diverse controls, etc.).
- `deterministic_prompt_hash(system, user, history)` → stored in `conformity_prompts.rendered_prompt_hash`.

We also store structured metadata for each prompt in:
- `conformity_prompt_metadata.metadata_json`

This is critical when you’re doing paper-level analysis: it gives you “receipts” for what was rendered, not just the final text.

## 8) Interpretability + Analysis Workflows (How the Pieces Fit)

This repo’s interpretability story is intentionally “stacked”:
- behavioral run produces prompts/outputs (+ optionally activations),
- post-hoc tooling produces probes/projections/logit-lens/interventions,
- analytics generates figures/tables.

### 8.1 Probes and projections (TVP/SVP)

Files:
- `src/vivarium/experiments/olmo_conformity/probes.py`
- `src/vivarium/experiments/olmo_conformity/vector_analysis.py`

Core steps:
1) capture probe training dataset activations into the same run DB (`conformity_*` tables)
2) train per-layer logistic probes
3) save probe weights to `artifacts/tables/{truth|social}_probe*.safetensors`
4) compute scalar projections for behavioral trials and store in `conformity_probe_projections`
5) analyze “turn layers” (first layer where SVP > TVP) and generate plots

Scientific rigor note:
- `vector_analysis.py` supports `variant=` to ensure you do not apply a probe trained on one model variant to another variant’s activations (representation spaces differ).
- `orchestration.py:run_full_experiment()` trains probes **per (model_id, variant)** for exactly this reason.

### 8.2 Logit lens

File: `src/vivarium/experiments/olmo_conformity/logit_lens.py`

OLMo-compatible logit lens:
- use captured `hook_resid_post` vectors (per layer),
- unembed via HF `lm_head.weight`/output embeddings,
- store top-k probs in `conformity_logit_lens`.

Think-token parsing:
- `parse_and_store_think_tokens()` stores coarse whitespace tokens from `<think>...</think>`.
- optional TL-based `compute_logit_lens_for_think_tokens()` is experimental for OLMo.

### 8.3 Interventions (“social vector subtraction”)

File: `src/vivarium/experiments/olmo_conformity/intervention.py`

Mechanism:
- load social probe vector `v_social` per target layer from safetensors.
- register a forward hook on that layer output:
  - `patched = hs - alpha * v_social`
- generate again under same prompt and compare to baseline output.
- store outputs + flip-to-truth results in DB.

### 8.4 Judge scoring

Two separate systems exist (historical layering):

1) **Judgeval** integration (during behavioral run)
   - `runner.py` optionally uses `judgeval` scorers.
   - this path is fragile in synchronous contexts (async scorers + `asyncio.run`).

2) **Ollama post-hoc judge** (recommended / paper-friendly)
   - `src/vivarium/experiments/olmo_conformity/ollama_judge.py`
   - CLI: `vvm olmo-conformity-judgeval`
   - writes labels into `conformity_outputs.parsed_answer_json` for the *first output per trial*.

Because the judge stores the rubric version (`JUDGE_PROMPT_VERSION`) and the exact transcript, it’s easier to audit and reproduce.

### 8.5 Reporting

Entry:
- `src/vivarium/experiments/olmo_conformity/analysis.py:generate_core_figures()`

It calls modules under `src/vivarium/analytics/` when available:
- behavioral metrics/figures
- probe metrics/figures
- intervention metrics/figures
- judge metrics (if present)
- answer logprob metrics (posthoc)

Outputs live under:
- `runs/<...>/artifacts/figures/`
- `runs/<...>/artifacts/tables/`

## 9) Scripts: What They’re For (and What They Assume)

There are two “script ecosystems”:

### 9.1 Experiment pipelines (temperature sweeps, HPC job runs)

Directory: `experiments/olmo_conformity/configs/`

Notable:
- `run_expanded_experiments.py`: runs the full temperature sweep, tracks metadata, and generates combined comparisons.
- `run_llm_judge_posthoc.py`: judge backfill across multiple runs.
- `run_interpretability_posthoc.py`: posthoc interpretability across runs.
- `job_*.sh`: HPC job wrappers for specific temperature points and phases.

These scripts assume:
- paths can be resolved from `paths.json` (scratch locations),
- the suite configs (e.g., `suite_expanded_temp0.0.json`) exist and are consistent.

### 9.2 Paper-quality audits and “receipts”

Directory: `scripts/`

Examples:
- `audit_answer_parsing.py`: sensitivity analysis of parsing/correctness labels; produces markdown receipts.
- `audit_llm_judge_labeling.py`: checks whether judge labels exist and compares judge vs rule-based stats.
- `audit_paper_numbers.py`: reproduces paper tables/figures from run DBs.
- `generate_turn_layer_temperature_heatmaps.py`: cross-temp heatmaps of “turn layers”.
- `sample_scoring_cases.py`: stratified sampling of outputs for manual inspection.
- `fix_correctness_evaluation.py`, `fix_social_conventions_ground_truth.py`: legacy repair scripts for older runs/labeling logic.

These scripts are extremely valuable for reviewer-facing rigor because they:
- do not require new model calls,
- operate deterministically on stored artifacts,
- provide concrete example excerpts for edge cases.

**Scripts folder inventory (what each file is for)**
- `scripts/audit_answer_parsing.md`: narrative explanation of the parsing audit (useful for onboarding/reviewers).
- `scripts/audit_answer_parsing.py`: audits `parsed_answer_text`/`is_correct` robustness across the paper runs; writes a markdown report.
- `scripts/audit_llm_judge_labeling.py`: audits judge labels stored in `conformity_outputs.parsed_answer_json` and compares to rule-based stats.
- `scripts/audit_paper_numbers.py`: recomputes paper-facing numbers/tables from run DBs (the “reproduce the paper” script).
- `scripts/backfill_probe_projections_from_canonical.py`: repairs/backfills projection tables for older runs (used when schemas/workflows evolved).
- `scripts/fix_correctness_evaluation.py`: legacy re-scoring script for older DBs (current scoring lives in `src/vivarium/experiments/olmo_conformity/scoring.py`).
- `scripts/fix_social_conventions_ground_truth.py`: repairs dataset labeling/ground-truth edge cases for older runs.
- `scripts/generate_code_documentation.py`: auto-generates codebase documentation snapshots (used for audits/sharing).
- `scripts/generate_complete_analysis.py`: runs the full analytics suite on a run directory and writes figures/tables/logs.
- `scripts/generate_pi_slides.py`: produces PI-facing slide artifacts from runs (presentation automation).
- `scripts/generate_turn_layer_temperature_heatmaps.py`: cross-run aggregation + heatmaps for turn-layer results.
- `scripts/sample_scoring_cases.py`: stratified sampling of trials for manual scoring validation (“show me receipts”).

## 10) Known Concerns / Sharp Edges (Read This Before You Change Things)

This section is intentionally blunt: it’s where an engineer can save days.

### 10.1 Doc drift in CLI examples

`README.md` contains a “full experiment” command line that mixes flags from `olmo-conformity-full` into `olmo-conformity`.

Reality:
- `vvm olmo-conformity` takes a suite config and runs behavioral trials (+ optional capture).
- `vvm olmo-conformity-full` is the orchestration command that accepts probe/intervention/report flags.

Recommendation:
- treat `src/vivarium/run.py` as authoritative for flags; update README when you change CLI.

### 10.2 Activation capture semantics vs “prompt boundary”

As described in §6.3, the default “last token” capture frequently corresponds to the last generation step, not the prompt boundary.

If your mechanistic claim implicitly assumes prompt-boundary representations, you need to enforce that in code.

### 10.3 Resume/re-run idempotency

The behavioral runner supports “resume” by reusing an existing run directory when `--run-id` matches.

But the runner does **not** enforce uniqueness on (run_id, model_id, item_id, condition_id).
- Re-running the same suite against the same run_id can duplicate trials and complicate analysis.

Suggested improvement:
- add an idempotency check before `insert_conformity_trial`, or
- enforce a UNIQUE constraint (run_id, model_id, variant, item_id, condition_id, seed, temperature) and upsert.

### 10.4 time_step assignment is quirky (potential off-by-one)

In `runner.py`, time_step is currently:
- `time_step = COUNT(conformity_trials WHERE run_id=...)` after inserting the trial row.

This means the first behavioral trial is step 1 (not 0).
It’s not wrong, but it’s easy to misread, and it’s a footgun if anyone assumes 0-based indexing.

### 10.5 TransformerLens “bridge” is experimental

The `TransformerLensGateway` fallback that loads HF weights and wraps them via `from_pretrained_no_processing("gpt2", hf_model=...)` has not been paper-validated for OLMo.

Do not assume:
- hook names exist/align,
- cached activations mean the same thing,
- unembedding/unembed APIs match.

For OLMo, prefer HF-hooked gateway everywhere (and extend it if you need more hooks).

### 10.6 Partial feature overlap: Judgeval vs Ollama judge

There are two judge systems; using both can confuse downstream analysis if you don’t know which populated `parsed_answer_json`.

In the paper workflow, prefer the Ollama post-hoc judge:
- it is explicit about prompt versioning and stores stable JSON.

### 10.7 Hooking edge cases for “fancy” components

HFHookedGateway supports a growing set of hook names beyond resid_post, but some paths are “best-effort” and should be tested if you rely on them (attention qkv splits, patterns, etc.).

If you expand capture to new components:
- add a small unit/integration test that asserts the hook names exist and activation_metadata rows populate for a tiny run.

## 11) Practical “How Do I Extend This?” Guide

### 11.1 Add a new behavioral dataset

1) Create a JSONL file under `experiments/olmo_conformity/datasets/<domain>/...jsonl` with fields:
   - `item_id` (stable string)
   - `domain` (string label)
   - `question` (string)
   - `ground_truth_text` (string or null)
   - `wrong_answer` (string; required for pressure conditions)
2) Add it to a suite config under `datasets[]`.
3) Run `vvm olmo-conformity --suite-config ...`.

### 11.2 Add a new condition type / prompt family

1) Add a condition entry in suite JSON with:
   - `name`: user-facing label
   - `params`: include `"type": "your_new_type"` + any new parameters
2) Implement rendering in `runner.py:_build_prompt_for_condition()`.
3) Update:
   - `PROMPT_CATALOG.md` (examples + rationale),
   - optionally add unit tests for deterministic rendering.

### 11.3 Add a new interpretability metric

Decide which “layer” of the stack it belongs to:
- purely DB-derived metrics → implement in `src/vivarium/analytics/`
- requires new model calls → implement as a post-hoc command in `src/vivarium/run.py` + a module under `src/vivarium/experiments/olmo_conformity/`

Then:
- add a DB table if needed (extend `TraceDb.init_conformity_schema()`),
- add insert helpers in `persistence.py`,
- add a posthoc CLI command or integrate into `olmo-conformity-posthoc`,
- add analysis hooks in `analysis.py:generate_core_figures()`.

## 12) Recommended Next Improvements (High ROI)

If you want to “make it even better”, the highest leverage changes are:

1) **Make behavioral runs idempotent for a given run_id**
   - prevents accidental duplicated trials on resume.
2) **Make activation capture semantics explicit**
   - offer `--capture-at prompt_end|first_token|last_token|all_tokens` and implement it.
3) **Unify judge scoring**
   - choose Ollama judge as the default path; demote/remove fragile inline judgeval path.
4) **Add one “smoke test” suite**
   - a tiny local suite that runs in <60s with mock/HF small model, asserts DB tables are populated consistently.
5) **Validate / remove the TL OLMo bridge**
   - either upstream real OLMo TL support, or clearly flag bridge as unsupported and avoid accidental use.
