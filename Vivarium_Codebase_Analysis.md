# Vivarium Codebase Analysis

**Prepared for:** Mahdi Saeedifar
**Date:** March 7, 2026
**Scope:** Full architectural review of `src/vivarium/` — 23,695 lines across 52 Python modules
**Purpose:** Inform the transition from an OLMo-specific experiment harness to a general-purpose LLM simulation framework

---

## Executive Summary

Vivarium is a functional, database-backed experiment harness that has successfully executed 189,953+ conformity trials across 7 OLMo model variants, 12 experimental conditions, and 6 temperature settings. It works. But it was built experiment-first, not framework-first, and that origin story is now the primary obstacle to scaling it into a general-purpose tool.

The core problems are threefold. First, the experiment-specific code (OLMo conformity) and the reusable framework code are entangled — 62% of the codebase lives in `experiments/olmo_conformity/` and cannot be reused for a different model family or experiment type without significant surgery. Second, the execution model is entirely sequential: one model, one item, one condition at a time, with no batching, no thread pooling, and no pipeline parallelism. Third, memory management is implicit rather than explicit — there is no lifecycle control over when models are loaded, held, or released, and the system accumulates state in ways that explain the significant memory footprint you observed during runs.

The good news: the persistence layer is sound, the gateway abstraction is well-designed, and several modules (output parsing, memory protocols, world engine) are already framework-quality code that just needs to be relocated and documented. The path from here to a model-driven, parallelized framework is achievable without a rewrite — it requires a disciplined refactoring in three phases.

---

## 1. Architecture Overview

### 1.1 Module Map

The 23,695 lines break down into four layers:

**Framework Core (4,751 lines, 20%)** — Model-agnostic infrastructure that could serve any LLM experiment:

| Module | Lines | Purpose |
|--------|-------|---------|
| `llm_gateway.py` | 1,606 | Multi-backend inference (LiteLLM, HuggingFace, TransformerLens, Mock) |
| `persistence.py` | 1,187 | SQLite database with 17-table schema, WAL journaling |
| `output_parsing.py` | 985 | Text normalization, quality classification (25+ configurable parameters) |
| `memory.py` | 475 | Memory protocols and implementations (Simple, ChromaDB) |
| `world_engine.py` | 293 | Core simulation loop |
| `settings.py` | 207 | Environment variable resolution with fallback chains |

**Experiment Layer (8,192 lines, 35%)** — OLMo conformity experiment, tightly coupled to that specific research question:

| Module | Lines | Purpose |
|--------|-------|---------|
| `runner.py` | 1,164 | Main execution loop (sequential triple-nested: models × items × conditions) |
| `cli.py` | 1,191 | 12 CLI subcommands, monolithic dispatcher |
| `enhanced_scoring.py` | 859 | 4-tier answer extraction and evaluation |
| `suite_completion.py` | 733 | Backfill missing trials (duplicates gateway creation from runner.py) |
| `orchestration.py` | 371 | End-to-end workflow driver |
| `prompts.py` | 242 | Prompt rendering for conformity conditions |
| Others (8 files) | 3,632 | Probes, interventions, logit lens, contrastive steering, etc. |

**Analytics Layer (5,331 lines, 23%)** — Post-hoc analysis and reporting:

| Module | Lines | Purpose |
|--------|-------|---------|
| `activations.py` | 797 | Activation tensor analysis |
| `reporting.py` | 696 | Scientific report generation |
| `behavioral.py` | 618 | Behavioral metrics computation |
| Others (10 files) | 3,220 | Correlations, interventions, probes, plotting, etc. |

**Legacy/Agent Layer (5,421 lines, 22%)** — Phase 1-4 agent code and supporting modules:

| Module | Lines | Purpose |
|--------|-------|---------|
| `run.py` | 640 | Phase-based experiment drivers (backwards compatibility) |
| `agent_langgraph.py` | 441 | LangGraph agent implementation |
| Others | 4,340 | Domain state, channels, tools, scheduling, provenance, etc. |

### 1.2 Dependency Flow

```
CLI (cli.py)
  └─→ runner.py / suite_completion.py / orchestration.py
        ├─→ llm_gateway.py (inference)
        ├─→ persistence.py (storage)
        ├─→ output_parsing.py (normalization)
        ├─→ enhanced_scoring.py (evaluation)
        ├─→ prompts.py (prompt rendering)
        └─→ olmo_utils.py (model resolution)
              └─→ settings.py (path resolution)
```

The critical observation: `runner.py` is the only orchestrator. Every trial flows through a single function (`run_suite()`, 692 lines) that handles configuration, database setup, dataset registration, model selection, prompt building, inference, scoring, and persistence — all in one sequential pass.

---

## 2. Memory Footprint Analysis

### 2.1 Where the Memory Goes

When you run a suite, the memory footprint is dominated by three things:

**Model weights (14–28 GB).** The `HuggingFaceHookedGateway` loads the full model into memory at initialization and holds it for the entire duration of that model's trial loop. For a 7B parameter model in fp16, that is approximately 14 GB on GPU/MPS or 28 GB on CPU. The model is never offloaded between conditions or between items — it sits in memory from the first trial to the last, even during database I/O and prompt construction.

**GPU/MPS cache fragmentation (2–4 GB).** PyTorch's memory allocator reserves GPU blocks and does not return them to the OS after tensors are freed. Over thousands of inference calls, the reserved-but-unused memory grows. The codebase calls `torch.mps.empty_cache()` between *models* (runner.py line 1147–1160) but never between *trials*. For a single-model surgical run like think_dpo, the cache is never cleared at all during the entire run.

**Activation capture hooks.** When activation capture is enabled, the `HuggingFaceHookedGateway` registers 256+ forward hooks at initialization (llm_gateway.py lines 1073–1239). Each hook closure captures a reference to the `CaptureContext` object. The hooks are never removed — there is no `__del__()` method or cleanup mechanism. While the captured activation tensors themselves are managed by `CaptureContext`, the hook handles and their closures persist for the lifetime of the gateway object.

**What is done well:** Trial results are written to SQLite immediately after each inference call (runner.py lines 1132–1142). The system does not accumulate a list of results in memory. This is the correct pattern for long-running experiments. However, every other aspect of the pipeline — model weights, tokenizers, GPU cache, hook handles — remains in memory without lifecycle management.

### 2.2 Per-Trial Memory Timeline

```
Trial Start
  ├── Build prompt (~2 KB, ephemeral)
  ├── Tokenize input (~50 KB, ephemeral)
  ├── Forward pass (250 MB peak, then freed — but GPU cache retains allocation)
  ├── Decode output (~2 KB, ephemeral)
  ├── Parse + score (~1 KB, ephemeral)
  └── Write to DB (freed immediately)

Trial End: net memory delta ≈ 0 KB in Python heap, but +50–200 KB in GPU reserved cache
```

Over 5,000 trials, the GPU cache fragmentation alone can accumulate 250 MB–1 GB of reserved-but-unused memory. The fix is straightforward: call `torch.mps.empty_cache()` (or `torch.cuda.empty_cache()`) every N trials (e.g., every 100). This is a one-line change.

### 2.3 Steady-State Memory Budget (7B Model, fp16, MPS)

| Component | Size | Lifecycle |
|-----------|------|-----------|
| Model weights | 14 GB | Held for entire model loop |
| Tokenizer | 50 MB | Held for entire model loop |
| GPU reserved cache | 2–4 GB | Grows monotonically, never cleared |
| Hook handles (if capturing) | 38 KB + closure refs | Never removed |
| SQLite connection | ~1 MB | Held for entire run |
| Python heap (prompts, parsing) | 10–50 MB | Ephemeral per trial |
| **Total steady-state** | **~16–18 GB** | |
| **Peak (during forward pass)** | **~18–20 GB** | |

---

## 3. Execution Model: Why It Is Slow

### 3.1 The Sequential Triple Loop

The core execution in `runner.py` is:

```python
for model in models:                    # Outer: 1-7 models
    load_model()                        # 14 GB, ~30 seconds
    for item in items:                  # Middle: 400 items
        for condition in conditions:    # Inner: 3-12 conditions
            prompt = build_prompt()     # ~5 ms
            response = gateway.chat()   # ~2 seconds (BLOCKING)
            score = parse_and_score()   # ~10 ms
            db.write(result)            # ~1 ms
```

For a full run (7 models × 400 items × 12 conditions × 6 temperatures), that is 201,600 trials at approximately 2 seconds each — roughly 112 hours of wall-clock time, purely sequential.

### 3.2 Parallelization Opportunities

There are three axes along which trials could be parallelized, each with different constraints:

**Within a single model (item-level parallelism).** Items are independent — trial (item_42, condition_A) does not depend on trial (item_42, condition_B) or trial (item_43, condition_A). A `ThreadPoolExecutor` with 4–8 workers could process multiple items concurrently while sharing the same loaded model. The bottleneck is GPU memory: each concurrent inference requires its own forward pass allocation. On a 48 GB M-series Mac, 2–3 concurrent 7B inferences at fp16 are feasible.

**Across models (model-level parallelism).** With sufficient memory, two models could be loaded simultaneously on different devices (e.g., one on GPU, one on CPU). This is the highest-impact optimization for multi-model runs but requires explicit device management that does not exist today.

**Across temperatures (config-level parallelism).** Different temperature runs are entirely independent and could be dispatched as separate processes. This is the simplest form of parallelism — just run 6 processes with different suite configs — but the codebase has no built-in support for it.

### 3.3 Per-Trial Overhead

Beyond the 2-second inference call, each trial incurs:

| Operation | Time | Notes |
|-----------|------|-------|
| Condition param DB query | ~0.5 ms | Redundant — same params queried for every item |
| `time_step` COUNT query | ~0.3 ms | Could be an in-memory counter |
| JSON deserialization (3×) | ~0.2 ms | Three separate `json.loads()` calls per trial |
| Trial metadata DB write | ~0.5 ms | Could be batched |
| Prompt metadata DB write | ~0.5 ms | Could be batched |
| Output DB write | ~0.5 ms | Could be batched |
| **Total overhead** | **~2.5 ms** | **Negligible at 2s/trial, significant at scale** |

For 200,000 trials, the overhead alone is ~8 minutes. Not catastrophic, but the condition param queries (same 12 conditions queried 200,000 times) are wasteful. Preloading them into a dict would eliminate 200,000 redundant SQL queries.

---

## 4. Code Redundancy and Anti-Patterns

### 4.1 Duplicated Gateway Creation

Both `runner.py` and `suite_completion.py` implement gateway creation independently. The `suite_completion.py` version (lines 45–92) is a near-copy of the logic in `runner.py` (lines 741–847), including the same OLMo detection, HuggingFace cache path resolution, and device selection. When you fixed the path resolution for `runner.py`, `suite_completion.py` still has the old logic.

**Impact:** Bug fixes must be applied twice. Path resolution bugs (like the `/scratch1` issue) can persist in one copy after being fixed in another.

**Fix:** Extract to a shared `gateway_factory.py` with a single `create_gateway()` function.

### 4.2 Monolithic CLI (1,191 lines)

The CLI module registers 12 subcommands in a single file with a cascading `if mode == "..."` dispatcher. Each subcommand has its own argument parser setup (30–80 lines) and handler function. Adding a new experiment type means editing this file, importing new handlers, and adding new mode strings to the global list.

**Fix:** Plugin-based command registry where each experiment registers its own CLI subcommands.

### 4.3 Enhanced Scoring God Class (859 lines)

`enhanced_scoring.py` handles four distinct responsibilities in a single module: structural zone detection (reason/premise/conclusion boundaries), 4-tier candidate extraction (formal wrappers → assertions → MCQ patterns → shotgun), type-aware evaluation (math, MCQ, factual, opinion), and guard clauses (refusal detection, negation guards, attribution guards, endorsement detection).

**Fix:** Split into four focused modules of 150–200 lines each.

### 4.4 Analytics Layer Boilerplate

All 13 analytics modules follow the same three-function pattern:

```python
def compute_XXX_metrics(trace_db, run_id, run_dir) -> Dict:
    df = pd.read_sql_query("SELECT ... FROM conformity_trials ...", trace_db.conn)
    # groupby, aggregate
    return metrics

def generate_XXX_graphs(trace_db, run_id, run_dir) -> Dict:
    # matplotlib figures
    return paths

def export_XXX_logs(trace_db, run_id, run_dir) -> Dict:
    # JSON/CSV export
    return paths
```

There is no base class, no shared query library, and no common reporting interface. Each module independently constructs its SQL queries, creating 15+ near-identical `SELECT ... FROM conformity_trials t JOIN conformity_conditions c ON ...` patterns.

**Fix:** Create an `AnalyticsModule` abstract base class and centralize the common SQL queries in `TraceDb` helper methods.

### 4.5 Dead and Orphaned Code

| Module | Lines | Status |
|--------|-------|--------|
| `memory.py` | 475 | Feature-complete but never integrated into any experiment |
| `agent_langgraph.py` | 441 | Phase 1 agent, not used by OLMo experiments |
| `domain_state.py` | 198 | Phase 1 state management, not used |
| `channel.py` | ~100 | Messaging abstraction, not used |
| `replay.py` | 132 | Replay system, not used |
| `scheduler.py` | 152 | Task scheduling, not used |
| `export.py` | 108 | Data export, not used by current experiments |
| **Total orphaned** | **~1,606** | **7% of codebase** |

These modules represent earlier design iterations. Some (like `memory.py`) are well-written and could be integrated into the new framework design. Others should be archived.

---

## 5. Database and Persistence

### 5.1 Schema Design

The 17-table schema is moderately normalized with strategic JSON denormalization:

**Core tables** (experiment-agnostic): `runs`, `trace`, `messages`, `activation_metadata`, `merkle_log`, `agent_states`

**Conformity tables** (experiment-specific): `conformity_datasets`, `conformity_items`, `conformity_conditions`, `conformity_trials`, `conformity_prompts`, `conformity_prompt_metadata`, `conformity_trial_metadata`, `conformity_trial_steps`, `conformity_outputs`, plus analysis tables for probes, projections, interventions, and steering.

The schema uses WAL journaling and `PRAGMA synchronous = NORMAL`, which is the correct balance of safety and performance for this workload. Foreign keys are enforced.

### 5.2 Write Patterns

All writes are single-row inserts wrapped in `with self.conn:` (implicit commit). There is exactly one batch method in the entire persistence layer: `insert_conformity_projection_rows()`, which uses `executemany()` for probe projection bulk inserts.

For a typical trial, the system performs approximately 9 database operations: trial insert, trial metadata upsert, prompt insert, prompt metadata upsert, output insert, plus condition parameter reads and time_step counts. At 200,000 trials, that is 1.8 million database operations.

**Opportunity:** Batching writes in groups of 50–100 would reduce SQLite overhead by 10–20× for the write operations. The current per-trial commit pattern is safe but wasteful.

### 5.3 Data Duplication

Response data is stored in three forms within `conformity_outputs`: `raw_text` (full LLM output), `parsed_answer_text` (extracted answer), and `parsed_answer_json` (structured version). Trial metadata is stored both as structured columns in `conformity_trials` and as JSON blobs in `conformity_trial_metadata`. This is a conscious tradeoff — the structured columns enable fast queries while the JSON blobs preserve full fidelity — but it means each trial's metadata is stored roughly twice.

---

## 6. What Works Well

Not everything needs to change. Several modules are well-designed and should be preserved:

**`output_parsing.py`** (985 lines) is a genuinely excellent text normalization and quality classification pipeline. It handles unicode normalization, whitespace cleanup, special token removal, structural garbage detection, encoding issue detection, prompt leakage detection, and degenerate repetition detection — all with 25+ configurable parameters. This module is framework-quality code that happens to live in the wrong directory. It should be promoted to a first-class framework component.

**`persistence.py`** (1,187 lines) is a solid database layer. WAL journaling, proper foreign keys, `row_factory` for dict-like access, and immediate writes prevent data loss on crashes. The schema is well-thought-out.

**`settings.py`** (207 lines) has a clean environment variable resolution pattern with proper fallback chains. The only issue is the `paths.json` interaction, which we already debugged.

**`world_engine.py`** (293 lines) is a clean, model-agnostic simulation loop that could serve as the foundation for the new framework.

**The gateway abstraction** in `llm_gateway.py` is well-designed. The four backend types (LiteLLM, HuggingFace Hooked, TransformerLens, Mock) share a common `.chat()` interface, and the rate limiter implementation is solid.

---

## 7. Recommendations for the Framework Transition

### 7.1 Vision: Model-Driven Configuration

The target architecture should allow a user to define an experiment entirely through a JSON configuration:

```json
{
  "framework_version": "2.0",
  "experiment": {
    "name": "conformity_replication",
    "type": "survey_simulation"
  },
  "models": [
    {
      "id": "allenai/Olmo-3-7B-Think-DPO",
      "backend": "huggingface",
      "device": "mps",
      "dtype": "float16"
    },
    {
      "id": "meta-llama/Llama-3-8B-Instruct",
      "backend": "huggingface",
      "device": "mps",
      "dtype": "float16"
    }
  ],
  "conditions": [...],
  "datasets": [...],
  "run": {
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "max_items_per_dataset": 50,
    "parallelism": {
      "item_workers": 4,
      "model_strategy": "sequential"
    }
  }
}
```

No Python code changes required to add a new model. No OLMo-specific branching. The framework reads the config and dispatches accordingly.

### 7.2 Proposed Architecture Layers

```
┌─────────────────────────────────────────────────┐
│  CLI / Config Layer                              │
│  - Plugin-based command registry                 │
│  - JSON schema validation                        │
│  - Environment-aware path resolution             │
├─────────────────────────────────────────────────┤
│  Orchestration Layer                             │
│  - Model lifecycle manager (load/hold/release)   │
│  - Trial dispatcher (ThreadPool / ProcessPool)   │
│  - State machine (pending → running → done)      │
│  - Progress tracking and checkpointing           │
├─────────────────────────────────────────────────┤
│  Experiment Layer (pluggable)                    │
│  - Prompt builders (registered per experiment)   │
│  - Scorers (registered per experiment)           │
│  - Condition handlers (registered per experiment)│
├─────────────────────────────────────────────────┤
│  Gateway Layer                                   │
│  - HuggingFace, LiteLLM, TransformerLens, Mock  │
│  - Rate limiting, retry logic                    │
│  - Device management (GPU/MPS/CPU allocation)    │
├─────────────────────────────────────────────────┤
│  Persistence Layer                               │
│  - Batched writes (configurable batch size)      │
│  - Centralized query library                     │
│  - Schema migration support                      │
├─────────────────────────────────────────────────┤
│  Analytics Layer                                 │
│  - AnalyticsModule base class                    │
│  - Pluggable metric/graph/export modules         │
│  - Shared SQL query library                      │
└─────────────────────────────────────────────────┘
```

### 7.3 Model Lifecycle Management

The single most impactful addition would be an explicit model lifecycle manager:

```python
class ModelManager:
    """Manages model loading, caching, and memory cleanup."""

    def __init__(self, max_loaded_models=1, device_budget_gb=16):
        self._loaded: OrderedDict[str, Gateway] = OrderedDict()
        self._max_loaded = max_loaded_models
        self._device_budget = device_budget_gb

    def acquire(self, model_id: str, config: dict) -> Gateway:
        """Load model if not cached, evict LRU if at capacity."""
        if model_id in self._loaded:
            self._loaded.move_to_end(model_id)
            return self._loaded[model_id]
        if len(self._loaded) >= self._max_loaded:
            self._evict_lru()
        gateway = self._create_gateway(model_id, config)
        self._loaded[model_id] = gateway
        return gateway

    def release(self, model_id: str):
        """Explicitly release model and reclaim memory."""
        if model_id in self._loaded:
            gw = self._loaded.pop(model_id)
            gw.cleanup()       # Remove hooks, delete tensors
            gc.collect()
            torch.mps.empty_cache()  # or torch.cuda.empty_cache()
```

This replaces the current pattern where models are loaded at loop start and held until loop end, with explicit memory tracking and LRU eviction.

### 7.4 Parallelization Strategy

**Phase 1 — Item-level parallelism (biggest win, lowest risk):**
Use `concurrent.futures.ThreadPoolExecutor` to process multiple items concurrently within a single model. The gateway is thread-safe (PyTorch inference is GIL-safe under `torch.no_grad()`). Start with 2 workers and scale based on available memory.

**Phase 2 — Config-level parallelism:**
Allow the framework to dispatch multiple temperature configs as separate processes, each with its own model instance. A simple process pool with `--temperature 0.8` flags would work.

**Phase 3 — Model-level parallelism (future):**
For machines with multiple GPUs or sufficient memory, load two models simultaneously on different devices. This requires the `ModelManager` to be device-aware.

### 7.5 Refactoring Phases

**Phase 1 — Decouple (2–3 days):** Extract `gateway_factory.py` from duplicated code. Split `enhanced_scoring.py` into four modules. Create `AnalyticsModule` base class. Centralize SQL queries.

**Phase 2 — Generalize (1 week):** Replace OLMo-specific branching in `runner.py` with model-agnostic gateway selection driven by config. Implement plugin-based CLI. Move `output_parsing.py` to framework layer.

**Phase 3 — Scale (1–2 weeks):** Implement `ModelManager` with lifecycle control. Add `ThreadPoolExecutor` for item-level parallelism. Add batch database writes. Implement GPU cache cleanup.

---

## 8. Line Count Reduction Opportunities

The current codebase is 23,695 lines. A disciplined refactoring could reduce this significantly:

| Action | Lines Removed | Lines Added | Net |
|--------|--------------|-------------|-----|
| Archive dead code (memory, agent, replay, etc.) | -1,606 | 0 | -1,606 |
| Deduplicate gateway creation | -92 | +30 | -62 |
| AnalyticsModule base class (eliminates boilerplate) | -1,500 | +200 | -1,300 |
| Centralize SQL queries | -400 | +150 | -250 |
| Plugin CLI (replaces monolithic dispatcher) | -600 | +200 | -400 |
| Split + simplify enhanced_scoring | -859 | +600 | -259 |
| Remove Phase 1/2/3 backwards compat from run.py | -300 | 0 | -300 |
| **Total** | **-5,357** | **+1,180** | **-4,177** |

Target: approximately 19,500 lines with better modularity, zero dead code, and no duplication.

---

## 9. Immediate Action Items

These can be done before the larger refactoring effort:

1. **Fix GPU cache fragmentation.** Add `torch.mps.empty_cache()` every 100 trials in `runner.py`. One line, immediate memory improvement.

2. **Preload condition parameters.** Load all condition params into a dict before the trial loop instead of querying per-trial. Eliminates 200,000 redundant SQL queries.

3. **Replace in-loop COUNT query with counter.** The `time_step` query (runner.py lines 991–994) runs a `SELECT COUNT(*)` per trial. Replace with an incrementing integer.

4. **Create `paths_local.json`.** Add a local paths config alongside the HPC one, and update `io.py` to check for `paths_local.json` first. Eliminates the need to set environment variables for local runs.

5. **Add hook cleanup to `HuggingFaceHookedGateway`.** Implement `__del__()` or a `cleanup()` method that removes registered hooks. Prevents memory leaks during long sessions.

---

## Appendix A: File-by-File Assessment

| File | Lines | Health | OLMo-Specific | Action |
|------|-------|--------|---------------|--------|
| `llm_gateway.py` | 1,606 | Good | 15% | Add lifecycle methods |
| `persistence.py` | 1,187 | Good | 30% | Add batch writes, centralize queries |
| `runner.py` | 1,164 | Fair | 80% | Generalize, add parallelism |
| `cli.py` | 1,191 | Poor | 100% | Replace with plugin registry |
| `output_parsing.py` | 985 | Excellent | 5% | Promote to framework layer |
| `enhanced_scoring.py` | 859 | Fair | 95% | Split into 4 modules |
| `suite_completion.py` | 733 | Poor | 100% | Extract gateway factory |
| `run.py` | 640 | Fair | 10% | Archive Phase 1/2/3 code |
| `memory.py` | 475 | Good | 0% | Integrate or document |
| `world_engine.py` | 293 | Good | 0% | Keep as-is |
| `settings.py` | 207 | Good | 0% | Keep as-is |
| Analytics (13 files) | 5,331 | Fair | 100% | Base class refactor |
| Legacy (7 files) | 1,606 | N/A | N/A | Archive |

## Appendix B: Key Code Locations

| Concern | File | Lines | Notes |
|---------|------|-------|-------|
| Main trial loop | `runner.py` | 883–1142 | Sequential, no batching |
| Gateway selection | `runner.py` | 741–847 | OLMo-specific branching |
| Gateway selection (duplicate) | `suite_completion.py` | 45–92 | Copy of above |
| Model loading | `llm_gateway.py` | 1025–1070 | No lifecycle management |
| Hook registration | `llm_gateway.py` | 1073–1239 | No removal mechanism |
| GPU cache cleanup | `runner.py` | 1147–1160 | Per-model only, not per-trial |
| Rate limiter accumulation | `llm_gateway.py` | 174, 200, 279 | `_request_times` list grows |
| Condition param queries | `runner.py` | 908 | Redundant per-trial SQL |
| `time_step` COUNT | `runner.py` | 991–994 | Redundant per-trial SQL |
| paths.json loading | `io.py` | 45–90 | No local override support |
| HPC path hardcoding | `configs/paths.json` | 1–7 | `/scratch1` paths |
