# Phase 1 Accomplishments: The Abstract Agent Machine - Designed for Olmo

## Introduction: Design Intuition Centered on Olmo

The Abstract Agent Machine is a deterministic simulation framework designed specifically to leverage the **unique open-source transparency of the Olmo model family** for mechanistic interpretability research. Unlike proprietary models where researchers are limited to behavioral observation, Olmo's "glass box" architecture enables us to observe and intervene at the level of individual neurons.

### Why Olmo is Central

The entire system architecture is built around Olmo's distinctive characteristics:

- **Full Weight Access**: Unlike closed APIs (GPT-4, Claude), Olmo provides direct access to model weights, enabling activation capture, probe training, and intervention experiments that are impossible with black-box models.

- **Multiple Variants for Comparative Analysis**: Olmo offers a controlled "model flow" with variants (Base, Instruct, Think, RL-Zero) that share a common architecture but differ in training methodology. This enables rigorous ablation studies on the effects of alignment techniques.

- **Standard Transformer Architecture**: Olmo's compatibility with standard interpretability tools (HuggingFace, TransformerLens) means we can apply well-established mechanistic interpretability techniques without custom adapters.

- **Open Training Data**: The Dolma 3 dataset is publicly available, allowing provenance tracking and understanding how training data influences model behavior.

### System Design Philosophy

Every component of the Abstract Agent Machine is optimized to enable mechanistic interpretability research on Olmo:

- **Deterministic Simulation**: Reproducible Olmo experiments with identical seeds producing identical traces, enabling rigorous scientific investigation.

- **Separation of Concerns**: Clean separation between Platform (WorldEngine) and Agent (Policy) enables seamless integration of Olmo models while maintaining deterministic execution.

- **Trace as Truth**: Complete audit trail of all Olmo behavior, with activation capture aligned to trace events, enabling correlation between neural activations and behavioral outputs.

- **Activation Capture Aligned to Events**: The system captures neural activations synchronized with trace events, enabling probe training and intervention experiments that link internal representations to behavior.

- **Rule Adjustment as Activation Steering**: The system enables "rule adjustment" through activation steering—modifying model behavior by directly manipulating internal representations. This is only possible with Olmo's open weights.

### The Olmo Conformity Experiment: Primary Use Case

The system's architecture is driven by the Olmo Conformity Experiment—a mechanistic interpretability study that tests how different Olmo variants respond to social pressure. This experiment demonstrates the full power of Olmo's transparency:

- **Behavioral Trials**: Testing how Base, Instruct, Think, and RL-Zero variants respond to incorrect social consensus
- **Activation Capture**: Capturing internal activations during trials for analysis
- **Probe Training**: Training linear probes to identify "truth vectors" and "social consensus vectors" in activation space
- **Interventions**: Using activation steering to modify behavior by subtracting social vectors from the residual stream
- **Comparative Analysis**: Comparing variants to understand how different training methodologies affect conformity

This experiment would be impossible with closed models—it requires the full transparency that only Olmo provides.

## Core Engine: WorldEngine (Designed for Olmo)

### Intuition

The `WorldEngine` is the central orchestrator of the Abstract Agent Machine, designed specifically to enable Olmo mechanistic interpretability research. It functions as a deterministic state machine that:

- Ensures reproducible Olmo experiments through deterministic step execution
- Enables activation capture synchronized with trace events
- Maintains trace alignment that allows correlation between neural activations and behavioral outputs
- Provides integrity checking through state hashing for experimental validation

The engine's design philosophy centers on enabling researchers to observe and intervene in Olmo's internal processes while maintaining scientific rigor through determinism and complete traceability.

### Implementation

**Location**: `src/aam/world_engine.py`

**Core Architecture**:

The `WorldEngine` orchestrates simulation steps with the following flow:

1. **Observation Building**: For each agent at each time step, the engine builds an observation containing:
   - Current time step
   - Agent identifier
   - Recent message history (for cognitive agents)
   - Available tools
   - Memory context (if memory system is enabled)

2. **Agent Decision Collection**: The engine calls each agent's `decide()` method with the observation, collecting action requests. Agents are processed in deterministic order (sorted by `agent_id`).

3. **Action Execution**: Actions are executed sequentially in deterministic order, with each action producing:
   - An `ActionResult` indicating success/failure
   - A `TraceEvent` recording the complete request/response

4. **State Hash Computation**: After all actions in a step are committed, the engine computes a SHA-256 hash of the environment state (messages, trace events) for integrity checking.

5. **Activation Capture Integration**: If a `CaptureContext` is provided, the engine flushes activation buffers after each step, ensuring activations are aligned to time steps for analysis.

**Key Methods**:

```python
def step(self, *, time_step: int) -> None:
    """Execute one simulation step with deterministic ordering."""
    # Build observations for all agents
    # Collect action requests (synchronously)
    # Sort deterministically by agent_id
    # Commit all requests

def build_observation(self, *, time_step: int, agent_id: str) -> Observation:
    """Build observation for an agent, including message history and tools."""
    # Fetch recent messages from database
    # Enrich with memory if available
    # Return structured observation

def execute(self, req: ActionRequest, *, timestamp: float) -> Tuple[ActionResult, TraceEvent]:
    """Execute a single action and generate trace event."""
    # Handle action types (noop, emit_event, post_message, domain actions)
    # Generate result and trace event

def commit_requests(self, *, time_step: int, reqs: List[ActionRequest]) -> None:
    """Commit all requests for a step, compute state hash, flush activations."""
    # Execute all actions
    # Compute state hash
    # Update trace events with state hash
    # Flush activation buffers
```

**Olmo Integration**:

The engine integrates with Olmo models through:

- **HuggingFaceHookedGateway**: When using local Olmo models, the gateway captures activations during inference, which are then flushed by the engine after each step.

- **Trace Alignment**: Each trial execution is assigned a `time_step` and `agent_id`, allowing activations to be precisely aligned with behavioral outputs in the database.

- **State Integrity**: The state hash ensures that experimental runs are reproducible and that activations correspond to the correct behavioral state.

**Example Usage**:

```python
from aam.world_engine import WorldEngine, WorldEngineConfig
from aam.channel import InMemoryChannel
from aam.persistence import TraceDb, TraceDbConfig
from aam.llm_gateway import HuggingFaceHookedGateway
from aam.interpretability import CaptureContext, CaptureConfig

# Setup database
trace_db = TraceDb(TraceDbConfig(db_path="experiment.db"))
trace_db.connect()
trace_db.init_schema()

# Setup activation capture for Olmo
cap_config = CaptureConfig(
    layers=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    components=["resid_post"],
    trigger_actions=["trial_execution"],
    token_position=-1,
)
cap_ctx = CaptureContext(
    output_dir="activations/",
    config=cap_config,
    dtype="float16",
    trace_db=trace_db,
)

# Setup Olmo gateway
gateway = HuggingFaceHookedGateway(
    model_id_or_path="allenai/Olmo-3-7B-Instruct",
    capture_context=cap_ctx,
)

# Create engine
engine = WorldEngine(
    config=WorldEngineConfig(run_id="olmo_experiment_001"),
    agents={},  # Agents configured separately
    channel=InMemoryChannel(),
    trace_db=trace_db,
    capture_context=cap_ctx,
)

# Run simulation
engine.run(steps=100)
```

### Troubleshooting

**Common Issues**:

1. **Activation Capture Not Working**:
   - Ensure `CaptureContext` is passed to both `HuggingFaceHookedGateway` and `WorldEngine`
   - Verify that `capture_activations=True` is set in experiment configuration
   - Check that activation output directory exists and is writable

2. **State Hash Mismatches**:
   - State hashes are computed after all actions in a step are committed
   - If hashes differ between runs, check for non-deterministic behavior (e.g., random number generation, API calls)
   - Verify that the same seed is used for both runs

3. **Trace Alignment Issues**:
   - Ensure `time_step` and `agent_id` are correctly assigned to trials
   - Check that `upsert_conformity_trial_step()` is called before activation capture
   - Verify that activation metadata includes correct `time_step` and `agent_id`

4. **Performance Issues**:
   - Activation capture adds significant overhead; use sparse sampling (specific layers/components)
   - Consider using `float16` dtype for activations to reduce memory usage
   - For large experiments, consider capturing activations only for specific trials

## Core Components (Olmo-Enabled)

Each component of the Abstract Agent Machine is designed to enable Olmo mechanistic interpretability research. This section explains how each component supports Olmo's transparency and enables the conformity experiment.

### 1. Persistence: SQLite Trace Database

**Intuition**: The persistence layer implements the "Trace as Truth" principle, providing a complete audit trail of all Olmo behavior. For the conformity experiment, it stores not just behavioral outputs but also activation metadata, probe artifacts, and intervention results, enabling comprehensive analysis.

**Implementation**: `src/aam/persistence.py`

**Core Tables**:

- **`runs`**: Experiment run metadata (seed, configuration)
- **`trace`**: Action requests and results with state hashes
- **`messages`**: Shared message feed for cognitive agents
- **`activation_metadata`**: References to activation files aligned to trace events

**Olmo Conformity Experiment Tables**:

- **`conformity_datasets`**: Datasets used (immutable facts, social conventions)
- **`conformity_items`**: Individual test items (questions, ground truth)
- **`conformity_conditions`**: Experimental conditions (control, synthetic_asch_history, authoritative_bias)
- **`conformity_trials`**: Individual trials (model variant, item, condition)
- **`conformity_prompts`**: Prompt templates and rendered prompts
- **`conformity_outputs`**: Model responses with correctness evaluation
- **`conformity_probes`**: Trained probe metadata (truth vector, social vector)
- **`conformity_probe_projections`**: Probe projections on trial activations
- **`conformity_interventions`**: Intervention configurations (alpha, target layers)
- **`conformity_intervention_results`**: Before/after intervention outputs
- **`conformity_think_tokens`**: Think token analysis for Olmo-Think variants
- **`conformity_logit_lens`**: Logit lens analysis across layers

**Key Features**:

- Deterministic JSON serialization ensures reproducibility
- WAL mode enables concurrent reads during analysis
- Foreign key constraints maintain data integrity
- Indexes optimize queries for analysis workflows

**Olmo-Specific Features**:

- Activation metadata table links activation files to trace events
- Probe artifacts stored as Safetensors files, referenced in database
- Trial step alignment enables precise activation-to-behavior correlation

**Troubleshooting**:

- **Database Lock Errors**: Ensure WAL mode is enabled; close connections properly
- **Missing Activation References**: Verify `insert_activation_metadata()` is called after activation capture
- **Probe Artifact Not Found**: Check that probe artifacts are saved before inserting probe records

### 2. Channel: Communication Abstraction

**Intuition**: The `Channel` protocol decouples agents from the platform, enabling clean integration of Olmo models. For the conformity experiment, this abstraction allows trials to be executed without direct coupling to the engine.

**Implementation**: `src/aam/channel.py`

**InMemoryChannel**: Thread-safe in-memory queue used for synchronous execution. Actions are collected during the "think" phase and then committed deterministically.

**Olmo Integration**: The channel enables Olmo models to submit action requests (trial responses) that are then processed by the engine, maintaining separation between model inference and platform execution.

**Troubleshooting**: Channel issues are rare; if actions are lost, check that `take_all()` is called after all agents have submitted requests.

### 3. Policy System: Agent Decision-Making

**Intuition**: The `AgentPolicy` protocol enables pluggable agent implementations. For Olmo experiments, agents are typically stateless—each trial is independent, and the policy is re-instantiated per trial to ensure determinism.

**Implementation**: `src/aam/policy.py`, `src/aam/agent_langgraph.py`

**Policy Types**:

- **RandomAgentPolicy**: Mock agent for Phase 1 testing (not used in Olmo experiments)
- **CognitiveAgentPolicy**: LLM-driven agent using LangGraph (can use Olmo via API)
- **SimpleCognitivePolicy**: Fallback without LangGraph dependency

**Olmo Integration**: For the conformity experiment, agents are typically not used in the traditional sense—instead, trials are executed directly via the gateway, with the engine managing the trial lifecycle.

**Seed Derivation**: Each agent receives a deterministic seed derived from the master seed and agent ID using SHA-256 hashing, ensuring reproducibility.

**Troubleshooting**: Policy issues are uncommon in Olmo experiments since trials execute directly. If using cognitive policies with Olmo, ensure the gateway is correctly configured.

### 4. Tools: Action Definitions

**Intuition**: Tools define the action space available to agents. For Olmo experiments, tools are typically not used—trials are direct model queries rather than tool-using agents.

**Implementation**: `src/aam/tools.py`

**Available Tools**: `post_message` (for cognitive agents in Phase 2 simulations)

**Olmo Integration**: Tools are not directly used in conformity experiments, but the tool system demonstrates the extensibility of the action space.

### 5. LLM Gateway: Model Interface

**Intuition**: The LLM Gateway provides a unified interface to different model serving methods. For Olmo, this is critical—different serving methods enable different capabilities (activation capture vs. convenience vs. efficiency).

**Implementation**: `src/aam/llm_gateway.py`

**Gateway Types**:

1. **HuggingFaceHookedGateway** (Primary for Olmo Interpretability):
   - Direct access to Olmo weights via HuggingFace
   - Activation hooks for mechanistic interpretability
   - Supports all Olmo variants (Base, Instruct, Think, RL-Zero)
   - Enables probe training and interventions
   - **Storage**: Models cached in `models/huggingface_cache/allenai_Olmo-3-*`

2. **TransformerLensGateway** (Alternative):
   - TransformerLens compatibility for models with TL support
   - Similar activation capture capabilities

3. **LiteLLMGateway** (API Access):
   - Multi-provider support (OpenAI, Anthropic, Ollama)
   - Used for Olmo via Ollama API or GGUF via llama-server
   - Rate limiting support for production APIs
   - **Storage**: Models served via Ollama (`~/.ollama/models/`) or llama-server

4. **MockLLMGateway** (Testing):
   - Deterministic offline LLM for testing
   - Not used in Olmo experiments

**Olmo-Specific Features**:

- Automatic model download and caching
- Activation hook registration for capture
- Support for Olmo-Think token parsing
- Variant detection and configuration

**Troubleshooting**:

- **Model Download Failures**: Check internet connection, disk space (~14GB for 7B models), and HuggingFace authentication
- **Activation Hooks Not Firing**: Verify `capture_context` is passed to gateway and `capture_activations=True` is set
- **Ollama API Errors**: Ensure Ollama is running and model is imported (`ollama pull olmo-3-7b-instruct`)

### 6. Interpretability: Activation Capture

**Intuition**: The interpretability layer enables full access to Olmo's internal activations, which is the core capability that makes mechanistic interpretability research possible. Without this, we could only observe behavior, not understand mechanisms.

**Implementation**: `src/aam/interpretability.py`

**Core Components**:

- **CaptureConfig**: Configuration for sparse sampling (layers, components, token positions)
- **CaptureContext**: Manages activation capture lifecycle and file storage
- **Activation Storage**: Safetensors format for efficient tensor storage

**Olmo Integration**:

- Activations captured from HuggingFace-hooked Olmo models
- Aligned to trace events via `time_step` and `agent_id`
- Stored as Safetensors files for probe training
- Metadata stored in database for querying

**Key Features**:

- Sparse sampling: Capture only specific layers/components to reduce storage
- Token position selection: Capture activations at specific token positions (e.g., last token)
- Dtype selection: `float16` or `float32` for memory/accuracy tradeoff
- Action filtering: Capture only for specific action types (e.g., `trial_execution`)

**Troubleshooting**:

- **Activation Files Not Created**: Check output directory permissions, verify hooks are registered
- **Memory Issues**: Use `float16` dtype, reduce layers/components, use sparse sampling
- **Alignment Issues**: Ensure `time_step` and `agent_id` are correctly set before capture

### 7. Memory System: Agent Context

**Intuition**: The memory system enables agents to maintain context across steps. For Olmo experiments, this is typically not used since trials are independent, but it demonstrates the system's extensibility.

**Implementation**: `src/aam/memory.py`

**Features**: Observation enrichment, action storage, summarization

**Olmo Integration**: Not directly used in conformity experiments, but available for future multi-turn Olmo interactions.

### 8. Domain State: Extensible Domain Logic

**Intuition**: Domain state handlers enable domain-specific action execution. For Olmo experiments, this is not used, but it shows how the system can be extended for different research domains.

**Implementation**: `src/aam/domain_state.py`

**Example**: `SocialMediaDomainHandler` demonstrates how domain-specific logic can be integrated.

### 9. Scheduler: Async Execution

**Intuition**: The barrier scheduler enables concurrent agent execution while maintaining determinism. For Olmo experiments, this enables parallel trial execution with rate limiting.

**Implementation**: `src/aam/scheduler.py`

**Features**: Async execution, retry logic, deterministic ordering

**Olmo Integration**: Enables efficient parallel trial execution with API rate limiting.

**Troubleshooting**: Retry logic handles transient API failures; check rate limit configuration if trials are failing.

## Model Serving Infrastructure (Critical for Olmo)

### Intuition

Olmo models can be served in three different ways, each enabling different capabilities:

1. **Local HuggingFace** (Primary for Interpretability): Full activation access, hooks for probes/interventions
2. **Ollama API** (Convenience): Fast API access, no activation capture overhead
3. **GGUF via Llama CPP** (Efficiency): Lower memory, faster inference, GPU acceleration

The system is designed to support all three methods, allowing researchers to choose the optimal approach for their specific needs.

### Implementation

#### Olmo Models - Local HuggingFace (Primary for Interpretability)

**Storage**: Models are cached in `models/huggingface_cache/allenai_Olmo-3-*` directory structure.

**Served via**: `HuggingFaceHookedGateway` - Direct access to Olmo weights with activation hooks.

**Advantages**:
- Full activation access for mechanistic interpretability
- Hooks for probe training and interventions
- Direct weight access for activation steering
- Supports all Olmo variants

**Setup**:
```python
from aam.llm_gateway import HuggingFaceHookedGateway
from aam.interpretability import CaptureContext, CaptureConfig

# Models auto-download on first use
# Or explicitly ensure download:
from aam.experiments.olmo_conformity.olmo_utils import ensure_olmo_model_downloaded

ensure_olmo_model_downloaded(
    model_id="allenai/Olmo-3-7B-Instruct",
    models_dir="models",
    import_to_ollama=False,  # Don't import to Ollama
)

# Create gateway with activation capture
cap_config = CaptureConfig(layers=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
cap_ctx = CaptureContext(output_dir="activations/", config=cap_config, trace_db=trace_db)

gateway = HuggingFaceHookedGateway(
    model_id_or_path="allenai/Olmo-3-7B-Instruct",
    capture_context=cap_ctx,
)
```

**Usage in Experiments**:
- Default when no `--api-base` is specified
- Required for activation capture
- Required for probe training and interventions

#### Olmo Models - Ollama API (Convenience)

**Storage**: Models stored in Ollama's internal storage (`~/.ollama/models/`).

**Served via**: `LiteLLMGateway` with `api_base=http://localhost:11434/v1`.

**Advantages**:
- Fast API access without activation capture overhead
- Easy model management via Ollama CLI
- Supports multiple models simultaneously

**Setup**:
```bash
# Import Olmo model to Ollama
ollama pull olmo-3-7b-instruct

# Or use helper function:
python -c "
from aam.experiments.olmo_conformity.olmo_utils import ensure_olmo_model_downloaded
ensure_olmo_model_downloaded('allenai/Olmo-3-7B-Instruct', import_to_ollama=True)
"
```

**Usage**:
```python
from aam.llm_gateway import LiteLLMGateway

gateway = LiteLLMGateway(
    api_base="http://localhost:11434/v1",
    api_key=None,  # Not needed for local Ollama
)
```

**CLI Usage**:
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --api-base http://localhost:11434/v1 \
  --model olmo-3-7b-instruct
```

**Limitations**: Cannot capture activations (API is black-box).

#### Olmo Models - GGUF via Llama CPP (Efficiency)

**Storage**: GGUF files in `models/*.gguf` (converted from HuggingFace).

**Served via**: `llama-server` binary → `LiteLLMGateway` with `api_base=http://127.0.0.1:8081/v1`.

**Advantages**:
- Lower memory usage (quantized models)
- Faster inference
- GPU acceleration (Metal on Apple Silicon, CUDA on Linux)
- Platform-specific optimizations

**Setup**:
```bash
# Convert Olmo models to GGUF (one-time)
python experiments/olmo_conformity/download_and_convert_olmo_models.py

# This downloads HuggingFace models and converts them to GGUF format
# Output: models/olmo-3-7b-instruct.gguf, etc.
```

**Serving**:
```bash
# Terminal 1: Start llama-server
python -m aam.run llama serve models/olmo-3-7b-instruct.gguf

# Terminal 2: Run experiment
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --api-base http://127.0.0.1:8081/v1 \
  --api-key local \
  --model olmo-3-7b-instruct.gguf
```

**Configuration**:
- Default port: 8081
- GPU acceleration: Automatic (Metal on Apple Silicon, CUDA if available)
- Context size: Configurable via `LlamaServerConfig`

**Limitations**: Cannot capture activations (served via API).

#### Other Models - API Serving

**Any OpenAI-compatible API** via `LiteLLMGateway`:
- OpenAI, Anthropic, etc.
- Rate limiting support for production APIs
- Works with any model served via Ollama

**Example**:
```python
gateway = LiteLLMGateway(
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    rate_limit_config=RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=90000,
    ),
)
```

#### Other Models - Local GGUF via Llama CPP

**Model Discovery**: `src/aam/model_discovery.py`

Discovers GGUF models from:
- **Ollama**: `~/.ollama/models/` (parses manifests)
- **LM Studio**: `~/Library/Application Support/LM Studio/models/` (macOS)

**Export to Local Directory**:
```bash
# List discovered models
python -m aam.run llama list

# Export to models/ directory (symlinks by default)
python -m aam.run llama export

# Models exported as: models/ollama__library_smollm2_135m.gguf, etc.
```

**Serving**:
```bash
# Serve any discovered model
python -m aam.run llama serve models/ollama__library_smollm2_135m.gguf
```

**Platform-Specific GPU Acceleration**:
- **Apple Silicon (M1/M2/M3)**: Metal acceleration by default (all layers on GPU)
- **Linux with CUDA**: CUDA acceleration if available
- **CPU-only**: Falls back to CPU if no GPU available

### Troubleshooting

**Model Download Failures**:
- Check internet connection and HuggingFace authentication
- Verify sufficient disk space (~14GB for 7B models, ~60GB for 32B)
- Check HuggingFace cache directory permissions

**Ollama Integration Issues**:
- Ensure Ollama is installed and running: `ollama serve`
- Verify model is imported: `ollama list`
- Check Ollama API is accessible: `curl http://localhost:11434/api/tags`

**Llama CPP Build Problems**:
- Ensure llama.cpp is cloned in `third_party/llama.cpp/`
- Build llama-server: `cd third_party/llama.cpp && make llama-server`
- Check platform-specific build requirements (Metal on macOS, CUDA on Linux)

**GPU Acceleration Configuration**:
- **Apple Silicon**: Metal acceleration is automatic; check with `python -m aam.run llama serve --help`
- **Linux CUDA**: Ensure CUDA toolkit is installed and llama.cpp is built with CUDA support
- **CPU-only**: Set `n_gpu_layers=0` in `LlamaServerConfig`

**Model Discovery Failures**:
- Check that Ollama or LM Studio is installed
- Verify model directories exist and are readable
- Check manifest parsing (Ollama) or file scanning (LM Studio) permissions

## Rule Adjustment System (Olmo's Open Weights Enable This)

### Intuition

The "rule adjustment" system enables modifying Olmo's behavior by directly manipulating its internal representations through activation steering. This is only possible because Olmo provides open weights—closed models (GPT-4, Claude) cannot be modified in this way.

The system implements three key mechanisms:

1. **Probe Training**: Learn linear directions in activation space that correspond to concepts (truth, social consensus)
2. **Vector Analysis**: Track how these vectors interact across layers during inference
3. **Interventions**: Modify behavior by subtracting learned vectors from activations at specific layers

This enables researchers to answer questions like: "Can we make Olmo more truthful by steering away from social consensus?"—questions that are impossible to answer with closed models.

### Implementation

#### Probe Training

**Location**: `src/aam/experiments/olmo_conformity/probes.py`

**Process**:

1. **Activation Capture**: Capture activations from Olmo models on labeled datasets (truth vs. false statements, social consensus vs. opposition)

2. **Linear Probe Training**: Train logistic regression classifiers on activations to distinguish truth from falsehood, or social consensus from opposition:
   ```python
   # Train truth probe
   truth_probe = train_probe_from_captured_activations(
       trace_db=trace_db,
       run_id=run_id,
       train_dataset_id=truth_dataset_id,
       model_id="allenai/Olmo-3-7B-Instruct",
       probe_kind="truth",
       layers=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
       component="hook_resid_post",
       token_position=-1,
   )
   ```

3. **Probe Artifact Storage**: Save probe weights as Safetensors files for use in interventions:
   ```python
   # Probe weights saved as: artifacts/probe_truth_layer_15.safetensors
   # Contains: layer_15.weight (the "truth vector")
   ```

4. **Projection Computation**: Compute projections of trial activations onto probe vectors to track how truth/social vectors evolve across layers:
   ```python
   compute_and_store_probe_projections_for_trials(
       trace_db=trace_db,
       run_id=run_id,
       probe_id=truth_probe_id,
       artifact_path="artifacts/probe_truth.safetensors",
   )
   ```

**Key Functions**:
- `capture_probe_dataset_to_db()`: Capture activations for probe training dataset
- `train_probe_from_captured_activations()`: Train linear probe on captured activations
- `compute_and_store_probe_projections_for_trials()`: Compute projections for analysis

**Olmo-Specific Considerations**:
- Probes are trained on specific Olmo variants (Base, Instruct, Think, RL-Zero)
- Think variants require special handling for `<think>` tokens
- Probe accuracy varies by layer (typically highest in middle-to-late layers)

#### Vector Analysis

**Location**: `src/aam/experiments/olmo_conformity/vector_analysis.py`

**Process**:

1. **Truth Vector Training**: Train probe to identify truthfulness in activations
2. **Social Consensus Vector Training**: Train probe to identify social consensus signals
3. **Projection Tracking**: Track how truth and social vectors interact across layers during trials
4. **Collision Point Identification**: Identify layers where social vector overrides truth vector

**Key Function**:
```python
run_truth_social_vector_analysis(
    trace_db=trace_db,
    run_id=run_id,
    model_id="allenai/Olmo-3-7B-Instruct",
    truth_probe_dataset_path="datasets/truth_probe_train.jsonl",
    social_probe_dataset_path="datasets/social_conventions/minimal_items.jsonl",
    layers=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    component="hook_resid_post",
    token_position=-1,
    artifacts_dir="artifacts/",
)
```

**Output**: 
- Probe artifacts (Safetensors files)
- Projection data stored in `conformity_probe_projections` table
- Analysis visualizations (if enabled)

#### Interventions (Activation Steering)

**Location**: `src/aam/experiments/olmo_conformity/intervention.py`

**Process**:

1. **Load Probe Artifact**: Load trained social consensus vector from Safetensors file
2. **Baseline Generation**: Generate response without intervention (baseline behavior)
3. **Intervention Generation**: Generate response with activation steering:
   ```python
   # At target layers, subtract alpha * social_vector from residual stream
   def intervention_hook(activation, hook):
       return activation - (alpha * social_vector)[None, None, :]
   
   # Apply hook at layers 15-20
   model.add_hook("blocks.15.hook_resid_post", intervention_hook)
   model.add_hook("blocks.16.hook_resid_post", intervention_hook)
   # ... etc.
   ```
4. **Comparison**: Compare baseline vs. intervention outputs to measure effect

**Key Function**:
```python
run_intervention_sweep(
    trace_db=trace_db,
    run_id=run_id,
    model_id="allenai/Olmo-3-7B-Instruct",
    probe_artifact_path="artifacts/probe_social.safetensors",
    social_probe_id=social_probe_id,
    target_layers=[15, 16, 17, 18, 19, 20],
    component_hook="hook_resid_post",
    alpha_values=[0.5, 1.0, 2.0],  # Different intervention strengths
    max_new_tokens=64,
)
```

**Intervention Mechanism**:
- Vector subtraction: `x_layer ← x_layer - α * v_social`
- Alpha controls intervention strength (higher = stronger effect)
- Target layers are typically middle-to-late layers where social vector dominates

**Olmo-Specific Considerations**:
- Requires HuggingFace-hooked Olmo models (cannot use API)
- Think variants require special handling for `<think>` tokens
- Intervention effects vary by variant (Instruct more affected than Base)

### Troubleshooting

**Probe Training Issues**:
- **Low Accuracy**: Try different layers (typically 15-20 work best), ensure sufficient training data
- **Probe Artifact Not Found**: Verify probe training completed and artifact was saved
- **Activation Mismatch**: Ensure probe was trained on same model variant and layer structure

**Intervention Failures**:
- **No Effect**: Try higher alpha values, check target layers are correct
- **Model Errors**: Verify model is HuggingFace-hooked (not API), check hook registration
- **Memory Issues**: Reduce number of target layers, use smaller alpha values

**Vector Analysis Issues**:
- **Missing Projections**: Ensure probe projections were computed for trials
- **Incorrect Collision Points**: Check layer indices match model architecture

## Olmo Conformity Experiment (Primary Use Case)

### Intuition

The Olmo Conformity Experiment is the primary use case that drives the Abstract Agent Machine's architecture. It implements the "Synthetic Asch Paradigm"—a computational adaptation of Solomon Asch's classic social psychology experiments, designed for silicon cognition.

The experiment tests how different Olmo variants (Base, Instruct, Think, RL-Zero) respond to social pressure when presented with incorrect information. By leveraging Olmo's transparency, we can:

1. **Observe Behavior**: Measure how often each variant conforms to incorrect social consensus
2. **Understand Mechanisms**: Use probes to identify where in the network social pressure overrides truth
3. **Intervene**: Use activation steering to modify behavior and test causal hypotheses

This experiment would be impossible with closed models—it requires the full transparency that only Olmo provides.

### Implementation

**Location**: `src/aam/experiments/olmo_conformity/`

**Full Workflow**: `src/aam/experiments/olmo_conformity/orchestration.py`

#### Step 1: Behavioral Trials

**Location**: `src/aam/experiments/olmo_conformity/runner.py`

**Process**:

1. **Load Configuration**: Load suite config (datasets, conditions, models)
2. **Register Datasets**: Register immutable facts and social conventions datasets
3. **Register Conditions**: Register experimental conditions (control, synthetic_asch_history, authoritative_bias)
4. **Execute Trials**: For each model variant, item, and condition:
   - Build prompt with condition-specific formatting
   - Execute trial via gateway
   - Capture activations (if enabled)
   - Store response and correctness evaluation

**Configuration**:
```json
{
  "run": {
    "seed": 42,
    "temperature": 0.0,
    "max_items_per_dataset": 10
  },
  "datasets": [
    {
      "name": "immutable_facts",
      "version": "v1",
      "path": "experiments/olmo_conformity/datasets/immutable_facts/minimal_items.jsonl"
    }
  ],
  "conditions": [
    {
      "name": "control",
      "params": {"type": "control"}
    },
    {
      "name": "synthetic_asch_history",
      "params": {
        "type": "synthetic_asch_history",
        "confederates": 5,
        "confidence": "high"
      }
    }
  ],
  "models": [
    {
      "variant": "instruct",
      "model_id": "allenai/Olmo-3-7B-Instruct"
    }
  ]
}
```

**Running Trials**:
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --capture-activations \
  --capture-layers 10,11,12,13,14,15,16,17,18,19,20
```

**Output**:
- Database with `conformity_trials`, `conformity_outputs` tables
- Activation files (if enabled) in `runs/<timestamp>_<run_id>/activations/`
- Trial statistics and correctness rates

#### Step 2: Probe Training

**Location**: `src/aam/experiments/olmo_conformity/probes.py`

**Process**:

1. **Capture Probe Dataset Activations**: Run probe training dataset through model and capture activations
2. **Train Probes**: Train truth and social consensus probes on captured activations
3. **Store Probe Artifacts**: Save probe weights as Safetensors files
4. **Compute Projections**: Compute projections of trial activations onto probe vectors

**Running Probe Training**:
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <previous_run_id> \
  --train-probes \
  --truth-probe-dataset experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl \
  --social-probe-dataset experiments/olmo_conformity/datasets/social_conventions/minimal_items.jsonl \
  --probe-layers 10,11,12,13,14,15,16,17,18,19,20
```

**Output**:
- Probe artifacts in `runs/<run_id>/artifacts/probe_*.safetensors`
- Probe metadata in `conformity_probes` table
- Projection data in `conformity_probe_projections` table

#### Step 3: Interventions

**Location**: `src/aam/experiments/olmo_conformity/intervention.py`

**Process**:

1. **Load Probe Artifact**: Load trained social consensus vector
2. **Select Trials**: Select trials for intervention (typically incorrect responses under social pressure)
3. **Run Intervention Sweep**: For each trial and alpha value:
   - Generate baseline response
   - Generate intervention response (with activation steering)
   - Store both for comparison

**Running Interventions**:
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <previous_run_id> \
  --run-interventions \
  --social-probe-artifact runs/<run_id>/artifacts/probe_social.safetensors \
  --social-probe-id <probe_id> \
  --intervention-layers 15,16,17,18,19,20 \
  --intervention-alphas 0.5,1.0,2.0
```

**Output**:
- Intervention results in `conformity_interventions` and `conformity_intervention_results` tables
- Before/after outputs for comparison
- Intervention effectiveness metrics

#### Step 4: Analysis and Reporting

**Location**: `src/aam/experiments/olmo_conformity/analysis.py`

**Process**:

1. **Statistical Analysis**: Compute correctness rates by variant, condition, intervention
2. **Probe Analysis**: Analyze probe projections across layers
3. **Intervention Analysis**: Measure intervention effectiveness
4. **Visualization**: Generate figures and tables

**Running Analysis**:
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <run_id> \
  --generate-reports
```

**Output**:
- Figures in `runs/<run_id>/artifacts/figures/`
- Tables in `runs/<run_id>/artifacts/tables/`
- Analysis reports

#### End-to-End Workflow

**Using Orchestration**:
```python
from aam.experiments.olmo_conformity.orchestration import run_full_experiment, ExperimentConfig

config = ExperimentConfig(
    suite_config_path="experiments/olmo_conformity/configs/suite_small.json",
    runs_dir="runs/",
    capture_activations=True,
    capture_layers=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    truth_probe_dataset_path="experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl",
    social_probe_dataset_path="experiments/olmo_conformity/datasets/social_conventions/minimal_items.jsonl",
    run_interventions=True,
    generate_reports=True,
)

results = run_full_experiment(config)
```

**Output Structure**:
```
runs/
└── 20240101_120000_<run_id>/
    ├── simulation.db                    # SQLite database
    ├── activations/                     # Activation files (if captured)
    │   ├── step_0_agent_trial_xxx.safetensors
    │   └── ...
    ├── artifacts/
    │   ├── probe_truth.safetensors      # Probe artifacts
    │   ├── probe_social.safetensors
    │   ├── figures/                     # Analysis figures
    │   │   ├── correctness_by_variant.png
    │   │   └── ...
    │   └── tables/                       # Analysis tables
    │       └── ...
    └── exports/                          # Parquet exports (if enabled)
```

### Troubleshooting

**Trial Execution Issues**:
- **Model Not Found**: Ensure model is downloaded or Ollama is running with model imported
- **Activation Capture Fails**: Verify `--capture-activations` is set and model is HuggingFace-hooked (not API)
- **Rate Limiting**: Configure rate limits or use local models

**Probe Training Issues**:
- **Insufficient Data**: Ensure probe training dataset has sufficient examples (typically 100+)
- **Low Probe Accuracy**: Try different layers, check data quality
- **Activation Mismatch**: Ensure probe training uses same model variant as trials

**Intervention Issues**:
- **No Effect**: Try higher alpha values, verify target layers are correct
- **Model Errors**: Ensure model is HuggingFace-hooked, check hook registration
- **Memory Issues**: Reduce target layers or use smaller models

**Analysis Issues**:
- **Missing Data**: Verify all steps completed successfully
- **Database Errors**: Check database integrity, verify foreign key constraints

## Running the System (Olmo-Focused)

### Phase 1: Random Agents (Foundation)

**Purpose**: Validates core engine without Olmo dependencies.

**Command**:
```bash
python -m aam.run phase1 --steps 100 --agents 5 --seed 42 --db simulation.db
```

**Use Case**: Testing core engine functionality, not used in Olmo experiments.

### Phase 2: Cognitive Agents (Olmo via API)

**Purpose**: Run cognitive agents using Olmo via Ollama API.

**Model Setup - Option B: Ollama API**:
```bash
# Import Olmo to Ollama
ollama pull olmo-3-7b-instruct

# Run simulation
python -m aam.run phase2 \
  --steps 10 \
  --agents 2 \
  --seed 42 \
  --api-base http://localhost:11434/v1 \
  --model olmo-3-7b-instruct \
  --db simulation_phase2.db
```

**Limitations**: Cannot capture activations (API is black-box).

### Phase 3: Interpretability (Olmo with Activation Capture)

**Purpose**: Run Olmo with activation capture for mechanistic interpretability.

**Model Setup - Option A: Local HuggingFace**:
```bash
# Models auto-download on first use
# Or explicitly ensure download:
python -c "
from aam.experiments.olmo_conformity.olmo_utils import ensure_olmo_model_downloaded
ensure_olmo_model_downloaded('allenai/Olmo-3-7B-Instruct')
"

# Run with activation capture
python -m aam.run phase3 \
  --steps 10 \
  --agents 2 \
  --model-id allenai/Olmo-3-7B-Instruct \
  --layers 10,11,12,13,14,15,16,17,18,19,20 \
  --components resid_post \
  --trigger-actions post_message \
  --dtype float16
```

**Output**: Activation files in `runs/<run_id>/activations/`.

### Phase 4: Experiments (Barrier Scheduler)

**Purpose**: Run experiments with async execution and rate limiting.

**Command**:
```bash
python -m aam.run experiment --config experiments/config.json
```

**Use Case**: Large-scale experiments with multiple models, rate limiting.

### Olmo Conformity: Full Workflow

**Complete workflow from trials to analysis**:

#### Option A: Local HuggingFace (for Activation Capture)

```bash
# Step 1: Run behavioral trials with activation capture
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --capture-activations \
  --capture-layers 10,11,12,13,14,15,16,17,18,19,20

# Step 2: Train probes (uses captured activations)
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <run_id_from_step_1> \
  --train-probes \
  --truth-probe-dataset experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl \
  --social-probe-dataset experiments/olmo_conformity/datasets/social_conventions/minimal_items.jsonl

# Step 3: Run interventions
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <run_id_from_step_1> \
  --run-interventions \
  --social-probe-artifact runs/<run_id>/artifacts/probe_social.safetensors \
  --social-probe-id <probe_id_from_step_2>

# Step 4: Generate analysis
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <run_id_from_step_1> \
  --generate-reports
```

#### Option B: Ollama API (for Convenience)

```bash
# Import model to Ollama
ollama pull olmo-3-7b-instruct

# Run trials (no activation capture)
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --api-base http://localhost:11434/v1 \
  --model olmo-3-7b-instruct
```

#### Option C: GGUF via Llama CPP (for Efficiency)

```bash
# Terminal 1: Convert and serve model
python experiments/olmo_conformity/download_and_convert_olmo_models.py
python -m aam.run llama serve models/olmo-3-7b-instruct.gguf

# Terminal 2: Run trials
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --api-base http://127.0.0.1:8081/v1 \
  --api-key local \
  --model olmo-3-7b-instruct.gguf
```

**Note**: Options B and C cannot capture activations (required for probes/interventions).

## Output and Analysis

### Database Structure

**Core Tables**:
- `runs`: Experiment metadata
- `trace`: Action trace (for Phase 1/2 simulations)
- `messages`: Message feed (for Phase 2 simulations)
- `activation_metadata`: Activation file references

**Olmo Conformity Tables**:
- `conformity_datasets`: Datasets used
- `conformity_items`: Test items
- `conformity_conditions`: Experimental conditions
- `conformity_trials`: Individual trials
- `conformity_prompts`: Prompt templates
- `conformity_outputs`: Model responses
- `conformity_probes`: Trained probes
- `conformity_probe_projections`: Probe projections
- `conformity_interventions`: Intervention configurations
- `conformity_intervention_results`: Intervention results
- `conformity_think_tokens`: Think token analysis
- `conformity_logit_lens`: Logit lens analysis

### Trace Analysis

**Using Jupyter Notebook**: `trace_analysis.ipynb`

```python
import sqlite3
from aam.persistence import TraceDb, TraceDbConfig

db = TraceDb(TraceDbConfig(db_path="runs/20240101_120000_xxx/simulation.db"))
db.connect()

# Get trial statistics
trials = db.conn.execute("""
    SELECT variant, condition_id, 
           COUNT(*) as total,
           SUM(CASE WHEN o.is_correct = 1 THEN 1 ELSE 0 END) as correct
    FROM conformity_trials t
    JOIN conformity_outputs o ON t.trial_id = o.trial_id
    GROUP BY variant, condition_id
""").fetchall()
```

### Activation Analysis

**Loading Activations**:
```python
from safetensors.torch import load_file
import torch

# Load activation file
activation = load_file("runs/xxx/activations/step_0_agent_trial_xxx.safetensors")
# Shape: [batch, sequence, hidden_dim]

# Access specific layer
layer_15 = activation["layer_15"]  # [batch, sequence, hidden_dim]
```

**Probe Projections**:
```python
# Query probe projections from database
projections = db.conn.execute("""
    SELECT layer_index, projection_value
    FROM conformity_probe_projections
    WHERE trial_id = ? AND probe_id = ?
    ORDER BY layer_index
""", (trial_id, probe_id)).fetchall()
```

### Experiment Results

**Analysis Outputs**:
- **Figures**: `runs/<run_id>/artifacts/figures/`
  - Correctness by variant and condition
  - Probe projection trends across layers
  - Intervention effectiveness
- **Tables**: `runs/<run_id>/artifacts/tables/`
  - Statistical summaries
  - Probe accuracy metrics
  - Intervention comparison tables

**Parquet Exports** (if enabled):
```bash
python -m aam.run olmo-conformity \
  --suite-config experiments/olmo_conformity/configs/suite_small.json \
  --runs-dir runs/ \
  --run-id <run_id> \
  --export-parquet
```

Exports to `runs/<run_id>/exports/`:
- `trace.parquet`: Trace events
- `messages.parquet`: Messages
- `conformity_trials.parquet`: Trial data
- `conformity_outputs.parquet`: Output data

## Conclusion

The Abstract Agent Machine provides a complete framework for mechanistic interpretability research on the Olmo model family. By leveraging Olmo's unique open-source transparency, researchers can:

- **Observe**: Capture internal activations aligned to behavioral outputs
- **Understand**: Train probes to identify conceptual directions in activation space
- **Intervene**: Modify behavior through activation steering
- **Compare**: Analyze differences between Olmo variants (Base, Instruct, Think, RL-Zero)

The system's architecture—from the deterministic engine to the model serving infrastructure—is designed specifically to enable this research. The Olmo Conformity Experiment demonstrates the full power of this approach, showing how transparency enables scientific investigation that is impossible with closed models.

As Olmo continues to evolve and new variants are released, the Abstract Agent Machine provides a foundation for understanding how different training methodologies affect model behavior—questions that are critical for the safe and responsible development of AI systems.
