# Phase 1 & Phase 2 Accomplishments: The Core + Cognitive Layer

## Executive Summary

**Phase 1** successfully implements the foundational kernel of the Abstract Agent Machine as specified in the PRD. The system provides a **deterministic simulation engine** with **SQLite trace persistence**, enabling reproducible multi-agent experiments. The milestone of running a 100-step simulation with random agents and producing a valid Trace DB has been achieved.

**Phase 2** extends the system with the **Cognitive Layer**, integrating LangGraph for agent orchestration, LiteLLM for multi-provider LLM support, and local LLM serving via llama.cpp. Agents can now make intelligent decisions using real or mock LLMs, interact via tools (e.g., `post_message`), and maintain conversation history. The milestone of agents conversing and using basic tools has been achieved.

## Implementation Overview

### Phase 1 Core Components Delivered

1. **WorldEngine**: Deterministic state machine that orchestrates simulation steps
2. **Channel**: Communication abstraction (InMemoryChannel implementation)
3. **SQLite Persistence**: Append-only trace database with WAL mode
4. **AgentPolicy Interface**: Protocol for agent decision-making
5. **RandomAgentPolicy**: Mock agent implementation for Phase 1 testing
6. **CLI Runner**: Command-line interface for executing simulations
7. **Trace Analysis Notebook**: Jupyter notebook for analyzing simulation results

### Phase 2 Cognitive Layer Components Delivered

1. **LangGraph Integration**: Stateful agent graphs with tool calling support
2. **LiteLLMGateway**: Unified LLM client supporting OpenAI, Anthropic, and local servers
3. **CognitiveAgentPolicy**: LLM-driven agent decision-making with LangGraph
4. **SimpleCognitivePolicy**: Fallback implementation when LangGraph is unavailable
5. **Dual-Mode Action Parsing**: Native tool-calling + text fallback with JSON repair
6. **Message Persistence**: `messages` table for Phase 2 conversation history
7. **Local LLM Support**: llama.cpp integration for serving GGUF models
8. **Model Discovery**: Automatic discovery of GGUF models from Ollama and LM Studio
9. **MockLLMGateway**: Deterministic offline LLM for testing without API costs

## Architecture & Design Decisions

### Determinism Strategy

The system enforces determinism through:

- **Master Seed**: Single seed value controls all randomness
- **Per-Agent RNG**: Each agent receives a deterministic seed derived from `hash(master_seed, agent_id)`
- **Deterministic Action Ordering**: Actions are sorted by `agent_id` before execution
- **Deterministic Timestamps**: Logical timestamps derived from `time_step` and agent index (optional wall-clock mode available)

### Data Contracts (Pydantic V2)

All data structures are strictly typed using Pydantic V2 models:

- **`ActionRequest`**: Agent intent envelope with `run_id`, `time_step`, `agent_id`, `action_name`, `arguments`, optional `reasoning`, and `metadata`
- **`ActionResult`**: Platform response with `success`, `data`, `error`, and `trace_id`
- **`TraceEvent`**: Immutable event record with full request/response payloads
- **`RunMetadata`**: Run configuration snapshot for reproducibility

### SQLite Schema Design

The database implements the "Trace as Truth" principle:

**`runs` table:**
- `run_id` (PRIMARY KEY)
- `seed` (INTEGER)
- `created_at` (REAL)
- `config_json` (TEXT) - JSON blob of run configuration

**`trace` table:**
- `trace_id` (PRIMARY KEY)
- `run_id` (FOREIGN KEY → runs)
- `time_step` (INTEGER)
- `agent_id` (TEXT)
- `action_type` (TEXT)
- `info_json` (TEXT) - ActionRequest payload (deterministic JSON)
- `outcome_json` (TEXT) - ActionResult payload (deterministic JSON)
- `created_at` (REAL)
- `environment_state_hash` (TEXT, nullable)

**Indexes:**
- `idx_trace_run_step` on `(run_id, time_step)`
- `idx_trace_agent` on `(run_id, agent_id)`

**`messages` table (Phase 2):**
- `message_id` (PRIMARY KEY)
- `run_id` (FOREIGN KEY → runs)
- `time_step` (INTEGER)
- `author_id` (TEXT)
- `content` (TEXT)
- `created_at` (REAL)

**Indexes:**
- `idx_messages_run_step` on `(run_id, time_step)`
- `idx_messages_run_author` on `(run_id, author_id)`

**Database Configuration:**
- WAL (Write-Ahead Logging) mode enabled for concurrent reads
- Foreign key constraints enabled
- Deterministic JSON serialization (sorted keys, no whitespace)

## File Structure

```
abstractAgentMachine/
├── pyproject.toml              # Project metadata (uv-compatible, optional extras)
├── src/aam/
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Entry point for `python -m aam`
│   ├── types.py               # Pydantic data contracts
│   ├── persistence.py         # SQLite trace database + messages table
│   ├── channel.py             # Channel abstraction + InMemoryChannel
│   ├── policy.py              # AgentPolicy protocol + RandomAgentPolicy
│   ├── world_engine.py        # Deterministic simulation engine
│   ├── tools.py               # Tool definitions (post_message, etc.)
│   ├── llm_gateway.py         # LiteLLM gateway + MockLLMGateway
│   ├── text_parse.py          # Dual-mode text parsing utilities
│   ├── agent_langgraph.py     # CognitiveAgentPolicy + SimpleCognitivePolicy
│   ├── interpretability.py    # TransformerLens scaffolding (Phase 3)
│   ├── model_discovery.py     # GGUF model discovery (Ollama, LM Studio)
│   ├── llama_cpp.py           # llama.cpp server integration
│   └── run.py                 # CLI runner (phase1, phase2, llama subcommands)
├── trace_analysis.ipynb        # Jupyter notebook for DB analysis
├── models/                     # Symlinked GGUF models directory
└── third_party/llama.cpp/      # Cloned llama.cpp repository
```

## Component Details

### 1. Types (`src/aam/types.py`)

Defines all core data structures using Pydantic V2:

- **`ActionRequest`**: Complete agent action specification
- **`ActionResult`**: Platform execution result
- **`TraceEvent`**: Immutable historical record
- **`RunMetadata`**: Run configuration and metadata
- **`Observation`**: Type alias for agent observations (currently `Dict[str, Any]`)

All models include `json_dict()` methods for deterministic serialization.

### 2. Persistence (`src/aam/persistence.py`)

**`TraceDb` class:**
- Manages SQLite connection lifecycle
- Initializes schema (runs + trace tables + indexes)
- Provides `insert_run()` and `append_trace()` methods
- Enforces deterministic JSON serialization via `_json_dumps_deterministic()`

**Key Features:**
- WAL mode for performance
- Foreign key integrity
- Transaction-safe writes
- Connection pooling ready (single connection per instance)

### 3. Channel (`src/aam/channel.py`)

**`Channel` protocol:**
- Abstract interface for agent-platform communication
- `submit(request: ActionRequest)` method
- `take_all() -> List[ActionRequest]` method

**`InMemoryChannel` implementation:**
- Thread-safe in-memory queue
- Used for Phase 1 synchronous execution
- Extensible to distributed message brokers in future phases

### 4. Policy (`src/aam/policy.py`)

**`AgentPolicy` protocol:**
- `decide(run_id, time_step, agent_id, observation) -> ActionRequest`
- Stateless interface (agents are re-instantiated per step in Phase 1)

**`RandomAgentPolicy` implementation:**
- Mock agent for Phase 1 testing
- Chooses randomly from action set: `["noop", "emit_event"]`
- Uses deterministic `random.Random` instance per agent
- `stable_agent_seed()` function derives agent seed from master seed

### 5. World Engine (`src/aam/world_engine.py`)

**`WorldEngine` class:**

**Core Methods:**
- `step(time_step: int)`: Executes one simulation step
  - Builds observations for all agents
  - Collects action requests (synchronously in Phase 1)
  - Sorts requests deterministically by `agent_id`
  - Executes actions sequentially
  - Appends trace events to database
- `run(steps: int)`: Executes full simulation loop
- `execute(req: ActionRequest, timestamp: float) -> (ActionResult, TraceEvent)`: Executes single action

**Action Execution:**
- Phase 1 supports two actions:
  - `noop`: No-operation (always succeeds)
  - `emit_event`: Echo action (returns arguments as outcome)
- Unknown actions return error result
- All actions generate trace events

**Determinism:**
- Logical timestamps: `time_step + (agent_index / 1000.0)`
- Optional wall-clock timestamps via config flag
- Deterministic action ordering

### 6. Tools (`src/aam/tools.py`)

**Tool Definitions:**
- `ToolSpec` Pydantic model for structured tool definitions
- `post_message` tool: Allows agents to post messages to the shared feed
- `_tool_specs()` function: Returns list of available tools for agents

### 7. LLM Gateway (`src/aam/llm_gateway.py`)

**`LLMGateway` protocol:**
- Abstract interface for LLM interactions
- `complete()` method for generating responses

**`LiteLLMGateway` implementation:**
- Uses `litellm.completion()` for multi-provider support
- Supports OpenAI, Anthropic, and local OpenAI-compatible servers
- Automatically configures `api_base` and `api_key` for local servers
- Forces `openai/` prefix when `api_base` is set to use OpenAI-compatible client

**`MockLLMGateway` implementation:**
- Deterministic, seeded random behavior for offline testing
- Simulates tool calls (`post_message`) or `noop` responses
- Generates deterministic message content

### 8. Text Parsing (`src/aam/text_parse.py`)

**Dual-Mode Support:**
- `parse_tool_call_from_text()` function for text-based tool calling
- Strips markdown code blocks
- Uses `json_repair` for dirty JSON parsing
- Falls back to regex extraction if JSON parsing fails

### 9. Cognitive Agent Policy (`src/aam/agent_langgraph.py`)

**`CognitiveAgentPolicy` (LangGraph-based):**
- Builds LangGraph `StateGraph` with nodes: `build_messages`, `call_model`, `parse_response`
- Handles native tool calling and text parsing fallback
- Maintains conversation history in agent state
- Integrates with `LiteLLMGateway` or `MockLLMGateway`

**`SimpleCognitivePolicy` (Fallback):**
- Non-LangGraph implementation providing same core logic
- Used when LangGraph is not installed
- Maintains compatibility with Phase 2 features

**`default_cognitive_policy()` factory:**
- Dynamically chooses between LangGraph and simple implementation
- Ensures Phase 2 works with or without LangGraph installed

### 10. Model Discovery (`src/aam/model_discovery.py`)

**GGUF Model Discovery:**
- `GGUFModel` dataclass for normalized model information
- `_find_lmstudio_ggufs()`: Scans LM Studio model directory
- `_find_ollama_models()`: Discovers Ollama models via manifest parsing
- `list_discovered_models()`: Combines discovery from both sources
- `export_models_to_repo()`: Creates symlinks to local `models/` directory

### 11. LLaMA CPP Integration (`src/aam/llama_cpp.py`)

**Local LLM Serving:**
- `build_llama_cpp_server()`: Clones and builds llama.cpp repository
- `run_llama_server()`: Launches `llama-server` subprocess
- Provides OpenAI-compatible API endpoint for local GGUF models

### 12. CLI Runner (`src/aam/run.py`)

**Command-line interface with subcommands:**

**`phase1` subcommand:**
```bash
python -m aam.run phase1 [OPTIONS]
```
- Options: `--steps`, `--agents`, `--seed`, `--db`, `--run-id`, `--no-validate`, `--nondeterministic-timestamps`
- Runs Phase 1 simulation with `RandomAgentPolicy`

**`phase2` subcommand:**
```bash
python -m aam.run phase2 [OPTIONS]
```
- Options: All Phase 1 options plus:
  - `--model MODEL`: LLM model identifier (default: `gpt-3.5-turbo`)
  - `--mock-llm`: Use deterministic offline mock LLM
  - `--api-base URL`: Override API base for local servers (e.g., `http://127.0.0.1:8081/v1`)
  - `--api-key KEY`: API key for provider (optional for local servers)
  - `--message-history N`: Number of recent messages in observations (default: 20)
- Runs Phase 2 simulation with `CognitiveAgentPolicy`

**`llama` subcommand:**
```bash
python -m aam.run llama list          # List discovered GGUF models
python -m aam.run llama export       # Export models to models/
python -m aam.run llama serve MODEL  # Serve a GGUF model
```

**Workflow:**
1. Generate or accept `run_id`
2. Initialize SQLite database (create schema if needed)
3. Insert `RunMetadata` row
4. Create agents (random or cognitive based on mode)
5. Initialize `WorldEngine` with agents, channel, and database
6. Execute simulation loop
7. Validate database (unless `--no-validate`)
8. Print `run_id` and `db` path

**Validation:**
- Verifies `runs` table contains the run
- Verifies `trace` table contains exactly `steps * num_agents` rows
- Verifies `time_step` range is `0..(steps-1)`

### 13. Trace Analysis Notebook (`trace_analysis.ipynb`)

Jupyter notebook for analyzing simulation databases:

**Features:**
- Opens and inspects multiple simulation databases (Phase 1 and Phase 2)
- Lists tables and schemas (runs, trace, messages)
- Displays run metadata with configuration details
- Computes trace statistics (row counts, time step ranges, action distributions)
- Computes message statistics for Phase 2 runs
- Compares two databases (row counts + deterministic SHA-256 digests)
- Visualizations: Action type distribution, time step progression
- Message analysis: View recent messages, author statistics
- Sample trace row inspection with parsed JSON

**Helper Functions:**
- `open_db()`: Opens SQLite database with row factory
- `get_tables()`: Lists all tables in database
- `get_runs()`: Retrieves all run metadata
- `get_latest_run()`: Gets most recent run
- `trace_stats()`: Computes summary statistics for trace table
- `message_stats()`: Computes summary statistics for messages table
- `trace_digest()`: Generates deterministic hash of trace content
- `get_trace_rows()`: Retrieves trace events with parsed JSON
- `get_messages()`: Retrieves recent messages for a run

## Testing & Validation

### Milestone Achievement

The Phase 1 milestone has been validated:

✅ **100-step simulation runs successfully**
✅ **Valid SQLite Trace DB produced**
✅ **Deterministic execution verified** (same seed → same trace)
✅ **Database validation passes** (row counts, time step ranges)

### Example Usage

**Phase 1:**
```bash
# Run 100-step simulation with 5 random agents
python -m aam.run phase1 --steps 100 --agents 5 --seed 42 --db simulation.db

# Run quick test (3 steps, 2 agents)
python -m aam.run phase1 --steps 3 --agents 2 --seed 42 --db simulation_test.db
```

**Phase 2:**
```bash
# Run with mock LLM (offline, deterministic)
python -m aam.run phase2 --steps 10 --agents 2 --seed 42 --db simulation_phase2_mock.db --mock-llm

# Run with OpenAI API
python -m aam.run phase2 --steps 10 --agents 2 --model gpt-3.5-turbo --db simulation_phase2.db

# Run with local llama-server
python -m aam.run llama serve models/ollama__library_smollm2_135m.gguf  # In one terminal
python -m aam.run phase2 --steps 5 --agents 2 --api-base http://127.0.0.1:8081/v1 --api-key local --model ollama__library_smollm2_135m.gguf --db simulation_llamacpp_phase2.db
```

**Model Management:**
```bash
# List discovered GGUF models
python -m aam.run llama list

# Export models to local directory
python -m aam.run llama export
```

**Analysis:**
```bash
# Analyze results in Jupyter
jupyter notebook trace_analysis.ipynb
```

### Determinism Verification

Running the same simulation twice with identical parameters produces:
- Identical `trace` table row counts
- Identical `time_step` sequences
- Identical action types and arguments (when using deterministic timestamps)
- Identical SHA-256 trace digests (via notebook analysis)

## Dependencies

**Phase 1 Dependencies (Minimal):**
- `pydantic>=2.6`: Data validation and serialization
- Python 3.11+: Standard library only otherwise

**Phase 2 Dependencies (Optional Extras):**
- `langgraph>=0.2.0`: Agent orchestration framework
- `litellm>=1.40.0`: Unified LLM client for multiple providers
- `json-repair`: Dirty JSON parsing for text fallback mode

**Installation:**
```bash
# Phase 1 only (minimal)
uv sync

# Phase 2 (with cognitive layer)
uv sync --extra cognitive

# With interpretability (Phase 3, future)
uv sync --extra cognitive --extra interpretability
```

**Optional (for analysis notebook):**
- `pandas`: Data analysis
- `matplotlib`: Visualization

## Design Principles Implemented

### 1. Separation of Concerns
- **Platform (WorldEngine)**: Authoritative state machine, deterministic execution
- **Agent (Policy)**: Stateless decision-making, no direct state mutation
- **Channel**: Communication abstraction, decouples agent from platform

### 2. Trace as Truth
- Database state is derivative of trace events
- Trace is append-only, immutable
- Full replay capability (future phase)

### 3. Determinism
- Reproducible execution given same seed
- Deterministic action ordering
- Deterministic JSON serialization
- Logical timestamps (optional wall-clock mode)

### 4. Extensibility
- `AgentPolicy` protocol allows pluggable agent implementations
- `Channel` abstraction supports distributed communication
- Database schema supports future domain tables
- Minimal dependencies enable easy integration

## Limitations & Future Work

### Phase 1 Limitations (Resolved in Phase 2)

1. ✅ **LLM Integration**: Now supported via LangGraph and LiteLLM
2. ✅ **Dual-Mode Action Parsing**: Implemented with text fallback
3. ✅ **Message Persistence**: `messages` table added for conversation history
4. ✅ **Local LLM Support**: llama.cpp integration for offline testing

### Remaining Limitations

1. **Synchronous Execution**: Agents execute sequentially (no true parallelism)
   - **Planned for Phase 4**: Async Barrier Scheduler
2. **Limited Domain State**: Only `messages` table (no `posts`, `users`, etc.)
   - **Planned for Phase 4**: Domain-specific tables and validation
3. **No Activation Capture**: No TransformerLens integration
   - **Planned for Phase 3**: TransformerLens hooks and Safetensors export
4. **No Memory System**: No vector DB or long-term memory
   - **Planned for Phase 4**: Vector DB integration for agent context
5. **Basic Observations**: Observations include `time_step`, `agent_id`, `messages`, `tools`
   - **Future**: More sophisticated observation construction

### Planned for Phase 3+

- **TransformerLens Integration**: Activation capture hooks
- **Safetensors Export**: High-dimensional tensor storage
- **Sparse Sampling**: Configurable layer/component capture
- **Async Barrier Scheduler**: True concurrent agent execution
- **Domain State**: Domain-specific tables and validation
- **Memory System**: Vector DB integration for agent context

## Code Quality

- **Type Safety**: Full type hints throughout
- **Error Handling**: Graceful failures with clear error messages
- **Documentation**: Inline docstrings and type annotations
- **Validation**: Pydantic models enforce data contracts
- **Testing**: CLI validation ensures correctness
- **Linting**: No linter errors (verified)

## Performance Characteristics

- **Database Writes**: Transaction-safe, WAL mode for efficiency
- **Memory Usage**: Minimal (no large in-memory buffers)
- **Determinism Overhead**: Negligible (sorting + deterministic JSON)
- **Scalability**: Tested with 100 steps × 5 agents (500 trace rows)

## Conclusion

**Phase 1** successfully delivers a **robust, deterministic simulation kernel** that satisfies the core requirements:

✅ Deterministic execution with reproducible traces
✅ SQLite persistence with proper schema and indexing
✅ Clean separation between agent policy and platform engine
✅ Extensible architecture ready for Phase 2 enhancements
✅ Validated milestone: 100-step simulation with trace DB

**Phase 2** extends the system with the **Cognitive Layer**, achieving the following:

✅ LangGraph integration for stateful agent orchestration
✅ LiteLLM gateway supporting multiple providers (OpenAI, Anthropic, local servers)
✅ Dual-mode action parsing (native tool-calling + text fallback)
✅ Message persistence for conversation history
✅ Local LLM support via llama.cpp for offline testing
✅ Model discovery from Ollama and LM Studio
✅ Mock LLM gateway for deterministic offline testing
✅ Validated milestone: Agents can converse and use basic tools

The system now provides a **complete foundation** for cognitive agent simulations, with the architecture ready for the interpretability layer (Phase 3) and scaling infrastructure (Phase 4) as specified in the PRD.

