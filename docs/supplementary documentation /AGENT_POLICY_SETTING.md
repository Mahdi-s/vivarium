# Agent Policy Setting in the Vivarium

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The AgentPolicy Protocol](#the-agentpolicy-protocol)
4. [Policy Implementations](#policy-implementations)
5. [Policy Instantiation Process](#policy-instantiation-process)
6. [Seed Derivation and Determinism](#seed-derivation-and-determinism)
7. [Policy Decision Flow](#policy-decision-flow)
8. [Configuration Options](#configuration-options)
9. [Examples](#examples)
10. [Advanced Topics](#advanced-topics)

---

## Overview

The Vivarium uses a **policy-based architecture** where each agent's decision-making logic is encapsulated in an `AgentPolicy` object. This design provides:

- **Separation of Concerns**: Agent logic is decoupled from the platform engine
- **Pluggability**: Different policy implementations can be swapped easily
- **Determinism**: Policies use seeded random number generators for reproducibility
- **Stateless Design**: Policies are stateless, making them easy to test and reason about

The policy system supports two main modes:
- **Phase 1**: Random agents (`RandomAgentPolicy`) for core simulation testing
- **Phase 2**: Cognitive agents (`CognitiveAgentPolicy` or `SimpleCognitivePolicy`) using LLMs for intelligent decision-making

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI (run.py)                            │
│  - Parses command-line arguments                             │
│  - Determines mode (phase1/phase2)                          │
│  - Creates policy instances                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Policy Instantiation                            │
│  - Derives per-agent seeds from master seed                  │
│  - Creates RandomAgentPolicy or CognitiveAgentPolicy        │
│  - Configures LLM gateway (if Phase 2)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              WorldEngine                                     │
│  - Stores agents dict: {agent_id: AgentPolicy}              │
│  - Calls policy.decide() each simulation step                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AgentPolicy.decide()                            │
│  - Receives observation (time_step, agent_id, messages)     │
│  - Returns ActionRequest                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## The AgentPolicy Protocol

The `AgentPolicy` protocol defines the interface that all policy implementations must follow:

```python
class AgentPolicy(Protocol):
    def decide(
        self, 
        *, 
        run_id: str, 
        time_step: int, 
        agent_id: str, 
        observation: Observation
    ) -> ActionRequest:
        ...
```

### Protocol Contract

**Input Parameters:**
- `run_id`: Unique identifier for the simulation run
- `time_step`: Current logical time step in the simulation
- `agent_id`: Unique identifier for this agent (e.g., `"agent_000"`)
- `observation`: Dictionary containing:
  - `time_step`: Current time step
  - `agent_id`: Agent's identifier
  - `messages`: Recent message history (Phase 2 only)
  - `tools`: List of available tool names (Phase 2 only)

**Output:**
- `ActionRequest`: A Pydantic model containing:
  - `run_id`, `time_step`, `agent_id`
  - `action_name`: The action to execute (e.g., `"noop"`, `"emit_event"`, `"post_message"`)
  - `arguments`: Dictionary of action-specific arguments
  - `reasoning`: Optional reasoning text (Phase 2)
  - `metadata`: Additional metadata about the decision

### Key Design Principles

1. **Stateless**: Policies don't maintain internal state between calls
2. **Deterministic**: Given the same inputs and seed, policies produce identical outputs
3. **Pure Function**: `decide()` has no side effects (except RNG state changes)

---

## Policy Implementations

### 1. RandomAgentPolicy (Phase 1)

**Location**: `src/aam/policy.py`

**Purpose**: Mock agent for Phase 1 testing and validation. Makes random decisions from a fixed action space.

**Implementation**:

```python
@dataclass(frozen=True)
class RandomAgentPolicy:
    rng: random.Random
    action_space: List[str]

    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest:
        action = self.rng.choice(self.action_space)
        if action == "emit_event":
            args = {
                "value": self.rng.randint(0, 1_000_000), 
                "seen_time_step": observation.get("time_step")
            }
        else:
            args = {}
        
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=action,
            arguments=args,
            reasoning=None,
            metadata={"policy": "RandomAgentPolicy"},
        )
```

**Features**:
- Uses a seeded `random.Random` instance for deterministic randomness
- Default action space: `["noop", "emit_event"]`
- `emit_event` generates random integer values
- Fully deterministic when using the same seed

**Action Space**:
- `noop`: No operation (always succeeds)
- `emit_event`: Emits an event with a random value

### 2. CognitiveAgentPolicy (Phase 2 - LangGraph)

**Location**: `src/aam/agent_langgraph.py`

**Purpose**: LLM-driven agent using LangGraph for orchestration. Supports native tool calling and text fallback.

**Implementation**:

```python
@dataclass(frozen=True)
class CognitiveAgentPolicy:
    gateway: LLMGateway
    model: str
    tools: List[ToolSpec]
    temperature: float = 0.2

    def __post_init__(self) -> None:
        # Build LangGraph StateGraph with three nodes:
        # 1. build_messages: Convert observation to OpenAI message format
        # 2. call_model: Call LLM via gateway
        # 3. parse_action: Extract action from LLM response
        ...
```

**Decision Flow** (LangGraph StateGraph):

1. **build_messages**: Converts observation to OpenAI message format
   - System prompt: "You are an agent in a simulation..."
   - User prompt: Includes agent_id, time_step, and message history

2. **call_model**: Calls LLM via gateway
   - Uses `gateway.chat()` with tools and tool_choice="auto"
   - Supports native tool calling (OpenAI format)

3. **parse_action**: Extracts action from LLM response
   - **Path A**: Native tool call → Extract tool name and arguments
   - **Path B**: Text fallback → Parse JSON from text response
   - **Path C**: Safe fallback → Default to "noop" if parsing fails

**Features**:
- LangGraph orchestration for complex decision flows
- Dual-mode action parsing (native tools + text fallback)
- Configurable temperature for LLM responses
- Tool support via `ToolSpec` definitions

### 3. SimpleCognitivePolicy (Phase 2 - Fallback)

**Location**: `src/aam/agent_langgraph.py`

**Purpose**: Non-LangGraph fallback for Phase 2. Provides same functionality as `CognitiveAgentPolicy` but without LangGraph dependency.

**Implementation**:

```python
@dataclass(frozen=True)
class SimpleCognitivePolicy:
    gateway: LLMGateway
    model: str
    tools: List[ToolSpec]
    temperature: float = 0.2

    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest:
        # Direct implementation without LangGraph
        # Same logic as CognitiveAgentPolicy but linear execution
        ...
```

**Features**:
- Same functionality as `CognitiveAgentPolicy`
- No LangGraph dependency (works without optional extras)
- Linear execution (no graph orchestration)
- Used automatically when LangGraph is not installed

### 4. Policy Factory Function

**Location**: `src/aam/agent_langgraph.py`

```python
def default_cognitive_policy(*, gateway: LLMGateway, model: str) -> CognitiveAgentPolicy:
    """
    Factory function that returns the best available cognitive policy.
    Prefers LangGraph implementation when available.
    """
    if langgraph_available():
        return CognitiveAgentPolicy(gateway=gateway, model=model, tools=_tool_specs())
    return SimpleCognitivePolicy(gateway=gateway, model=model, tools=_tool_specs())
```

**Purpose**: Automatically selects the best available cognitive policy implementation.

---

## Policy Instantiation Process

The policy instantiation process occurs in `src/aam/run.py` during simulation setup.

### Step-by-Step Process

#### 1. Parse Command-Line Arguments

```python
# Phase 1
p1 = sub.add_parser("phase1", help="Phase 1 core simulation (RandomAgent)")
p1.add_argument("--seed", type=int, default=42)
p1.add_argument("--agents", type=int, default=5)

# Phase 2
p2 = sub.add_parser("phase2", help="Phase 2 cognitive simulation")
p2.add_argument("--seed", type=int, default=42)
p2.add_argument("--agents", type=int, default=2)
p2.add_argument("--model", type=str, default="gpt-3.5-turbo")
p2.add_argument("--mock-llm", action="store_true")
p2.add_argument("--api-base", type=str, default=None)
p2.add_argument("--api-key", type=str, default=None)
```

#### 2. Determine Mode

```python
mode = args.mode or "phase1"  # Default to phase1 if no subcommand
```

#### 3. Create Agent Dictionary

```python
agents = {}
for i in range(args.agents):
    agent_id = f"agent_{i:03d}"  # e.g., "agent_000", "agent_001"
    agent_seed = stable_agent_seed(args.seed, agent_id)
    
    if mode == "phase1":
        # Phase 1: RandomAgentPolicy
        agents[agent_id] = RandomAgentPolicy(random.Random(agent_seed))
    else:
        # Phase 2: Cognitive policy
        gateway = (
            MockLLMGateway(seed=agent_seed)
            if args.mock_llm
            else LiteLLMGateway(api_base=args.api_base, api_key=args.api_key)
        )
        agents[agent_id] = default_cognitive_policy(gateway=gateway, model=args.model)
```

#### 4. Pass to WorldEngine

```python
engine = WorldEngine(
    config=WorldEngineConfig(
        run_id=run_id,
        deterministic_timestamps=(not args.nondeterministic_timestamps),
        message_history_limit=(args.message_history if mode == "phase2" else 0),
    ),
    agents=agents,  # Dictionary of {agent_id: AgentPolicy}
    channel=InMemoryChannel(),
    trace_db=trace_db,
)
```

### Agent ID Format

Agent IDs follow a consistent format:
- Pattern: `agent_{index:03d}`
- Examples: `agent_000`, `agent_001`, `agent_002`, ..., `agent_099`
- Purpose: Ensures deterministic sorting and consistent identification

---

## Seed Derivation and Determinism

### Master Seed

The master seed is provided via the `--seed` CLI argument (default: `42`). It's stored in the database as part of `RunMetadata` for reproducibility.

### Per-Agent Seed Derivation

Each agent receives a unique, deterministic seed derived from the master seed and agent ID:

```python
def stable_agent_seed(master_seed: int, agent_id: str) -> int:
    """
    Derive a stable per-agent seed (not affected by Python's hash randomization).
    """
    h = hashlib.sha256(f"{master_seed}:{agent_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)
```

**Process**:
1. Combine master seed and agent ID: `f"{master_seed}:{agent_id}"`
2. Hash with SHA-256: `hashlib.sha256(...).digest()`
3. Extract first 8 bytes: `h[:8]`
4. Convert to integer: `int.from_bytes(..., "big", signed=False)`

**Why SHA-256?**
- Python's built-in `hash()` can vary between runs (PYTHONHASHSEED)
- SHA-256 is deterministic across runs and platforms
- Ensures reproducibility regardless of Python version or environment

### Seed Usage

**Phase 1 (RandomAgentPolicy)**:
```python
agent_seed = stable_agent_seed(args.seed, agent_id)
agents[agent_id] = RandomAgentPolicy(random.Random(agent_seed))
```
- Each agent gets its own `random.Random` instance
- Used for choosing actions and generating random values

**Phase 2 (MockLLMGateway)**:
```python
agent_seed = stable_agent_seed(args.seed, agent_id)
gateway = MockLLMGateway(seed=agent_seed)
```
- Mock LLM uses seed for deterministic responses
- Ensures offline testing produces consistent results

**Phase 2 (Real LLM)**:
- Real LLMs (via `LiteLLMGateway`) don't use seeds directly
- Determinism comes from deterministic action ordering and observation construction

### Determinism Guarantees

With the same master seed:
- ✅ Same agent IDs → same per-agent seeds
- ✅ Same per-agent seeds → same RNG sequences
- ✅ Same RNG sequences → same agent decisions (Phase 1)
- ✅ Same decisions → same trace output
- ✅ Same trace output → bitwise identical database

---

## Policy Decision Flow

### Phase 1 Flow (RandomAgentPolicy)

```
┌─────────────────────────────────────────────────────────┐
│ WorldEngine.step(time_step)                              │
│   for agent_id in sorted(agent_ids):                     │
│     obs = build_observation(time_step, agent_id)        │
│     req = agents[agent_id].decide(                        │
│       run_id, time_step, agent_id, obs                   │
│     )                                                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ RandomAgentPolicy.decide()                               │
│   1. Choose action: rng.choice(["noop", "emit_event"]) │
│   2. Generate arguments (if emit_event)                  │
│   3. Return ActionRequest                               │
└─────────────────────────────────────────────────────────┘
```

### Phase 2 Flow (CognitiveAgentPolicy with LangGraph)

```
┌─────────────────────────────────────────────────────────┐
│ WorldEngine.step(time_step)                              │
│   for agent_id in sorted(agent_ids):                    │
│     obs = build_observation(time_step, agent_id)        │
│     req = agents[agent_id].decide(                        │
│       run_id, time_step, agent_id, obs                   │
│     )                                                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ CognitiveAgentPolicy.decide()                            │
│   state = {run_id, time_step, agent_id, observation}    │
│   out = _graph.invoke(state)                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ LangGraph StateGraph Execution                           │
│                                                           │
│   build_messages ──► call_model ──► parse_action         │
│        │                │                │              │
│        │                │                │              │
│   Convert obs      Call LLM via    Extract action       │
│   to messages      gateway.chat()   from response        │
│                    with tools                            │
└─────────────────────────────────────────────────────────┘
```

### Phase 2 Flow (SimpleCognitivePolicy - Fallback)

```
┌─────────────────────────────────────────────────────────┐
│ SimpleCognitivePolicy.decide()                           │
│   1. Convert observation to OpenAI messages             │
│   2. Call gateway.chat() with tools                     │
│   3. Try to extract tool call from response             │
│   4. If no tool call, parse text for JSON action        │
│   5. Fallback to "noop" if parsing fails                │
│   6. Return ActionRequest                               │
└─────────────────────────────────────────────────────────┘
```

### Observation Construction

The `WorldEngine.build_observation()` method constructs the observation passed to policies:

**Phase 1**:
```python
{
    "time_step": 0,
    "agent_id": "agent_000"
}
```

**Phase 2**:
```python
{
    "time_step": 0,
    "agent_id": "agent_000",
    "messages": [
        {"time_step": 0, "author_id": "agent_001", "content": "Hello!"},
        ...
    ],
    "tools": ["post_message"]
}
```

---

## Configuration Options

### Phase 1 Configuration

**CLI Arguments**:
- `--seed N`: Master random seed (default: 42)
- `--agents N`: Number of agents (default: 5)
- `--steps N`: Number of simulation steps (default: 100)

**Policy Configuration**:
- Action space: `["noop", "emit_event"]` (hardcoded)
- RNG: Seeded `random.Random` instance per agent

### Phase 2 Configuration

**CLI Arguments**:
- `--seed N`: Master random seed (default: 42)
- `--agents N`: Number of agents (default: 2)
- `--steps N`: Number of simulation steps (default: 10)
- `--model MODEL`: LLM model identifier (default: "gpt-3.5-turbo")
- `--mock-llm`: Use deterministic offline mock LLM
- `--api-base URL`: Override API base for local servers
- `--api-key KEY`: API key for provider
- `--message-history N`: Number of recent messages in observations (default: 20)

**Policy Configuration**:
- Temperature: `0.2` (hardcoded, can be customized)
- Tools: Retrieved via `_tool_specs()` (currently `["post_message"]`)
- Gateway: `MockLLMGateway` or `LiteLLMGateway`

### Customization Points

**To customize action space (Phase 1)**:
```python
agents[agent_id] = RandomAgentPolicy(
    random.Random(agent_seed),
    action_space=["noop", "emit_event", "custom_action"]
)
```

**To customize temperature (Phase 2)**:
```python
policy = CognitiveAgentPolicy(
    gateway=gateway,
    model=args.model,
    tools=_tool_specs(),
    temperature=0.7  # Higher creativity
)
```

**To add custom tools (Phase 2)**:
Modify `src/aam/tools.py` to add new `ToolSpec` definitions.

---

## Examples

### Example 1: Phase 1 with 3 Agents

```bash
vvm phase1 --steps 10 --agents 3 --seed 42
```

**Policy Instantiation**:
```python
# Master seed: 42
# Agents: agent_000, agent_001, agent_002

agents = {
    "agent_000": RandomAgentPolicy(random.Random(seed_000)),
    "agent_001": RandomAgentPolicy(random.Random(seed_001)),
    "agent_002": RandomAgentPolicy(random.Random(seed_002)),
}
```

**Decision Process** (Step 0):
```python
# agent_000.decide()
observation = {"time_step": 0, "agent_id": "agent_000"}
action = rng.choice(["noop", "emit_event"])  # e.g., "emit_event"
args = {"value": rng.randint(0, 1_000_000), "seen_time_step": 0}
return ActionRequest(action_name="emit_event", arguments=args, ...)
```

### Example 2: Phase 2 with Mock LLM

```bash
vvm phase2 --steps 5 --agents 2 --seed 42 --mock-llm
```

**Policy Instantiation**:
```python
# Master seed: 42
# Agents: agent_000, agent_001

for i in range(2):
    agent_id = f"agent_{i:03d}"
    agent_seed = stable_agent_seed(42, agent_id)
    gateway = MockLLMGateway(seed=agent_seed)
    agents[agent_id] = default_cognitive_policy(
        gateway=gateway, 
        model="gpt-3.5-turbo"
    )
```

**Decision Process** (Step 0):
```python
# agent_000.decide()
observation = {
    "time_step": 0,
    "agent_id": "agent_000",
    "messages": [],
    "tools": ["post_message"]
}

# LangGraph execution:
# 1. build_messages: Creates system + user prompts
# 2. call_model: MockLLMGateway.chat() returns tool call
# 3. parse_action: Extracts "post_message" with content
return ActionRequest(action_name="post_message", arguments={"content": "hello (123)"}, ...)
```

### Example 3: Phase 2 with Real LLM

```bash
vvm phase2 \
  --steps 5 \
  --agents 2 \
  --model gpt-3.5-turbo \
  --api-base http://127.0.0.1:8081/v1 \
  --api-key local
```

**Policy Instantiation**:
```python
gateway = LiteLLMGateway(api_base="http://127.0.0.1:8081/v1", api_key="local")
agents[agent_id] = default_cognitive_policy(gateway=gateway, model="gpt-3.5-turbo")
```

**Decision Process**:
- Same flow as Example 2, but uses real LLM API calls
- Responses may vary (non-deterministic) unless LLM supports deterministic mode

### Example 4: Custom Policy Implementation

To create a custom policy:

```python
from aam.policy import AgentPolicy
from aam.types import ActionRequest, Observation

class MyCustomPolicy:
    def __init__(self, some_config: str):
        self.config = some_config
    
    def decide(
        self, 
        *, 
        run_id: str, 
        time_step: int, 
        agent_id: str, 
        observation: Observation
    ) -> ActionRequest:
        # Custom decision logic
        if time_step % 2 == 0:
            action = "emit_event"
        else:
            action = "noop"
        
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=action,
            arguments={},
            reasoning=None,
            metadata={"policy": "MyCustomPolicy"},
        )

# Usage in run.py:
agents[agent_id] = MyCustomPolicy(some_config="value")
```

---

## Advanced Topics

### Policy State Management

**Current Design**: Policies are stateless
- No internal state between `decide()` calls
- State is managed by the platform (WorldEngine, TraceDb)
- Observations include all necessary context

**Future Considerations**:
- Long-term memory could be added as a separate component
- Vector DB integration for semantic search
- Reflection/summarization of past events

### Determinism in Phase 2

**Challenge**: Real LLMs are non-deterministic by nature.

**Current Approach**:
- Deterministic action ordering (sorted by agent_id)
- Deterministic observation construction
- Deterministic tool parsing
- Mock LLM provides deterministic testing

**Future Considerations**:
- LLM providers may support deterministic mode
- Temperature=0 helps but doesn't guarantee determinism
- "Logical determinism" (same prompt → same response) may be acceptable

### Policy Composition

**Current Design**: One policy per agent.

**Future Possibilities**:
- Policy ensembles (multiple policies vote)
- Hierarchical policies (meta-policy selects sub-policy)
- Adaptive policies (policy changes based on performance)

### Performance Considerations

**Phase 1**:
- Very fast (random number generation)
- No external dependencies
- Scales to thousands of agents

**Phase 2**:
- Slower (LLM API calls)
- Network latency is the bottleneck
- Rate limiting considerations
- Async execution planned for Phase 4

### Testing Policies

**Unit Testing**:
```python
def test_random_policy():
    rng = random.Random(42)
    policy = RandomAgentPolicy(rng)
    obs = {"time_step": 0, "agent_id": "test"}
    req = policy.decide(run_id="test", time_step=0, agent_id="test", observation=obs)
    assert req.action_name in ["noop", "emit_event"]
```

**Integration Testing**:
- Use `--mock-llm` for deterministic Phase 2 testing
- Verify trace output matches expected patterns
- Test with different seeds to ensure variety

---

## Summary

The agent policy system in the Vivarium provides:

1. **Protocol-Based Design**: `AgentPolicy` protocol ensures consistent interface
2. **Multiple Implementations**: Random (Phase 1) and Cognitive (Phase 2) policies
3. **Deterministic Seeding**: Per-agent seeds derived from master seed
4. **Stateless Architecture**: Policies are pure functions, easy to test
5. **Flexible Configuration**: CLI arguments control policy behavior
6. **Extensibility**: Easy to add custom policy implementations

The system balances simplicity (Phase 1 random agents) with sophistication (Phase 2 LLM-driven agents), while maintaining determinism and reproducibility throughout.

