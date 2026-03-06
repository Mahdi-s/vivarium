# Testing Guide: PRD Implementation Verification

This guide provides commands to test all implemented features from the PRD analysis.

## Prerequisites

1. Install dependencies:
```bash
uv sync --extra cognitive --extra interpretability
# or: pip install -e .[cognitive,interpretability]
```

2. Optional: Install JSON repair for better text parsing:
```bash
pip install json-repair
```

## Test Commands

### 1. Basic Simulation Test (Phase 1 - Random Agents)

Test core determinism and trace persistence:

```bash
PYTHONPATH=src vvm phase1 --steps 10 --agents 3 --seed 42 --db test_phase1.db
```

**Expected Output:**
- `run_id=<uuid>`
- `db=test_phase1.db`
- Database created with trace events
- Verify determinism: Run twice with same seed, should produce identical results

**Verify:**
```bash
# Run again with same parameters
PYTHONPATH=src vvm phase1 --steps 10 --agents 3 --seed 42 --db test_phase1_verify.db

# Compare databases (should be identical)
sqlite3 test_phase1.db "SELECT COUNT(*) FROM trace;" 
sqlite3 test_phase1_verify.db "SELECT COUNT(*) FROM trace;"
```

### 2. Cognitive Simulation with Mock LLM (Phase 2)

Test LangGraph integration and dual-mode action parsing:

```bash
PYTHONPATH=src vvm phase2 \
  --steps 5 --agents 2 --seed 42 \
  --mock-llm \
  --db test_phase2_mock.db \
  --message-history 10
```

**Expected Output:**
- `run_id=<uuid>`
- `db=test_phase2_mock.db`
- Messages table populated with agent interactions
- Trace events with `post_message` actions

**Verify:**
```bash
sqlite3 test_phase2_mock.db "SELECT action_type, COUNT(*) FROM trace GROUP BY action_type;"
sqlite3 test_phase2_mock.db "SELECT COUNT(*) FROM messages;"
```

### 3. Local LLM Test with llama.cpp (Phase 2)

Test with actual GGUF model via llama.cpp server:

**Terminal 1 - Start llama.cpp server:**
```bash
# Use a small model for testing
PYTHONPATH=src vvm llama serve models/ollama__library_smollm2_135m.gguf
```

**Terminal 2 - Run simulation:**
```bash
PYTHONPATH=src vvm phase2 \
  --steps 3 --agents 2 --seed 42 \
  --api-base http://127.0.0.1:8081/v1 \
  --api-key local \
  --model ollama__library_smollm2_135m.gguf \
  --db test_llamacpp.db \
  --message-history 5
```

**Expected Output:**
- Agents make decisions using the local model
- Messages posted to shared feed
- Trace events recorded

**Note:** The llama.cpp server must be running. If port 8081 is in use, check the server output for the actual port.

### 4. Rate Limiting Test

Test rate limiting and backpressure (requires OpenAI API key or mock):

```python
# Create test_rate_limit.py
from aam.llm_gateway import LiteLLMGateway, RateLimitConfig
from aam.tools import post_message_tool

gateway = LiteLLMGateway(
    rate_limit_config=RateLimitConfig(
        max_concurrent_requests=2,
        requests_per_minute=10,
        tokens_per_minute=1000,
        enable_context_degradation=True,
    )
)

# Test with mock (or real API)
messages = [{"role": "user", "content": "Hello"}]
# This will respect rate limits
```

### 5. Replay Functionality Test

Test counterfactual analysis and state reconstruction:

```bash
# First, create a simulation
PYTHONPATH=src vvm phase2 \
  --steps 10 --agents 2 --seed 42 \
  --mock-llm \
  --db test_replay.db

# Then use Python to replay
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from aam.persistence import TraceDb, TraceDbConfig
from aam.world_engine import WorldEngine, WorldEngineConfig
from aam.replay import ReplayEngine
from aam.channel import InMemoryChannel

db = TraceDb(TraceDbConfig(db_path="test_replay.db"))
db.connect()

# Get run_id from database
run_meta = db.conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
if run_meta:
    run_id = run_meta[0]
    print(f"Replaying run_id={run_id}")
    
    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id),
        agents={},
        channel=InMemoryChannel(),
        trace_db=db,
    )
    
    replay = ReplayEngine(trace_db=db, engine=engine)
    
    # Replay to step 5
    replay.replay_to_step(time_step=5, rebuild_state=True)
    
    # Get state snapshot
    state = replay.get_state_at_step(time_step=5)
    print(f"State at step 5: {len(state['messages'])} messages, {state['trace_count']} trace events")
EOF
```

**Expected Output:**
- State successfully reconstructed
- Messages table rebuilt from trace
- State snapshot shows correct counts

### 6. Memory System Test

Test long-term memory and reflection:

```python
# Create test_memory.py
import sys
sys.path.insert(0, 'src')
from aam.memory import SimpleMemorySystem, MemoryManager
from aam.types import Observation

memory = SimpleMemorySystem()
manager = MemoryManager(memory)

# Store some memories
manager.store_action(agent_id="agent_001", time_step=0, action_name="post_message", arguments={"content": "Hello"})
manager.store_action(agent_id="agent_001", time_step=1, action_name="post_message", arguments={"content": "World"})
manager.store_observation(agent_id="agent_001", time_step=2, observation={"messages": [{"content": "Hi"}]})

# Retrieve memories
memories = memory.retrieve(agent_id="agent_001", query="Hello", limit=5)
print(f"Retrieved {len(memories)} memories")

# Get summary
summary = memory.summarize(agent_id="agent_001", up_to_time_step=2)
print(f"Summary: {summary}")

# Get short-term context
context = memory.get_short_term_context(agent_id="agent_001", time_step=2, limit=10)
print(f"Context: {len(context)} items")
```

### 7. Domain State Test

Test generic domain state system:

```python
# Create test_domain_state.py
import sys
sys.path.insert(0, 'src')
from aam.domain_state import GenericDomainState, SocialMediaDomainHandler
from aam.persistence import TraceDb, TraceDbConfig

db = TraceDb(TraceDbConfig(db_path="test_domain.db"))
db.connect()
db.init_schema()

domain_state = GenericDomainState(db)
domain_state.register_handler("social_media", SocialMediaDomainHandler())

# Test creating a post
result = domain_state.handle_action(
    domain="social_media",
    action_name="create_post",
    arguments={"content": "Test post"},
    run_id="test_run",
    time_step=0,
    agent_id="agent_001"
)
print(f"Create post result: {result}")

# Test liking a post
if result.get("success"):
    post_id = result["data"]["post_id"]
    like_result = domain_state.handle_action(
        domain="social_media",
        action_name="like_post",
        arguments={"post_id": post_id},
        run_id="test_run",
        time_step=1,
        agent_id="agent_002"
    )
    print(f"Like post result: {like_result}")

# Get state snapshot
snapshot = domain_state.get_state_snapshot(domain="social_media", run_id="test_run", time_step=1)
print(f"State snapshot: {snapshot}")
```

### 8. Parquet Export Test

Test Parquet export for analysis:

```bash
# First create a simulation
PYTHONPATH=src vvm phase2 \
  --steps 10 --agents 2 --seed 42 \
  --mock-llm \
  --db test_export.db

# Export to Parquet
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from aam.persistence import TraceDb, TraceDbConfig
from aam.export import export_trace_to_parquet, export_messages_to_parquet

db = TraceDb(TraceDbConfig(db_path="test_export.db"))
db.connect()

# Get run_id
run_meta = db.conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
if run_meta:
    run_id = run_meta[0]
    export_trace_to_parquet(db, run_id=run_id, output_path="trace.parquet")
    export_messages_to_parquet(db, run_id=run_id, output_path="messages.parquet")
    print("Exported to Parquet files")
EOF
```

**Expected Output:**
- `trace.parquet` and `messages.parquet` files created
- Can be read with pandas: `pd.read_parquet("trace.parquet")`

### 9. Environment State Hash Test

Verify state hash computation:

```bash
PYTHONPATH=src vvm phase2 \
  --steps 5 --agents 2 --seed 42 \
  --mock-llm \
  --db test_hash.db

# Check state hashes
sqlite3 test_hash.db "SELECT time_step, environment_state_hash FROM trace WHERE environment_state_hash IS NOT NULL LIMIT 5;"
```

**Expected Output:**
- State hashes computed for each trace event
- Same time_step should have same hash (after all actions committed)

### 10. Retry Logic Test

Test retry mechanism in scheduler:

```python
# The retry logic is automatically used in Phase 4 experiments
# It will retry failed agent decisions up to 2 additional times
# Create test_retry.json:
cat > test_retry.json << 'EOF'
{
  "run": {"steps": 5, "agents": 2, "seed": 42},
  "scheduler": {"per_agent_timeout_s": 2.0, "max_concurrency": 2},
  "policy": {"kind": "cognitive", "model": "mock", "mock_llm": true}
}
EOF

PYTHONPATH=src vvm experiment --config test_retry.json
```

**Expected Output:**
- Simulation completes even if some agents timeout
- Fallback `noop` actions recorded with retry metadata

## Comprehensive Test Script

Run all tests in sequence:

```bash
#!/bin/bash
# test_all.sh

set -e

echo "=== Test 1: Phase 1 Determinism ==="
PYTHONPATH=src vvm phase1 --steps 10 --agents 3 --seed 42 --db test1.db
PYTHONPATH=src vvm phase1 --steps 10 --agents 3 --seed 42 --db test1_verify.db
COUNT1=$(sqlite3 test1.db "SELECT COUNT(*) FROM trace;")
COUNT2=$(sqlite3 test1_verify.db "SELECT COUNT(*) FROM trace;")
if [ "$COUNT1" == "$COUNT2" ]; then
    echo "✓ Determinism test passed"
else
    echo "✗ Determinism test failed"
    exit 1
fi

echo "=== Test 2: Phase 2 Mock LLM ==="
PYTHONPATH=src vvm phase2 --steps 5 --agents 2 --seed 42 --mock-llm --db test2.db
MSG_COUNT=$(sqlite3 test2.db "SELECT COUNT(*) FROM messages;")
if [ "$MSG_COUNT" -gt 0 ]; then
    echo "✓ Mock LLM test passed ($MSG_COUNT messages)"
else
    echo "✗ Mock LLM test failed"
    exit 1
fi

echo "=== Test 3: State Hash ==="
PYTHONPATH=src vvm phase2 --steps 3 --agents 2 --seed 42 --mock-llm --db test3.db
HASH_COUNT=$(sqlite3 test3.db "SELECT COUNT(*) FROM trace WHERE environment_state_hash IS NOT NULL;")
if [ "$HASH_COUNT" -gt 0 ]; then
    echo "✓ State hash test passed ($HASH_COUNT hashes computed)"
else
    echo "✗ State hash test failed"
    exit 1
fi

echo "=== All tests passed! ==="
```

## Expected Results Summary

After running all tests, you should have:

1. **Deterministic simulations** - Same seed produces identical results
2. **Trace persistence** - All actions recorded in SQLite
3. **Message feed** - Agents can communicate via messages
4. **State hashes** - Integrity checking enabled
5. **Replay capability** - Can reconstruct state from trace
6. **Memory system** - Long-term and short-term memory working
7. **Domain state** - Generic system for custom domains
8. **Parquet export** - Efficient analysis format
9. **Rate limiting** - API protection (when configured)
10. **Retry logic** - Fault tolerance for agent failures

## Troubleshooting

- **Import errors**: Make sure `PYTHONPATH=src` is set
- **Model not found**: Check `models/` directory has GGUF files
- **llama.cpp server**: Ensure it's built in `third_party/llama.cpp/build/bin/`
- **Database locked**: Close other connections or delete `.db-wal` files
- **Parquet export fails**: Install `pandas` and `pyarrow`: `pip install pandas pyarrow`

