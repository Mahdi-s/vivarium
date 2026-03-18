#!/bin/bash
set -euo pipefail

echo "=== Verifying Epic 4: Hybrid BDI Framing & RLSF ==="

# -------------------------------------------------------------------
# 1. BDI prompt structure present
# -------------------------------------------------------------------
echo -n "1. BDI prompt sections present: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 5, 'messages': [], 'tools': ['post_message', 'noop']}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
system = msgs[0]['content']
assert '[BELIEFS]' in system, 'Missing [BELIEFS]'
assert '[DESIRES]' in system, 'Missing [DESIRES]'
assert '[INTENTIONS]' in system, 'Missing [INTENTIONS]'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 2. BDI with persona → persona in DESIRES
# -------------------------------------------------------------------
echo -n "2. Persona injected into [DESIRES]: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {
    'time_step': 0, 'messages': [],
    'agent_state': {'persona': 'Your Identity & Demographic Profile:\n  - Age: 34\n  - Gender: Female'},
}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
system = msgs[0]['content']
# Persona must appear AFTER [DESIRES] and BEFORE [INTENTIONS]
desires_idx = system.index('[DESIRES]')
intentions_idx = system.index('[INTENTIONS]')
assert 'Age: 34' in system[desires_idx:intentions_idx], 'Persona not in DESIRES section'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 3. Without persona → default desire text
# -------------------------------------------------------------------
echo -n "3. Default desire without persona: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': []}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
system = msgs[0]['content']
desires_idx = system.index('[DESIRES]')
intentions_idx = system.index('[INTENTIONS]')
desires_section = system[desires_idx:intentions_idx]
assert 'cooperatively' in desires_section, f'Default desire missing: {desires_section[:100]}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 4. Memory context injected into BELIEFS
# -------------------------------------------------------------------
echo -n "4. Memory context in [BELIEFS]: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {
    'time_step': 2, 'messages': [],
    'memory_context': [
        {'time_step': 1, 'content': \"Feedback: Action 'fly' failed with error: Unknown action\"},
    ],
}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
system = msgs[0]['content']
beliefs_idx = system.index('[BELIEFS]')
desires_idx = system.index('[DESIRES]')
beliefs_section = system[beliefs_idx:desires_idx]
assert 'Past experiences' in beliefs_section, 'Memory context not in BELIEFS'
assert 'fly' in beliefs_section, 'Feedback content not in BELIEFS'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 5. Reasoning field required in JSON schema
# -------------------------------------------------------------------
echo -n "5. Reasoning required in JSON schema: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': []}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
system = msgs[0]['content']
assert '\"reasoning\"' in system, 'reasoning field not in schema'
assert 'chain of thought' in system.lower(), 'No chain-of-thought instruction'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 6. MemoryManager.store_feedback works
# -------------------------------------------------------------------
echo -n "6. store_feedback stores correctly: "
python3 -c "
from vivarium.memory import SimpleMemorySystem, MemoryManager

mem_sys = SimpleMemorySystem()
mgr = MemoryManager(mem_sys)
mgr.store_feedback(agent_id='a1', time_step=3, action_name='fly', error='Unknown action')

entries = mem_sys.get_short_term_context(agent_id='a1', time_step=4, limit=10)
assert len(entries) == 1, f'Expected 1 entry, got {len(entries)}'
e = entries[0]
assert e['metadata']['type'] == 'feedback', f'Wrong type: {e[\"metadata\"]}'
assert e['metadata']['action_name'] == 'fly'
assert 'failed' in e['content']
assert 'Unknown action' in e['content']
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 7. RLSF: WorldEngine stores feedback on failed action
# -------------------------------------------------------------------
echo -n "7. RLSF feedback on failed action: "
python3 -c "
import json, tempfile, time, uuid, os
from vivarium.channel import InMemoryChannel
from vivarium.memory import SimpleMemorySystem, MemoryManager
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import RandomAgentPolicy
from vivarium.types import ActionRequest, RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig
import random

with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'test.db')
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = f'test_{uuid.uuid4().hex[:8]}'
    trace_db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    mem_sys = SimpleMemorySystem()
    mem_mgr = MemoryManager(mem_sys)

    policy = RandomAgentPolicy(rng=random.Random(42), action_space=['noop'])
    agents = {'a1': policy}

    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True),
        agents=agents, channel=InMemoryChannel(), trace_db=trace_db,
        memory_manager=mem_mgr,
    )

    # Manually inject a failing action request
    bad_req = ActionRequest(
        run_id=run_id, time_step=0, agent_id='a1',
        action_name='unknown_action', arguments={}, reasoning=None, metadata={},
    )
    engine.commit_requests(time_step=0, reqs=[bad_req])

    entries = mem_sys.get_short_term_context(agent_id='a1', time_step=1, limit=10)
    feedback_entries = [e for e in entries if e.get('metadata', {}).get('type') == 'feedback']
    assert len(feedback_entries) == 1, f'Expected 1 feedback, got {len(feedback_entries)}'
    assert 'unknown_action' in feedback_entries[0]['content']
    trace_db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 8. RLSF does NOT store feedback on success
# -------------------------------------------------------------------
echo -n "8. No feedback on successful action: "
python3 -c "
import json, tempfile, time, uuid, os
from vivarium.channel import InMemoryChannel
from vivarium.memory import SimpleMemorySystem, MemoryManager
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import RandomAgentPolicy
from vivarium.types import ActionRequest, RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig
import random

with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'test.db')
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = f'test_{uuid.uuid4().hex[:8]}'
    trace_db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    mem_sys = SimpleMemorySystem()
    mem_mgr = MemoryManager(mem_sys)

    policy = RandomAgentPolicy(rng=random.Random(42), action_space=['noop'])
    agents = {'a1': policy}

    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True),
        agents=agents, channel=InMemoryChannel(), trace_db=trace_db,
        memory_manager=mem_mgr,
    )

    # Successful action
    good_req = ActionRequest(
        run_id=run_id, time_step=0, agent_id='a1',
        action_name='noop', arguments={}, reasoning=None, metadata={},
    )
    engine.commit_requests(time_step=0, reqs=[good_req])

    entries = mem_sys.get_short_term_context(agent_id='a1', time_step=1, limit=10)
    feedback_entries = [e for e in entries if e.get('metadata', {}).get('type') == 'feedback']
    assert len(feedback_entries) == 0, f'Expected 0 feedback, got {len(feedback_entries)}'
    trace_db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 9. store_action still called for all actions (no regression)
# -------------------------------------------------------------------
echo -n "9. store_action still called (no regression): "
python3 -c "
import json, tempfile, time, uuid, os
from vivarium.channel import InMemoryChannel
from vivarium.memory import SimpleMemorySystem, MemoryManager
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import RandomAgentPolicy
from vivarium.types import ActionRequest, RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig
import random

with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'test.db')
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = f'test_{uuid.uuid4().hex[:8]}'
    trace_db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    mem_sys = SimpleMemorySystem()
    mem_mgr = MemoryManager(mem_sys)

    policy = RandomAgentPolicy(rng=random.Random(42), action_space=['noop'])
    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True),
        agents={'a1': policy}, channel=InMemoryChannel(), trace_db=trace_db,
        memory_manager=mem_mgr,
    )

    # Both success and failure
    reqs = [
        ActionRequest(run_id=run_id, time_step=0, agent_id='a1',
                      action_name='noop', arguments={}, reasoning=None, metadata={}),
        ActionRequest(run_id=run_id, time_step=0, agent_id='a1',
                      action_name='bad_action', arguments={}, reasoning=None, metadata={}),
    ]
    engine.commit_requests(time_step=0, reqs=reqs)

    entries = mem_sys.get_short_term_context(agent_id='a1', time_step=1, limit=20)
    action_entries = [e for e in entries if e.get('metadata', {}).get('type') == 'action']
    assert len(action_entries) == 2, f'Expected 2 action entries, got {len(action_entries)}'
    trace_db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 10. Tool-call mode also has BDI structure
# -------------------------------------------------------------------
echo -n "10. Tool-call mode has BDI structure: "
python3 -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': []}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=False)
system = msgs[0]['content']
assert '[BELIEFS]' in system
assert '[DESIRES]' in system
assert '[INTENTIONS]' in system
assert 'tools' in system.lower() or 'Tools' in system
print('✓')
" || echo "✗"

echo "=== Epic 4 Verification Complete ==="
