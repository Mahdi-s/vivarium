#!/bin/bash
set -euo pipefail

echo "=== Verifying Epic 3: Scalability via LLM Archetypes ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# -------------------------------------------------------------------
# 1. ArchetypeAgentPolicy importable
# -------------------------------------------------------------------
echo -n "1. ArchetypeAgentPolicy importable: "
python3 -c "
from vivarium.policy import ArchetypeAgentPolicy, _canonical_json
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 2. _canonical_json produces stable output
# -------------------------------------------------------------------
echo -n "2. _canonical_json stable output: "
python3 -c "
from vivarium.policy import _canonical_json
a = _canonical_json({'b': 2, 'a': 1})
b = _canonical_json({'a': 1, 'b': 2})
assert a == b, f'{a} != {b}'
assert a == '{\"a\":1,\"b\":2}', f'Unexpected: {a}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 3. Archetype hash is stable (same inputs → same hash)
# -------------------------------------------------------------------
echo -n "3. Archetype hash stability: "
python3 -c "
from vivarium.policy import ArchetypeAgentPolicy
obs = {
    'time_step': 5,
    'messages': [{'time_step': 4, 'author_id': 'a1', 'content': 'hello'}],
    'agent_state': {'persona': 'Age: 34, Gender: Female'},
}
h1 = ArchetypeAgentPolicy._compute_archetype_hash(obs, 5)
h2 = ArchetypeAgentPolicy._compute_archetype_hash(obs, 5)
assert h1 == h2, 'Same inputs must yield same hash'
assert len(h1) == 64, f'Expected SHA-256 hex, got len={len(h1)}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 4. Archetype hash differs when profile differs
# -------------------------------------------------------------------
echo -n "4. Archetype hash differs for different profiles: "
python3 -c "
from vivarium.policy import ArchetypeAgentPolicy
obs_a = {
    'time_step': 0, 'messages': [],
    'agent_state': {'persona': 'Age: 34'},
}
obs_b = {
    'time_step': 0, 'messages': [],
    'agent_state': {'persona': 'Age: 52'},
}
ha = ArchetypeAgentPolicy._compute_archetype_hash(obs_a, 0)
hb = ArchetypeAgentPolicy._compute_archetype_hash(obs_b, 0)
assert ha != hb, 'Different profiles must hash differently'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 5. Cache hit: MockLLMGateway called only once for same archetype
# -------------------------------------------------------------------
echo -n "5. Cache hit (LLM called once per archetype): "
python3 -c "
import json
from vivarium.policy import ArchetypeAgentPolicy

call_count = 0
class CountingGateway:
    supports_tool_calls = False
    def chat(self, **kwargs):
        global call_count
        call_count += 1
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': json.dumps({
                        'distribution': {'post_message': 0.6, 'noop': 0.4},
                        'post_message_contents': ['Hello!', 'Hi there!'],
                    })
                }
            }]
        }

gw = CountingGateway()
policy = ArchetypeAgentPolicy(gateway=gw, model='test', master_seed=42)

# Same archetype obs for 5 agents
obs = {'time_step': 0, 'messages': [], 'agent_state': {'persona': 'Age: 34'}}
for i in range(5):
    policy.decide(run_id='r', time_step=0, agent_id=f'agent_{i:03d}', observation=obs)

assert call_count == 1, f'Expected 1 LLM call, got {call_count}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 6. Cache invalidation on time_step change
# -------------------------------------------------------------------
echo -n "6. Cache invalidation on time_step change: "
python3 -c "
import json
from vivarium.policy import ArchetypeAgentPolicy

call_count = 0
class CountingGateway:
    supports_tool_calls = False
    def chat(self, **kwargs):
        global call_count
        call_count += 1
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': json.dumps({
                        'distribution': {'noop': 1.0},
                        'post_message_contents': [],
                    })
                }
            }]
        }

gw = CountingGateway()
policy = ArchetypeAgentPolicy(gateway=gw, model='test', master_seed=42)
obs = {'time_step': 0, 'messages': [], 'agent_state': {'persona': 'Age: 34'}}

policy.decide(run_id='r', time_step=0, agent_id='a0', observation=obs)
assert call_count == 1

# Same time_step → cache hit
policy.decide(run_id='r', time_step=0, agent_id='a1', observation=obs)
assert call_count == 1

# New time_step → cache miss (invalidated)
obs2 = {'time_step': 1, 'messages': [], 'agent_state': {'persona': 'Age: 34'}}
policy.decide(run_id='r', time_step=1, agent_id='a0', observation=obs2)
assert call_count == 2, f'Expected 2, got {call_count}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 7. Distribution parsing: valid JSON
# -------------------------------------------------------------------
echo -n "7. Distribution parsing (valid JSON): "
python3 -c "
import json
from vivarium.policy import ArchetypeAgentPolicy

policy = ArchetypeAgentPolicy.__new__(ArchetypeAgentPolicy)
object.__setattr__(policy, 'action_space', ['post_message', 'noop'])

resp = {
    'choices': [{
        'message': {
            'role': 'assistant',
            'content': json.dumps({
                'distribution': {'post_message': 0.7, 'noop': 0.3},
                'post_message_contents': ['Hello!'],
                'reasoning': 'test',
            })
        }
    }]
}
result = policy._parse_distribution(resp)
dist = result['distribution']
assert abs(dist['post_message'] - 0.7) < 0.01
assert abs(dist['noop'] - 0.3) < 0.01
assert result['post_message_contents'] == ['Hello!']
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 8. Distribution parsing: malformed → fallback
# -------------------------------------------------------------------
echo -n "8. Distribution parsing (malformed → noop fallback): "
python3 -c "
from vivarium.policy import ArchetypeAgentPolicy

policy = ArchetypeAgentPolicy.__new__(ArchetypeAgentPolicy)
object.__setattr__(policy, 'action_space', ['post_message', 'noop'])

# Garbage response
resp = {'choices': [{'message': {'role': 'assistant', 'content': 'I cannot do that.'}}]}
result = policy._parse_distribution(resp)
assert result['distribution'] == {'noop': 1.0}, f'Got {result}'

# Missing choices
result2 = policy._parse_distribution({})
assert result2['distribution'] == {'noop': 1.0}

# Unknown actions filtered out
import json
resp3 = {
    'choices': [{
        'message': {
            'role': 'assistant',
            'content': json.dumps({'distribution': {'fly_away': 0.8, 'noop': 0.2}})
        }
    }]
}
result3 = policy._parse_distribution(resp3)
assert 'fly_away' not in result3['distribution']
assert abs(result3['distribution'].get('noop', 0) - 1.0) < 0.01
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 9. Sampling determinism (same agent_id → same action)
# -------------------------------------------------------------------
echo -n "9. Sampling determinism: "
python3 -c "
from vivarium.policy import ArchetypeAgentPolicy

policy = ArchetypeAgentPolicy.__new__(ArchetypeAgentPolicy)
object.__setattr__(policy, 'action_space', ['post_message', 'noop'])
object.__setattr__(policy, 'master_seed', 42)

dist_resp = {
    'distribution': {'post_message': 0.5, 'noop': 0.5},
    'post_message_contents': ['Hello!', 'Hi!'],
}
r1 = policy._sample_action(dist_resp, 'agent_007')
r2 = policy._sample_action(dist_resp, 'agent_007')
assert r1 == r2, f'Same agent must get same result: {r1} vs {r2}'

# Different agent may (statistically) get different result
# but at minimum the function works without error
r3 = policy._sample_action(dist_resp, 'agent_042')
assert isinstance(r3, tuple) and len(r3) == 2
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 10. Integration with WorldEngine using MockLLMGateway
# -------------------------------------------------------------------
echo -n "10. WorldEngine integration (MockLLMGateway): "
python3 -c "
import json, tempfile, time, uuid
from vivarium.policy import ArchetypeAgentPolicy
from vivarium.agent_state import EmpiricalAgentStateSpace
from vivarium.channel import InMemoryChannel
from vivarium.llm_gateway import MockLLMGateway
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig

with tempfile.TemporaryDirectory() as td:
    import os
    db_path = os.path.join(td, 'test.db')
    profiles_path = os.path.join(td, 'profiles.jsonl')
    with open(profiles_path, 'w') as f:
        f.write(json.dumps({'age': 34, 'gender': 'Female'}) + '\n')
        f.write(json.dumps({'age': 52, 'gender': 'Male'}) + '\n')

    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = f'test_{uuid.uuid4().hex[:8]}'
    trace_db.insert_run(RunMetadata(
        run_id=run_id, seed=42, created_at=time.time(), config={},
    ))

    agent_state = EmpiricalAgentStateSpace(profiles_path, master_seed=42)
    gw = MockLLMGateway(seed=42)

    policy = ArchetypeAgentPolicy(
        gateway=gw, model='mock', master_seed=42,
        action_space=['post_message', 'noop'],
    )
    # 6 agents, 2 profiles → at most 2 archetype LLM calls per step
    agents = {f'agent_{i:03d}': policy for i in range(6)}

    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True, deterministic_ids=True),
        agents=agents,
        channel=InMemoryChannel(),
        trace_db=trace_db,
        agent_state_space=agent_state,
    )
    engine.run(steps=2)

    rows = trace_db.conn.execute(
        'SELECT agent_id, action_type FROM trace WHERE run_id = ?', (run_id,)
    ).fetchall()
    assert len(rows) == 12, f'Expected 12 trace rows (6 agents × 2 steps), got {len(rows)}'

    # Verify metadata contains archetype info
    info_rows = trace_db.conn.execute(
        'SELECT info_json FROM trace WHERE run_id = ? LIMIT 1', (run_id,)
    ).fetchall()
    info = json.loads(info_rows[0]['info_json'])
    meta = info.get('metadata', {})
    assert meta.get('policy') == 'ArchetypeAgentPolicy', f'Got {meta}'
    assert 'archetype_hash' in meta

    trace_db.close()
    print('✓')
" || echo "✗"

echo "=== Epic 3 Verification Complete ==="
