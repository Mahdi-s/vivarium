#!/bin/bash
set -euo pipefail

echo "=== Verifying Epic 2: Empirical Digital Twin Initialization ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# -------------------------------------------------------------------
# Create temporary test fixtures
# -------------------------------------------------------------------

# CSV fixture
cat > "$TMPDIR/demo_profiles.csv" <<'CSVEOF'
age,gender,education,occupation,political_leaning,region
34,Female,Masters,Software Engineer,Moderate,Northeast
52,Male,PhD,Professor,Liberal,West Coast
28,Non-binary,Bachelors,Nurse,Conservative,Midwest
45,Male,High School,Mechanic,Conservative,South
61,Female,PhD,Retired Scientist,Liberal,Northeast
CSVEOF

# JSONL fixture
cat > "$TMPDIR/demo_profiles.jsonl" <<'JSONLEOF'
{"age": 34, "gender": "Female", "education": "Masters", "occupation": "Software Engineer"}
{"age": 52, "gender": "Male", "education": "PhD", "occupation": "Professor"}
{"age": 28, "gender": "Non-binary", "education": "Bachelors", "occupation": "Nurse"}
JSONLEOF

# JSON (array) fixture
cat > "$TMPDIR/demo_profiles.json" <<'JSONEOF'
[
  {"age": 40, "gender": "Male", "education": "Bachelors", "occupation": "Accountant"},
  {"age": 29, "gender": "Female", "education": "Masters", "occupation": "Designer"}
]
JSONEOF

# -------------------------------------------------------------------
# 1. EmpiricalAgentStateSpace loads CSV
# -------------------------------------------------------------------
echo -n "1. Load CSV profiles: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.csv')
assert len(e._profiles) == 5, f'Expected 5 profiles, got {len(e._profiles)}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 2. EmpiricalAgentStateSpace loads JSONL
# -------------------------------------------------------------------
echo -n "2. Load JSONL profiles: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.jsonl')
assert len(e._profiles) == 3, f'Expected 3 profiles, got {len(e._profiles)}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 3. EmpiricalAgentStateSpace loads JSON array
# -------------------------------------------------------------------
echo -n "3. Load JSON profiles: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.json')
assert len(e._profiles) == 2, f'Expected 2 profiles, got {len(e._profiles)}'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 4. Deterministic init_state assignment (same agent → same profile)
# -------------------------------------------------------------------
echo -n "4. Deterministic assignment: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.csv', master_seed=42)
s1 = e.init_state('agent_007')
s2 = e.init_state('agent_007')
assert s1 == s2, 'Same agent_id must yield identical state'
# Different agent should (very likely) get a different profile
s3 = e.init_state('agent_042')
# At minimum the structure is correct
assert 'profile' in s1 and '_profile_index' in s1
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 5. observe() produces persona text block
# -------------------------------------------------------------------
echo -n "5. Observe produces persona: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.csv', master_seed=42)
state = e.init_state('agent_007')
obs = e.observe(agent_id='agent_007', state=state, time_step=0)
assert 'persona' in obs, 'observe() must return a persona key'
persona = obs['persona']
assert 'Your Identity & Demographic Profile:' in persona
assert 'You MUST embody this persona' in persona
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 6. transition() is immutable (returns current_state unchanged)
# -------------------------------------------------------------------
echo -n "6. Immutable transition: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
from vivarium.types import ActionRequest, ActionResult
e = EmpiricalAgentStateSpace('$TMPDIR/demo_profiles.csv')
state = e.init_state('agent_001')
dummy_action = ActionRequest(
    run_id='r', time_step=0, agent_id='agent_001',
    action_name='noop', arguments={}, reasoning=None, metadata={}
)
dummy_result = ActionResult(
    success=True, data={}, trace_id='fake-trace-id'
)
new_state = e.transition(
    agent_id='agent_001', current_state=state,
    action=dummy_action, result=dummy_result, time_step=1
)
assert new_state is state, 'transition must return the same object (immutable)'
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 7. _openai_messages_from_observation injects persona into system prompt
# -------------------------------------------------------------------
echo -n "7. Persona injected into LLM system prompt: "
python -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {
    'time_step': 1,
    'messages': [],
    'agent_state': {
        'persona': 'Your Identity & Demographic Profile:\n  - Age: 34\n  - Gender: Female'
    },
}
msgs = _openai_messages_from_observation(
    agent_id='agent_007', observation=obs, require_json_action=True
)
system_content = msgs[0]['content']
assert 'Your Identity & Demographic Profile:' in system_content, \
    'System prompt must contain persona'
assert 'Age: 34' in system_content
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 8. Without agent_state, system prompt is unchanged (no regression)
# -------------------------------------------------------------------
echo -n "8. No regression without agent_state: "
python -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': []}
msgs = _openai_messages_from_observation(
    agent_id='agent_007', observation=obs, require_json_action=False
)
system_content = msgs[0]['content']
# BDI structure present even without agent_state
assert '[BELIEFS]' in system_content
assert '[DESIRES]' in system_content
assert '[INTENTIONS]' in system_content
# No persona-specific text should appear (only default desire)
assert 'Demographic Profile' not in system_content
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 9. Empty dataset raises ValueError
# -------------------------------------------------------------------
echo -n "9. Empty dataset raises ValueError: "
python -c "
import tempfile, os
from vivarium.agent_state import EmpiricalAgentStateSpace
# Create empty CSV
empty_csv = os.path.join('$TMPDIR', 'empty.csv')
with open(empty_csv, 'w') as f:
    f.write('col1,col2\n')  # header only, no rows
try:
    EmpiricalAgentStateSpace(empty_csv)
    print('✗ (no error raised)')
except ValueError:
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 10. NoOpAgentStateSpace still works (backward compat)
# -------------------------------------------------------------------
echo -n "10. NoOpAgentStateSpace backward compat: "
python -c "
from vivarium.agent_state import NoOpAgentStateSpace
n = NoOpAgentStateSpace()
assert n.init_state('x') == {}
assert n.observe(agent_id='x', state={}, time_step=0) == {}
print('✓')
" || echo "✗"

echo "=== Epic 2 Verification Complete ==="
