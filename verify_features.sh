#!/bin/bash
echo "=== Verifying All Features ==="

# ---------------------------------------------------------------------------
# Original feature checks (pre-Epic)
# ---------------------------------------------------------------------------

# 1. Settings module
echo -n "1. Settings module: "
python -c "from vivarium.settings import settings; print('✓')" || echo "✗"

# 2. Scientific report
echo -n "2. Scientific report generator: "
python -c "from vivarium.analytics.reporting import ScientificReportGenerator; print('✓')" || echo "✗"

# 3. Sparse capture config
echo -n "3. Sparse capture config: "
python -c "from vivarium.interpretability import CaptureConfig; c=CaptureConfig.from_dict({'layers':[0],'components':['resid'],'trigger_actions':['test'],'layer_sample_rate':0.5}); print('✓')" || echo "✗"

# 4. CoT indexing
echo -n "4. CoT indexing: "
python -c "from vivarium.interpretability import CaptureContext, CaptureConfig; c=CaptureConfig(layers=[0],components=['resid'],trigger_actions=['test']); ctx=CaptureContext(output_dir='/tmp',config=c); print('✓' if hasattr(ctx,'mark_cot_region') else '✗')" || echo "✗"

# 5. Dual-stack detection
echo -n "5. Dual-stack detection: "
python -c "from vivarium.llm_gateway import select_local_gateway; print('✓' if 'scientific_mode' in select_local_gateway.__code__.co_varnames else '✗')" || echo "✗"

# 6. Probe dataset diversity
echo -n "6. Probe dataset diversity: "
python -c "import json; items=[json.loads(l) for l in open('experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl')]; prefixes=[i.get('source',{}).get('prefix_template') for i in items if i.get('label')==1]; print('✓' if len(set(prefixes)) >= 5 else '✗')" || echo "✗"

# ---------------------------------------------------------------------------
# Epic 1: Core interpretability schema (vivarium_* tables)
# ---------------------------------------------------------------------------

echo -n "7. Epic 1 — Interpretability schema (vivarium_* tables): "
python -c "
from vivarium.persistence import TraceDb, TraceDbConfig
import tempfile, os
with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 't.db')))
    db.connect(); db.init_schema()
    tables = [r[0] for r in db.conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
    needed = ['vivarium_probes', 'vivarium_probe_projections', 'vivarium_answer_logprobs',
              'vivarium_interventions', 'vivarium_intervention_results']
    ok = all(t in tables for t in needed)
    db.close()
    print('✓' if ok else '✗')
" || echo "✗"

# ---------------------------------------------------------------------------
# Epic 2: Empirical Digital Twin (EmpiricalAgentStateSpace)
# ---------------------------------------------------------------------------

echo -n "8. Epic 2 — EmpiricalAgentStateSpace: "
python -c "
from vivarium.agent_state import EmpiricalAgentStateSpace
import tempfile, json, os
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, 'profiles.jsonl')
    with open(p, 'w') as f:
        f.write(json.dumps({'age': 34, 'gender': 'Female'}) + '\n')
    e = EmpiricalAgentStateSpace(p, master_seed=42)
    s = e.init_state('a1')
    obs = e.observe(agent_id='a1', state=s, time_step=0)
    print('✓' if 'persona' in obs else '✗')
" || echo "✗"

echo -n "9. Epic 2 — Persona in BDI [DESIRES]: "
python -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': [], 'agent_state': {'persona': 'Age: 34'}}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
sys = msgs[0]['content']
print('✓' if '[DESIRES]' in sys and 'Age: 34' in sys else '✗')
" || echo "✗"

# ---------------------------------------------------------------------------
# Epic 3: Archetype scaling
# ---------------------------------------------------------------------------

echo -n "10. Epic 3 — ArchetypeAgentPolicy importable: "
python -c "from vivarium.policy import ArchetypeAgentPolicy; print('✓')" || echo "✗"

echo -n "11. Epic 3 — Cache deduplication (1 LLM call for N agents): "
python -c "
import json
from vivarium.policy import ArchetypeAgentPolicy

call_count = 0
class CountGW:
    supports_tool_calls = False
    def chat(self, **kw):
        global call_count; call_count += 1
        return {'choices':[{'message':{'role':'assistant','content':json.dumps({'distribution':{'noop':1.0},'post_message_contents':[]})}}]}

p = ArchetypeAgentPolicy(gateway=CountGW(), model='t', master_seed=42)
obs = {'time_step':0,'messages':[],'agent_state':{'persona':'X'}}
for i in range(5):
    p.decide(run_id='r', time_step=0, agent_id=f'a{i}', observation=obs)
print('✓' if call_count == 1 else '✗')
" || echo "✗"

# ---------------------------------------------------------------------------
# Epic 4: BDI Framing + RLSF
# ---------------------------------------------------------------------------

echo -n "12. Epic 4 — BDI prompt structure: "
python -c "
from vivarium.agent_langgraph import _openai_messages_from_observation
obs = {'time_step': 0, 'messages': []}
msgs = _openai_messages_from_observation(agent_id='a1', observation=obs, require_json_action=True)
sys = msgs[0]['content']
ok = '[BELIEFS]' in sys and '[DESIRES]' in sys and '[INTENTIONS]' in sys and '\"reasoning\"' in sys
print('✓' if ok else '✗')
" || echo "✗"

echo -n "13. Epic 4 — RLSF store_feedback: "
python -c "
from vivarium.memory import SimpleMemorySystem, MemoryManager
m = SimpleMemorySystem(); mgr = MemoryManager(m)
mgr.store_feedback(agent_id='a1', time_step=0, action_name='fly', error='Unknown')
entries = m.get_short_term_context(agent_id='a1', time_step=1, limit=10)
fb = [e for e in entries if e.get('metadata',{}).get('type') == 'feedback']
print('✓' if len(fb) == 1 and 'fly' in fb[0]['content'] else '✗')
" || echo "✗"

# ---------------------------------------------------------------------------
# Epic 5: Mode Collapse Detection
# ---------------------------------------------------------------------------

echo -n "14. Epic 5 — Validation module importable: "
python -c "
from vivarium.analytics.validation import compute_mode_collapse_entropy, compute_empirical_divergence, compute_entropy_series
print('✓')
" || echo "✗"

echo -n "15. Epic 5 — Entropy correctness (uniform=1.0, collapsed=0.0): "
python -c "
import tempfile, os, time
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata, TraceEvent
from vivarium.analytics.validation import compute_mode_collapse_entropy

with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 't.db')))
    db.connect(); db.init_schema()

    db.insert_run(RunMetadata(run_id='u', seed=0, created_at=time.time(), config={}))
    for i in range(10):
        db.append_trace(TraceEvent(trace_id=f'u{i}', run_id='u', time_step=0,
            timestamp=float(i), agent_id=f'a{i}',
            action_type='noop' if i<5 else 'post_message', info={}, outcome={}))
    eu = compute_mode_collapse_entropy(db, 'u')

    db.insert_run(RunMetadata(run_id='c', seed=0, created_at=time.time(), config={}))
    for i in range(10):
        db.append_trace(TraceEvent(trace_id=f'c{i}', run_id='c', time_step=0,
            timestamp=float(i), agent_id=f'a{i}', action_type='noop', info={}, outcome={}))
    ec = compute_mode_collapse_entropy(db, 'c')
    db.close()
    print('✓' if abs(eu - 1.0) < 0.01 and ec == 0.0 else '✗')
" || echo "✗"

echo -n "16. Epic 5 — ScientificReport WARNING on low entropy: "
python -c "
from vivarium.types import ScientificReport
r = ScientificReport(run_id='t', duration_seconds=1.0, integrity_verified=True, mode_collapse_entropy=0.2)
print('✓' if 'WARNING' in r.summary() and 'MODE COLLAPSE' in r.summary() else '✗')
" || echo "✗"

echo "=== Verification Complete (16 checks) ==="
