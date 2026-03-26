#!/bin/bash
set -euo pipefail

echo "=== Verifying Epic 5: Automated Mode Collapse & Validation Metrics ==="

# -------------------------------------------------------------------
# 1. validation.py importable
# -------------------------------------------------------------------
echo -n "1. validation module importable: "
python3 -c "
from vivarium.analytics.validation import (
    compute_mode_collapse_entropy,
    compute_empirical_divergence,
    compute_entropy_series,
)
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 2. ScientificReport has new fields
# -------------------------------------------------------------------
echo -n "2. ScientificReport has mode_collapse_entropy field: "
python3 -c "
from vivarium.types import ScientificReport
r = ScientificReport(
    run_id='test', duration_seconds=1.0, integrity_verified=True,
    mode_collapse_entropy=0.3, empirical_divergence=0.12,
)
assert r.mode_collapse_entropy == 0.3
assert r.empirical_divergence == 0.12
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 3. Mode collapse entropy: uniform distribution = max entropy
# -------------------------------------------------------------------
echo -n "3. Entropy: uniform actions → high entropy: "
python3 -c "
import math, tempfile, os, time, uuid
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata, TraceEvent
from vivarium.analytics.validation import compute_mode_collapse_entropy

with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 'test.db')))
    db.connect()
    db.init_schema()
    run_id = 'test_uniform'
    db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    # 10 agents, 5 each for 'noop' and 'post_message' → entropy = 1.0
    for i in range(10):
        action = 'noop' if i < 5 else 'post_message'
        db.append_trace(TraceEvent(
            trace_id=f't{i}', run_id=run_id, time_step=0, timestamp=float(i),
            agent_id=f'a{i}', action_type=action, info={}, outcome={},
        ))

    ent = compute_mode_collapse_entropy(db, run_id)
    assert abs(ent - 1.0) < 0.01, f'Expected ~1.0, got {ent}'
    db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 4. Mode collapse entropy: all same action → entropy = 0
# -------------------------------------------------------------------
echo -n "4. Entropy: identical actions → zero entropy: "
python3 -c "
import tempfile, os, time
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata, TraceEvent
from vivarium.analytics.validation import compute_mode_collapse_entropy

with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 'test.db')))
    db.connect()
    db.init_schema()
    run_id = 'test_collapse'
    db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    for i in range(10):
        db.append_trace(TraceEvent(
            trace_id=f't{i}', run_id=run_id, time_step=0, timestamp=float(i),
            agent_id=f'a{i}', action_type='noop', info={}, outcome={},
        ))

    ent = compute_mode_collapse_entropy(db, run_id)
    assert ent == 0.0, f'Expected 0.0, got {ent}'
    db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 5. Entropy per time_step filter
# -------------------------------------------------------------------
echo -n "5. Entropy filtered by time_step: "
python3 -c "
import tempfile, os, time
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata, TraceEvent
from vivarium.analytics.validation import compute_mode_collapse_entropy

with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 'test.db')))
    db.connect()
    db.init_schema()
    run_id = 'test_step'
    db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    # Step 0: all noop (entropy=0)
    for i in range(5):
        db.append_trace(TraceEvent(
            trace_id=f's0_{i}', run_id=run_id, time_step=0, timestamp=float(i),
            agent_id=f'a{i}', action_type='noop', info={}, outcome={},
        ))
    # Step 1: mixed (entropy>0)
    for i in range(5):
        action = 'noop' if i < 2 else 'post_message'
        db.append_trace(TraceEvent(
            trace_id=f's1_{i}', run_id=run_id, time_step=1, timestamp=float(i),
            agent_id=f'a{i}', action_type=action, info={}, outcome={},
        ))

    ent0 = compute_mode_collapse_entropy(db, run_id, time_step=0)
    ent1 = compute_mode_collapse_entropy(db, run_id, time_step=1)
    assert ent0 == 0.0, f'Step 0 should be 0, got {ent0}'
    assert ent1 > 0.5, f'Step 1 should be > 0.5, got {ent1}'
    db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 6. Empirical divergence (KS test)
# -------------------------------------------------------------------
echo -n "6. Empirical divergence (identical → 0, different → high): "
python3 -c "
from vivarium.analytics.validation import compute_empirical_divergence

# Identical distributions → KS ~0
same = compute_empirical_divergence([1,2,3,4,5], [1,2,3,4,5])
assert same['ks_statistic'] < 0.3, f'Same data should be ~0, got {same}'

# Very different distributions → KS ~1
diff = compute_empirical_divergence([0,0,0,0,0], [100,100,100,100,100])
assert diff['ks_statistic'] > 0.8, f'Different data should be ~1, got {diff}'

# Empty data → fallback
empty = compute_empirical_divergence([], [1,2,3])
assert empty['ks_statistic'] == 1.0
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 7. Entropy series
# -------------------------------------------------------------------
echo -n "7. Entropy series across time steps: "
python3 -c "
import tempfile, os, time
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata, TraceEvent
from vivarium.analytics.validation import compute_entropy_series

with tempfile.TemporaryDirectory() as td:
    db = TraceDb(TraceDbConfig(db_path=os.path.join(td, 'test.db')))
    db.connect()
    db.init_schema()
    run_id = 'test_series'
    db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    for t in range(3):
        for i in range(4):
            action = 'noop' if (i + t) % 2 == 0 else 'post_message'
            db.append_trace(TraceEvent(
                trace_id=f't{t}_{i}', run_id=run_id, time_step=t, timestamp=float(i),
                agent_id=f'a{i}', action_type=action, info={}, outcome={},
            ))

    series = compute_entropy_series(db, run_id)
    assert len(series) == 3, f'Expected 3 steps, got {len(series)}'
    assert all('entropy' in s and 'n_actions' in s for s in series)
    assert series[0]['time_step'] == 0
    db.close()
    print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 8. ScientificReport summary() WARNING on low entropy
# -------------------------------------------------------------------
echo -n "8. Summary WARNING on mode collapse: "
python3 -c "
from vivarium.types import ScientificReport

# Low entropy → WARNING
r_low = ScientificReport(
    run_id='test', duration_seconds=1.0, integrity_verified=True,
    mode_collapse_entropy=0.2,
)
summary = r_low.summary()
assert 'WARNING' in summary, f'Expected WARNING in summary: {summary}'
assert 'MODE COLLAPSE' in summary

# Normal entropy → no warning
r_ok = ScientificReport(
    run_id='test', duration_seconds=1.0, integrity_verified=True,
    mode_collapse_entropy=1.5,
)
summary_ok = r_ok.summary()
assert 'WARNING' not in summary_ok
assert 'Action Entropy' in summary_ok
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 9. Exports from analytics __init__
# -------------------------------------------------------------------
echo -n "9. Validation exports from analytics package: "
python3 -c "
from vivarium.analytics import (
    compute_mode_collapse_entropy,
    compute_empirical_divergence,
    compute_entropy_series,
)
assert callable(compute_mode_collapse_entropy)
assert callable(compute_empirical_divergence)
assert callable(compute_entropy_series)
print('✓')
" || echo "✗"

# -------------------------------------------------------------------
# 10. Integration: entropy computed on real WorldEngine trace
# -------------------------------------------------------------------
echo -n "10. Integration with WorldEngine trace: "
python3 -c "
import json, tempfile, time, uuid, os, random
from vivarium.channel import InMemoryChannel
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import RandomAgentPolicy
from vivarium.types import RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig
from vivarium.analytics.validation import compute_mode_collapse_entropy

with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'test.db')
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = f'test_{uuid.uuid4().hex[:8]}'
    trace_db.insert_run(RunMetadata(run_id=run_id, seed=42, created_at=time.time(), config={}))

    policy = RandomAgentPolicy(rng=random.Random(42), action_space=['noop', 'emit_event'])
    agents = {f'a{i}': policy for i in range(6)}

    engine = WorldEngine(
        config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True),
        agents=agents, channel=InMemoryChannel(), trace_db=trace_db,
    )
    engine.run(steps=2)

    ent = compute_mode_collapse_entropy(trace_db, run_id)
    assert ent >= 0.0, f'Entropy must be non-negative, got {ent}'
    # With RandomAgentPolicy and 2 actions, entropy should be > 0
    assert ent > 0.0, f'Expected positive entropy with random policy, got {ent}'

    trace_db.close()
    print('✓')
" || echo "✗"

echo "=== Epic 5 Verification Complete ==="
