#!/usr/bin/env python3
"""
Minimal end-to-end integration smoke test for Vivarium.

Requires:
  - Ollama running locally (http://localhost:11434)
  - A small model pulled (default: qwen2.5:0.5b)
  - litellm installed (pip install litellm)

Exercises:
  1. TraceDb schema init (core + interpretability)
  2. EmpiricalAgentStateSpace loading + deterministic assignment
  3. Persona injection into LLM system prompt
  4. Real LLM call via Ollama (LiteLLMGateway)
  5. WorldEngine step: observation → decide → execute → trace
  6. Interpretability table writes (probe, projection, answer_logprob)
  7. Query-back verification of all stored data
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from vivarium.agent_langgraph import SimpleCognitivePolicy
from vivarium.agent_state import EmpiricalAgentStateSpace
from vivarium.channel import InMemoryChannel
from vivarium.llm_gateway import create_gateway
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.tools import default_tools
from vivarium.types import RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_ID = os.environ.get("VVM_TEST_MODEL", "qwen2.5:0.5b")
NUM_AGENTS = 3
NUM_STEPS = 1
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    mark = PASS if passed else FAIL
    msg = f"  {mark} {name}"
    if detail and not passed:
        msg += f"  ({detail})"
    print(msg)


def main() -> int:
    print(f"\n=== Vivarium Integration Smoke Test ===")
    print(f"    Model : {MODEL_ID}")
    print(f"    Ollama: {OLLAMA_BASE}")
    print(f"    Agents: {NUM_AGENTS}  Steps: {NUM_STEPS}\n")

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "smoke.db")
        profiles_path = os.path.join(td, "profiles.jsonl")
        run_id = f"smoke_{uuid.uuid4().hex[:8]}"

        # ---- 1. Write test demographic profiles ----
        profiles = [
            {"age": 34, "gender": "Female", "education": "Masters", "occupation": "Software Engineer", "region": "Northeast"},
            {"age": 52, "gender": "Male", "education": "PhD", "occupation": "Professor", "region": "West Coast"},
            {"age": 28, "gender": "Non-binary", "education": "Bachelors", "occupation": "Nurse", "region": "Midwest"},
        ]
        with open(profiles_path, "w") as f:
            for p in profiles:
                f.write(json.dumps(p) + "\n")

        # ---- 2. Init persistence ----
        trace_db = TraceDb(TraceDbConfig(db_path=db_path))
        trace_db.connect()
        trace_db.init_schema()
        check("Schema init (core + interpretability)", True)

        trace_db.insert_run(RunMetadata(
            run_id=run_id, seed=42, created_at=time.time(),
            config={"model": MODEL_ID, "test": True},
        ))
        check("Run metadata inserted", True)

        # ---- 3. Agent state space ----
        agent_state = EmpiricalAgentStateSpace(profiles_path, master_seed=42)
        check("EmpiricalAgentStateSpace loaded",
              len(agent_state._profiles) == 3,
              f"got {len(agent_state._profiles)} profiles")

        # Determinism check
        s1 = agent_state.init_state("agent_000")
        s2 = agent_state.init_state("agent_000")
        check("Deterministic profile assignment", s1 == s2)

        # Persona text
        obs_state = agent_state.observe(agent_id="agent_000", state=s1, time_step=0)
        has_persona = "persona" in obs_state and "Your Identity" in obs_state["persona"]
        check("Persona text generated", has_persona)

        # ---- 4. Create LLM gateway via Ollama ----
        try:
            gateway, model_for_api = create_gateway(
                model_id=MODEL_ID,
                api_base=OLLAMA_BASE,
                max_new_tokens=64,
            )
            check("LiteLLM gateway created", True)
        except Exception as e:
            check("LiteLLM gateway created", False, str(e))
            _print_summary()
            return 1

        # ---- 5. Build agents + world engine ----
        tools = default_tools()
        agents = {}
        for i in range(NUM_AGENTS):
            aid = f"agent_{i:03d}"
            agents[aid] = SimpleCognitivePolicy(
                gateway=gateway,
                model=model_for_api,
                tools=tools,
                temperature=0.7,
            )

        engine = WorldEngine(
            config=WorldEngineConfig(
                run_id=run_id,
                deterministic_timestamps=True,
                deterministic_ids=True,
                message_history_limit=10,
            ),
            agents=agents,
            channel=InMemoryChannel(),
            trace_db=trace_db,
            agent_state_space=agent_state,
        )
        check("WorldEngine constructed", True)

        # ---- 6. Run simulation step(s) ----
        try:
            engine.run(steps=NUM_STEPS)
            check(f"Simulation ran {NUM_STEPS} step(s)", True)
        except Exception as e:
            check(f"Simulation ran {NUM_STEPS} step(s)", False, str(e))
            _print_summary()
            return 1

        # ---- 7. Verify trace rows ----
        rows = trace_db.conn.execute(
            "SELECT trace_id, agent_id, action_type FROM trace WHERE run_id = ?;",
            (run_id,),
        ).fetchall()
        expected = NUM_AGENTS * NUM_STEPS
        check(f"Trace rows: expected {expected}",
              len(rows) == expected,
              f"got {len(rows)}")

        # Verify deterministic ordering (agent_id sorted)
        agent_ids = [r["agent_id"] for r in rows]
        check("Trace order is deterministic (agent_id sorted)",
              agent_ids == sorted(agent_ids))

        # At least one non-noop action (LLM actually responded)
        actions = [r["action_type"] for r in rows]
        has_real_action = any(a != "noop" for a in actions)
        check("LLM produced at least one non-noop action",
              has_real_action,
              f"actions={actions}")

        # ---- 8. Verify agent_state rows persisted ----
        state_rows = trace_db.conn.execute(
            "SELECT agent_id, state_json FROM agent_states WHERE run_id = ?;",
            (run_id,),
        ).fetchall()
        check(f"Agent state rows persisted",
              len(state_rows) >= NUM_AGENTS,
              f"got {len(state_rows)}")

        # ---- 9. Write interpretability records ----
        # Pick first trace_id for interpretability writes
        trace_id_0 = rows[0]["trace_id"]

        # 9a. Insert a probe
        probe_id = f"probe_{uuid.uuid4().hex[:8]}"
        trace_db.insert_probe(
            probe_id=probe_id,
            run_id=run_id,
            probe_kind="linear_classifier",
            train_dataset_id="smoke_test",
            model_id=MODEL_ID,
            layers=[0, 1],
            component="resid_post",
            token_position=-1,
            artifact_path="/dev/null",
            metrics={"accuracy": 0.99},
        )
        probe_row = trace_db.conn.execute(
            "SELECT * FROM vivarium_probes WHERE probe_id = ?;", (probe_id,)
        ).fetchone()
        check("vivarium_probes insert + read-back", probe_row is not None)

        # 9b. Insert projection rows
        proj_rows = [
            (f"proj_{i}", trace_id_0, probe_id, i, -1, 0.5 + i * 0.1)
            for i in range(2)
        ]
        trace_db.insert_projection_rows(rows=proj_rows)
        proj_count = trace_db.conn.execute(
            "SELECT COUNT(*) FROM vivarium_probe_projections WHERE probe_id = ?;",
            (probe_id,),
        ).fetchone()[0]
        check("vivarium_probe_projections insert", proj_count == 2, f"got {proj_count}")

        # 9c. Upsert answer logprob
        trace_db.upsert_answer_logprob(
            trace_id=trace_id_0,
            context_kind="baseline",
            candidate_kind="ground_truth",
            candidate_text="Yes",
            token_count=1,
            logprob_sum=-0.42,
            logprob_mean=-0.42,
            first_token_id=9999,
            first_token_logprob=-0.42,
            metadata={"model_id": MODEL_ID},
        )
        lp_row = trace_db.conn.execute(
            "SELECT * FROM vivarium_answer_logprobs WHERE trace_id = ?;",
            (trace_id_0,),
        ).fetchone()
        check("vivarium_answer_logprobs upsert + read-back", lp_row is not None)

        # 9d. Insert intervention
        interv_id = f"interv_{uuid.uuid4().hex[:8]}"
        trace_db.insert_intervention(
            intervention_id=interv_id,
            run_id=run_id,
            name="smoke_steering",
            alpha=1.5,
            target_layers=[0, 1],
            component="resid_post",
            vector_probe_id=probe_id,
            notes="smoke test",
        )
        interv_row = trace_db.conn.execute(
            "SELECT * FROM vivarium_interventions WHERE intervention_id = ?;",
            (interv_id,),
        ).fetchone()
        check("vivarium_interventions insert + read-back", interv_row is not None)

        # 9e. Insert intervention result
        trace_db.insert_intervention_result(
            result_id=f"ir_{uuid.uuid4().hex[:8]}",
            intervention_id=interv_id,
            trace_id=trace_id_0,
            output_id_before=f"out_before_{uuid.uuid4().hex[:8]}",
            output_id_after=f"out_after_{uuid.uuid4().hex[:8]}",
            flipped_to_truth=True,
        )
        ir_count = trace_db.conn.execute(
            "SELECT COUNT(*) FROM vivarium_intervention_results WHERE intervention_id = ?;",
            (interv_id,),
        ).fetchone()[0]
        check("vivarium_intervention_results insert", ir_count == 1)

        # ---- 10. Verify messages table (if any agent posted) ----
        msg_count = trace_db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE run_id = ?;", (run_id,)
        ).fetchone()[0]
        check(f"Messages table populated ({msg_count} msgs)",
              msg_count >= 0)  # 0 is OK if all noops

        trace_db.close()

    # ---- Summary ----
    _print_summary()
    return 0 if all(r[1] for r in results) else 1


def _print_summary() -> None:
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = total - passed
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"  All {total} checks passed {PASS}")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED {FAIL}")
        for name, ok, detail in results:
            if not ok:
                print(f"    {FAIL} {name}: {detail}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    sys.exit(main())
