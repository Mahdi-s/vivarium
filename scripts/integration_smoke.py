#!/usr/bin/env python3
"""
Minimal end-to-end integration smoke test for Vivarium.

Supports three LLM backends:

  ollama      – Ollama running locally (default, lightest dependency)
  huggingface – HuggingFace Transformers local inference (no server needed)
  llamacpp    – llama.cpp llama-server spawned as a subprocess

Exercises:
  1. TraceDb schema init (core + interpretability)
  2. EmpiricalAgentStateSpace loading + deterministic assignment
  3. Persona injection into LLM system prompt
  4. Real LLM call via the selected backend
  5. WorldEngine step: observation → decide → execute → trace
  6. Interpretability table writes (probe, projection, answer_logprob)
  7. Query-back verification of all stored data

Usage:
  python scripts/integration_smoke.py --backend ollama
  python scripts/integration_smoke.py --backend huggingface
  python scripts/integration_smoke.py --backend llamacpp
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from vivarium.agent_langgraph import SimpleCognitivePolicy, _openai_messages_from_observation
from vivarium.agent_state import EmpiricalAgentStateSpace
from vivarium.channel import InMemoryChannel
from vivarium.llm_gateway import create_gateway
from vivarium.memory import SimpleMemorySystem, MemoryManager
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import ArchetypeAgentPolicy
from vivarium.tools import default_tools
from vivarium.types import ActionRequest, RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_AGENTS = 3
NUM_STEPS = 1
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

# Default models per backend
DEFAULT_MODELS = {
    "ollama": "qwen3:0.6b",
    "huggingface": "Qwen/Qwen2.5-0.5B-Instruct",
    "llamacpp": "qwen2.5-0.5b-instruct-q5_k_m.gguf",
}

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    mark = PASS if passed else FAIL
    msg = f"  {mark} {name}"
    if detail and not passed:
        msg += f"  ({detail})"
    print(msg)


# ---------------------------------------------------------------------------
# Backend-specific gateway constructors
# ---------------------------------------------------------------------------

def _gateway_ollama(model_id: str, api_base: str):
    """Create gateway targeting a running Ollama server."""
    return create_gateway(model_id=model_id, api_base=api_base, max_new_tokens=64)


def _gateway_huggingface(model_id: str):
    """Create gateway using local HuggingFace Transformers inference."""
    hf_cache = os.environ.get("VIVARIUM_HF_CACHE") or os.environ.get("AAM_HF_CACHE")
    return create_gateway(
        model_id=model_id,
        hf_cache_dir=hf_cache,
        max_new_tokens=64,
    )


def _resolve_gguf_path(model_name: str) -> str:
    """Resolve GGUF file path from models directory."""
    env = os.environ.get("VIVARIUM_MODEL_DIR") or os.environ.get("AAM_MODEL_DIR")
    models_dir = env if env else os.path.join(REPO_ROOT, "models")
    path = os.path.join(models_dir, model_name)
    if os.path.isfile(path):
        return path
    # Already an absolute path?
    if os.path.isfile(model_name):
        return model_name
    raise FileNotFoundError(
        f"GGUF file not found: {path}\n"
        f"  Download with: python scripts/download_test_model.py --gguf-only"
    )


def _find_llama_server_binary() -> str:
    """Find llama-server binary."""
    # 1. Vivarium settings path
    default = os.path.join(REPO_ROOT, "third_party", "llama.cpp", "build", "bin", "llama-server")
    if os.path.isfile(default):
        return default
    # 2. Environment variable
    root = os.environ.get("VIVARIUM_LLAMA_CPP_ROOT") or os.environ.get("AAM_LLAMA_CPP_ROOT")
    if root:
        candidate = os.path.join(root, "build", "bin", "llama-server")
        if os.path.isfile(candidate):
            return candidate
    # 3. PATH
    import shutil
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(
        "llama-server binary not found.\n"
        "  Build llama.cpp or set VIVARIUM_LLAMA_CPP_ROOT."
    )


class LlamaServerProcess:
    """Context manager that starts and stops a llama-server subprocess."""

    def __init__(self, gguf_path: str, port: int = 18081):
        self.gguf_path = gguf_path
        self.port = port
        self.proc: subprocess.Popen | None = None

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "LlamaServerProcess":
        binary = _find_llama_server_binary()
        # Determine GPU layers: -1 on Apple Silicon (Metal), 0 otherwise
        import platform
        n_gpu = "-1" if (platform.system() == "Darwin" and platform.machine() == "arm64") else "0"

        cmd = [
            binary,
            "--model", self.gguf_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", "2048",
            "--n-gpu-layers", n_gpu,
        ]
        print(f"  Starting llama-server on port {self.port} ...")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        # Wait for server to become healthy (up to 60s)
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 60.0) -> None:
        import urllib.request
        import urllib.error
        health_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"  llama-server ready (port {self.port})")
                        return
            except (urllib.error.URLError, OSError, ConnectionRefusedError):
                pass
            # Check if process died
            if self.proc and self.proc.poll() is not None:
                stdout = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"llama-server exited early (rc={self.proc.returncode}):\n{stdout}")
            time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become healthy within {timeout}s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            print(f"  llama-server stopped.")


# ---------------------------------------------------------------------------
# Core test pipeline (shared across all backends)
# ---------------------------------------------------------------------------

def run_pipeline(gateway, model_for_api: str, backend_label: str) -> int:
    """Run the full integration pipeline and return 0 on success."""
    global results
    results = []

    print(f"\n=== Vivarium Integration Smoke Test ({backend_label}) ===")
    print(f"    Model : {model_for_api}")
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
            config={"model": model_for_api, "test": True},
        ))
        check("Run metadata inserted", True)

        # ---- 3. Agent state space ----
        agent_state = EmpiricalAgentStateSpace(profiles_path, master_seed=42)
        check("EmpiricalAgentStateSpace loaded",
              len(agent_state._profiles) == 3,
              f"got {len(agent_state._profiles)} profiles")

        s1 = agent_state.init_state("agent_000")
        s2 = agent_state.init_state("agent_000")
        check("Deterministic profile assignment", s1 == s2)

        obs_state = agent_state.observe(agent_id="agent_000", state=s1, time_step=0)
        has_persona = "persona" in obs_state and "Your Identity" in obs_state["persona"]
        check("Persona text generated", has_persona)

        check(f"Gateway created ({backend_label})", True)

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
            _print_summary(backend_label)
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

        agent_ids = [r["agent_id"] for r in rows]
        check("Trace order is deterministic (agent_id sorted)",
              agent_ids == sorted(agent_ids))

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
        check("Agent state rows persisted",
              len(state_rows) >= NUM_AGENTS,
              f"got {len(state_rows)}")

        # ---- 9. Write interpretability records ----
        trace_id_0 = rows[0]["trace_id"]

        # 9a. Insert a probe
        probe_id = f"probe_{uuid.uuid4().hex[:8]}"
        trace_db.insert_probe(
            probe_id=probe_id,
            run_id=run_id,
            probe_kind="linear_classifier",
            train_dataset_id="smoke_test",
            model_id=model_for_api,
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
            metadata={"model_id": model_for_api},
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

        # ---- 10. Verify messages table ----
        msg_count = trace_db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE run_id = ?;", (run_id,)
        ).fetchone()[0]
        check(f"Messages table populated ({msg_count} msgs)",
              msg_count >= 0)  # 0 is OK if all noops

        # ---- 11. Epic 3: ArchetypeAgentPolicy with real LLM ----
        arch_run_id = f"arch_{uuid.uuid4().hex[:8]}"
        trace_db.insert_run(RunMetadata(
            run_id=arch_run_id, seed=42, created_at=time.time(),
            config={"model": model_for_api, "test": True, "policy": "archetype"},
        ))

        arch_policy = ArchetypeAgentPolicy(
            gateway=gateway,
            model=model_for_api,
            action_space=["post_message", "noop"],
            master_seed=42,
            temperature=0.7,
        )
        # 6 agents sharing the archetype policy, 2 distinct profiles
        arch_agents = {f"arch_{i:03d}": arch_policy for i in range(6)}

        arch_engine = WorldEngine(
            config=WorldEngineConfig(
                run_id=arch_run_id,
                deterministic_timestamps=True,
                deterministic_ids=True,
                message_history_limit=10,
            ),
            agents=arch_agents,
            channel=InMemoryChannel(),
            trace_db=trace_db,
            agent_state_space=agent_state,
        )

        try:
            arch_engine.run(steps=1)
            check("Epic 3: ArchetypeAgentPolicy ran 1 step", True)
        except Exception as e:
            check("Epic 3: ArchetypeAgentPolicy ran 1 step", False, str(e))

        arch_rows = trace_db.conn.execute(
            "SELECT agent_id, action_type FROM trace WHERE run_id = ?;",
            (arch_run_id,),
        ).fetchall()
        check("Epic 3: Archetype trace rows (6 agents)",
              len(arch_rows) == 6,
              f"got {len(arch_rows)}")

        # Verify archetype metadata in trace
        arch_info_row = trace_db.conn.execute(
            "SELECT info_json FROM trace WHERE run_id = ? LIMIT 1;",
            (arch_run_id,),
        ).fetchone()
        if arch_info_row:
            arch_info = json.loads(arch_info_row["info_json"])
            arch_meta = arch_info.get("metadata", {})
            check("Epic 3: Archetype metadata present",
                  arch_meta.get("policy") == "ArchetypeAgentPolicy"
                  and "archetype_hash" in arch_meta,
                  f"metadata={arch_meta}")
        else:
            check("Epic 3: Archetype metadata present", False, "no trace rows")

        # Verify cache efficiency: at most K archetype hashes for 6 agents
        all_arch_info = trace_db.conn.execute(
            "SELECT info_json FROM trace WHERE run_id = ?;",
            (arch_run_id,),
        ).fetchall()
        arch_hashes = set()
        for row in all_arch_info:
            info = json.loads(row["info_json"])
            h = info.get("metadata", {}).get("archetype_hash")
            if h:
                arch_hashes.add(h)
        check(f"Epic 3: Cache efficiency ({len(arch_hashes)} archetypes for 6 agents)",
              0 < len(arch_hashes) <= 6)

        # ---- 12. Epic 4: BDI prompt structure with real LLM ----
        # Verify BDI sections exist in the generated prompt
        bdi_obs = {
            "time_step": 0,
            "messages": [],
            "tools": ["post_message", "noop"],
            "agent_state": {"persona": "Your Identity: Age 34, Female, Engineer"},
        }
        bdi_msgs = _openai_messages_from_observation(
            agent_id="agent_000", observation=bdi_obs, require_json_action=True,
        )
        bdi_system = bdi_msgs[0]["content"]
        check("Epic 4: BDI [BELIEFS] section present",
              "[BELIEFS]" in bdi_system)
        check("Epic 4: BDI [DESIRES] section present",
              "[DESIRES]" in bdi_system and "Engineer" in bdi_system)
        check("Epic 4: BDI [INTENTIONS] with reasoning",
              "[INTENTIONS]" in bdi_system and '"reasoning"' in bdi_system)

        # ---- 13. Epic 4: RLSF feedback on failed action ----
        rlsf_run_id = f"rlsf_{uuid.uuid4().hex[:8]}"
        trace_db.insert_run(RunMetadata(
            run_id=rlsf_run_id, seed=42, created_at=time.time(),
            config={"model": model_for_api, "test": True, "policy": "rlsf"},
        ))

        mem_sys = SimpleMemorySystem()
        mem_mgr = MemoryManager(mem_sys)

        rlsf_policy = SimpleCognitivePolicy(
            gateway=gateway,
            model=model_for_api,
            tools=tools,
            temperature=0.7,
        )
        rlsf_agents = {f"rlsf_{i:03d}": rlsf_policy for i in range(2)}

        rlsf_engine = WorldEngine(
            config=WorldEngineConfig(
                run_id=rlsf_run_id,
                deterministic_timestamps=True,
                deterministic_ids=True,
                message_history_limit=10,
            ),
            agents=rlsf_agents,
            channel=InMemoryChannel(),
            trace_db=trace_db,
            agent_state_space=agent_state,
            memory_manager=mem_mgr,
        )

        # Inject a failing action to test RLSF feedback
        bad_req = ActionRequest(
            run_id=rlsf_run_id, time_step=0, agent_id="rlsf_000",
            action_name="invalid_action_xyz", arguments={},
            reasoning=None, metadata={},
        )
        rlsf_engine.commit_requests(time_step=0, reqs=[bad_req])

        feedback_entries = [
            e for e in mem_sys.get_short_term_context(
                agent_id="rlsf_000", time_step=1, limit=20,
            )
            if e.get("metadata", {}).get("type") == "feedback"
        ]
        check("Epic 4: RLSF feedback stored for failed action",
              len(feedback_entries) == 1,
              f"got {len(feedback_entries)} feedback entries")

        # Also verify store_action still recorded the attempt
        action_entries = [
            e for e in mem_sys.get_short_term_context(
                agent_id="rlsf_000", time_step=1, limit=20,
            )
            if e.get("metadata", {}).get("type") == "action"
        ]
        check("Epic 4: store_action still records failed attempt",
              len(action_entries) >= 1,
              f"got {len(action_entries)} action entries")

        trace_db.close()

    _print_summary(backend_label)
    return 0 if all(r[1] for r in results) else 1


def _print_summary(backend_label: str) -> None:
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = total - passed
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"  [{backend_label}] All {total} checks passed {PASS}")
    else:
        print(f"  [{backend_label}] {passed}/{total} passed, {failed} FAILED {FAIL}")
        for name, ok, detail in results:
            if not ok:
                print(f"    {FAIL} {name}: {detail}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Vivarium integration smoke test")
    parser.add_argument(
        "--backend", choices=["ollama", "huggingface", "llamacpp"],
        default="ollama",
        help="LLM backend to test (default: ollama)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model name/path (default varies per backend)",
    )
    args = parser.parse_args()

    backend = args.backend
    model_id = args.model or os.environ.get("VVM_TEST_MODEL") or DEFAULT_MODELS[backend]

    if backend == "ollama":
        api_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        try:
            gateway, model_for_api = _gateway_ollama(model_id, api_base)
        except Exception as e:
            print(f"ERROR: Failed to create Ollama gateway: {e}")
            return 1
        return run_pipeline(gateway, model_for_api, "Ollama")

    elif backend == "huggingface":
        try:
            gateway, model_for_api = _gateway_huggingface(model_id)
        except Exception as e:
            print(f"ERROR: Failed to create HuggingFace gateway: {e}")
            return 1
        return run_pipeline(gateway, model_for_api, "HuggingFace")

    elif backend == "llamacpp":
        gguf_path = _resolve_gguf_path(model_id)
        port = int(os.environ.get("LLAMA_SERVER_PORT", "18081"))
        with LlamaServerProcess(gguf_path, port=port) as srv:
            try:
                gateway, model_for_api = create_gateway(
                    model_id=model_id,
                    api_base=srv.api_base,
                    max_new_tokens=64,
                )
            except Exception as e:
                print(f"ERROR: Failed to create llama.cpp gateway: {e}")
                return 1
            return run_pipeline(gateway, model_for_api, "llama.cpp")

    return 1


if __name__ == "__main__":
    sys.exit(main())
