import json


def test_merkle_logger_deterministic():
    from aam.provenance import MerkleLogger

    ml = MerkleLogger()
    leaf1, root1 = ml.add_step(step_id="0", agent_id="a", prompt_hash="p", activation_hash="x")
    leaf2, root2 = ml.add_step(step_id="1", agent_id="a", prompt_hash="p2", activation_hash="y")

    # Stable shapes / non-empty hashes
    assert isinstance(leaf1, str) and len(leaf1) == 64
    assert isinstance(root1, str) and len(root1) == 64
    assert isinstance(root2, str) and len(root2) == 64
    assert root2 != root1

    # Deterministic: same sequence yields same root
    ml2 = MerkleLogger()
    _leaf1b, _root1b = ml2.add_step(step_id="0", agent_id="a", prompt_hash="p", activation_hash="x")
    _leaf2b, root2b = ml2.add_step(step_id="1", agent_id="a", prompt_hash="p2", activation_hash="y")
    assert root2b == root2


def test_capture_context_flush_writes_metadata_and_merkle(tmp_path):
    import time

    import torch
    from safetensors import safe_open

    from aam.interpretability import CaptureConfig, CaptureContext
    from aam.persistence import TraceDb, TraceDbConfig
    from aam.types import RunMetadata

    db_path = str(tmp_path / "test.db")
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    run_id = "run_test_1"
    trace_db.insert_run(RunMetadata(run_id=run_id, seed=0, created_at=time.time(), config={"k": "v"}))

    out_dir = str(tmp_path / "activations")
    cap = CaptureContext(
        output_dir=out_dir,
        config=CaptureConfig(layers=[0], components=["resid_post"], trigger_actions=["probe_capture"]),
        dtype="float16",
        trace_db=trace_db,
    )

    # Simulate a committed activation tensor for a single agent.
    cap.on_action_decided(run_id=run_id, time_step=0, agent_id="agentA", model_id="modelX", action_name="probe_capture")
    cap._committed_by_step[0] = {  # type: ignore[attr-defined]
        "agentA.blocks.0.hook_resid_post": torch.zeros((4096,), dtype=torch.float16)
    }

    shard_path = cap.flush_step(time_step=0)
    assert shard_path is not None

    with safe_open(shard_path, framework="pt") as f:
        md = f.metadata()
        assert md is not None
        # Required fields
        assert md.get("run_id") == run_id
        assert md.get("step_id") == "0"
        assert md.get("model_id") == "modelX"
        assert "provenance_hash" in md
        assert "merkle_root_at_step" in md

    row = trace_db.conn.execute("SELECT COUNT(*) AS n FROM merkle_log WHERE run_id = ?", (run_id,)).fetchone()
    assert int(row["n"]) >= 1


def test_select_local_gateway_routes_olmo(monkeypatch):
    # Avoid actually loading a HF model by monkeypatching the gateway classes.
    import aam.llm_gateway as g

    calls = {"hf": 0, "tl": 0}

    class DummyHF:
        def __init__(self, **kwargs):
            calls["hf"] += 1
            self.kwargs = kwargs

    class DummyTL:
        def __init__(self, **kwargs):
            calls["tl"] += 1
            self.kwargs = kwargs

    monkeypatch.setattr(g, "HuggingFaceHookedGateway", DummyHF)
    monkeypatch.setattr(g, "TransformerLensGateway", DummyTL)

    _gw = g.select_local_gateway(model_id_or_path="allenai/Olmo-3-7B-Instruct")
    assert calls["hf"] == 1
    assert calls["tl"] == 0

