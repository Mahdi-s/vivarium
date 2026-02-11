from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


# Ensure local imports work when running `python -m unittest` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class ProvenanceTests(unittest.TestCase):
    def test_merkle_logger_deterministic(self) -> None:
        from aam.provenance import MerkleLogger

        ml = MerkleLogger()
        leaf1, root1 = ml.add_step(step_id="0", agent_id="a", prompt_hash="p", activation_hash="x")
        _leaf2, root2 = ml.add_step(step_id="1", agent_id="a", prompt_hash="p2", activation_hash="y")

        # Stable shapes / non-empty hashes
        self.assertIsInstance(leaf1, str)
        self.assertEqual(len(leaf1), 64)
        self.assertIsInstance(root1, str)
        self.assertEqual(len(root1), 64)
        self.assertIsInstance(root2, str)
        self.assertEqual(len(root2), 64)
        self.assertNotEqual(root2, root1)

        # Deterministic: same sequence yields same root
        ml2 = MerkleLogger()
        _leaf1b, _root1b = ml2.add_step(step_id="0", agent_id="a", prompt_hash="p", activation_hash="x")
        _leaf2b, root2b = ml2.add_step(step_id="1", agent_id="a", prompt_hash="p2", activation_hash="y")
        self.assertEqual(root2b, root2)

    def test_capture_context_flush_writes_metadata_and_merkle(self) -> None:
        import torch
        from safetensors import safe_open

        from aam.interpretability import CaptureConfig, CaptureContext
        from aam.persistence import TraceDb, TraceDbConfig
        from aam.types import RunMetadata

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
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
            self.assertIsNotNone(shard_path)

            with safe_open(shard_path, framework="pt") as f:
                md = f.metadata()
                self.assertIsNotNone(md)
                # Required fields
                self.assertEqual(md.get("run_id"), run_id)
                self.assertEqual(md.get("step_id"), "0")
                self.assertEqual(md.get("model_id"), "modelX")
                self.assertIn("provenance_hash", md)
                self.assertIn("merkle_root_at_step", md)

            row = trace_db.conn.execute("SELECT COUNT(*) AS n FROM merkle_log WHERE run_id = ?", (run_id,)).fetchone()
            self.assertIsNotNone(row)
            self.assertGreaterEqual(int(row["n"]), 1)

    def test_select_local_gateway_routes_olmo(self) -> None:
        # Avoid actually loading a HF model by patching the gateway classes.
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

        with patch.object(g, "HuggingFaceHookedGateway", DummyHF), patch.object(g, "TransformerLensGateway", DummyTL):
            _gw = g.select_local_gateway(model_id_or_path="allenai/Olmo-3-7B-Instruct")
            self.assertEqual(calls["hf"], 1)
            self.assertEqual(calls["tl"], 0)


if __name__ == "__main__":
    unittest.main()

