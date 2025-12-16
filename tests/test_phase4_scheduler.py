import asyncio
import os
import sys
import tempfile
import unittest


# Ensure local imports work when running `python -m unittest` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


from aam.agent_langgraph import SimpleCognitivePolicy  # noqa: E402
from aam.llm_gateway import MockLLMGateway  # noqa: E402
from aam.persistence import TraceDb, TraceDbConfig  # noqa: E402
from aam.scheduler import BarrierScheduler, BarrierSchedulerConfig  # noqa: E402
from aam.tools import default_tools  # noqa: E402
from aam.types import ActionRequest, RunMetadata  # noqa: E402
from aam.world_engine import WorldEngine, WorldEngineConfig  # noqa: E402


class _DelayedEmitEventPolicy:
    def __init__(self, *, delay_s: float):
        self._delay_s = float(delay_s)

    async def adecide(self, *, run_id: str, time_step: int, agent_id: str, observation):  # type: ignore[no-untyped-def]
        _ = observation
        await asyncio.sleep(self._delay_s)
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name="emit_event",
            arguments={"agent_id": agent_id, "time_step": time_step},
            reasoning=None,
            metadata={"policy": "DelayedEmitEvent"},
        )


class _SleepForeverPolicy:
    async def adecide(self, *, run_id: str, time_step: int, agent_id: str, observation):  # type: ignore[no-untyped-def]
        _ = (run_id, time_step, agent_id, observation)
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


class Phase4BarrierSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_commit_order_is_agent_id_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "simulation.db")
            run_id = "test_run"
            trace_db = TraceDb(TraceDbConfig(db_path=db_path))
            trace_db.connect()
            trace_db.init_schema()
            trace_db.insert_run(RunMetadata(run_id=run_id, seed=123, created_at=0.0, config={"test": True}))

            engine = WorldEngine(
                config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True, message_history_limit=0),
                agents={},
                channel=None,  # type: ignore[arg-type]
                trace_db=trace_db,
                capture_context=None,
            )

            agents = {
                "agent_000": _DelayedEmitEventPolicy(delay_s=0.05),
                "agent_001": _DelayedEmitEventPolicy(delay_s=0.00),
                "agent_002": _DelayedEmitEventPolicy(delay_s=0.02),
            }
            scheduler = BarrierScheduler(
                config=BarrierSchedulerConfig(per_agent_timeout_s=5.0, max_concurrency=3, sort_mode="agent_id", seed=0),
                engine=engine,
                agents=agents,
            )

            await scheduler.run(steps=1)

            rows = trace_db.conn.execute(
                "SELECT agent_id FROM trace WHERE run_id = ? AND time_step = 0 ORDER BY created_at ASC;",
                (run_id,),
            ).fetchall()
            got = [r["agent_id"] for r in rows]
            self.assertEqual(got, ["agent_000", "agent_001", "agent_002"])

            trace_db.close()

    async def test_timeout_produces_noop_not_missing_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "simulation.db")
            run_id = "test_timeout"
            trace_db = TraceDb(TraceDbConfig(db_path=db_path))
            trace_db.connect()
            trace_db.init_schema()
            trace_db.insert_run(RunMetadata(run_id=run_id, seed=123, created_at=0.0, config={"test": True}))

            engine = WorldEngine(
                config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True, message_history_limit=0),
                agents={},
                channel=None,  # type: ignore[arg-type]
                trace_db=trace_db,
                capture_context=None,
            )

            agents = {
                "agent_000": _SleepForeverPolicy(),
                "agent_001": _DelayedEmitEventPolicy(delay_s=0.0),
            }
            scheduler = BarrierScheduler(
                config=BarrierSchedulerConfig(per_agent_timeout_s=0.01, max_concurrency=2, sort_mode="agent_id", seed=0),
                engine=engine,
                agents=agents,
            )

            await scheduler.run(steps=1)

            rows = trace_db.conn.execute(
                "SELECT agent_id, action_type FROM trace WHERE run_id = ? AND time_step = 0 ORDER BY created_at ASC;",
                (run_id,),
            ).fetchall()
            got = {r["agent_id"]: r["action_type"] for r in rows}
            self.assertEqual(len(got), 2)
            self.assertEqual(got["agent_000"], "noop")

            trace_db.close()

    async def test_50_agent_offline_smoke_mock_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "simulation.db")
            run_id = "test_50"
            trace_db = TraceDb(TraceDbConfig(db_path=db_path))
            trace_db.connect()
            trace_db.init_schema()
            trace_db.insert_run(RunMetadata(run_id=run_id, seed=123, created_at=0.0, config={"test": True}))

            engine = WorldEngine(
                config=WorldEngineConfig(run_id=run_id, deterministic_timestamps=True, message_history_limit=5),
                agents={},
                channel=None,  # type: ignore[arg-type]
                trace_db=trace_db,
                capture_context=None,
            )

            tools = default_tools()
            agents = {}
            for i in range(50):
                agent_id = f"agent_{i:03d}"
                gateway = MockLLMGateway(seed=i)
                agents[agent_id] = SimpleCognitivePolicy(gateway=gateway, model="mock", tools=tools)

            scheduler = BarrierScheduler(
                config=BarrierSchedulerConfig(per_agent_timeout_s=5.0, max_concurrency=50, sort_mode="agent_id", seed=0),
                engine=engine,
                agents=agents,
            )

            await scheduler.run(steps=2)
            (count,) = trace_db.conn.execute("SELECT COUNT(*) FROM trace WHERE run_id = ?;", (run_id,)).fetchone()
            self.assertEqual(count, 2 * 50)

            trace_db.close()


if __name__ == "__main__":
    unittest.main()


