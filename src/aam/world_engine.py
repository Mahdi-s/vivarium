from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from aam.channel import Channel
from aam.persistence import TraceDb
from aam.policy import AgentPolicy
from aam.types import ActionRequest, ActionResult, Observation, TraceEvent
from aam.tools import default_tools

if TYPE_CHECKING:  # pragma: no cover
    from aam.interpretability import CaptureContext
    from aam.memory import MemoryManager
    from aam.domain_state import GenericDomainState


@dataclass(frozen=True)
class WorldEngineConfig:
    run_id: str
    deterministic_timestamps: bool = True
    message_history_limit: int = 20


class WorldEngine:
    def __init__(
        self,
        *,
        config: WorldEngineConfig,
        agents: Dict[str, AgentPolicy],
        channel: Channel,
        trace_db: TraceDb,
        capture_context: Optional["CaptureContext"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        domain_state: Optional["GenericDomainState"] = None,
    ):
        self._config = config
        self._agents = dict(agents)
        self._channel = channel
        self._trace_db = trace_db
        self._capture = capture_context
        self._memory = memory_manager
        self._domain_state = domain_state

    @property
    def agent_ids(self) -> List[str]:
        return sorted(self._agents.keys())

    @property
    def run_id(self) -> str:
        return self._config.run_id

    def build_observation(self, *, time_step: int, agent_id: str) -> Observation:
        # Phase 2: include shared message feed context + available tools.
        messages = self._trace_db.fetch_recent_messages(
            run_id=self._config.run_id,
            up_to_time_step=time_step,
            limit=self._config.message_history_limit,
        )
        tools = [t.name for t in default_tools()]
        obs: Observation = {"time_step": time_step, "agent_id": agent_id, "messages": messages, "tools": tools}
        
        # Enrich with memory if available (FR-05)
        if self._memory is not None:
            obs = self._memory.enrich_observation(agent_id=agent_id, time_step=time_step, observation=obs)
        
        return obs

    def _now(self, *, time_step: int, agent_index: int) -> float:
        if not self._config.deterministic_timestamps:
            return time.time()
        # Deterministic logical timestamp: stable across reruns.
        return float(time_step) + (agent_index / 1000.0)

    def _compute_state_hash(self, *, time_step: int) -> str:
        """
        Compute Merkle root / hash of environment state for integrity checking.
        
        Includes:
        - All messages up to time_step
        - All trace events up to time_step
        """
        # Collect state components
        messages = self._trace_db.fetch_recent_messages(
            run_id=self._config.run_id, up_to_time_step=time_step, limit=10000
        )
        
        trace_events = self._trace_db.fetch_trace_events(
            run_id=self._config.run_id, from_time_step=0, to_time_step=time_step
        )
        
        # Create deterministic JSON representation
        state_data = {
            "time_step": time_step,
            "messages": sorted(messages, key=lambda m: (m.get("time_step", 0), m.get("created_at", 0))),
            "trace_count": len(trace_events),
            "trace_ids": sorted([e.trace_id for e in trace_events]),
        }
        
        # Compute SHA256 hash
        state_json = json.dumps(state_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(state_json.encode("utf-8")).hexdigest()

    def execute(self, req: ActionRequest, *, timestamp: float) -> Tuple[ActionResult, TraceEvent]:
        trace_id = str(uuid.uuid4())
        if req.action_name == "noop":
            outcome = {"ok": True}
            res = ActionResult(success=True, data=outcome, error=None, trace_id=trace_id)
        elif req.action_name == "emit_event":
            outcome = {"ok": True, "echo": req.arguments}
            res = ActionResult(success=True, data=outcome, error=None, trace_id=trace_id)
        elif req.action_name == "post_message":
            content = str(req.arguments.get("content", "")).strip()
            if not content:
                res = ActionResult(
                    success=False,
                    data=None,
                    error="post_message requires non-empty 'content'",
                    trace_id=trace_id,
                )
            else:
                message_id = str(uuid.uuid4())
                self._trace_db.insert_message(
                    message_id=message_id,
                    run_id=req.run_id,
                    time_step=req.time_step,
                    author_id=req.agent_id,
                    content=content,
                    created_at=timestamp,
                )
                res = ActionResult(
                    success=True,
                    data={"ok": True, "message_id": message_id},
                    error=None,
                    trace_id=trace_id,
                )
        else:
            # Try domain state handlers if available
            if self._domain_state is not None:
                # Extract domain from action_name (e.g., "social_media:create_post")
                if ":" in req.action_name:
                    domain, action = req.action_name.split(":", 1)
                    outcome = self._domain_state.handle_action(
                        domain=domain,
                        action_name=action,
                        arguments=req.arguments,
                        run_id=req.run_id,
                        time_step=req.time_step,
                        agent_id=req.agent_id,
                    )
                    res = ActionResult(
                        success=outcome.get("success", False),
                        data=outcome.get("data"),
                        error=outcome.get("error"),
                        trace_id=trace_id,
                    )
                else:
                    res = ActionResult(
                        success=False,
                        data=None,
                        error=f"Unknown action_name: {req.action_name}",
                        trace_id=trace_id,
                    )
            else:
                res = ActionResult(
                    success=False,
                    data=None,
                    error=f"Unknown action_name: {req.action_name}",
                    trace_id=trace_id,
                )

        # State hash will be computed after all actions in the step are committed
        # For now, set to None (will be updated in commit_requests)
        event = TraceEvent(
            trace_id=trace_id,
            run_id=req.run_id,
            time_step=req.time_step,
            timestamp=timestamp,
            agent_id=req.agent_id,
            action_type=req.action_name,
            info=req.json_dict(),
            outcome=res.json_dict(),
            environment_state_hash=None,  # Will be set after step completion
        )
        return res, event

    def commit_requests(self, *, time_step: int, reqs: List[ActionRequest]) -> None:
        """
        Deterministic "Commit" phase: execute actions sequentially and append trace.

        The caller is responsible for providing deterministically ordered requests.
        """
        # Phase 3: flush activation buffers once per step (file aligned by time_step).
        if self._capture is not None:
            self._capture.flush_step(time_step=time_step)

        events = []
        for idx, req in enumerate(reqs):
            ts = self._now(time_step=time_step, agent_index=idx)
            _, event = self.execute(req, timestamp=ts)
            events.append(event)
            
            # Store action in memory if available (FR-05)
            if self._memory is not None:
                self._memory.store_action(
                    agent_id=req.agent_id,
                    time_step=time_step,
                    action_name=req.action_name,
                    arguments=req.arguments,
                )

        # Compute state hash after all actions are committed
        state_hash = self._compute_state_hash(time_step=time_step)
        
        # Update events with state hash and append to trace
        for event in events:
            # Create new event with state hash
            event_with_hash = TraceEvent(
                trace_id=event.trace_id,
                run_id=event.run_id,
                time_step=event.time_step,
                timestamp=event.timestamp,
                agent_id=event.agent_id,
                action_type=event.action_type,
                info=event.info,
                outcome=event.outcome,
                environment_state_hash=state_hash,
            )
            self._trace_db.append_trace(event_with_hash)

    def step(self, *, time_step: int) -> None:
        # Phase 1 "Think" phase is synchronous; ordering is deterministic.
        for agent_id in self.agent_ids:
            obs = self.build_observation(time_step=time_step, agent_id=agent_id)
            req = self._agents[agent_id].decide(
                run_id=self._config.run_id,
                time_step=time_step,
                agent_id=agent_id,
                observation=obs,
            )
            self._channel.submit(req)

        # Deterministic sort prior to sequential commit.
        reqs = sorted(self._channel.take_all(), key=lambda r: r.agent_id)

        self.commit_requests(time_step=time_step, reqs=reqs)

    def run(self, *, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        for t in range(steps):
            self.step(time_step=t)


