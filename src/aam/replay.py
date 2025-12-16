from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from aam.persistence import TraceDb
from aam.types import ActionRequest, TraceEvent
from aam.world_engine import WorldEngine, WorldEngineConfig


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for replaying a simulation from trace events."""

    run_id: str
    from_time_step: int = 0
    to_time_step: Optional[int] = None
    rebuild_state: bool = True  # If True, rebuild domain state (messages table) from trace


class ReplayEngine:
    """
    Replay engine for rebuilding simulation state from trace events.

    This enables counterfactual analysis by allowing researchers to:
    1. Load state up to a specific time_step
    2. Modify agent policies or seeds
    3. Continue simulation from that point
    """

    def __init__(self, *, trace_db: TraceDb, engine: WorldEngine):
        self._trace_db = trace_db
        self._engine = engine

    def replay_to_step(self, *, time_step: int, rebuild_state: bool = True) -> None:
        """
        Replay trace events up to (and including) the specified time_step.

        If rebuild_state is True, this will reconstruct the messages table
        from trace events. This is necessary for counterfactual analysis.
        """
        events = self._trace_db.fetch_trace_events(
            run_id=self._engine.run_id, from_time_step=0, to_time_step=time_step
        )

        if rebuild_state:
            # Clear existing domain state for this run (messages table)
            self._trace_db.conn.execute(
                "DELETE FROM messages WHERE run_id = ?;", (self._engine.run_id,)
            )

        # Replay events in order
        for event in events:
            self._replay_event(event, rebuild_state=rebuild_state)

    def _replay_event(self, event: TraceEvent, *, rebuild_state: bool) -> None:
        """
        Replay a single trace event, reconstructing domain state if requested.
        """
        if not rebuild_state:
            return

        # Extract action request from event info
        info = event.info
        action_type = event.action_type

        # Reconstruct domain state based on action type
        if action_type == "post_message":
            outcome = event.outcome
            if outcome.get("success") and outcome.get("data"):
                # Extract message data from outcome
                message_id = outcome["data"].get("message_id")
                if message_id:
                    # Check if message already exists (idempotent replay)
                    existing = self._trace_db.conn.execute(
                        "SELECT 1 FROM messages WHERE message_id = ?;", (message_id,)
                    ).fetchone()
                    if existing is None:
                        # Reconstruct message from trace
                        # The content is in the original action request arguments
                        content = info.get("arguments", {}).get("content", "")
                        if content:
                            self._trace_db.insert_message(
                                message_id=message_id,
                                run_id=event.run_id,
                                time_step=event.time_step,
                                author_id=event.agent_id,
                                content=content,
                                created_at=event.timestamp,
                            )

    def get_state_at_step(self, *, time_step: int) -> Dict:
        """
        Get a snapshot of simulation state at a specific time_step.

        Returns a dictionary with:
        - messages: List of messages up to time_step
        - trace_count: Number of trace events up to time_step
        """
        messages = self._trace_db.fetch_recent_messages(
            run_id=self._engine.run_id, up_to_time_step=time_step, limit=1000
        )

        trace_count = self._trace_db.conn.execute(
            "SELECT COUNT(*) FROM trace WHERE run_id = ? AND time_step <= ?;",
            (self._engine.run_id, time_step),
        ).fetchone()[0]

        return {
            "time_step": time_step,
            "messages": messages,
            "trace_count": trace_count,
        }

    def extract_action_request(self, event: TraceEvent) -> ActionRequest:
        """
        Extract an ActionRequest from a TraceEvent for counterfactual replay.

        This allows modifying the action and re-executing it.
        """
        info = event.info
        return ActionRequest(
            run_id=info.get("run_id", event.run_id),
            time_step=info.get("time_step", event.time_step),
            agent_id=info.get("agent_id", event.agent_id),
            action_name=info.get("action_name", event.action_type),
            arguments=info.get("arguments", {}),
            reasoning=info.get("reasoning"),
            metadata=info.get("metadata", {}),
        )

