"""
Agent state space extension for Vivarium.

Allows experiments to define mutable per-agent state that evolves across time steps.
Complements DomainStateHandler (world/environment state) with agent-internal state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from vivarium.types import ActionRequest, ActionResult, JsonDict


class AgentStateSpace(Protocol):
    """
    Protocol for defining the shape and evolution of an agent's internal state.

    Experiments can implement this to track agent-specific state (e.g., conviction,
    social pressure accumulation) that evolves over time and influences observations.
    """

    def init_state(self, agent_id: str) -> JsonDict:
        """Return initial state for a new agent."""
        ...

    def transition(
        self,
        *,
        agent_id: str,
        current_state: JsonDict,
        action: ActionRequest,
        result: ActionResult,
        time_step: int,
    ) -> JsonDict:
        """Compute next state after an action is committed."""
        ...

    def observe(
        self,
        *,
        agent_id: str,
        state: JsonDict,
        time_step: int,
    ) -> JsonDict:
        """Extract the observable portion of state to inject into the agent's observation."""
        ...


class NoOpAgentStateSpace:
    """
    Default no-op implementation: agents have no internal state.
    """

    def init_state(self, agent_id: str) -> JsonDict:
        return {}

    def transition(
        self,
        *,
        agent_id: str,
        current_state: JsonDict,
        action: ActionRequest,
        result: ActionResult,
        time_step: int,
    ) -> JsonDict:
        return current_state

    def observe(
        self,
        *,
        agent_id: str,
        state: JsonDict,
        time_step: int,
    ) -> JsonDict:
        return {}
