from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Protocol

from aam.types import ActionRequest, Observation


class AgentPolicy(Protocol):
    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest: ...


class AsyncAgentPolicy(Protocol):
    async def adecide(
        self, *, run_id: str, time_step: int, agent_id: str, observation: Observation
    ) -> ActionRequest: ...


def stable_agent_seed(master_seed: int, agent_id: str) -> int:
    """
    Derive a stable per-agent seed (not affected by Python's hash randomization).
    """
    h = hashlib.sha256(f"{master_seed}:{agent_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


@dataclass(frozen=True)
class RandomAgentPolicy:
    rng: random.Random
    action_space: List[str]

    def __init__(self, rng: random.Random, action_space: List[str] | None = None):
        object.__setattr__(self, "rng", rng)
        object.__setattr__(self, "action_space", action_space or ["noop", "emit_event"])

    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest:
        action = self.rng.choice(self.action_space)
        if action == "emit_event":
            args = {"value": self.rng.randint(0, 1_000_000), "seen_time_step": observation.get("time_step")}
        else:
            args = {}

        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=action,
            arguments=args,
            reasoning=None,
            metadata={"policy": "RandomAgentPolicy"},
        )


