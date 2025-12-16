from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from aam.policy import AsyncAgentPolicy
from aam.types import ActionRequest, Observation
from aam.world_engine import WorldEngine


SortMode = Literal["agent_id", "seeded_shuffle"]


@dataclass(frozen=True)
class BarrierSchedulerConfig:
    per_agent_timeout_s: float = 60.0
    max_concurrency: int = 50
    sort_mode: SortMode = "agent_id"
    seed: int = 0


class BarrierScheduler:
    """
    PRD Phase 4 scheduler:
    Think concurrently (parallel), then commit sequentially (deterministic).
    """

    def __init__(
        self,
        *,
        config: BarrierSchedulerConfig,
        engine: WorldEngine,
        agents: Dict[str, AsyncAgentPolicy],
    ) -> None:
        self._config = config
        self._engine = engine
        self._agents = dict(agents)
        if config.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        if config.per_agent_timeout_s <= 0:
            raise ValueError("per_agent_timeout_s must be > 0")
        self._sem = asyncio.Semaphore(int(config.max_concurrency))

    @property
    def agent_ids(self) -> List[str]:
        return sorted(self._agents.keys())

    def _fallback_request(
        self, *, time_step: int, agent_id: str, reason: str, detail: Optional[str] = None
    ) -> ActionRequest:
        meta = {"policy": "BarrierSchedulerFallback", "scheduler_reason": reason}
        if detail:
            meta["scheduler_detail"] = detail
        return ActionRequest(
            run_id=self._engine.run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name="noop",
            arguments={},
            reasoning=None,
            metadata=meta,
        )

    async def _think_one(
        self, *, time_step: int, agent_id: str, observation: Observation, max_retries: int = 2
    ) -> ActionRequest:
        """
        Execute agent decision with retry logic (NFR-03).
        
        Retries up to max_retries times (3 total attempts) before falling back to noop.
        """
        policy = self._agents[agent_id]
        async with self._sem:
            last_error = None
            for attempt in range(max_retries + 1):  # 0, 1, 2 = 3 attempts total
                try:
                    return await asyncio.wait_for(
                        policy.adecide(
                            run_id=self._engine.run_id,
                            time_step=time_step,
                            agent_id=agent_id,
                            observation=observation,
                        ),
                        timeout=float(self._config.per_agent_timeout_s),
                    )
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        # Retry on timeout
                        continue
                    return self._fallback_request(
                        time_step=time_step,
                        agent_id=agent_id,
                        reason="timeout",
                        detail=f"per_agent_timeout_s={self._config.per_agent_timeout_s}, attempts={attempt+1}",
                    )
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        # Retry on exception
                        continue
                    return self._fallback_request(
                        time_step=time_step,
                        agent_id=agent_id,
                        reason="exception",
                        detail=f"{repr(e)}, attempts={attempt+1}",
                    )
            
            # Should not reach here, but handle edge case
            return self._fallback_request(
                time_step=time_step,
                agent_id=agent_id,
                reason="exhausted_retries",
                detail=f"max_retries={max_retries}, last_error={repr(last_error)}",
            )

    def _sort_requests(self, *, time_step: int, reqs: List[ActionRequest]) -> List[ActionRequest]:
        if self._config.sort_mode == "agent_id":
            return sorted(reqs, key=lambda r: r.agent_id)

        # Deterministic shuffle seeded by (seed, time_step), then stable-tie by agent_id.
        rng = random.Random(int(self._config.seed) ^ (int(time_step) * 1_000_003))
        keys: Dict[str, float] = {aid: rng.random() for aid in self.agent_ids}
        return sorted(reqs, key=lambda r: (keys.get(r.agent_id, 0.0), r.agent_id))

    async def step(self, *, time_step: int) -> None:
        # Broadcast: build deterministic observation snapshots.
        observations: Dict[str, Observation] = {
            agent_id: self._engine.build_observation(time_step=time_step, agent_id=agent_id) for agent_id in self.agent_ids
        }

        # Parallel think: run all agent decisions concurrently (bounded by semaphore).
        tasks = [
            asyncio.create_task(self._think_one(time_step=time_step, agent_id=aid, observation=observations[aid]))
            for aid in self.agent_ids
        ]

        # Barrier: wait for all tasks (timeouts are handled inside _think_one).
        reqs = await asyncio.gather(*tasks)

        # Deterministic sort + sequential commit.
        ordered = self._sort_requests(time_step=time_step, reqs=list(reqs))
        self._engine.commit_requests(time_step=time_step, reqs=ordered)

    async def run(self, *, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        for t in range(steps):
            await self.step(time_step=t)


