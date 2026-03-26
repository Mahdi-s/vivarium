from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

from vivarium.types import ActionRequest, Observation

if TYPE_CHECKING:  # pragma: no cover
    from vivarium.llm_gateway import LLMGateway

JsonDict = Dict[str, Any]


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


# ---------------------------------------------------------------------------
# Archetype policy helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: object) -> str:
    """Deterministic JSON serialisation (mirrors WorldEngine._stable_json)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[JsonDict]:
    """Best-effort extraction of the first JSON object from *text*."""
    cleaned = _FENCE_RE.sub("", text.strip())
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ArchetypeAgentPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchetypeAgentPolicy:
    """
    Scalability via LLM Archetypes (AgentTorch §3.1).

    Agents that share the same demographic profile *and* see the same
    environmental context are grouped into an **archetype**.  The LLM is
    queried exactly once per archetype to produce a probability distribution
    over the available actions.  Each agent then *samples* from that
    distribution using its own deterministic RNG — no additional LLM call
    required.

    This reduces the number of LLM calls from N (agents) to K (distinct
    archetypes per step), where K << N.
    """

    gateway: "LLMGateway"
    model: str
    action_space: List[str]
    master_seed: int = 0
    temperature: float = 0.7

    def __init__(
        self,
        gateway: "LLMGateway",
        model: str,
        action_space: Optional[List[str]] = None,
        master_seed: int = 0,
        temperature: float = 0.7,
    ) -> None:
        object.__setattr__(self, "gateway", gateway)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "action_space", action_space or ["post_message", "noop"])
        object.__setattr__(self, "master_seed", master_seed)
        object.__setattr__(self, "temperature", temperature)
        # Mutable internal state
        object.__setattr__(self, "_cache", {})           # archetype_hash → parsed distribution
        object.__setattr__(self, "_current_time_step", -1)

    # ------------------------------------------------------------------
    # Archetype hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_archetype_hash(observation: Observation, time_step: int) -> str:
        """
        Derive a cache key from the agent's demographic archetype and the
        current environmental context.

        The ``agent_id`` is deliberately excluded so that agents sharing the
        same profile AND seeing the same message feed map to a single hash.
        """
        agent_state = observation.get("agent_state", {})
        messages = observation.get("messages", [])
        material = (
            _canonical_json(agent_state)
            + "|"
            + _canonical_json(messages)
            + "|"
            + str(time_step)
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # LLM prompt construction
    # ------------------------------------------------------------------

    def _build_distribution_prompt(self, observation: Observation) -> List[JsonDict]:
        """Build OpenAI-format messages that ask for an action distribution."""
        action_list = ", ".join(f'"{a}"' for a in self.action_space)

        system = (
            "You are a behavioral modeling engine for a multi-agent simulation.\n"
            "Given a demographic archetype and conversational context, you must "
            "output a JSON probability distribution over the available actions.\n\n"
            "Output ONLY a JSON object with this exact schema (no markdown, no extra text):\n"
            "{\n"
            '  "distribution": {"action_name": probability, ...},\n'
            '  "post_message_contents": ["example message 1", "example message 2"],\n'
            '  "reasoning": "short explanation"\n'
            "}\n\n"
            "Rules:\n"
            f"- Available actions: [{action_list}]\n"
            "- Probabilities must sum to 1.0\n"
            "- Only include actions from the available list\n"
            '- If "post_message" has nonzero probability, provide 2-4 representative '
            "message strings in post_message_contents\n"
            '- If "post_message" has zero probability, set post_message_contents to []\n'
        )

        # Persona injection
        agent_state = observation.get("agent_state")
        persona_block = ""
        if agent_state and isinstance(agent_state, dict):
            persona = agent_state.get("persona", "")
            if persona:
                persona_block = f"\n{persona}\n"

        # Message feed
        msgs = observation.get("messages", []) or []
        history_lines = []
        for m in msgs:
            history_lines.append(
                f"[t={m.get('time_step')}] {m.get('author_id')}: {m.get('content')}"
            )
        history = "\n".join(history_lines) if history_lines else "(no messages yet)"

        user = (
            f"Demographic Archetype:{persona_block}\n"
            f"time_step={observation.get('time_step', 0)}\n\n"
            f"Shared message feed:\n{history}\n\n"
            f"Available actions: [{action_list}]\n\n"
            "Return the probability distribution as JSON."
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_distribution(self, resp: JsonDict) -> JsonDict:
        """
        Parse the LLM response into a validated distribution dict.

        Returns ``{"distribution": {...}, "post_message_contents": [...]}``
        or the noop fallback on any failure.
        """
        fallback: JsonDict = {"distribution": {"noop": 1.0}, "post_message_contents": []}

        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return fallback

        parsed = _extract_json_object(text)
        if parsed is None:
            return fallback

        dist = parsed.get("distribution")
        if not isinstance(dist, dict) or not dist:
            return fallback

        # Filter to known actions and cast values to float
        filtered: Dict[str, float] = {}
        for action, prob in dist.items():
            if action in self.action_space:
                try:
                    filtered[action] = float(prob)
                except (ValueError, TypeError):
                    pass

        if not filtered:
            return fallback

        # Normalize so probabilities sum to 1.0
        total = sum(filtered.values())
        if total <= 0:
            return fallback
        normalized = {a: p / total for a, p in filtered.items()}

        # Extract post_message_contents
        contents = parsed.get("post_message_contents", [])
        if not isinstance(contents, list):
            contents = []
        contents = [str(c) for c in contents if c]

        return {"distribution": normalized, "post_message_contents": contents}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_action(self, dist_resp: JsonDict, agent_id: str) -> Tuple[str, JsonDict]:
        """
        Deterministically sample an action for *agent_id* from the
        archetype's cached distribution.
        """
        dist: Dict[str, float] = dist_resp.get("distribution", {"noop": 1.0})
        contents: List[str] = dist_resp.get("post_message_contents", [])

        actions = list(dist.keys())
        weights = [dist[a] for a in actions]

        rng = random.Random(stable_agent_seed(self.master_seed, agent_id))
        chosen = rng.choices(actions, weights=weights, k=1)[0]

        if chosen == "post_message":
            if contents:
                content = rng.choice(contents)
                return chosen, {"content": content}
            # No content templates — fall back to noop
            return "noop", {}

        return chosen, {}

    # ------------------------------------------------------------------
    # AgentPolicy protocol
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        run_id: str,
        time_step: int,
        agent_id: str,
        observation: Observation,
    ) -> ActionRequest:
        # Cache invalidation on new time step
        if time_step != self._current_time_step:
            self._cache.clear()
            object.__setattr__(self, "_current_time_step", time_step)

        arch_hash = self._compute_archetype_hash(observation, time_step)
        cache_hit = arch_hash in self._cache

        if not cache_hit:
            messages = self._build_distribution_prompt(observation)
            resp = self.gateway.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            self._cache[arch_hash] = self._parse_distribution(resp)

        action_name, arguments = self._sample_action(self._cache[arch_hash], agent_id)

        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=action_name,
            arguments=arguments,
            reasoning=None,
            metadata={
                "policy": "ArchetypeAgentPolicy",
                "archetype_hash": arch_hash,
                "cache_hit": cache_hit,
            },
        )


