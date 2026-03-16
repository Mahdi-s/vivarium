"""
Agent state space extension for Vivarium.

Allows experiments to define mutable per-agent state that evolves across time steps.
Complements DomainStateHandler (world/environment state) with agent-internal state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Protocol

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


class EmpiricalAgentStateSpace:
    """
    Empirical Digital Twin initialization.

    Loads real demographic profiles from a CSV or JSONL file and
    deterministically assigns one profile per agent using a stable hash.
    Demographics are immutable across time steps; ``observe()`` formats
    the assigned profile into a *Persona* text block suitable for injection
    into an LLM system prompt.
    """

    def __init__(self, dataset_path: str, *, master_seed: int = 0) -> None:
        self._master_seed = master_seed
        self._profiles: List[JsonDict] = self._load_profiles(dataset_path)
        if not self._profiles:
            raise ValueError(f"No profiles loaded from {dataset_path}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profiles(path: str) -> List[JsonDict]:
        """Load demographic profiles from CSV or JSONL."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return EmpiricalAgentStateSpace._load_csv(path)
        elif ext in (".jsonl", ".ndjson"):
            return EmpiricalAgentStateSpace._load_jsonl(path)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            raise ValueError(f"JSON file must contain a list of objects: {path}")
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for dataset_path. "
                "Expected .csv, .jsonl, .ndjson, or .json"
            )

    @staticmethod
    def _load_csv(path: str) -> List[JsonDict]:
        rows: List[JsonDict] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _load_jsonl(path: str) -> List[JsonDict]:
        rows: List[JsonDict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # ------------------------------------------------------------------
    # Deterministic assignment
    # ------------------------------------------------------------------

    def _stable_index(self, agent_id: str) -> int:
        """
        Deterministically map *agent_id* → profile index using SHA-256.

        Mirrors ``vivarium.policy.stable_agent_seed`` but returns an index
        into ``self._profiles``.
        """
        h = hashlib.sha256(
            f"{self._master_seed}:{agent_id}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(h[:8], "big", signed=False)
        return seed % len(self._profiles)

    # ------------------------------------------------------------------
    # AgentStateSpace protocol
    # ------------------------------------------------------------------

    def init_state(self, agent_id: str) -> JsonDict:
        """Assign a demographic profile deterministically to *agent_id*."""
        idx = self._stable_index(agent_id)
        profile = dict(self._profiles[idx])  # shallow copy
        return {"profile": profile, "_profile_index": idx}

    def transition(
        self,
        *,
        agent_id: str,
        current_state: JsonDict,
        action: ActionRequest,
        result: ActionResult,
        time_step: int,
    ) -> JsonDict:
        # Demographics are immutable – state never changes.
        return current_state

    def observe(
        self,
        *,
        agent_id: str,
        state: JsonDict,
        time_step: int,
    ) -> JsonDict:
        """
        Format the demographic profile into a *Persona* text block.

        Returns a dict with key ``"persona"`` containing a human-readable
        multi-line string ready for injection into the LLM system prompt.
        """
        profile: JsonDict = state.get("profile", {})
        if not profile:
            return {}

        lines = ["Your Identity & Demographic Profile:"]
        for key, value in profile.items():
            # Pretty-print the key: snake_case → Title Case
            label = key.replace("_", " ").title()
            lines.append(f"  - {label}: {value}")

        lines.append("")
        lines.append(
            "You MUST embody this persona consistently. "
            "Let these traits naturally inform your reasoning, opinions, and communication style."
        )
        return {"persona": "\n".join(lines)}
