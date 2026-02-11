from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from aam.llm_gateway import LLMGateway
from aam.text_parse import parse_action_json
from aam.tools import ToolSpec, default_tools
from aam.types import ActionRequest, Observation


JsonDict = Dict[str, Any]


def _openai_messages_from_observation(
    *, agent_id: str, observation: Observation, require_json_action: bool
) -> List[JsonDict]:
    time_step = int(observation.get("time_step", 0))
    msgs = observation.get("messages", []) or []

    if require_json_action:
        system = (
            "You are an agent in a simulation. You must decide ONE action each step.\n"
            "Output ONLY a JSON object, with no surrounding markdown.\n"
            'Schema: {"action": "<action_name>", "args": {...}, "reasoning": "<optional>"}\n'
            'If unsure, output: {"action":"noop","args":{}}'
        )
    else:
        system = (
            "You are an agent in a simulation. "
            "You must decide ONE action each step. "
            "Use tools when available."
        )

    # Provide message feed as context (simple text form for Phase 2 MVP).
    history_lines = []
    for m in msgs:
        history_lines.append(f"[t={m.get('time_step')}] {m.get('author_id')}: {m.get('content')}")
    history = "\n".join(history_lines) if history_lines else "(no messages yet)"

    user = (
        f"agent_id={agent_id}\n"
        f"time_step={time_step}\n\n"
        "Shared message feed:\n"
        f"{history}\n\n"
        + (
            "Decide an action."
            if require_json_action
            else "If you post, write a short, helpful message."
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _tool_specs() -> List[ToolSpec]:
    return default_tools()


def _extract_tool_call(resp: JsonDict) -> Optional[tuple[str, JsonDict]]:
    """
    Extract (tool_name, args_dict) from an OpenAI-ish completion response.
    """
    try:
        msg = resp["choices"][0]["message"]
    except Exception:
        return None

    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return None

    fc = tool_calls[0].get("function", {})
    name = fc.get("name")
    args_raw = fc.get("arguments", "{}")
    if not name:
        return None
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
    except Exception:
        args = {}
    return str(name), args


def _extract_text(resp: JsonDict) -> str:
    try:
        msg = resp["choices"][0]["message"]
        return str(msg.get("content") or "")
    except Exception:
        return ""


def langgraph_available() -> bool:
    try:
        import langgraph  # type: ignore # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class CognitiveAgentPolicy:
    """
    Phase 2 policy:
    - LangGraph orchestrates the cognitive steps (optional dependency).
    - LiteLLM provides model access (optional dependency via the gateway).
    - Dual-mode: tool calls if available; else text JSON parsing fallback.
    """

    gateway: LLMGateway
    model: str
    tools: List[ToolSpec]
    temperature: float = 0.2
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            from langgraph.graph import StateGraph  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LangGraph is not installed. Install extras: `pip install -e .[cognitive]`"
            ) from e

        # Build a tiny StateGraph: observation -> messages -> model -> parse -> action
        class _State(TypedDict, total=False):
            run_id: str
            time_step: int
            agent_id: str
            observation: Observation
            messages: List[JsonDict]
            llm_response: JsonDict
            action_name: str
            action_args: JsonDict
            reasoning: Optional[str]

        g: Any = StateGraph(_State)

        def build_messages(state: _State) -> _State:
            obs = state["observation"]
            supports_tools = bool(getattr(self.gateway, "supports_tool_calls", True))
            state["messages"] = _openai_messages_from_observation(
                agent_id=state["agent_id"],
                observation=obs,
                require_json_action=(not supports_tools),
            )
            return state

        def call_model(state: _State) -> _State:
            supports_tools = bool(getattr(self.gateway, "supports_tool_calls", True))
            resp = self.gateway.chat(
                model=self.model,
                messages=state["messages"],
                tools=(self.tools if supports_tools else None),
                tool_choice=("auto" if supports_tools else None),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            state["llm_response"] = resp
            return state

        def parse_action(state: _State) -> _State:
            resp = state["llm_response"]
            supports_tools = bool(getattr(self.gateway, "supports_tool_calls", True))
            tool = _extract_tool_call(resp)
            if supports_tools and tool is not None:
                name, args = tool
                state["action_name"] = name
                state["action_args"] = args
                state["reasoning"] = None
                cb = getattr(self.gateway, "on_action_decided", None)
                if callable(cb):
                    cb(
                        run_id=state["run_id"],
                        time_step=state["time_step"],
                        agent_id=state["agent_id"],
                        action_name=str(name),
                    )
                return state

            # Text fallback: ask for JSON action schema.
            text = _extract_text(resp)
            parsed = parse_action_json(text)
            if parsed and "action" in parsed:
                state["action_name"] = str(parsed.get("action"))
                state["action_args"] = dict(parsed.get("args") or {})
                state["reasoning"] = str(parsed.get("reasoning")) if parsed.get("reasoning") else None
                cb = getattr(self.gateway, "on_action_decided", None)
                if callable(cb):
                    cb(
                        run_id=state["run_id"],
                        time_step=state["time_step"],
                        agent_id=state["agent_id"],
                        action_name=str(state["action_name"]),
                    )
                return state

            # Safe fallback
            state["action_name"] = "noop"
            state["action_args"] = {}
            state["reasoning"] = None
            cb = getattr(self.gateway, "on_action_decided", None)
            if callable(cb):
                cb(
                    run_id=state["run_id"],
                    time_step=state["time_step"],
                    agent_id=state["agent_id"],
                    action_name="noop",
                )
            return state

        g.add_node("build_messages", build_messages)
        g.add_node("call_model", call_model)
        g.add_node("parse_action", parse_action)

        g.set_entry_point("build_messages")
        g.add_edge("build_messages", "call_model")
        g.add_edge("call_model", "parse_action")
        g.set_finish_point("parse_action")

        object.__setattr__(self, "_graph", g.compile())

    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest:
        state = {
            "run_id": run_id,
            "time_step": time_step,
            "agent_id": agent_id,
            "observation": observation,
        }
        out = self._graph.invoke(state)
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=str(out.get("action_name") or "noop"),
            arguments=dict(out.get("action_args") or {}),
            reasoning=out.get("reasoning"),
            metadata={"model": self.model, "policy": "CognitiveAgentPolicy"},
        )

    async def adecide(
        self, *, run_id: str, time_step: int, agent_id: str, observation: Observation
    ) -> ActionRequest:
        state = {
            "run_id": run_id,
            "time_step": time_step,
            "agent_id": agent_id,
            "observation": observation,
        }
        ainvoke = getattr(self._graph, "ainvoke", None)
        if callable(ainvoke):
            out = await ainvoke(state)
        else:
            out = await asyncio.to_thread(self._graph.invoke, state)

        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name=str(out.get("action_name") or "noop"),
            arguments=dict(out.get("action_args") or {}),
            reasoning=out.get("reasoning"),
            metadata={"model": self.model, "policy": "CognitiveAgentPolicy"},
        )


@dataclass(frozen=True)
class SimpleCognitivePolicy:
    """
    Fallback Phase 2 policy when LangGraph is not installed.
    Uses the same gateway + tool/text parsing logic, but without graph orchestration.
    """

    gateway: LLMGateway
    model: str
    tools: List[ToolSpec]
    temperature: float = 0.2
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def decide(self, *, run_id: str, time_step: int, agent_id: str, observation: Observation) -> ActionRequest:
        supports_tools = bool(getattr(self.gateway, "supports_tool_calls", True))
        messages = _openai_messages_from_observation(
            agent_id=agent_id, observation=observation, require_json_action=(not supports_tools)
        )
        resp = self.gateway.chat(
            model=self.model,
            messages=messages,
            tools=(self.tools if supports_tools else None),
            tool_choice=("auto" if supports_tools else None),
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        tool = _extract_tool_call(resp)
        if supports_tools and tool is not None:
            name, args = tool
            cb = getattr(self.gateway, "on_action_decided", None)
            if callable(cb):
                cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name=str(name))
            return ActionRequest(
                run_id=run_id,
                time_step=time_step,
                agent_id=agent_id,
                action_name=name,
                arguments=args,
                reasoning=None,
                metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
            )

        text = _extract_text(resp)
        parsed = parse_action_json(text)
        if parsed and "action" in parsed:
            action_name = str(parsed.get("action") or "noop")
            cb = getattr(self.gateway, "on_action_decided", None)
            if callable(cb):
                cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name=action_name)
            return ActionRequest(
                run_id=run_id,
                time_step=time_step,
                agent_id=agent_id,
                action_name=action_name,
                arguments=dict(parsed.get("args") or {}),
                reasoning=str(parsed.get("reasoning")) if parsed.get("reasoning") else None,
                metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
            )

        cb = getattr(self.gateway, "on_action_decided", None)
        if callable(cb):
            cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name="noop")
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name="noop",
            arguments={},
            reasoning=None,
            metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
        )

    async def adecide(
        self, *, run_id: str, time_step: int, agent_id: str, observation: Observation
    ) -> ActionRequest:
        supports_tools = bool(getattr(self.gateway, "supports_tool_calls", True))
        messages = _openai_messages_from_observation(
            agent_id=agent_id, observation=observation, require_json_action=(not supports_tools)
        )

        achat = getattr(self.gateway, "achat", None)
        if callable(achat):
            resp = await achat(
                model=self.model,
                messages=messages,
                tools=(self.tools if supports_tools else None),
                tool_choice=("auto" if supports_tools else None),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        else:
            resp = await asyncio.to_thread(
                self.gateway.chat,
                model=self.model,
                messages=messages,
                tools=(self.tools if supports_tools else None),
                tool_choice=("auto" if supports_tools else None),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        tool = _extract_tool_call(resp)
        if supports_tools and tool is not None:
            name, args = tool
            cb = getattr(self.gateway, "on_action_decided", None)
            if callable(cb):
                cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name=str(name))
            return ActionRequest(
                run_id=run_id,
                time_step=time_step,
                agent_id=agent_id,
                action_name=name,
                arguments=args,
                reasoning=None,
                metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
            )

        text = _extract_text(resp)
        parsed = parse_action_json(text)
        if parsed and "action" in parsed:
            action_name = str(parsed.get("action") or "noop")
            cb = getattr(self.gateway, "on_action_decided", None)
            if callable(cb):
                cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name=action_name)
            return ActionRequest(
                run_id=run_id,
                time_step=time_step,
                agent_id=agent_id,
                action_name=action_name,
                arguments=dict(parsed.get("args") or {}),
                reasoning=str(parsed.get("reasoning")) if parsed.get("reasoning") else None,
                metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
            )

        cb = getattr(self.gateway, "on_action_decided", None)
        if callable(cb):
            cb(run_id=run_id, time_step=time_step, agent_id=agent_id, action_name="noop")
        return ActionRequest(
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            action_name="noop",
            arguments={},
            reasoning=None,
            metadata={"model": self.model, "policy": "SimpleCognitivePolicy"},
        )


def default_cognitive_policy(
    *, gateway: LLMGateway, model: str, temperature: float = 0.2, top_k: Optional[int] = None, top_p: Optional[float] = None
) -> CognitiveAgentPolicy:
    # Prefer LangGraph implementation when available; otherwise fall back (still Phase 2, but no graph).
    if langgraph_available():
        return CognitiveAgentPolicy(
            gateway=gateway,
            model=model,
            tools=_tool_specs(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    return SimpleCognitivePolicy(
        gateway=gateway,
        model=model,
        tools=_tool_specs(),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
