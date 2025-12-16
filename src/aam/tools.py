from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """
    Minimal, OpenAI-compatible tool spec (works with LiteLLM's OpenAI-style interface).
    """

    name: str
    description: str
    parameters: JsonDict

    def as_openai_tool(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def post_message_tool() -> ToolSpec:
    return ToolSpec(
        name="post_message",
        description="Post a chat message into the shared world message feed.",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Message text to post."},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    )


def noop_tool() -> ToolSpec:
    return ToolSpec(
        name="noop",
        description="Do nothing this step.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )


def default_tools() -> List[ToolSpec]:
    return [post_message_tool(), noop_tool()]


