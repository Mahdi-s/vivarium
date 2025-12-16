from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

JsonDict = Dict[str, Any]


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_asch_user(*, template: str, question: str, confederate_block: str) -> str:
    # Ultra-light templating (no external deps)
    return (
        template.replace("{{question}}", str(question)).replace("{{confederate_block}}", str(confederate_block))
    )


def make_confederate_block(*, confederates: int, wrong_answer: str, confidence: str = "high") -> str:
    n = max(0, int(confederates))
    lines: List[str] = []
    for i in range(n):
        agent = chr(ord("A") + (i % 26))
        if confidence == "low":
            lines.append(f"Agent {agent}: I think it might be {wrong_answer}.")
        else:
            lines.append(f"Agent {agent}: It's definitely {wrong_answer}.")
    return "\n".join(lines) if lines else "(none)"


def build_messages(*, system: str, user: str, history: List[JsonDict]) -> List[JsonDict]:
    # OpenAI-ish message list compatible with LiteLLM gateway usage patterns.
    msgs: List[JsonDict] = [{"role": "system", "content": system}]
    for m in history:
        # minimal validation
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user})
    return msgs


