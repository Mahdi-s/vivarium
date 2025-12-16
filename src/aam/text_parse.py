from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


JsonDict = Dict[str, Any]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def parse_action_json(text: str) -> Optional[JsonDict]:
    """
    Parse a model text response into an action dict.

    Expected shape:
      {"action":"post_message","args":{"content":"..."}}

    This is a best-effort parser:
    - strips ```json fences
    - extracts the first {...} block if extra text exists
    - attempts JSON repair if basic parsing fails
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None

    # Try direct JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract first JSON object substring
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Try JSON repair library if available
        try:
            import json_repair  # type: ignore

            repaired = json_repair.repair_json(candidate)
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except ImportError:
            # json_repair not installed, fall back to regex rescue
            pass
        except Exception:
            # JSON repair failed, try regex rescue
            pass

    # Regex rescue: extract action and args separately
    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', candidate, re.IGNORECASE)
    args_match = re.search(r'"args"\s*:\s*(\{.*?\})', candidate, re.DOTALL | re.IGNORECASE)
    if action_match and args_match:
        try:
            action_name = action_match.group(1)
            args_str = args_match.group(1)
            args = json.loads(args_str)
            return {"action": action_name, "args": args}
        except Exception:
            pass

    return None


