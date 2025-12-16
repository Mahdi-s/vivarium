from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from aam.persistence import TraceDb


JsonDict = Dict[str, Any]


def _require_tl() -> Any:
    try:
        from transformer_lens import HookedTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Logit lens requires TransformerLens. Install extras: `pip install -e .[interpretability]`") from e
    return HookedTransformer


def _messages_to_prompt(messages: List[JsonDict]) -> str:
    lines: List[str] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        lines.append(f"{role.upper()}:\n{content}\n")
    lines.append("ASSISTANT:\n")
    return "\n".join(lines)


def compute_logit_lens_topk_for_trial(
    *,
    trace_db: TraceDb,
    trial_id: str,
    model_id: str,
    layers: List[int],
    k: int = 10,
) -> int:
    """
    Best-effort logit lens: for each requested layer, take the residual stream at the last prompt position
    (blocks.<L>.hook_resid_post) and unembed it to logits, then store top-k tokens/probs in conformity_logit_lens.

    Notes:
    - This is not generation-time logit lens over the full CoT; it’s a compact probe at the prompt boundary.
    - It’s still useful for detecting “early truth vs late conformity” drift across layers.
    """
    HookedTransformer = _require_tl()

    # Pull prompt for trial
    row = trace_db.conn.execute(
        """
        SELECT p.system_prompt, p.user_prompt, p.chat_history_json
        FROM conformity_prompts p
        WHERE p.trial_id = ?
        ORDER BY p.created_at ASC
        LIMIT 1;
        """,
        (trial_id,),
    ).fetchone()
    if row is None:
        return 0

    try:
        history = json.loads(row["chat_history_json"] or "[]")
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []

    messages: List[JsonDict] = [{"role": "system", "content": str(row["system_prompt"])}]
    for m in history:
        if isinstance(m, dict):
            messages.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": str(row["user_prompt"])})

    prompt = _messages_to_prompt(messages)

    model = HookedTransformer.from_pretrained(model_id)
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    inserted = 0
    for layer in layers:
        key = f"blocks.{int(layer)}.hook_resid_post"
        if key not in cache:
            continue
        resid = cache[key][0, -1, :]
        # Standard-ish logit lens: unembed residual directly (best-effort)
        try:
            logits = model.unembed(resid)
        except Exception:
            # fallback: use output logits from full forward pass if accessible
            continue
        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(int(k))
        toks = [model.to_string(int(i)) for i in topi.detach().cpu().tolist()]
        vals = topv.detach().cpu().tolist()
        topk = [{"token": t, "prob": float(p)} for t, p in zip(toks, vals)]

        trace_db.conn.execute(
            """
            INSERT INTO conformity_logit_lens(logit_id, trial_id, layer_index, token_index, topk_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (str(uuid.uuid4()), trial_id, int(layer), 0, json.dumps(topk, ensure_ascii=False), time.time()),
        )
        inserted += 1

    trace_db.conn.commit()
    return inserted


def parse_and_store_think_tokens(*, trace_db: TraceDb, trial_id: str) -> int:
    """
    Parses <think>...</think> from the trial's latest raw output and stores as coarse tokens (whitespace-split).
    This is a lightweight fallback when true tokenization isn't available/desired.
    """
    row = trace_db.conn.execute(
        """
        SELECT o.raw_text
        FROM conformity_outputs o
        WHERE o.trial_id = ?
        ORDER BY o.created_at DESC
        LIMIT 1;
        """,
        (trial_id,),
    ).fetchone()
    if row is None:
        return 0
    raw = str(row["raw_text"] or "")
    lo = raw.find("<think>")
    hi = raw.find("</think>")
    if lo == -1 or hi == -1 or hi <= lo:
        return 0
    inner = raw[lo + len("<think>") : hi].strip()
    if not inner:
        return 0
    parts = inner.split()
    now = time.time()
    inserted = 0
    for i, tok in enumerate(parts):
        trace_db.conn.execute(
            """
            INSERT INTO conformity_think_tokens(think_id, trial_id, token_index, token_text, token_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (str(uuid.uuid4()), trial_id, int(i), str(tok), None, now),
        )
        inserted += 1
    trace_db.conn.commit()
    return inserted


