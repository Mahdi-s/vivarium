from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from aam.persistence import TraceDb


JsonDict = Dict[str, Any]


def _require_tl() -> Any:
    try:
        from transformer_lens import HookedTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Intervention requires TransformerLens. Install extras: `pip install -e .[interpretability]`") from e
    return HookedTransformer


def _require_safetensors() -> Any:
    try:
        from safetensors.torch import load_file  # type: ignore
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Intervention requires torch + safetensors.") from e
    return load_file, torch


def _messages_to_prompt(messages: List[JsonDict]) -> str:
    lines: List[str] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        lines.append(f"{role.upper()}:\n{content}\n")
    lines.append("ASSISTANT:\n")
    return "\n".join(lines)


def _load_trial_messages(*, trace_db: TraceDb, trial_id: str) -> List[JsonDict]:
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
        raise ValueError(f"Missing conformity_prompts for trial_id={trial_id}")

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
    return messages


def _extract_text_from_generation(out: Any, prompt: str) -> str:
    s = out if isinstance(out, str) else str(out)
    # Some TL versions return full prompt + completion; trim if so.
    if s.startswith(prompt):
        return s[len(prompt) :]
    return s


def register_social_vector_intervention(
    *,
    trace_db: TraceDb,
    run_id: str,
    name: str,
    alpha: float,
    target_layers: List[int],
    component: str,
    vector_probe_id: str,
    notes: Optional[str] = None,
) -> str:
    intervention_id = str(uuid.uuid4())
    trace_db.insert_conformity_intervention(
        intervention_id=intervention_id,
        run_id=run_id,
        name=name,
        alpha=alpha,
        target_layers=target_layers,
        component=component,
        vector_probe_id=vector_probe_id,
        notes=notes,
    )
    return intervention_id


def run_intervention_sweep(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    probe_artifact_path: str,
    social_probe_id: str,
    target_layers: List[int],
    component_hook: str = "hook_resid_post",
    alpha_values: List[float] = [0.5, 1.0, 2.0],
    max_new_tokens: int = 64,
    normalize_vector: bool = True,
    trial_filter_sql: Optional[str] = None,
) -> int:
    """
    Executes activation steering by subtracting alpha * v_social at specified layers.
    Writes:
      - conformity_interventions
      - conformity_outputs (before/after)
      - conformity_intervention_results
    Returns number of result rows inserted.
    """
    HookedTransformer = _require_tl()
    load_file, torch = _require_safetensors()

    weights = load_file(probe_artifact_path)
    model = HookedTransformer.from_pretrained(model_id)

    # Choose trials (default: all immutable-fact trials with is_correct not NULL)
    base_query = """
      SELECT t.trial_id
      FROM conformity_trials t
      WHERE t.run_id = ?
    """
    if trial_filter_sql:
        base_query += f" AND ({trial_filter_sql})"
    base_query += " ORDER BY t.created_at ASC;"
    trials = trace_db.conn.execute(base_query, (run_id,)).fetchall()
    if not trials:
        return 0

    inserted = 0
    now = time.time()

    # Precompute vectors by layer
    vec_by_layer: Dict[int, Any] = {}
    for layer in target_layers:
        w = weights.get(f"layer_{int(layer)}.weight")
        if w is None:
            continue
        v = w.detach().to(torch.float32)
        if normalize_vector:
            denom = torch.norm(v) + 1e-8
            v = v / denom
        vec_by_layer[int(layer)] = v

    for alpha in alpha_values:
        intervention_id = register_social_vector_intervention(
            trace_db=trace_db,
            run_id=run_id,
            name=f"social_subtract_{alpha:g}",
            alpha=float(alpha),
            target_layers=list(target_layers),
            component=str(component_hook),
            vector_probe_id=social_probe_id,
            notes="Subtract normalized probe weight vector at resid_post (best-effort)",
        )

        for tr in trials:
            trial_id = str(tr["trial_id"])
            messages = _load_trial_messages(trace_db=trace_db, trial_id=trial_id)
            prompt = _messages_to_prompt(messages)

            # Baseline generation (no hooks)
            t0 = time.time()
            out_before = model.generate(prompt, max_new_tokens=int(max_new_tokens), temperature=0.0)
            latency_before = (time.time() - t0) * 1000.0
            text_before = _extract_text_from_generation(out_before, prompt)

            output_before_id = str(uuid.uuid4())
            trace_db.insert_conformity_output(
                output_id=output_before_id,
                trial_id=trial_id,
                raw_text=str(text_before),
                parsed_answer_text=None,
                parsed_answer_json=None,
                is_correct=None,
                refusal_flag=False,
                latency_ms=latency_before,
                token_usage_json=None,
                created_at=now,
            )

            # Intervention generation
            def make_hook(v: Any) -> Any:
                def _hook_fn(act: Any, hook: Any) -> Any:
                    # act shape [batch, pos, d_model] typically
                    try:
                        return act - (float(alpha) * v)[None, None, :]
                    except Exception:
                        return act
                return _hook_fn

            hooks: List[Tuple[str, Any]] = []
            for layer, v in vec_by_layer.items():
                # component_hook is expected as already 'hook_resid_post' or a full suffix
                hooks.append((f"blocks.{int(layer)}.{component_hook}", make_hook(v)))

            t1 = time.time()
            with model.hooks(fwd_hooks=hooks):
                out_after = model.generate(prompt, max_new_tokens=int(max_new_tokens), temperature=0.0)
            latency_after = (time.time() - t1) * 1000.0
            text_after = _extract_text_from_generation(out_after, prompt)

            output_after_id = str(uuid.uuid4())
            trace_db.insert_conformity_output(
                output_id=output_after_id,
                trial_id=trial_id,
                raw_text=str(text_after),
                parsed_answer_text=None,
                parsed_answer_json=None,
                is_correct=None,
                refusal_flag=False,
                latency_ms=latency_after,
                token_usage_json=None,
                created_at=now,
            )

            # flipped_to_truth is unknown here unless immutable fact + parsing is enabled; keep NULL.
            trace_db.insert_conformity_intervention_result(
                result_id=str(uuid.uuid4()),
                trial_id=trial_id,
                intervention_id=intervention_id,
                output_id_before=output_before_id,
                output_id_after=output_after_id,
                flipped_to_truth=None,
                created_at=now,
            )
            inserted += 1

    return inserted


