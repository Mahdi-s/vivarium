from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aam.experiments.olmo_conformity.olmo_utils import get_olmo_model_config
from aam.experiments.olmo_conformity.prompts import build_messages
from aam.llm_gateway import HuggingFaceHookedGateway, select_local_gateway
from aam.persistence import TraceDb


JsonDict = Dict[str, Any]

ANSWER_LOGPROB_VERSION = "2026-02-11_answer_logprobs_v1"


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Answer logprobs require torch.") from e
    return torch, F


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(str(text or "").encode("utf-8"))
    return h.hexdigest()


def extract_observed_think_prefix(raw_text: str) -> Optional[str]:
    """
    Extract the assistant prefix up to and including the last </think> tag, plus any
    immediately following whitespace (so the next token starts where the answer starts).

    Returns None if no </think> tag is present.
    """
    t = str(raw_text or "")
    low = t.lower()
    tag = "</think>"
    idx = low.rfind(tag)
    if idx == -1:
        return None
    end = idx + len(tag)
    while end < len(t) and t[end].isspace():
        end += 1
    return t[:end]


@dataclass(frozen=True)
class AnswerLogprobContext:
    kind: str
    prefix_text: str
    prefix_source: str  # "none"|"empty_think"|"observed_think_prefix"
    prefix_metadata: JsonDict


def _build_contexts_for_trial(
    *,
    model_id: str,
    raw_text: str,
    include_empty_think: bool,
    include_observed_think: bool,
) -> List[AnswerLogprobContext]:
    contexts: List[AnswerLogprobContext] = []

    contexts.append(
        AnswerLogprobContext(
            kind="assistant_start",
            prefix_text="",
            prefix_source="none",
            prefix_metadata={},
        )
    )

    observed = extract_observed_think_prefix(raw_text)
    if include_observed_think and observed:
        contexts.append(
            AnswerLogprobContext(
                kind="observed_think_prefix",
                prefix_text=observed,
                prefix_source="observed_think_prefix",
                prefix_metadata={
                    "observed_prefix_chars": len(observed),
                    "observed_prefix_sha256": _sha256_text(observed),
                },
            )
        )

    if include_empty_think and bool(get_olmo_model_config(str(model_id)).get("has_think_tokens", False)):
        empty = "<think>\n</think>\n"
        contexts.append(
            AnswerLogprobContext(
                kind="empty_think_prefix",
                prefix_text=empty,
                prefix_source="empty_think",
                prefix_metadata={
                    "empty_prefix_chars": len(empty),
                    "empty_prefix_sha256": _sha256_text(empty),
                },
            )
        )

    return contexts


def compute_and_store_answer_logprobs_for_model(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    trial_ids: Sequence[str],
    include_empty_think: bool = True,
    include_observed_think: bool = True,
    include_alternate_answer: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Compute and store answer-level logprob metrics for a single model_id over a set of trials.

    Stores one row per (trial_id, context_kind, candidate_kind) in `conformity_answer_logprobs`.
    """
    if not trial_ids:
        return {"model_id": str(model_id), "n_trials": 0, "inserted": 0, "skipped": 0, "errors": 0}

    gateway = select_local_gateway(
        model_id_or_path=str(model_id),
        capture_context=None,
        max_new_tokens=1,
        scientific_mode=False,
    )
    if not isinstance(gateway, HuggingFaceHookedGateway):
        raise RuntimeError(
            f"Answer logprob analysis currently requires HuggingFaceHookedGateway, got {gateway.__class__.__name__}"
        )

    tokenizer = getattr(gateway, "_tokenizer", None)
    model = getattr(gateway, "_model", None)
    if tokenizer is None or model is None:
        raise RuntimeError("Gateway missing tokenizer/model")

    # Prefer model.device when available; fall back to gateway.device.
    device = getattr(model, "device", None) or getattr(gateway, "device", None) or "cpu"

    existing: set[tuple[str, str, str]] = set()
    if skip_existing:
        rows = trace_db.conn.execute(
            """
            SELECT a.trial_id, a.context_kind, a.candidate_kind
            FROM conformity_answer_logprobs a
            JOIN conformity_trials t ON t.trial_id = a.trial_id
            WHERE t.run_id = ? AND t.model_id = ?;
            """,
            (str(run_id), str(model_id)),
        ).fetchall()
        existing = {(str(r["trial_id"]), str(r["context_kind"]), str(r["candidate_kind"])) for r in rows}

    inserted = 0
    skipped = 0
    errors = 0

    allowed_trials: Optional[set[str]] = None
    if trial_ids:
        allowed_trials = {str(t) for t in trial_ids}

    trial_rows = trace_db.conn.execute(
        """
        SELECT
          t.trial_id,
          t.variant,
          c.name AS condition_name,
          i.ground_truth_text,
          p.prompt_id,
          p.system_prompt,
          p.user_prompt,
          p.chat_history_json,
          p.rendered_prompt_hash,
          pm.metadata_json AS prompt_metadata_json,
          o.output_id,
          o.raw_text
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_items i ON i.item_id = t.item_id
        JOIN conformity_prompts p ON p.prompt_id = (
          SELECT prompt_id FROM conformity_prompts
          WHERE trial_id = t.trial_id
          ORDER BY created_at ASC
          LIMIT 1
        )
        LEFT JOIN conformity_prompt_metadata pm ON pm.prompt_id = p.prompt_id
        JOIN conformity_outputs o ON o.output_id = (
          SELECT output_id FROM conformity_outputs
          WHERE trial_id = t.trial_id
          ORDER BY created_at DESC
          LIMIT 1
        )
        WHERE t.run_id = ? AND t.model_id = ? AND c.name NOT LIKE '%probe_capture%'
        ORDER BY t.created_at ASC, t.trial_id ASC;
        """,
        (str(run_id), str(model_id)),
    ).fetchall()

    for row in trial_rows:
        tid = str(row["trial_id"])
        if allowed_trials is not None and tid not in allowed_trials:
            continue

        condition_name = str(row["condition_name"] or "")

        gt = row["ground_truth_text"]
        ground_truth = (str(gt) if gt is not None else "").strip()

        prompt_meta: JsonDict = {}
        try:
            raw = row["prompt_metadata_json"]
            if raw:
                prompt_meta = json.loads(str(raw))
        except Exception:
            prompt_meta = {}

        wrong = str(prompt_meta.get("wrong_answer") or "").strip()
        alt = str(prompt_meta.get("alternate_answer") or "").strip()

        if not ground_truth or not wrong:
            continue

        # Candidate set
        candidates: List[Tuple[str, str]] = [("ground_truth", ground_truth), ("wrong_answer", wrong)]
        if include_alternate_answer and alt:
            candidates.append(("alternate_answer", alt))

        # Render messages -> prompt string using the same gateway formatting as inference.
        system_prompt = str(row["system_prompt"] or "")
        user_prompt = str(row["user_prompt"] or "")
        history: List[JsonDict] = []
        try:
            parsed = json.loads(str(row["chat_history_json"] or "[]"))
            if isinstance(parsed, list):
                history = [m for m in parsed if isinstance(m, dict)]
        except Exception:
            history = []

        messages = build_messages(system=system_prompt, user=user_prompt, history=history)
        prompt_str, used_chat_template = gateway._messages_to_prompt(messages)  # type: ignore[attr-defined]

        raw_text = str(row["raw_text"] or "")
        contexts = _build_contexts_for_trial(
            model_id=str(model_id),
            raw_text=raw_text,
            include_empty_think=bool(include_empty_think),
            include_observed_think=bool(include_observed_think),
        )

        for ctx in contexts:
            context_text = str(prompt_str) + str(ctx.prefix_text or "")
            try:
                ctx_enc = tokenizer(str(context_text), return_tensors="pt")
                context_ids = [int(x) for x in ctx_enc["input_ids"][0].detach().cpu().tolist()]
            except Exception:
                errors += 1
                continue

            # Compute the shared context cache once per (trial, context_kind).
            torch, F = _require_torch()
            ctx_t = torch.tensor([list(context_ids)], dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    base_out = model(input_ids=ctx_t, use_cache=True)
                    base_logits = base_out.logits[:, -1, :].to(torch.float32)
                    base_past = getattr(base_out, "past_key_values", None)
                    if base_past is None:
                        raise RuntimeError("Model did not return past_key_values.")
            except Exception:
                errors += 1
                continue

            for cand_kind, cand_text in candidates:
                if skip_existing:
                    if (str(tid), str(ctx.kind), str(cand_kind)) in existing:
                        skipped += 1
                        continue

                try:
                    full_enc = tokenizer(str(context_text) + str(cand_text), return_tensors="pt")
                    full_ids = [int(x) for x in full_enc["input_ids"][0].detach().cpu().tolist()]
                    if len(full_ids) < len(context_ids) or full_ids[: len(context_ids)] != context_ids:
                        comp = tokenizer(str(cand_text), add_special_tokens=False)
                        comp_ids = [int(x) for x in (comp.get("input_ids") or [])]
                    else:
                        comp_ids = full_ids[len(context_ids) :]

                    # Compute logprobs reusing the base cache.
                    token_logprobs: List[float] = []
                    if comp_ids:
                        logits = base_logits
                        past = base_past
                        for i, tok in enumerate(list(comp_ids)):
                            tok_i = int(tok)
                            logp = float(F.log_softmax(logits, dim=-1)[0, tok_i].item())
                            token_logprobs.append(logp)
                            if i == (len(comp_ids) - 1):
                                break
                            nxt_in = torch.tensor([[tok_i]], dtype=torch.long, device=device)
                            out2 = model(input_ids=nxt_in, past_key_values=past, use_cache=True)
                            logits = out2.logits[:, -1, :].to(torch.float32)
                            past = getattr(out2, "past_key_values", None)
                            if past is None:
                                raise RuntimeError("Model stopped returning past_key_values during cached evaluation.")

                    n_tok = int(len(comp_ids))
                    if n_tok <= 0:
                        continue

                    logp_sum = float(sum(token_logprobs))
                    logp_mean = float(logp_sum / float(n_tok)) if n_tok else 0.0
                    first_token_id = int(comp_ids[0]) if comp_ids else None
                    first_token_logp = float(token_logprobs[0]) if token_logprobs else None

                    meta: JsonDict = {
                        "answer_logprob_version": ANSWER_LOGPROB_VERSION,
                        "run_id": str(run_id),
                        "trial_id": str(tid),
                        "model_id": str(model_id),
                        "variant": str(row["variant"] or ""),
                        "condition_name": str(condition_name),
                        "prompt_id": str(row["prompt_id"] or ""),
                        "rendered_prompt_hash": str(row["rendered_prompt_hash"] or ""),
                        "used_chat_template": bool(used_chat_template),
                        "context": {
                            "context_kind": str(ctx.kind),
                            "prefix_source": str(ctx.prefix_source),
                            "prefix_metadata": (ctx.prefix_metadata or {}),
                            "context_token_count": int(len(context_ids)),
                        },
                        "candidate": {
                            "candidate_kind": str(cand_kind),
                            "candidate_text_sha256": _sha256_text(str(cand_text)),
                            "completion_token_ids": [int(x) for x in comp_ids[:64]],
                        },
                        "output": {
                            "output_id": str(row["output_id"] or ""),
                        },
                    }

                    trace_db.upsert_conformity_answer_logprob(
                        trial_id=str(tid),
                        context_kind=str(ctx.kind),
                        candidate_kind=str(cand_kind),
                        candidate_text=str(cand_text),
                        token_count=int(n_tok),
                        logprob_sum=float(logp_sum),
                        logprob_mean=float(logp_mean),
                        first_token_id=(None if first_token_id is None else int(first_token_id)),
                        first_token_logprob=(None if first_token_logp is None else float(first_token_logp)),
                        metadata=meta,
                        created_at=time.time(),
                    )
                    if skip_existing:
                        existing.add((str(tid), str(ctx.kind), str(cand_kind)))
                    inserted += 1
                except Exception:
                    errors += 1
                    continue

    return {
        "model_id": str(model_id),
        "n_trials": int(len(trial_ids)),
        "inserted": int(inserted),
        "skipped": int(skipped),
        "errors": int(errors),
    }


def compute_and_store_answer_logprobs_for_run(
    *,
    trace_db: TraceDb,
    run_id: str,
    include_empty_think: bool = True,
    include_observed_think: bool = True,
    include_alternate_answer: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute logprobs for every model_id present in the run.
    """
    rows = trace_db.conn.execute(
        """
        SELECT DISTINCT model_id
        FROM conformity_trials
        WHERE run_id = ?
        ORDER BY model_id ASC;
        """,
        (str(run_id),),
    ).fetchall()
    model_ids = [str(r["model_id"]) for r in rows]

    out: Dict[str, Any] = {"run_id": str(run_id), "by_model_id": {}, "total_inserted": 0, "total_errors": 0}
    for model_id in model_ids:
        tids = trace_db.conn.execute(
            """
            SELECT trial_id
            FROM conformity_trials
            WHERE run_id = ? AND model_id = ?
            ORDER BY created_at ASC, trial_id ASC;
            """,
            (str(run_id), str(model_id)),
        ).fetchall()
        trial_ids = [str(r["trial_id"]) for r in tids]

        res = compute_and_store_answer_logprobs_for_model(
            trace_db=trace_db,
            run_id=str(run_id),
            model_id=str(model_id),
            trial_ids=trial_ids,
            include_empty_think=bool(include_empty_think),
            include_observed_think=bool(include_observed_think),
            include_alternate_answer=bool(include_alternate_answer),
            skip_existing=bool(skip_existing),
        )
        out["by_model_id"][model_id] = res
        out["total_inserted"] += int(res.get("inserted") or 0)
        out["total_errors"] += int(res.get("errors") or 0)
    return out
