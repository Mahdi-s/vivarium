from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from aam.interpretability import CaptureConfig, CaptureContext
from aam.llm_gateway import select_local_gateway
from aam.persistence import TraceDb

from .probes import compute_and_store_probe_projections_for_trials
from .prompts import build_messages
from .vector_analysis import detect_turn_layers, generate_vector_collision_plots


JsonDict = Dict[str, Any]


def _fetch_latest_probe(
    *,
    trace_db: TraceDb,
    run_id: str,
    probe_kind: str,
    model_id: str,
) -> Optional[Dict[str, str]]:
    row = trace_db.conn.execute(
        """
        SELECT probe_id, artifact_path
        FROM conformity_probes
        WHERE run_id = ? AND probe_kind = ? AND model_id = ?
        ORDER BY created_at DESC
        LIMIT 1;
        """,
        (str(run_id), str(probe_kind), str(model_id)),
    ).fetchone()
    if row is None:
        return None
    return {"probe_id": str(row["probe_id"]), "artifact_path": str(row["artifact_path"])}


def repair_trial_activations(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    run_dir: str,
    layers: List[int],
    component: str = "resid_post",
    token_position: int = -1,
    dtype: str = "float16",
    max_new_tokens: int = 128,
) -> int:
    """
    Re-capture activations for behavioral trials in an existing run.

    Why needed:
    Earlier versions of probe capture reused low time_step indices and overwrote
    step_*.safetensors shards created during trial execution. This repair step
    replays trial prompts and writes fresh shards (never overwriting existing files),
    inserting newer activation_metadata rows so downstream queries pick the latest.
    """
    # Capture into the same run_dir/activations folder so metadata stays co-located.
    activations_dir = os.path.join(str(run_dir), "activations")
    os.makedirs(activations_dir, exist_ok=True)

    # CaptureContext uses short component names (e.g. "resid_post") but stores
    # component="hook_resid_post" in activation_metadata after hook parsing.
    cap_cfg = CaptureConfig(
        layers=list(layers),
        components=[str(component)],
        trigger_actions=["trial_execution"],
        token_position=int(token_position),
    )
    cap_ctx = CaptureContext(output_dir=activations_dir, config=cap_cfg, dtype=str(dtype), trace_db=trace_db)
    gateway = select_local_gateway(model_id_or_path=str(model_id), capture_context=cap_ctx, max_new_tokens=int(max_new_tokens))

    trials = trace_db.conn.execute(
        """
        SELECT t.trial_id, t.model_id, s.time_step, s.agent_id
        FROM conformity_trials t
        JOIN conformity_trial_steps s ON s.trial_id = t.trial_id
        WHERE t.run_id = ? AND s.agent_id LIKE 'trial_%'
        ORDER BY s.time_step ASC;
        """,
        (str(run_id),),
    ).fetchall()
    if not trials:
        return 0

    repaired = 0
    for tr in trials:
        trial_id = str(tr["trial_id"])
        time_step = int(tr["time_step"])
        agent_id = str(tr["agent_id"])

        prow = trace_db.conn.execute(
            """
            SELECT system_prompt, user_prompt, chat_history_json
            FROM conformity_prompts
            WHERE trial_id = ?
            ORDER BY created_at ASC
            LIMIT 1;
            """,
            (trial_id,),
        ).fetchone()
        if prow is None:
            continue

        system_prompt = str(prow["system_prompt"] or "")
        user_prompt = str(prow["user_prompt"] or "")
        history: List[JsonDict] = []
        try:
            raw = prow["chat_history_json"]
            if raw:
                history = json.loads(raw)
        except Exception:
            history = []

        msgs = build_messages(system=system_prompt, user=user_prompt, history=history)

        cap_ctx.begin_inference()
        _ = gateway.chat(model=str(model_id), messages=msgs, tools=None, tool_choice=None, temperature=0.0)
        cap_ctx.on_action_decided(
            run_id=str(run_id),
            time_step=int(time_step),
            agent_id=str(agent_id),
            model_id=str(model_id),
            action_name="trial_execution",
        )
        cap_ctx.flush_step(time_step=int(time_step))
        repaired += 1

    return repaired


def resume_from_projections(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    run_dir: str,
    layers: List[int],
    component: str = "hook_resid_post",
    repair_activations_first: bool = True,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Resume the pipeline from the projection computation step for an existing run.
    Assumes truth/social probe artifacts already exist for this run.
    """
    if repair_activations_first:
        # Runner used components like "resid_post" but stored component="hook_resid_post" in metadata.
        _ = repair_trial_activations(
            trace_db=trace_db,
            run_id=run_id,
            model_id=model_id,
            run_dir=run_dir,
            layers=layers,
            component="resid_post",
            max_new_tokens=max_new_tokens,
        )

    truth = _fetch_latest_probe(trace_db=trace_db, run_id=run_id, probe_kind="truth", model_id=model_id)
    if truth is None:
        raise RuntimeError("Could not find a truth probe for this run. Run probe training first.")

    social = _fetch_latest_probe(trace_db=trace_db, run_id=run_id, probe_kind="social", model_id=model_id)

    # Avoid duplicating projections if this resume command is rerun after a partial failure.
    expected_trials_row = trace_db.conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM conformity_trials t
        JOIN conformity_trial_steps s ON s.trial_id = t.trial_id
        WHERE t.run_id = ? AND s.agent_id LIKE 'trial_%';
        """,
        (str(run_id),),
    ).fetchone()
    expected_trials = int(expected_trials_row["c"]) if expected_trials_row is not None else 0
    expected_rows = int(expected_trials) * int(len(layers))

    def _existing_projection_rows(probe_id: str) -> int:
        row = trace_db.conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM conformity_probe_projections p
            JOIN conformity_trials t ON t.trial_id = p.trial_id
            WHERE t.run_id = ? AND p.probe_id = ?;
            """,
            (str(run_id), str(probe_id)),
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    truth_inserted = 0
    if expected_rows > 0 and _existing_projection_rows(str(truth["probe_id"])) >= expected_rows:
        truth_inserted = 0
    else:
        truth_inserted = compute_and_store_probe_projections_for_trials(
            trace_db=trace_db,
            run_id=run_id,
            probe_id=str(truth["probe_id"]),
            probe_artifact_path=str(truth["artifact_path"]),
            model_id=str(model_id),
            component=str(component),
            layers=list(layers),
        )

    social_inserted = 0
    social_probe_id: Optional[str] = None
    if social is not None:
        social_probe_id = str(social["probe_id"])
        if expected_rows > 0 and _existing_projection_rows(str(social_probe_id)) >= expected_rows:
            social_inserted = 0
        else:
            social_inserted = compute_and_store_probe_projections_for_trials(
                trace_db=trace_db,
                run_id=run_id,
                probe_id=str(social_probe_id),
                probe_artifact_path=str(social["artifact_path"]),
                model_id=str(model_id),
                component=str(component),
                layers=list(layers),
            )

    turn_layers = detect_turn_layers(
        trace_db=trace_db,
        run_id=run_id,
        truth_probe_id=str(truth["probe_id"]),
        social_probe_id=social_probe_id,
        layers=list(layers),
    )

    artifacts_dir = os.path.join(str(run_dir), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    plots = generate_vector_collision_plots(
        trace_db=trace_db,
        run_id=run_id,
        truth_probe_id=str(truth["probe_id"]),
        social_probe_id=social_probe_id,
        layers=list(layers),
        output_dir=str(artifacts_dir),
    )

    return {
        "truth_probe_id": str(truth["probe_id"]),
        "social_probe_id": social_probe_id,
        "truth_projections_inserted": int(truth_inserted),
        "social_projections_inserted": int(social_inserted),
        "turn_layers": turn_layers,
        "plots": plots,
    }

