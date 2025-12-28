from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aam.interpretability import CaptureConfig, CaptureContext
from aam.llm_gateway import select_local_gateway
from aam.persistence import TraceDb

from .io import deterministic_prompt_hash, read_jsonl, sha256_file
from .prompts import build_messages


JsonDict = Dict[str, Any]


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Probe pipeline requires numpy.") from e
    return np


def _require_torch_and_safetensors() -> Any:
    try:
        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Probe pipeline requires torch + safetensors. Install extras: `pip install -e .[interpretability]`") from e
    return torch, save_file


@dataclass(frozen=True)
class ProbeCaptureSpec:
    model_id: str
    layers: List[int]
    component: str
    token_position: int = -1
    dtype: str = "float16"
    agent_id: str = "probe_agent"


def _sigmoid(z: Any) -> Any:
    np = _require_numpy()
    return 1.0 / (1.0 + np.exp(-z))


def _train_logreg_l2(
    *,
    X: Any,
    y: Any,
    l2: float = 1e-3,
    lr: float = 0.1,
    steps: int = 300,
) -> Tuple[Any, float]:
    """
    Minimal logistic regression with L2 via gradient descent.
    Returns (w, b).
    """
    np = _require_numpy()
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(int(steps)):
        logits = X @ w + b
        p = _sigmoid(logits)
        # gradients
        gw = (X.T @ (p - y)) / n + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, float(b)


def _accuracy(*, X: Any, y: Any, w: Any, b: float) -> float:
    np = _require_numpy()
    p = _sigmoid(X @ w + b)
    pred = (p >= 0.5).astype(np.int64)
    return float((pred == y).mean())


def capture_probe_dataset_to_db(
    *,
    trace_db: TraceDb,
    run_id: str,
    dataset_name: str,
    dataset_version: str,
    dataset_path: str,
    capture: ProbeCaptureSpec,
    system_prompt: str,
    condition_name: str = "probe_capture",
) -> str:
    """
    Runs a TL model over labeled statements, captures activations, and registers:
      - conformity_datasets/items
      - conformity_conditions
      - conformity_trials/prompts/outputs
      - conformity_trial_steps (for activation alignment)
      - activation_metadata + activations/*.safetensors via CaptureContext
    Returns dataset_id.
    """
    repo_root = os.getcwd()
    abs_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(repo_root, dataset_path)
    items = read_jsonl(abs_path)
    dataset_id = str(uuid.uuid4())
    trace_db.upsert_conformity_dataset(
        dataset_id=dataset_id,
        name=dataset_name,
        version=dataset_version,
        path=dataset_path,
        sha256=sha256_file(abs_path),
    )

    condition_id = str(uuid.uuid4())
    trace_db.upsert_conformity_condition(condition_id=condition_id, name=condition_name, params={"type": "probe_capture"})

    # CaptureContext expects an output directory; use run dir co-located with DB.
    db_dir = os.path.dirname(trace_db._config.db_path)  # type: ignore[attr-defined]
    activations_dir = os.path.join(db_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    cap_cfg = CaptureConfig(
        layers=list(capture.layers),
        components=[str(capture.component)],
        trigger_actions=["probe_capture"],
        token_position=int(capture.token_position),
    )
    cap_ctx = CaptureContext(output_dir=activations_dir, config=cap_cfg, dtype=str(capture.dtype), trace_db=trace_db)
    model_id_str = str(capture.model_id)
    gateway = select_local_gateway(model_id_or_path=model_id_str, capture_context=cap_ctx)
    variant = "huggingface" if "olmo" in model_id_str.lower() else "transformerlens"

    # IMPORTANT:
    # Do NOT reuse trial time_step indices; the main behavioral runner already populated
    # steps for this run. If we reuse low indices here, we will overwrite activation shards
    # (same filename) and corrupt downstream projection / logit-lens computations.
    base_ts_row = trace_db.conn.execute(
        """
        SELECT COALESCE(MAX(s.time_step), -1) AS max_ts
        FROM conformity_trial_steps s
        JOIN conformity_trials t ON t.trial_id = s.trial_id
        WHERE t.run_id = ?;
        """,
        (run_id,),
    ).fetchone()
    base_ts = int(base_ts_row["max_ts"]) + 1 if base_ts_row is not None else 0

    # Deterministic (within this capture): time_step = base offset + item index
    for i, it in enumerate(items):
        time_step = int(base_ts + i)
        item_id = str(it.get("item_id") or f"{dataset_name}_{i:06d}")
        label = it.get("label")
        
        # Labels are required for probe training, but we allow capturing activations
        # even if labels are missing (items without labels will be skipped during training)
        if label not in (0, 1, True, False):
            # For social probes, items might not have labels yet - allow capture but warn
            if "social" in dataset_name.lower():
                print(f"  Warning: Item {item_id} missing label - will be skipped during probe training")
                label = None
            else:
                raise ValueError(f"Probe dataset item missing label 0/1: item_id={item_id}")

        trace_db.insert_conformity_item(
            item_id=item_id,
            dataset_id=dataset_id,
            domain=str(it.get("domain") or "probe"),
            question=str(it.get("text") or it.get("question") or ""),
            ground_truth_text=None,
            ground_truth_json={"label": int(bool(label))} if label is not None else None,
            source_json=(it.get("source") if isinstance(it.get("source"), dict) else None),
        )

        trial_id = str(uuid.uuid4())
        trace_db.insert_conformity_trial(
            trial_id=trial_id,
            run_id=run_id,
            model_id=str(capture.model_id),
            variant=variant,
            item_id=item_id,
            condition_id=condition_id,
            seed=0,
            temperature=0.0,
        )
        trace_db.upsert_conformity_trial_step(trial_id=trial_id, time_step=time_step, agent_id=str(capture.agent_id))

        user_prompt = str(it.get("text") or it.get("question") or "")
        history: List[JsonDict] = []
        phash = deterministic_prompt_hash(system=system_prompt, user=user_prompt, history=history)
        trace_db.insert_conformity_prompt(
            prompt_id=str(uuid.uuid4()),
            trial_id=trial_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_history=history,
            rendered_prompt_hash=phash,
        )

        # Run model; activations buffered by hooks
        msgs = build_messages(system=system_prompt, user=user_prompt, history=history)
        cap_ctx.begin_inference()
        t0 = time.time()
        resp = gateway.chat(model=str(capture.model_id), messages=msgs, tools=None, tool_choice=None, temperature=0.0)
        latency_ms = (time.time() - t0) * 1000.0

        # Commit activations and flush to shard file aligned to time_step
        cap_ctx.on_action_decided(
            run_id=run_id,
            time_step=time_step,
            agent_id=str(capture.agent_id),
            model_id=str(capture.model_id),
            action_name="probe_capture",
        )
        cap_ctx.flush_step(time_step=time_step)

        raw_text = ""
        try:
            raw_text = str(resp["choices"][0]["message"].get("content") or "")
        except Exception:
            raw_text = str(resp)
        trace_db.insert_conformity_output(
            output_id=str(uuid.uuid4()),
            trial_id=trial_id,
            raw_text=raw_text,
            parsed_answer_text=None,
            parsed_answer_json=None,
            is_correct=None,
            refusal_flag=False,
            latency_ms=latency_ms,
            token_usage_json=None,
        )

    return dataset_id


def train_probe_from_captured_activations(
    *,
    trace_db: TraceDb,
    run_id: str,
    train_dataset_id: str,
    model_id: str,
    probe_kind: str,
    layers: List[int],
    component: str,
    token_position: int,
    output_artifact_path: str,
) -> str:
    """
    Trains per-layer logistic probes on captured activations from the probe dataset.
    Saves weights to a safetensors file and registers a row in conformity_probes.
    Returns probe_id.
    """
    np = _require_numpy()
    torch, save_file = _require_torch_and_safetensors()

    # Pull labeled items
    items = trace_db.conn.execute(
        """
        SELECT item_id, question, ground_truth_json
        FROM conformity_items
        WHERE dataset_id = ?
        ORDER BY item_id ASC;
        """,
        (train_dataset_id,),
    ).fetchall()
    if not items:
        raise ValueError(f"No conformity_items found for dataset_id={train_dataset_id}")

    # Build label map
    import json as _json

    labels_by_item: Dict[str, int] = {}
    for r in items:
        gj = r["ground_truth_json"]
        if not gj:
            continue
        try:
            parsed = _json.loads(gj)
            lab = parsed.get("label")
            if lab is None:
                continue
            labels_by_item[str(r["item_id"])] = 1 if int(lab) else 0
        except Exception:
            continue
    
    if not labels_by_item:
        raise ValueError(
            f"No labeled items found in dataset {train_dataset_id}. "
            f"Probe training requires items with 'label' field (0 or 1) in ground_truth_json. "
            f"Found {len(items)} items, but none had valid labels."
        )

    # Link trials to time_step/agent_id for activation alignment
    trial_rows = trace_db.conn.execute(
        """
        SELECT t.trial_id, t.item_id, s.time_step, s.agent_id
        FROM conformity_trials t
        JOIN conformity_trial_steps s ON s.trial_id = t.trial_id
        WHERE t.run_id = ? AND t.item_id IN (SELECT item_id FROM conformity_items WHERE dataset_id = ?)
        ORDER BY s.time_step ASC;
        """,
        (run_id, train_dataset_id),
    ).fetchall()
    if not trial_rows:
        raise ValueError("No captured trials found for probe training (missing conformity_trial_steps join).")

    # Load safetensors vectors for each layer into X, y
    from safetensors.torch import load_file  # type: ignore

    metrics: Dict[str, Any] = {"layers": list(layers), "component": component, "token_position": int(token_position)}
    tensors_to_save: Dict[str, Any] = {}

    for layer in layers:
        X_list: List[Any] = []
        y_list: List[int] = []
        for tr in trial_rows:
            item_id = str(tr["item_id"])
            y = labels_by_item.get(item_id)
            if y is None:
                continue
            ts = int(tr["time_step"])
            agent_id = str(tr["agent_id"])

            # Find the activation record for this (run, step, agent, layer, component)
            rec = trace_db.conn.execute(
                """
                SELECT shard_file_path, tensor_key
                FROM activation_metadata
                WHERE run_id = ? AND time_step = ? AND agent_id = ? AND model_id = ? AND layer_index = ? AND component = ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (run_id, ts, agent_id, model_id, int(layer), str(component)),
            ).fetchone()
            if rec is None:
                continue
            shard = str(rec["shard_file_path"])
            key = str(rec["tensor_key"])
            buf = load_file(shard)
            vec_t = buf[key]
            vec = vec_t.detach().cpu().to(torch.float32).numpy()
            X_list.append(vec)
            y_list.append(int(y))

        if len(X_list) < 10:
            raise RuntimeError(f"Not enough training examples with activations for layer={layer} (got {len(X_list)})")

        X = np.stack(X_list, axis=0).astype(np.float64)
        y_arr = np.array(y_list, dtype=np.int64)

        w, b = _train_logreg_l2(X=X, y=y_arr, l2=1e-3, lr=0.1, steps=400)
        acc = _accuracy(X=X, y=y_arr, w=w, b=b)
        metrics[f"train_acc_layer_{layer}"] = acc

        tensors_to_save[f"layer_{layer}.weight"] = torch.tensor(w, dtype=torch.float32)
        tensors_to_save[f"layer_{layer}.bias"] = torch.tensor([b], dtype=torch.float32)

    os.makedirs(os.path.dirname(output_artifact_path), exist_ok=True)
    save_file(tensors_to_save, output_artifact_path)

    probe_id = str(uuid.uuid4())
    trace_db.insert_conformity_probe(
        probe_id=probe_id,
        run_id=run_id,
        probe_kind=probe_kind,
        train_dataset_id=train_dataset_id,
        model_id=model_id,
        layers=layers,
        component=component,
        token_position=token_position,
        artifact_path=output_artifact_path,
        metrics=metrics,
    )
    return probe_id


def compute_and_store_probe_projections_for_trials(
    *,
    trace_db: TraceDb,
    run_id: str,
    probe_id: str,
    probe_artifact_path: str,
    model_id: str,
    component: str,
    layers: List[int],
) -> int:
    """
    Computes scalar projections for each (trial, layer) based on the probe weights.
    Requires activation_metadata aligned via conformity_trial_steps.
    Returns number of projection rows inserted.
    """
    torch, _ = _require_torch_and_safetensors()
    from safetensors.torch import load_file  # type: ignore

    weights = load_file(probe_artifact_path)

    trials = trace_db.conn.execute(
        """
        SELECT t.trial_id, s.time_step, s.agent_id
        FROM conformity_trials t
        JOIN conformity_trial_steps s ON s.trial_id = t.trial_id
        WHERE t.run_id = ?
        ORDER BY s.time_step ASC;
        """,
        (run_id,),
    ).fetchall()
    if not trials:
        return 0

    rows_to_insert: List[Tuple[str, str, str, int, Optional[int], float]] = []

    for tr in trials:
        trial_id = str(tr["trial_id"])
        ts = int(tr["time_step"])
        agent_id = str(tr["agent_id"])

        for layer in layers:
            w = weights.get(f"layer_{layer}.weight")
            b = weights.get(f"layer_{layer}.bias")
            if w is None or b is None:
                continue

            rec = trace_db.conn.execute(
                """
                SELECT shard_file_path, tensor_key
                FROM activation_metadata
                WHERE run_id = ? AND time_step = ? AND agent_id = ? AND model_id = ? AND layer_index = ? AND component = ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (run_id, ts, agent_id, model_id, int(layer), str(component)),
            ).fetchone()
            if rec is None:
                continue

            buf = load_file(str(rec["shard_file_path"]))
            vec = buf[str(rec["tensor_key"])].detach().cpu().to(torch.float32)
            score = float((vec * w).sum().item() + float(b[0].item()))
            rows_to_insert.append((str(uuid.uuid4()), trial_id, probe_id, int(layer), None, score))

    if rows_to_insert:
        trace_db.insert_conformity_projection_rows(rows=rows_to_insert)
    return len(rows_to_insert)


