"""
Contrastive Vector Steering (Representation Engineering / CAA).

Computes a "deference vector" from the difference in residual-stream
activations between Authority (sycophantic) and Control (truthful) trials,
then causally tests whether injecting that vector into Control trials can
induce sycophantic behaviour.

Grounding:
  [R3] Turner et al., "Steering Language Models With Activation Engineering"
       (2023, arXiv:2308.10248)
  [R4] Panickssery et al., "Steering Llama 2 via Contrastive Activation
       Addition" (ACL 2024, arXiv:2312.06681)
       - Layer 15 for Llama-2-13B, layer 13 for 7B; multiplier +/-2
       - Steering vector added at all token positions after user prompt
       - Normalised to unit norm before scaling by alpha
  [R5] Zou et al., "Representation Engineering" (2023, arXiv:2310.01405)
  [R6] Hao et al., "Patterns and Mechanisms of Contrastive Activation
       Engineering" (ICLR 2025 Workshops, arXiv:2505.03189)
       - Diminishing returns at ~80 sample pairs
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from vivarium.persistence import TraceDb
from vivarium.llm_gateway import HuggingFaceHookedGateway
from vivarium.output_parsing import OutputParsingConfig, classify_output

from .enhanced_scoring import score_single_output

JsonDict = Dict[str, Any]

MAX_PAIRS_CAP = 80  # [R6] diminishing returns beyond ~80 pairs


def _require_torch_and_safetensors() -> Any:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Contrastive steering requires torch + safetensors. "
            "Install extras: `pip install -e .[interpretability]`"
        ) from e
    return torch, load_file, save_file


def _load_trial_messages(*, trace_db: TraceDb, trial_id: str) -> Tuple[List[JsonDict], str, str]:
    row = trace_db.conn.execute(
        """
        SELECT p.system_prompt, p.user_prompt, p.chat_history_json
        FROM conformity_prompts p
        WHERE p.trial_id = ?
        ORDER BY p.created_at ASC LIMIT 1;
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
    system_prompt = str(row["system_prompt"])
    user_prompt = str(row["user_prompt"])
    messages: List[JsonDict] = [{"role": "system", "content": system_prompt}]
    for m in history:
        if isinstance(m, dict):
            messages.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": user_prompt})
    return messages, system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Step 1: Compute deference vector
# ---------------------------------------------------------------------------

def compute_deference_vector(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    layers: List[int],
    component: str = "hook_resid_post",
    output_dir: str,
    min_pairs: int = 50,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute v_deference[L] = mean(h_L^authority) - mean(h_L^control)
    for each requested layer.

    Uses captured activations from ``activation_metadata`` / safetensors shards.
    Saves per-layer vectors as safetensors files under *output_dir*.

    Returns dict with ``vector_paths`` (layer -> path), ``n_pairs``, and metadata.
    """
    torch, load_file, save_file = _require_torch_and_safetensors()

    vec_dir = os.path.join(output_dir, "deference_vectors")
    os.makedirs(vec_dir, exist_ok=True)

    variant_filter = "AND t.variant = ?" if variant else ""
    base_params: list = [run_id, model_id]
    if variant:
        base_params.append(variant)

    # Find matching (item_id) pairs of Control vs Authority trials.
    pairs = trace_db.conn.execute(
        f"""
        SELECT
          ctrl.trial_id AS control_trial_id,
          auth.trial_id AS authority_trial_id,
          ctrl.item_id
        FROM conformity_trials ctrl
        JOIN conformity_conditions cc ON cc.condition_id = ctrl.condition_id
        JOIN conformity_trials auth ON auth.item_id = ctrl.item_id AND auth.run_id = ctrl.run_id AND auth.model_id = ctrl.model_id
        JOIN conformity_conditions ca ON ca.condition_id = auth.condition_id
        WHERE ctrl.run_id = ? AND ctrl.model_id = ? {variant_filter}
          AND cc.name = 'control'
          AND ca.name IN ('authoritative_bias', 'asch_history_5')
        ORDER BY ctrl.created_at ASC
        LIMIT ?;
        """,
        (*base_params, MAX_PAIRS_CAP),
    ).fetchall()

    if len(pairs) < min_pairs:
        return {
            "skipped": True,
            "reason": f"Only {len(pairs)} pairs available (need >= {min_pairs})",
            "n_pairs": len(pairs),
            "vector_paths": {},
        }

    vector_paths: Dict[int, str] = {}
    layer_stats: Dict[int, Dict[str, int]] = {}

    for layer in layers:
        ctrl_vecs: List[Any] = []
        auth_vecs: List[Any] = []

        for pair in pairs:
            ctrl_tid = str(pair["control_trial_id"])
            auth_tid = str(pair["authority_trial_id"])

            for tid, collector in [(ctrl_tid, ctrl_vecs), (auth_tid, auth_vecs)]:
                step_row = trace_db.conn.execute(
                    "SELECT time_step, agent_id FROM conformity_trial_steps WHERE trial_id = ? LIMIT 1;",
                    (tid,),
                ).fetchone()
                if step_row is None:
                    continue
                rec = trace_db.conn.execute(
                    """
                    SELECT shard_file_path, tensor_key
                    FROM activation_metadata
                    WHERE run_id = ? AND time_step = ? AND agent_id = ? AND layer_index = ? AND component = ?
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (run_id, int(step_row["time_step"]), str(step_row["agent_id"]), int(layer), component),
                ).fetchone()
                if rec is None:
                    continue
                try:
                    tensors = load_file(str(rec["shard_file_path"]))
                    vec = tensors[str(rec["tensor_key"])].to(torch.float32)
                    collector.append(vec)
                except Exception:
                    continue

        n_usable = min(len(ctrl_vecs), len(auth_vecs))
        layer_stats[layer] = {"n_ctrl": len(ctrl_vecs), "n_auth": len(auth_vecs), "n_usable": n_usable}

        if n_usable == 0:
            continue

        ctrl_mean = torch.stack(ctrl_vecs[:n_usable]).mean(dim=0)
        auth_mean = torch.stack(auth_vecs[:n_usable]).mean(dim=0)
        v_deference = auth_mean - ctrl_mean

        # Normalise to unit norm per [R4]
        norm = torch.norm(v_deference) + 1e-8
        v_deference = v_deference / norm

        vec_path = os.path.join(vec_dir, f"layer_{layer}.safetensors")
        save_file({"deference_vector": v_deference}, vec_path)
        vector_paths[layer] = vec_path

    return {
        "skipped": False,
        "n_pairs": len(pairs),
        "vector_paths": vector_paths,
        "layer_stats": layer_stats,
    }


# ---------------------------------------------------------------------------
# Step 2: Run causal steering test
# ---------------------------------------------------------------------------

def run_contrastive_steering_test(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    vector_paths: Dict[int, str],
    alpha_values: List[float],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    variant: Optional[str] = None,
    trial_filter_sql: Optional[str] = None,
) -> int:
    """
    For each Control trial (model answers truthfully), inject
    alpha * v_deference at the specified layer and check whether
    the model flips to the sycophantic answer.

    Following [R4], the steering vector is injected at a single layer
    per sweep but at ALL token positions during generation (the hook
    fires on every forward pass through that layer).

    Writes to ``conformity_contrastive_steering`` and ``conformity_outputs``.
    Returns total result rows inserted.
    """
    torch, load_file, _ = _require_torch_and_safetensors()

    gateway = HuggingFaceHookedGateway(
        model_id_or_path=model_id, capture_context=None, max_new_tokens=max_new_tokens,
    )
    output_parse_cfg = OutputParsingConfig()

    variant_filter = "AND t.variant = ?" if variant else ""
    base_params: list = [run_id]
    if variant:
        base_params.append(variant)

    base_query = f"""
        SELECT t.trial_id, i.ground_truth_text, t.seed
        FROM conformity_trials t
        JOIN conformity_items i ON i.item_id = t.item_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ? {variant_filter}
          AND c.name = 'control'
          AND i.ground_truth_text IS NOT NULL
    """
    if trial_filter_sql:
        base_query += f" AND ({trial_filter_sql})"
    base_query += " ORDER BY t.created_at ASC;"
    trials = trace_db.conn.execute(base_query, tuple(base_params)).fetchall()
    if not trials:
        return 0

    # Pre-load deference vectors
    vecs_by_layer: Dict[int, Any] = {}
    for layer, path in vector_paths.items():
        try:
            data = load_file(str(path))
            vecs_by_layer[int(layer)] = data["deference_vector"].to(torch.float32)
        except Exception:
            continue

    inserted = 0
    now = time.time()

    for layer_idx, v_def in vecs_by_layer.items():
        for alpha in alpha_values:
            for tr in trials:
                trial_id = str(tr["trial_id"])
                ground_truth = str(tr["ground_truth_text"]) if tr["ground_truth_text"] is not None else None
                seed = int(tr["seed"]) if tr["seed"] is not None else None
                messages, system_prompt, user_prompt = _load_trial_messages(trace_db=trace_db, trial_id=trial_id)

                # --- baseline (no steering) ---
                resp_before = gateway.chat(
                    model=model_id, messages=messages, tools=None, tool_choice=None,
                    temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                )
                text_before = ""
                try:
                    text_before = str(resp_before["choices"][0]["message"].get("content") or "")
                except Exception:
                    text_before = str(resp_before)
                classified_before = classify_output(
                    raw_text=text_before, cfg=output_parse_cfg,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    expected_answer_texts=([ground_truth] if ground_truth else []),
                    token_logprobs=None,
                )
                sr_before = score_single_output(
                    raw_text=text_before, ground_truth_text=ground_truth,
                    wrong_answer=None, condition_name="unknown", dataset_name="unknown",
                )

                output_before_id = str(uuid.uuid4())
                trace_db.insert_conformity_output(
                    output_id=output_before_id, trial_id=trial_id, raw_text=text_before,
                    parsed_answer_text=sr_before.parsed_answer_text, parsed_answer_json=None,
                    is_correct=sr_before.is_correct, refusal_flag=sr_before.refusal_flag,
                    latency_ms=0.0,
                    token_usage_json={"_output_quality": {"label": classified_before.label.value, "metadata": classified_before.metadata}},
                    created_at=now,
                )

                # --- steered generation ---
                def _make_steering_hook(vec: Any, a: float) -> Any:
                    def _hook(_module: Any, _inp: Any, out: Any) -> Any:
                        try:
                            hs = out[0] if isinstance(out, (tuple, list)) else out
                            patched = hs + (a * vec)[None, None, :]
                            if isinstance(out, tuple):
                                return (patched,) + tuple(out[1:])
                            if isinstance(out, list):
                                return [patched] + list(out[1:])
                            return patched
                        except Exception:
                            return out
                    return _hook

                handle = gateway.register_intervention_hook(
                    layer_idx=layer_idx,
                    hook_fn=_make_steering_hook(v_def, float(alpha)),
                )

                try:
                    resp_after = gateway.chat(
                        model=model_id, messages=messages, tools=None, tool_choice=None,
                        temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                    )
                finally:
                    try:
                        handle.remove()
                    except Exception:
                        pass

                text_after = ""
                try:
                    text_after = str(resp_after["choices"][0]["message"].get("content") or "")
                except Exception:
                    text_after = str(resp_after)
                classified_after = classify_output(
                    raw_text=text_after, cfg=output_parse_cfg,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    expected_answer_texts=([ground_truth] if ground_truth else []),
                    token_logprobs=None,
                )
                sr_after = score_single_output(
                    raw_text=text_after, ground_truth_text=ground_truth,
                    wrong_answer=None, condition_name="unknown", dataset_name="unknown",
                )

                output_after_id = str(uuid.uuid4())
                trace_db.insert_conformity_output(
                    output_id=output_after_id, trial_id=trial_id, raw_text=text_after,
                    parsed_answer_text=sr_after.parsed_answer_text, parsed_answer_json=None,
                    is_correct=sr_after.is_correct, refusal_flag=sr_after.refusal_flag,
                    latency_ms=0.0,
                    token_usage_json={"_output_quality": {"label": classified_after.label.value, "metadata": classified_after.metadata}},
                    created_at=now,
                )

                flipped: Optional[bool] = None
                if sr_before.is_correct is not None and sr_after.is_correct is not None:
                    flipped = bool(sr_before.is_correct) and (not bool(sr_after.is_correct))

                trace_db.conn.execute(
                    """
                    INSERT INTO conformity_contrastive_steering(
                      steering_id, run_id, layer_index, alpha, trial_id,
                      output_id_before, output_id_after,
                      flipped_to_sycophantic, deference_vector_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(uuid.uuid4()), run_id, int(layer_idx), float(alpha), trial_id,
                        output_before_id, output_after_id,
                        (1 if flipped else 0) if flipped is not None else None,
                        str(vector_paths.get(layer_idx, "")),
                        now,
                    ),
                )
                inserted += 1

    trace_db.conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Step 3: Visualisation
# ---------------------------------------------------------------------------

def plot_contrastive_steering_results(
    *,
    trace_db: TraceDb,
    run_id: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate contrastive steering visualisations:
      1. Dose-response curve (flip-to-sycophantic rate vs alpha)
      2. Layer-sweep heatmap (flip rate by layer x alpha)
      3. Negative-alpha sanity check

    Returns dict mapping plot name -> file path.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        return {}

    interp_dir = os.path.join(output_dir, "interpretability", "contrastive_steering")
    os.makedirs(interp_dir, exist_ok=True)

    rows = trace_db.conn.execute(
        """
        SELECT layer_index, alpha, flipped_to_sycophantic
        FROM conformity_contrastive_steering
        WHERE run_id = ? AND flipped_to_sycophantic IS NOT NULL;
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    plots: Dict[str, str] = {}

    # --- 1. Dose-response curve (aggregate across layers) ---
    dose = df.groupby("alpha").agg(
        flip_rate=("flipped_to_sycophantic", "mean"),
        n=("flipped_to_sycophantic", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dose["alpha"], dose["flip_rate"], "o-", linewidth=2, color="#9C27B0")
    ax.set_xlabel("Steering Coefficient (alpha)")
    ax.set_ylabel("Flip-to-Sycophantic Rate")
    ax.set_title("Contrastive Steering Dose-Response")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(interp_dir, "steering_flip_rates.png")
    plt.savefig(p, dpi=150)
    plt.close()
    plots["steering_flip_rates"] = p

    # --- 2. Layer x alpha heatmap ---
    pivot = df.pivot_table(
        index="layer_index", columns="alpha", values="flipped_to_sycophantic", aggfunc="mean",
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{a:g}" for a in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(l) for l in pivot.index])
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Layer Index")
        ax.set_title("Steering Flip Rate by Layer and Alpha")
        plt.colorbar(im, ax=ax, label="Flip-to-Sycophantic Rate")
        plt.tight_layout()
        p = os.path.join(interp_dir, "steering_layer_heatmap.png")
        plt.savefig(p, dpi=150)
        plt.close()
        plots["steering_layer_heatmap"] = p

    # --- 3. Negative-alpha sanity check ---
    neg = df[df["alpha"] < 0]
    pos = df[df["alpha"] > 0]
    if not neg.empty and not pos.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        neg_rate = neg.groupby("alpha")["flipped_to_sycophantic"].mean()
        pos_rate = pos.groupby("alpha")["flipped_to_sycophantic"].mean()
        ax.bar(
            [f"neg a={a:g}" for a in neg_rate.index],
            neg_rate.values, color="#4CAF50", label="Negative alpha (should be low)",
        )
        ax.bar(
            [f"pos a={a:g}" for a in pos_rate.index],
            pos_rate.values, color="#F44336", label="Positive alpha",
        )
        ax.set_ylabel("Flip-to-Sycophantic Rate")
        ax.set_title("Sanity Check: Negative vs Positive Alpha")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p = os.path.join(interp_dir, "steering_sanity_negative.png")
        plt.savefig(p, dpi=150)
        plt.close()
        plots["steering_sanity_negative"] = p

    # --- CSV export ---
    csv_path = os.path.join(interp_dir, "steering_results.csv")
    df.to_csv(csv_path, index=False)
    plots["steering_csv"] = csv_path

    return plots
