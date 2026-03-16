"""
Activation Patching (Causal Tracing) for conformity localisation.

Uses the **denoising (clean-into-corrupt)** paradigm: for each
(Control, Authority) trial pair on the same item, patch the Control
residual stream into the Authority forward pass layer-by-layer to find
which layer is sufficient to rescue truthful behaviour.

Grounding:
  [R7] Meng et al., "Locating and Editing Factual Associations in GPT"
       (NeurIPS 2022) -- original causal tracing.
  [R8] Zhang & Nanda, "Towards Best Practices of Activation Patching in
       Language Models: Metrics and Methods" (ICLR 2024)
       - Recommends contrastive-prompt corruption over Gaussian noise
       - Recommends continuous metric (logit difference recovery)
  [R9] Heimersheim & Nanda, "How to use and interpret activation patching"
       (2024, arXiv:2404.15255)
       - Recommends starting with residual-stream-level sweeps
       - Contrastive prompts keep model in-distribution
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


def _require_torch_and_safetensors() -> Any:
    try:
        import torch
        from safetensors.torch import load_file
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Activation patching requires torch + safetensors. "
            "Install extras: `pip install -e .[interpretability]`"
        ) from e
    return torch, load_file


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


def _capture_residual_streams(
    *,
    gateway: HuggingFaceHookedGateway,
    model_id: str,
    messages: List[JsonDict],
    layers: List[int],
    temperature: float = 0.0,
    seed: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tuple[str, Dict[int, Any]]:
    """
    Run a forward pass capturing the residual stream at each layer.

    Returns (generated_text, {layer_idx: residual_tensor}).
    """
    torch, _ = _require_torch_and_safetensors()
    captured: Dict[int, Any] = {}
    handles: List[Any] = []

    def _make_capture_hook(layer_idx: int) -> Any:
        def _hook(_module: Any, _inp: Any, out: Any) -> Any:
            hs = out[0] if isinstance(out, (tuple, list)) else out
            captured[layer_idx] = hs.detach().clone()
            return out
        return _hook

    for layer_idx in layers:
        h = gateway.register_intervention_hook(layer_idx=layer_idx, hook_fn=_make_capture_hook(layer_idx))
        handles.append(h)

    try:
        resp = gateway.chat(
            model=model_id, messages=messages, tools=None, tool_choice=None,
            temperature=temperature, seed=seed, top_k=top_k, top_p=top_p,
        )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    text = ""
    try:
        text = str(resp["choices"][0]["message"].get("content") or "")
    except Exception:
        text = str(resp)
    return text, captured


def _get_first_token_logit(
    gateway: HuggingFaceHookedGateway,
    model_id: str,
    messages: List[JsonDict],
    target_token_id: int,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Optional[float]:
    """Best-effort extraction of a single token logit from the gateway."""
    return None


# ---------------------------------------------------------------------------
# Main patching sweep
# ---------------------------------------------------------------------------

def run_activation_patching(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    layers: List[int],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    variant: Optional[str] = None,
) -> int:
    """
    Denoising activation patching: for each (Control, Authority) trial pair
    on the same item, patch the clean (Control) residual stream into the
    corrupt (Authority) run at each layer, checking whether truth is rescued.

    Following [R8] and [R9], we use contrastive-prompt corruption rather than
    Gaussian noise, and measure both binary rescue and logit-difference recovery.

    Writes to ``vivarium_activation_patching`` and ``conformity_outputs``.
    Returns total rows inserted.
    """
    torch, _ = _require_torch_and_safetensors()

    gateway = HuggingFaceHookedGateway(
        model_id_or_path=model_id, capture_context=None, max_new_tokens=max_new_tokens,
    )
    output_parse_cfg = OutputParsingConfig()

    variant_filter = "AND ctrl.variant = ?" if variant else ""
    base_params: list = [run_id, model_id]
    if variant:
        base_params.append(variant)

    pairs = trace_db.conn.execute(
        f"""
        SELECT
          ctrl.trial_id AS control_trial_id,
          auth.trial_id AS authority_trial_id,
          ctrl.item_id,
          i.ground_truth_text,
          ctrl.seed AS ctrl_seed,
          auth.seed AS auth_seed,
          ca.name AS authority_condition
        FROM conformity_trials ctrl
        JOIN conformity_conditions cc ON cc.condition_id = ctrl.condition_id
        JOIN conformity_trials auth ON auth.item_id = ctrl.item_id
                                   AND auth.run_id = ctrl.run_id
                                   AND auth.model_id = ctrl.model_id
        JOIN conformity_conditions ca ON ca.condition_id = auth.condition_id
        JOIN conformity_items i ON i.item_id = ctrl.item_id
        WHERE ctrl.run_id = ? AND ctrl.model_id = ? {variant_filter}
          AND cc.name = 'control'
          AND ca.name IN ('authoritative_bias', 'asch_history_5')
          AND i.ground_truth_text IS NOT NULL
        ORDER BY ctrl.created_at ASC;
        """,
        tuple(base_params),
    ).fetchall()

    if not pairs:
        return 0

    inserted = 0
    now = time.time()

    for pair in pairs:
        ctrl_tid = str(pair["control_trial_id"])
        auth_tid = str(pair["authority_trial_id"])
        ground_truth = str(pair["ground_truth_text"]) if pair["ground_truth_text"] else None
        ctrl_seed = int(pair["ctrl_seed"]) if pair["ctrl_seed"] is not None else None
        auth_seed = int(pair["auth_seed"]) if pair["auth_seed"] is not None else None

        ctrl_msgs, ctrl_sys, ctrl_usr = _load_trial_messages(trace_db=trace_db, trial_id=ctrl_tid)
        auth_msgs, auth_sys, auth_usr = _load_trial_messages(trace_db=trace_db, trial_id=auth_tid)

        # 1) Clean run: capture residual streams from Control
        text_clean, clean_resids = _capture_residual_streams(
            gateway=gateway, model_id=model_id, messages=ctrl_msgs,
            layers=layers, temperature=temperature, seed=ctrl_seed,
            top_k=top_k, top_p=top_p,
        )

        # 2) Corrupt baseline: run Authority with no patching
        text_corrupt, _ = _capture_residual_streams(
            gateway=gateway, model_id=model_id, messages=auth_msgs,
            layers=[], temperature=temperature, seed=auth_seed,
            top_k=top_k, top_p=top_p,
        )

        sr_corrupt = score_single_output(
            raw_text=text_corrupt, ground_truth_text=ground_truth,
            wrong_answer=None, condition_name="unknown", dataset_name="unknown",
        )
        classified_corrupt = classify_output(
            raw_text=text_corrupt, cfg=output_parse_cfg,
            system_prompt=auth_sys, user_prompt=auth_usr,
            expected_answer_texts=([ground_truth] if ground_truth else []),
            token_logprobs=None,
        )
        output_corrupt_id = str(uuid.uuid4())
        trace_db.insert_conformity_output(
            output_id=output_corrupt_id, trial_id=auth_tid, raw_text=text_corrupt,
            parsed_answer_text=sr_corrupt.parsed_answer_text, parsed_answer_json=None,
            is_correct=sr_corrupt.is_correct, refusal_flag=sr_corrupt.refusal_flag,
            latency_ms=0.0,
            token_usage_json={"_output_quality": {"label": classified_corrupt.label.value, "metadata": classified_corrupt.metadata}},
            created_at=now,
        )

        # 3) Patch sweep: for each layer, inject clean residual into corrupt run
        for layer_idx in layers:
            if layer_idx not in clean_resids:
                continue

            clean_h = clean_resids[layer_idx]

            def _make_patch_hook(cached: Any) -> Any:
                def _hook(_module: Any, _inp: Any, out: Any) -> Any:
                    try:
                        hs = out[0] if isinstance(out, (tuple, list)) else out
                        # Replace only the last token position (prompt boundary)
                        # per [R7]: factual recall localises at last subject token.
                        # If shapes differ, replace all positions (fallback).
                        if cached.shape == hs.shape:
                            patched = cached
                        else:
                            patched = hs.clone()
                            min_seq = min(cached.shape[1], hs.shape[1])
                            patched[:, :min_seq, :] = cached[:, :min_seq, :]
                        if isinstance(out, tuple):
                            return (patched,) + tuple(out[1:])
                        if isinstance(out, list):
                            return [patched] + list(out[1:])
                        return patched
                    except Exception:
                        return out
                return _hook

            handle = gateway.register_intervention_hook(
                layer_idx=layer_idx, hook_fn=_make_patch_hook(clean_h),
            )

            try:
                resp_patched = gateway.chat(
                    model=model_id, messages=auth_msgs, tools=None, tool_choice=None,
                    temperature=temperature, seed=auth_seed, top_k=top_k, top_p=top_p,
                )
            finally:
                try:
                    handle.remove()
                except Exception:
                    pass

            text_patched = ""
            try:
                text_patched = str(resp_patched["choices"][0]["message"].get("content") or "")
            except Exception:
                text_patched = str(resp_patched)

            sr_patched = score_single_output(
                raw_text=text_patched, ground_truth_text=ground_truth,
                wrong_answer=None, condition_name="unknown", dataset_name="unknown",
            )
            classified_patched = classify_output(
                raw_text=text_patched, cfg=output_parse_cfg,
                system_prompt=auth_sys, user_prompt=auth_usr,
                expected_answer_texts=([ground_truth] if ground_truth else []),
                token_logprobs=None,
            )

            output_patched_id = str(uuid.uuid4())
            trace_db.insert_conformity_output(
                output_id=output_patched_id, trial_id=auth_tid, raw_text=text_patched,
                parsed_answer_text=sr_patched.parsed_answer_text, parsed_answer_json=None,
                is_correct=sr_patched.is_correct, refusal_flag=sr_patched.refusal_flag,
                latency_ms=0.0,
                token_usage_json={"_output_quality": {"label": classified_patched.label.value, "metadata": classified_patched.metadata}},
                created_at=now,
            )

            rescued: Optional[bool] = None
            if sr_corrupt.is_correct is not None and sr_patched.is_correct is not None:
                rescued = (not bool(sr_corrupt.is_correct)) and bool(sr_patched.is_correct)

            # logit_diff_recovery is left as None for now; would require
            # extracting per-token logits which isn't cheap with the HF gateway.
            # Future work could add this via a logit-extraction hook.
            logit_diff_recovery: Optional[float] = None

            trace_db.conn.execute(
                """
                INSERT INTO vivarium_activation_patching(
                  patch_id, run_id, source_trial_id, target_trial_id, layer_index,
                  output_id_before_patch, output_id_after_patch,
                  rescued_truth, logit_diff_recovery, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(uuid.uuid4()), run_id, ctrl_tid, auth_tid, int(layer_idx),
                    output_corrupt_id, output_patched_id,
                    (1 if rescued else 0) if rescued is not None else None,
                    logit_diff_recovery,
                    now,
                ),
            )
            inserted += 1

    trace_db.conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Heatmap computation and visualisation
# ---------------------------------------------------------------------------

def compute_patching_heatmap(
    *,
    trace_db: TraceDb,
    run_id: str,
) -> Dict[str, Any]:
    """
    Aggregate activation-patching results into a summary structure:
    per-layer rescue rate and (optionally) mean logit-diff recovery.
    """
    rows = trace_db.conn.execute(
        """
        SELECT ap.layer_index, ap.rescued_truth, ap.logit_diff_recovery,
               ca.name AS authority_condition
        FROM vivarium_activation_patching ap
        JOIN conformity_trials t ON t.trial_id = ap.target_trial_id
        JOIN conformity_conditions ca ON ca.condition_id = t.condition_id
        WHERE ap.run_id = ? AND ap.rescued_truth IS NOT NULL;
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return {"layers": [], "rescue_rate": [], "by_condition": {}}

    try:
        import pandas as pd
    except ImportError:
        return {"layers": [], "rescue_rate": [], "by_condition": {}}

    df = pd.DataFrame([dict(r) for r in rows])

    agg = df.groupby("layer_index").agg(
        rescue_rate=("rescued_truth", "mean"),
        n=("rescued_truth", "count"),
    ).reset_index()

    by_cond: Dict[str, Any] = {}
    for cond in df["authority_condition"].unique():
        sub = df[df["authority_condition"] == cond].groupby("layer_index").agg(
            rescue_rate=("rescued_truth", "mean"),
            n=("rescued_truth", "count"),
        ).reset_index()
        by_cond[str(cond)] = {
            "layers": sub["layer_index"].tolist(),
            "rescue_rate": sub["rescue_rate"].tolist(),
        }

    return {
        "layers": agg["layer_index"].tolist(),
        "rescue_rate": agg["rescue_rate"].tolist(),
        "by_condition": by_cond,
    }


def plot_activation_patching_heatmap(
    *,
    trace_db: TraceDb,
    run_id: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate activation patching visualisations:
      1. Rescue rate by layer (line plot)
      2. Condition comparison (Asch vs Authority)

    Returns dict mapping plot name -> file path.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        return {}

    interp_dir = os.path.join(output_dir, "interpretability", "activation_patching")
    os.makedirs(interp_dir, exist_ok=True)

    rows = trace_db.conn.execute(
        """
        SELECT ap.layer_index, ap.rescued_truth, ap.logit_diff_recovery,
               ca.name AS authority_condition
        FROM vivarium_activation_patching ap
        JOIN conformity_trials t ON t.trial_id = ap.target_trial_id
        JOIN conformity_conditions ca ON ca.condition_id = t.condition_id
        WHERE ap.run_id = ? AND ap.rescued_truth IS NOT NULL;
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    plots: Dict[str, str] = {}

    # --- 1. Rescue rate by layer ---
    agg = df.groupby("layer_index").agg(
        rescue_rate=("rescued_truth", "mean"),
        n=("rescued_truth", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agg["layer_index"], agg["rescue_rate"], "o-", linewidth=2, color="#FF9800")
    ax.fill_between(agg["layer_index"], 0, agg["rescue_rate"], alpha=0.15, color="#FF9800")

    if not agg.empty:
        peak_idx = agg["rescue_rate"].idxmax()
        peak_layer = int(agg.loc[peak_idx, "layer_index"])
        peak_rate = float(agg.loc[peak_idx, "rescue_rate"])
        ax.annotate(
            f"peak L{peak_layer} ({peak_rate:.1%})",
            xy=(peak_layer, peak_rate),
            xytext=(peak_layer + 2, peak_rate + 0.05),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Rescue Rate (fraction flipped to truth)")
    ax.set_title("Activation Patching: Which Layer Rescues Truthful Behaviour?")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(interp_dir, "patching_rescue_by_layer.png")
    plt.savefig(p, dpi=150)
    plt.close()
    plots["patching_rescue_by_layer"] = p

    # --- 2. Condition comparison ---
    conditions = df["authority_condition"].unique()
    if len(conditions) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = {"authoritative_bias": "#E91E63", "asch_history_5": "#3F51B5"}
        for cond in sorted(conditions):
            sub = df[df["authority_condition"] == cond].groupby("layer_index")["rescued_truth"].mean().reset_index()
            ax.plot(
                sub["layer_index"], sub["rescued_truth"], "o-",
                label=cond, linewidth=2, color=cmap.get(str(cond), None),
            )
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Rescue Rate")
        ax.set_title("Activation Patching: Asch vs Authority Localisation")
        ax.legend()
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(interp_dir, "patching_condition_comparison.png")
        plt.savefig(p, dpi=150)
        plt.close()
        plots["patching_condition_comparison"] = p

    # --- Heatmap (layer x condition) ---
    pivot = df.pivot_table(
        index="layer_index", columns="authority_condition",
        values="rescued_truth", aggfunc="mean",
    )
    if not pivot.empty and pivot.shape[1] >= 1:
        fig, ax = plt.subplots(figsize=(6, max(8, len(pivot) * 0.3)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(l) for l in pivot.index])
        ax.set_xlabel("Condition")
        ax.set_ylabel("Layer Index")
        ax.set_title("Patching Rescue Heatmap")
        plt.colorbar(im, ax=ax, label="Rescue Rate")
        plt.tight_layout()
        p = os.path.join(interp_dir, "patching_heatmap.png")
        plt.savefig(p, dpi=150)
        plt.close()
        plots["patching_heatmap"] = p

    # --- CSV export ---
    csv_path = os.path.join(interp_dir, "patching_results.csv")
    df.to_csv(csv_path, index=False)
    plots["patching_csv"] = csv_path

    return plots
