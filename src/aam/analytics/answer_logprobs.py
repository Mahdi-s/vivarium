"""
Answer-logprob analytics for Olmo Conformity Experiment.

Uses posthoc-computed `conformity_answer_logprobs` to quantify how model preference
shifts between the ground-truth answer and the conforming (injected wrong) answer
across conditions, prompt scenarios, and model variants.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("pandas, numpy, and matplotlib are required for answer-logprob analytics")

from aam.analytics.plotting_style import (
    create_figure,
    rotate_labels_if_needed,
    save_figure,
    setup_publication_style,
    wrap_long_labels,
)
from aam.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from aam.persistence import TraceDb


def _is_behavioral_condition_name(name: Any) -> bool:
    try:
        s = str(name or "")
    except Exception:
        return False
    return "probe_capture" not in s


def _safe_json_load(s: Any) -> Dict[str, Any]:
    try:
        if s is None:
            return {}
        out = json.loads(str(s))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def compute_answer_logprob_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute answer-logprob metrics for a run.

    Core derived quantities (per trial, per context_kind):
    - delta_logprob_sum = logp(correct) - logp(conforming_wrong)
    - p_correct_2way = exp(logp(correct)) / (exp(logp(correct)) + exp(logp(wrong)))

    Returns:
      {
        run_id,
        metrics: { ... },
        statistics: { ... }
      }
    """
    _ = run_dir  # kept for signature parity with other analytics modules

    df = pd.read_sql_query(
        """
        SELECT
          a.trial_id,
          a.context_kind,
          a.candidate_kind,
          a.candidate_text,
          a.token_count,
          a.logprob_sum,
          a.logprob_mean,
          t.variant,
          t.model_id,
          c.name AS condition_name,
          pm.metadata_json AS prompt_metadata_json
        FROM conformity_answer_logprobs a
        JOIN conformity_trials t ON t.trial_id = a.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        LEFT JOIN conformity_prompts p ON p.prompt_id = (
          SELECT prompt_id FROM conformity_prompts
          WHERE trial_id = t.trial_id
          ORDER BY created_at ASC
          LIMIT 1
        )
        LEFT JOIN conformity_prompt_metadata pm ON pm.prompt_id = p.prompt_id
        WHERE t.run_id = ?
        ORDER BY a.trial_id, a.context_kind, a.candidate_kind;
        """,
        trace_db.conn,
        params=(str(run_id),),
    )

    if df.empty:
        return {
            "run_id": str(run_id),
            "metrics": {},
            "statistics": {"message": "No conformity_answer_logprobs rows found"},
        }

    # Exclude probe-capture conditions from paper-facing plots.
    df = df[df["condition_name"].apply(_is_behavioral_condition_name)].copy()
    if df.empty:
        return {
            "run_id": str(run_id),
            "metrics": {},
            "statistics": {"message": "Only probe-capture conditions present in answer logprobs"},
        }

    # Prompt scenario metadata (best-effort)
    meta = df["prompt_metadata_json"].apply(_safe_json_load)
    df["prompt_family"] = meta.apply(lambda m: str(m.get("prompt_family") or "unknown"))
    df["tone"] = meta.apply(lambda m: str(m.get("tone") or "unknown"))
    df["consensus"] = meta.apply(lambda m: str(m.get("consensus") or "unknown"))
    df["distillation"] = meta.apply(lambda m: bool(m.get("distillation", False)))
    df["devils_advocate"] = meta.apply(lambda m: bool(m.get("devils_advocate", False)))
    df["claim_style"] = meta.apply(lambda m: str(m.get("claim_style") or ""))

    # Focus metrics: correct vs conforming wrong answer (2-way)
    df_2 = df[df["candidate_kind"].isin(["ground_truth", "wrong_answer"])].copy()
    if df_2.empty:
        return {
            "run_id": str(run_id),
            "metrics": {},
            "statistics": {"message": "No ground_truth/wrong_answer rows found in conformity_answer_logprobs"},
        }

    pivot = (
        df_2.pivot_table(
            index=[
                "trial_id",
                "context_kind",
                "variant",
                "model_id",
                "condition_name",
                "prompt_family",
                "tone",
                "consensus",
                "distillation",
                "devils_advocate",
                "claim_style",
            ],
            columns="candidate_kind",
            values="logprob_sum",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["ground_truth", "wrong_answer"])
    )

    if pivot.empty:
        return {
            "run_id": str(run_id),
            "metrics": {},
            "statistics": {"message": "No trials had both ground_truth and wrong_answer logprobs"},
        }

    pivot["delta_logprob_sum"] = pivot["ground_truth"].astype(float) - pivot["wrong_answer"].astype(float)
    # Stable 2-way probability via logsumexp:
    # p = exp(gt) / (exp(gt)+exp(wrong)) = exp(gt - logaddexp(gt, wrong))
    pivot["p_correct_2way"] = np.exp(
        pivot["ground_truth"].astype(float) - np.logaddexp(pivot["ground_truth"].astype(float), pivot["wrong_answer"].astype(float))
    )
    pivot["prefers_correct"] = (pivot["delta_logprob_sum"] > 0).astype(int)

    by_condition = (
        pivot.groupby(["variant", "context_kind", "condition_name"], as_index=False)
        .agg(
            n_trials=("trial_id", "count"),
            mean_p_correct=("p_correct_2way", "mean"),
            mean_delta_logprob=("delta_logprob_sum", "mean"),
            frac_prefers_correct=("prefers_correct", "mean"),
        )
        .sort_values(["variant", "context_kind", "condition_name"])
    )

    by_prompt_scenario = (
        pivot.groupby(
            [
                "variant",
                "context_kind",
                "prompt_family",
                "tone",
                "consensus",
                "distillation",
                "devils_advocate",
                "claim_style",
            ],
            as_index=False,
        )
        .agg(
            n_trials=("trial_id", "count"),
            mean_p_correct=("p_correct_2way", "mean"),
            mean_delta_logprob=("delta_logprob_sum", "mean"),
        )
        .sort_values(["variant", "context_kind", "prompt_family", "tone", "consensus"])
    )

    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "metrics": {
            "by_condition": by_condition.to_dict("records"),
            "by_prompt_scenario": by_prompt_scenario.to_dict("records"),
        },
        "statistics": {
            "n_answer_logprob_rows": int(len(df)),
            "n_trials_2way": int(pivot["trial_id"].nunique()),
            "context_kinds": sorted(pivot["context_kind"].unique().tolist()),
            "variants": sorted(pivot["variant"].unique().tolist()),
            "conditions": sorted(pivot["condition_name"].unique().tolist()),
        },
    }
    return metrics


def generate_answer_logprob_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate publication-ready figures for answer-logprob metrics.

    Outputs:
    - Heatmap of mean p_correct (2-way) by (variant x condition), per context_kind.
    - Heatmap of mean delta_logprob by (variant x condition), per context_kind.
    """
    if metrics is None:
        metrics = compute_answer_logprob_metrics(trace_db, run_id, run_dir)

    paths = ensure_logs_dir(run_dir)
    setup_publication_style()

    figures: Dict[str, str] = {}

    rows = (metrics.get("metrics") or {}).get("by_condition") or []
    if not rows:
        return figures

    df = pd.DataFrame(rows)
    if df.empty:
        return figures

    for ctx_kind in sorted(df["context_kind"].unique().tolist()):
        sub = df[df["context_kind"] == ctx_kind].copy()
        if sub.empty:
            continue

        # Heatmap: p_correct
        pivot_p = sub.pivot(index="variant", columns="condition_name", values="mean_p_correct")
        fig, ax = create_figure(size_key="wide")
        im = ax.imshow(pivot_p.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"Correct-vs-Conforming Preference (p_correct, 2-way) | context={ctx_kind}")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Model Variant")
        ax.set_yticks(range(len(pivot_p.index)))
        ax.set_yticklabels(pivot_p.index.tolist())
        xlabels = wrap_long_labels([str(x) for x in pivot_p.columns.tolist()], max_length=22)
        ax.set_xticks(range(len(pivot_p.columns)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Mean p_correct")

        fig_path = os.path.join(paths["figures_dir"], f"answer_p_correct_heatmap_{ctx_kind}")
        saved = save_figure(fig, fig_path)
        figures[f"answer_p_correct_heatmap_{ctx_kind}"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)

        # Heatmap: delta_logprob
        pivot_d = sub.pivot(index="variant", columns="condition_name", values="mean_delta_logprob")
        fig, ax = create_figure(size_key="wide")
        vmax = float(np.nanmax(np.abs(pivot_d.values))) if np.isfinite(pivot_d.values).any() else 1.0
        vmax = max(1e-6, vmax)
        im = ax.imshow(pivot_d.values, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set_title(f"Correct-vs-Conforming Preference (Δ logp) | context={ctx_kind}")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Model Variant")
        ax.set_yticks(range(len(pivot_d.index)))
        ax.set_yticklabels(pivot_d.index.tolist())
        xlabels = wrap_long_labels([str(x) for x in pivot_d.columns.tolist()], max_length=22)
        ax.set_xticks(range(len(pivot_d.columns)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Mean Δ logp (gt - wrong)")

        fig_path = os.path.join(paths["figures_dir"], f"answer_delta_logprob_heatmap_{ctx_kind}")
        saved = save_figure(fig, fig_path)
        figures[f"answer_delta_logprob_heatmap_{ctx_kind}"] = saved.get("png", saved.get("pdf", ""))
        plt.close(fig)

        # Heatmap: Δ p_correct relative to control (within variant)
        try:
            if "control" in set(pivot_p.columns.tolist()):
                ctrl = pivot_p["control"]
                delta = pivot_p.subtract(ctrl, axis=0)
                # Drop the control column for readability (it's identically 0).
                if "control" in delta.columns:
                    delta = delta.drop(columns=["control"])
                if not delta.empty:
                    fig, ax = create_figure(size_key="wide")
                    vmax = float(np.nanmax(np.abs(delta.values))) if np.isfinite(delta.values).any() else 1.0
                    vmax = max(1e-6, vmax)
                    im = ax.imshow(delta.values, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
                    ax.set_title(f"Δ p_correct vs control | context={ctx_kind}")
                    ax.set_xlabel("Condition (non-control)")
                    ax.set_ylabel("Model Variant")
                    ax.set_yticks(range(len(delta.index)))
                    ax.set_yticklabels(delta.index.tolist())
                    xlabels = wrap_long_labels([str(x) for x in delta.columns.tolist()], max_length=22)
                    ax.set_xticks(range(len(delta.columns)))
                    ax.set_xticklabels(xlabels, rotation=45, ha="right")
                    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ p_correct (condition - control)")

                    fig_path = os.path.join(paths["figures_dir"], f"answer_delta_p_correct_from_control_{ctx_kind}")
                    saved = save_figure(fig, fig_path)
                    figures[f"answer_delta_p_correct_from_control_{ctx_kind}"] = saved.get("png", saved.get("pdf", ""))
                    plt.close(fig)
        except Exception:
            pass

    # Optional compact bar: mean p_correct aggregated across all non-control conditions
    try:
        df2 = df.copy()
        df2["is_control"] = df2["condition_name"].astype(str).eq("control")
        non_control = df2[~df2["is_control"]].copy()
        if not non_control.empty:
            agg = (
                non_control.groupby(["variant", "context_kind"], as_index=False)["mean_p_correct"]
                .mean()
                .rename(columns={"mean_p_correct": "mean_p_correct_non_control"})
            )
            fig, ax = create_figure(size_key="single")
            for ctx_kind in sorted(agg["context_kind"].unique().tolist()):
                s = agg[agg["context_kind"] == ctx_kind].set_index("variant")["mean_p_correct_non_control"]
                s.plot(kind="bar", ax=ax, alpha=0.7, label=ctx_kind)
            ax.set_ylabel("Mean p_correct (non-control)")
            ax.set_title("Answer Preference Summary (non-control)")
            rotate_labels_if_needed(ax, axis="x")
            ax.legend(title="Context")
            fig_path = os.path.join(paths["figures_dir"], "answer_p_correct_summary_non_control")
            saved = save_figure(fig, fig_path)
            figures["answer_p_correct_summary_non_control"] = saved.get("png", saved.get("pdf", ""))
            plt.close(fig)
    except Exception:
        pass

    return figures


def export_answer_logprob_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export metrics JSON + CSV tables for downstream paper writing.
    """
    if metrics is None:
        metrics = compute_answer_logprob_metrics(trace_db, run_id, run_dir)

    paths = ensure_logs_dir(run_dir)

    metrics_path = os.path.join(paths["logs_dir"], "metrics_answer_logprobs.json")
    save_metrics_json(metrics, metrics_path)

    by_condition = ((metrics.get("metrics") or {}).get("by_condition") or [])
    if by_condition:
        save_table_csv(by_condition, os.path.join(paths["tables_dir"], "answer_logprobs_by_condition.csv"))

    by_scenario = ((metrics.get("metrics") or {}).get("by_prompt_scenario") or [])
    if by_scenario:
        save_table_csv(by_scenario, os.path.join(paths["tables_dir"], "answer_logprobs_by_prompt_scenario.csv"))
