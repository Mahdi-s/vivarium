from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aam.persistence import TraceDb


def _require_plotting() -> Any:
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plot generation requires pandas + matplotlib.") from e
    return pd, plt


@dataclass(frozen=True)
class PlotPaths:
    figures_dir: str
    tables_dir: str


def _ensure_plot_dirs(run_dir: str) -> PlotPaths:
    figures_dir = os.path.join(run_dir, "artifacts", "figures")
    tables_dir = os.path.join(run_dir, "artifacts", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return PlotPaths(figures_dir=figures_dir, tables_dir=tables_dir)


def generate_core_figures(*, trace_db: TraceDb, run_id: str, run_dir: str) -> Dict[str, str]:
    """
    Generates the key plots specified in the design doc (behavioral + intervention scaffolding).
    Returns mapping figure_name -> output_path.
    """
    pd, plt = _require_plotting()
    paths = _ensure_plot_dirs(run_dir)
    
    # Check for Judge Eval scores in parsed_answer_json
    has_judgeval = False
    try:
        judgeval_check = trace_db.conn.execute(
            """
            SELECT COUNT(*) FROM conformity_outputs 
            WHERE trial_id IN (SELECT trial_id FROM conformity_trials WHERE run_id = ?)
            AND parsed_answer_json IS NOT NULL;
            """,
            (run_id,)
        ).fetchone()[0]
        has_judgeval = judgeval_check > 0
    except Exception:
        pass

    # Conformity rate by variant/condition (bar)
    df = pd.read_sql_query(
        """
        SELECT t.variant,
               c.name AS condition_name,
               o.is_correct
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    out: Dict[str, str] = {}
    if not df.empty:
        # is_correct is nullable; drop NAs for immutable facts
        df2 = df.dropna(subset=["is_correct"]).copy()
        if not df2.empty:
            summary = (
                df2.groupby(["variant", "condition_name"], as_index=False)["is_correct"].mean().rename(columns={"is_correct": "mean_is_correct"})
            )
            summary_path = os.path.join(paths.tables_dir, "conformity_rate_by_variant.csv")
            summary.to_csv(summary_path, index=False)

            fig_path = os.path.join(paths.figures_dir, "conformity_rate_by_variant.png")
            ax = summary.pivot(index="variant", columns="condition_name", values="mean_is_correct").plot(kind="bar")
            ax.set_ylabel("Mean correctness")
            ax.set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            out["conformity_rate_by_variant"] = fig_path

    # Intervention effect size placeholder (if present)
    df_int = pd.read_sql_query(
        """
        SELECT r.flipped_to_truth, i.alpha, i.name
        FROM conformity_intervention_results r
        JOIN conformity_interventions i ON i.intervention_id = r.intervention_id
        JOIN conformity_trials t ON t.trial_id = r.trial_id
        WHERE t.run_id = ?
        """,
        trace_db.conn,
        params=(run_id,),
    )
    if not df_int.empty:
        df_int2 = df_int.dropna(subset=["flipped_to_truth"]).copy()
        if not df_int2.empty:
            df_int2["flipped_to_truth"] = df_int2["flipped_to_truth"].astype(int)
            summ = df_int2.groupby(["name", "alpha"], as_index=False)["flipped_to_truth"].mean().rename(columns={"flipped_to_truth": "flip_rate"})
            summ_path = os.path.join(paths.tables_dir, "intervention_effect_size.csv")
            summ.to_csv(summ_path, index=False)

            fig_path = os.path.join(paths.figures_dir, "intervention_effect_size.png")
            ax = summ.pivot(index="alpha", columns="name", values="flip_rate").plot(kind="line", marker="o")
            ax.set_ylabel("Flip-to-truth rate")
            ax.set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            out["intervention_effect_size"] = fig_path

    # Judge Eval metrics visualization (if available)
    if has_judgeval:
        try:
            df_judgeval = pd.read_sql_query(
                """
                SELECT 
                    t.variant,
                    c.name AS condition_name,
                    json_extract(o.parsed_answer_json, '$.conformity') as conformity_score,
                    json_extract(o.parsed_answer_json, '$.truthfulness') as truthfulness_score,
                    json_extract(o.parsed_answer_json, '$.rationalization') as rationalization_score,
                    o.is_correct
                FROM conformity_trials t
                JOIN conformity_conditions c ON c.condition_id = t.condition_id
                JOIN conformity_outputs o ON o.trial_id = t.trial_id
                WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
                """,
                trace_db.conn,
                params=(run_id,),
            )
            
            if not df_judgeval.empty:
                # Plot 1: Conformity scores by variant/condition
                df_conv = df_judgeval.dropna(subset=["conformity_score"]).copy()
                if not df_conv.empty:
                    df_conv["conformity_score"] = pd.to_numeric(df_conv["conformity_score"], errors="coerce")
                    summary_conv = (
                        df_conv.groupby(["variant", "condition_name"], as_index=False)["conformity_score"]
                        .mean()
                        .rename(columns={"conformity_score": "mean_conformity"})
                    )
                    
                    fig_path = os.path.join(paths.figures_dir, "judgeval_conformity_scores.png")
                    ax = summary_conv.pivot(index="variant", columns="condition_name", values="mean_conformity").plot(kind="bar")
                    ax.set_ylabel("Mean Conformity Score (Judge Eval)")
                    ax.set_ylim(0.0, 1.0)
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=150)
                    plt.close()
                    out["judgeval_conformity_scores"] = fig_path
                
                # Plot 2: Truthfulness vs Correctness correlation
                df_truth = df_judgeval.dropna(subset=["truthfulness_score", "is_correct"]).copy()
                if not df_truth.empty:
                    df_truth["truthfulness_score"] = pd.to_numeric(df_truth["truthfulness_score"], errors="coerce")
                    df_truth["is_correct"] = df_truth["is_correct"].astype(float)
                    
                    fig_path = os.path.join(paths.figures_dir, "judgeval_truthfulness_correlation.png")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(df_truth["truthfulness_score"], df_truth["is_correct"], alpha=0.5)
                    ax.set_xlabel("Judge Eval Truthfulness Score")
                    ax.set_ylabel("Actual Correctness")
                    ax.set_title("Judge Eval Truthfulness vs Actual Correctness")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=150)
                    plt.close()
                    out["judgeval_truthfulness_correlation"] = fig_path
                
                # Export Judge Eval summary table
                summary_path = os.path.join(paths.tables_dir, "judgeval_summary.csv")
                summary = df_judgeval.groupby(["variant", "condition_name"], as_index=False).agg({
                    "conformity_score": "mean",
                    "truthfulness_score": "mean",
                    "rationalization_score": "mean",
                }).round(3)
                summary.to_csv(summary_path, index=False)
                out["judgeval_summary_table"] = summary_path
        except Exception as e:
            print(f"Warning: Could not generate Judge Eval plots: {e}")

    return out


