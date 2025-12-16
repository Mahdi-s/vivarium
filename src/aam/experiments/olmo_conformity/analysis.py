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

    return out


