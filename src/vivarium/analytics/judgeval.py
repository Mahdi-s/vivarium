"""
Judge Eval analytics for Olmo Conformity Experiment.

Computes a simple agreement table between the LLM judge's binary labels
(is_correct, refusal_flag, wrong_answer_endorsed) and the rule-based
labels from scoring.py. No visualizations -- just counts and rates.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:
    raise RuntimeError("pandas is required for judgeval analytics")

from vivarium.analytics.utils import ensure_logs_dir, save_metrics_json, save_table_csv
from vivarium.persistence import TraceDb


_FIRST_OUTPUT_CTE = """\
WITH first_outputs AS (
    SELECT trial_id, MIN(created_at) AS min_created_at
    FROM conformity_outputs
    GROUP BY trial_id
),
first_output_ids AS (
    SELECT MIN(o.output_id) AS output_id, o.trial_id
    FROM conformity_outputs o
    JOIN first_outputs fo ON fo.trial_id = o.trial_id AND fo.min_created_at = o.created_at
    GROUP BY o.trial_id
)"""


def compute_judgeval_metrics(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
) -> Dict[str, Any]:
    """
    Compute agreement counts between LLM judge and rule-based labels.

    Returns a dict with overall match/mismatch counts for each label,
    plus a per-cell (variant x condition) agreement table.
    """
    judgeval_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_outputs o
        JOIN conformity_trials t ON t.trial_id = o.trial_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        (run_id,),
    ).fetchone()[0]

    if judgeval_count == 0:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No Judge Eval scores found for this run"},
        }

    df = pd.read_sql_query(
        f"""
        {_FIRST_OUTPUT_CTE}
        SELECT
            t.trial_id,
            t.variant,
            c.name AS condition_name,
            o.is_correct AS manual_is_correct,
            o.refusal_flag AS manual_refusal_flag,
            json_extract(o.parsed_answer_json, '$.is_correct') AS judge_is_correct,
            json_extract(o.parsed_answer_json, '$.refusal_flag') AS judge_refusal_flag,
            json_extract(o.parsed_answer_json, '$.wrong_answer_endorsed') AS judge_wrong_answer_endorsed
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        JOIN first_output_ids foi ON foi.trial_id = t.trial_id
        JOIN conformity_outputs o ON o.output_id = foi.output_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        trace_db.conn,
        params=(run_id,),
    )

    if df.empty:
        return {
            "run_id": run_id,
            "metrics": {},
            "statistics": {"message": "No Judge Eval data available"},
        }

    for col in ["manual_is_correct", "manual_refusal_flag",
                 "judge_is_correct", "judge_refusal_flag", "judge_wrong_answer_endorsed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    metrics: Dict[str, Any] = {"run_id": run_id, "metrics": {}, "statistics": {}}

    # --- is_correct agreement ---
    ic = df[df["manual_is_correct"].notna() & df["judge_is_correct"].notna()].copy()
    if not ic.empty:
        m = ic["manual_is_correct"].astype(int)
        j = ic["judge_is_correct"].astype(int)
        metrics["metrics"]["is_correct"] = {
            "n": int(len(ic)),
            "match": int((m == j).sum()),
            "mismatch": int((m != j).sum()),
            "agreement_rate": float((m == j).mean()),
        }

    # --- refusal_flag agreement ---
    rf = df[df["manual_refusal_flag"].notna() & df["judge_refusal_flag"].notna()].copy()
    if not rf.empty:
        m_rf = rf["manual_refusal_flag"].astype(int)
        j_rf = rf["judge_refusal_flag"].astype(int)
        metrics["metrics"]["refusal_flag"] = {
            "n": int(len(rf)),
            "match": int((m_rf == j_rf).sum()),
            "mismatch": int((m_rf != j_rf).sum()),
            "agreement_rate": float((m_rf == j_rf).mean()),
        }

    # --- wrong_answer_endorsed (judge-only rate, no manual counterpart in DB) ---
    wa = df[df["judge_wrong_answer_endorsed"].notna()].copy()
    if not wa.empty:
        metrics["metrics"]["wrong_answer_endorsed"] = {
            "n": int(len(wa)),
            "endorsed_count": int(wa["judge_wrong_answer_endorsed"].sum()),
            "endorsement_rate": float(wa["judge_wrong_answer_endorsed"].mean()),
        }

    # --- Per-cell agreement table (variant x condition) ---
    ic2 = df[df["manual_is_correct"].notna() & df["judge_is_correct"].notna()].copy()
    if not ic2.empty:
        ic2["agree"] = (ic2["manual_is_correct"].astype(int) == ic2["judge_is_correct"].astype(int)).astype(int)
        cell_table = (
            ic2.groupby(["variant", "condition_name"], as_index=False)
            .agg(n=("trial_id", "count"), match=("agree", "sum"))
        )
        cell_table["mismatch"] = cell_table["n"] - cell_table["match"]
        cell_table["agreement_rate"] = (cell_table["match"] / cell_table["n"]).round(3)
        metrics["metrics"]["is_correct_by_cell"] = cell_table.to_dict("records")

    metrics["statistics"] = {
        "n_judged": int(len(df)),
        "variants": sorted(df["variant"].unique().tolist()),
        "conditions": sorted(df["condition_name"].unique().tolist()),
    }

    return metrics


def generate_judgeval_graphs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """No-op -- judgeval does not produce visualizations."""
    return {}


def export_judgeval_logs(
    trace_db: TraceDb,
    run_id: str,
    run_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Export agreement table to JSON and CSV."""
    if metrics is None:
        metrics = compute_judgeval_metrics(trace_db, run_id, run_dir)

    paths = ensure_logs_dir(run_dir)

    json_path = os.path.join(paths["logs_dir"], "metrics_judgeval.json")
    save_metrics_json(metrics, json_path)

    csv_paths: Dict[str, str] = {}

    if "is_correct_by_cell" in metrics.get("metrics", {}):
        csv_path = os.path.join(paths["tables_dir"], "judgeval_agreement.csv")
        save_table_csv(metrics["metrics"]["is_correct_by_cell"], csv_path)
        csv_paths["judgeval_agreement"] = csv_path

    return {"metrics_json": json_path, **csv_paths}
