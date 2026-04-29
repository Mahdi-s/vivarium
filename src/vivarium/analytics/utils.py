"""
Shared utilities for analytics modules.

Provides database loading, logging directory management, and metric export functions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from vivarium.persistence import TraceDb, TraceDbConfig


def load_simulation_db(run_dir: str) -> TraceDb:
    """
    Load TraceDb from a run directory.
    
    Args:
        run_dir: Path to run directory (contains simulation.db)
        
    Returns:
        TraceDb instance (connected)
    """
    db_path = os.path.join(run_dir, "simulation.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    config = TraceDbConfig(db_path=db_path)
    db = TraceDb(config)
    db.connect()
    return db


def get_run_metadata(trace_db: TraceDb, run_id: str) -> Dict[str, Any]:
    """
    Extract run configuration and metadata.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict with run metadata
    """
    row = trace_db.conn.execute(
        "SELECT seed, created_at, config_json FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    
    if row is None:
        raise ValueError(f"Run {run_id} not found")
    
    config = json.loads(row["config_json"])
    return {
        "run_id": run_id,
        "seed": row["seed"],
        "created_at": row["created_at"],
        "config": config,
    }


def ensure_logs_dir(run_dir: str) -> Dict[str, str]:
    """
    Create logs directory structure for a run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dict with paths: logs_dir, tables_dir, figures_dir
    """
    base = Path(run_dir) / "artifacts" / "logs"
    tables_dir = base / "tables"
    figures_dir = Path(run_dir) / "artifacts" / "figures"
    
    base.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "logs_dir": str(base),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_table_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save table data to CSV file.
    
    Args:
        data: List of row dictionaries
        output_path: Output file path
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for CSV export")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def check_missing_prerequisites(trace_db: TraceDb, run_id: str) -> Dict[str, bool]:
    """
    Check which prerequisites exist for full analytics.
    
    Args:
        trace_db: TraceDb instance
        run_id: Run ID
        
    Returns:
        Dict mapping prerequisite name -> exists (bool)
    """
    missing = {}

    def _table_exists(table_name: str) -> bool:
        row = trace_db.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    def _has_column(table_name: str, column_name: str) -> bool:
        if not _table_exists(table_name):
            return False
        rows = trace_db.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        # pragma table_info columns: cid, name, type, notnull, dflt_value, pk
        return any(r[1] == column_name for r in rows)
    
    # Check for Judge Eval scores
    judgeval_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_outputs o
        JOIN conformity_trials t ON t.trial_id = o.trial_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        (run_id,),
    ).fetchone()[0]
    missing["judgeval_scores"] = judgeval_count > 0
    
    # Check for probes
    probe_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM vivarium_probes WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    missing["probes"] = probe_count > 0
    
    # Check for probe projections
    if _has_column("vivarium_probe_projections", "trial_id"):
        projection_count = trace_db.conn.execute(
            """
            SELECT COUNT(*) FROM vivarium_probe_projections p
            JOIN conformity_trials t ON t.trial_id = p.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    elif _has_column("vivarium_probe_projections", "trace_id"):
        projection_count = trace_db.conn.execute(
            """
            SELECT COUNT(*)
            FROM vivarium_probe_projections p
            JOIN trace tr ON tr.trace_id = p.trace_id
            JOIN conformity_trial_steps s ON s.time_step = tr.time_step AND s.agent_id = tr.agent_id
            JOIN conformity_trials t ON t.trial_id = s.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        projection_count = 0
    missing["probe_projections"] = projection_count > 0
    
    # Check for interventions
    intervention_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM vivarium_interventions WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    missing["interventions"] = intervention_count > 0
    
    # Check for intervention results
    if _has_column("vivarium_intervention_results", "trial_id"):
        intervention_result_count = trace_db.conn.execute(
            """
            SELECT COUNT(*) FROM vivarium_intervention_results r
            JOIN conformity_trials t ON t.trial_id = r.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    elif _has_column("vivarium_intervention_results", "trace_id"):
        intervention_result_count = trace_db.conn.execute(
            """
            SELECT COUNT(*)
            FROM vivarium_intervention_results r
            JOIN trace tr ON tr.trace_id = r.trace_id
            JOIN conformity_trial_steps s ON s.time_step = tr.time_step AND s.agent_id = tr.agent_id
            JOIN conformity_trials t ON t.trial_id = s.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        intervention_result_count = 0
    missing["intervention_results"] = intervention_result_count > 0
    
    # Check for think tokens
    if _has_column("vivarium_think_tokens", "trial_id"):
        think_count = trace_db.conn.execute(
            """
            SELECT COUNT(*) FROM vivarium_think_tokens tt
            JOIN conformity_trials t ON t.trial_id = tt.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    elif _has_column("vivarium_think_tokens", "trace_id"):
        think_count = trace_db.conn.execute(
            """
            SELECT COUNT(*)
            FROM vivarium_think_tokens tt
            JOIN trace tr ON tr.trace_id = tt.trace_id
            JOIN conformity_trial_steps s ON s.time_step = tr.time_step AND s.agent_id = tr.agent_id
            JOIN conformity_trials t ON t.trial_id = s.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        think_count = 0
    missing["think_tokens"] = think_count > 0
    
    # Check for logit lens
    logit_count = trace_db.conn.execute(
        """
        SELECT COUNT(*) FROM conformity_logit_lens ll
        JOIN conformity_trials t ON t.trial_id = ll.trial_id
        WHERE t.run_id = ?
        """,
        (run_id,),
    ).fetchone()[0]
    missing["logit_lens"] = logit_count > 0

    # Check for answer logprob probes (correct vs conforming)
    if _has_column("vivarium_answer_logprobs", "trial_id"):
        answer_logprob_count = trace_db.conn.execute(
            """
            SELECT COUNT(*) FROM vivarium_answer_logprobs a
            JOIN conformity_trials t ON t.trial_id = a.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    elif _has_column("vivarium_answer_logprobs", "trace_id"):
        answer_logprob_count = trace_db.conn.execute(
            """
            SELECT COUNT(*)
            FROM vivarium_answer_logprobs a
            JOIN trace tr ON tr.trace_id = a.trace_id
            JOIN conformity_trial_steps s ON s.time_step = tr.time_step AND s.agent_id = tr.agent_id
            JOIN conformity_trials t ON t.trial_id = s.trial_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        answer_logprob_count = 0
    missing["answer_logprobs"] = answer_logprob_count > 0
    
    # Check for activation metadata (for attention capture)
    activation_count = trace_db.conn.execute(
        "SELECT COUNT(*) FROM activation_metadata WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    missing["activation_capture"] = activation_count > 0
    
    return missing


def save_missing_prerequisites_log(missing: Dict[str, bool], output_path: str) -> None:
    """
    Save missing prerequisites log.
    
    Args:
        missing: Dict from check_missing_prerequisites
        output_path: Output file path
    """
    # Convert to list of missing items
    missing_list = [name for name, exists in missing.items() if not exists]
    
    log_data = {
        "prerequisites_checked": missing,
        "missing_items": missing_list,
        "all_present": len(missing_list) == 0,
    }
    
    save_metrics_json(log_data, output_path)
