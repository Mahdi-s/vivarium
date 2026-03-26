"""
Automated validation metrics for detecting mode collapse and empirical
divergence in multi-agent simulations (Epic 5).

Scientific Rationale:
    The biggest existential threat to LLM ABMs is the "Scientific Validation
    Crisis" (Section 7.1).  Relying on an LLM to "judge" its own plausibility
    is methodologically circular.  These functions automatically benchmark
    synthetic outputs to detect if the agent population has collapsed into a
    uniform average.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from vivarium.persistence import TraceDb


# ---------------------------------------------------------------------------
# Mode-collapse entropy
# ---------------------------------------------------------------------------

def compute_mode_collapse_entropy(
    db: TraceDb,
    run_id: str,
    *,
    time_step: Optional[int] = None,
) -> float:
    """
    Compute the Shannon entropy of the ``action_type`` distribution across
    the agent population for a given run.

    .. math::

        H = -\\sum_i p_i \\log_2 p_i

    A value near **0** means all agents chose the same action (severe mode
    collapse).  The maximum entropy is ``log2(K)`` where *K* is the number
    of distinct actions available.

    Args:
        db: Connected TraceDb instance.
        run_id: Experiment run identifier.
        time_step: If provided, restrict to a single time step.
                   Otherwise aggregate across all steps.

    Returns:
        Shannon entropy in bits.  Returns 0.0 if no trace rows found.
    """
    if time_step is not None:
        rows = db.conn.execute(
            "SELECT action_type FROM trace WHERE run_id = ? AND time_step = ?;",
            (run_id, time_step),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT action_type FROM trace WHERE run_id = ?;",
            (run_id,),
        ).fetchall()

    if not rows:
        return 0.0

    counts = Counter(row["action_type"] for row in rows)
    total = sum(counts.values())

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


# ---------------------------------------------------------------------------
# Empirical divergence (Kolmogorov–Smirnov)
# ---------------------------------------------------------------------------

def compute_empirical_divergence(
    simulated_data: Sequence[float],
    empirical_data: Sequence[float],
) -> Dict[str, float]:
    """
    Compare the distribution of simulated agent actions against a
    ground-truth empirical reference using the two-sample
    Kolmogorov–Smirnov test.

    Args:
        simulated_data: Numeric array of simulated outcomes
                        (e.g. action frequencies per archetype).
        empirical_data: Numeric array of real-world reference outcomes.

    Returns:
        Dictionary with:
        - ``ks_statistic``: KS test statistic (0 = identical distributions).
        - ``p_value``: p-value; low values reject the null hypothesis
          that the two samples come from the same distribution.
    """
    from scipy.stats import ks_2samp  # type: ignore

    if len(simulated_data) == 0 or len(empirical_data) == 0:
        return {"ks_statistic": 1.0, "p_value": 0.0}

    stat, p_value = ks_2samp(simulated_data, empirical_data)
    return {"ks_statistic": float(stat), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Per-step entropy series (for convergence analysis)
# ---------------------------------------------------------------------------

def compute_entropy_series(
    db: TraceDb,
    run_id: str,
) -> List[Dict[str, Any]]:
    """
    Compute Shannon entropy of the action distribution at each time step.

    Useful for detecting *when* mode collapse begins during a simulation.

    Returns:
        List of ``{"time_step": int, "entropy": float, "n_actions": int}``
        sorted by time_step.
    """
    rows = db.conn.execute(
        "SELECT DISTINCT time_step FROM trace WHERE run_id = ? ORDER BY time_step;",
        (run_id,),
    ).fetchall()

    series: List[Dict[str, Any]] = []
    for row in rows:
        ts = row["time_step"]
        ent = compute_mode_collapse_entropy(db, run_id, time_step=ts)

        # Count distinct actions at this step
        n_row = db.conn.execute(
            "SELECT COUNT(DISTINCT action_type) AS n FROM trace WHERE run_id = ? AND time_step = ?;",
            (run_id, ts),
        ).fetchone()
        n_actions = n_row["n"] if n_row else 0

        series.append({"time_step": ts, "entropy": ent, "n_actions": n_actions})

    return series
