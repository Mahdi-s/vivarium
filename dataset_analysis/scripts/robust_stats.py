"""Robust statistics module for the Dolci data audit.

PRE-REGISTRATION NOTICE
-----------------------
All thresholds and effect-size labels in this module were registered BEFORE
any phase-level analysis ran on the full corpus (Task 2 of the audit plan).
To change any threshold or formula after Task 4 has run you MUST add an
"amendment" comment that documents:
  - what changed
  - why (new evidence, methodological error, etc.)
  - who approved it

Implements:
  robust_summary         — median-centric descriptive statistics
  cliffs_delta           — non-parametric effect size, range [-1, 1]
  cohens_d_paired        — parametric paired effect size (secondary statistic)
  hodges_lehmann         — robust location estimator (median of Walsh averages)
  paired_permutation_p   — sign-flip permutation test, two-sided p-value
  bootstrap_ci           — percentile bootstrap confidence interval
  practical_significance_label — Romano et al. 2006 thresholds

References
----------
Romano, J. P., Shaikh, A. M., & Wolf, M. (2006). Improving Nonparametric
Tests via Robust Estimation. NBER Working Paper.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def robust_summary(values: Sequence[float]) -> dict:
    """Return a median-centric summary dict for *values*.

    Keys: n, n_zero, zero_inflation_rate, mean, stdev,
          P5, P25, median, P75, P95, P99, max, iqr.
    """
    vs = [float(v) for v in values]
    n = len(vs)
    if n == 0:
        return {"n": 0, "n_zero": 0, "zero_inflation_rate": 0.0,
                "mean": float("nan"), "stdev": float("nan"),
                "P5": float("nan"), "P25": float("nan"),
                "median": float("nan"), "P75": float("nan"),
                "P95": float("nan"), "P99": float("nan"),
                "max": float("nan"), "iqr": float("nan")}
    arr = np.array(vs, dtype=float)
    n_zero = int(np.sum(arr == 0.0))
    # Use "higher" interpolation so tail percentiles (P95, P99) snap to an
    # actual observed value rather than interpolating below a sparse tail value.
    # This is semantically correct for zero-inflated discrete distributions.
    pct = dict(zip(
        [5, 25, 50, 75, 95, 99],
        np.percentile(arr, [5, 25, 50, 75, 95, 99], method="higher"),
    ))
    return {
        "n": n,
        "n_zero": n_zero,
        "zero_inflation_rate": n_zero / n,
        "mean": float(np.mean(arr)),
        "stdev": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "P5": float(pct[5]),
        "P25": float(pct[25]),
        "median": float(pct[50]),
        "P75": float(pct[75]),
        "P95": float(pct[95]),
        "P99": float(pct[99]),
        "max": float(np.max(arr)),
        "iqr": float(pct[75] - pct[25]),
    }


# ---------------------------------------------------------------------------
# Cliff's delta  —  two internal paths, public dispatcher
# ---------------------------------------------------------------------------

_BROADCAST_THRESHOLD = 5_000_000  # n_a * n_b above which we use binary search


def _cliffs_delta_broadcast(a: np.ndarray, b: np.ndarray) -> float:
    """Broadcast (O(n_a * n_b) memory) path — exact."""
    diff = a[:, None] - b[None, :]   # shape (n_a, n_b)
    n_gt = float(np.sum(diff > 0))
    n_lt = float(np.sum(diff < 0))
    return (n_gt - n_lt) / (len(a) * len(b))


def _cliffs_delta_binsearch(a: np.ndarray, b: np.ndarray) -> float:
    """Binary-search (O(n_a log n_b) time, O(n_b) memory) path.

    Derivation from first principles:
      b_sorted = sorted(b)
      For each a_i:
        count(b < a_i)  = searchsorted(b_sorted, a_i, side="left")
        count(b <= a_i) = searchsorted(b_sorted, a_i, side="right")
        count(b > a_i)  = len(b) - searchsorted(b_sorted, a_i, side="right")

      n_gt = sum over a_i of count(b < a_i)      [i.e., a_i > b]
      n_lt = sum over a_i of (len(b) - count(b <= a_i)) ... wait,
             n_lt = count of (a_i, b_j) pairs where a_i < b_j
                  = sum over a_i of count(b_j > a_i)
                  = sum over a_i of (len(b) - searchsorted(b_sorted, a_i, "right"))

    So:
      n_gt = sum(searchsorted(b_sorted, a_i, "left"))
      n_lt = len(a)*len(b) - sum(searchsorted(b_sorted, a_i, "right"))
    """
    b_sorted = np.sort(b)
    n_a, n_b = len(a), len(b)
    # count of b < a_i for each a_i  →  n_a > b contributions
    n_gt = float(np.sum(np.searchsorted(b_sorted, a, side="left")))
    # count of b > a_i for each a_i  →  n_a < b contributions
    n_lt = float(n_a * n_b - np.sum(np.searchsorted(b_sorted, a, side="right")))
    return (n_gt - n_lt) / (n_a * n_b)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1].

    δ =  (P(a > b) - P(a < b))
    Positive → a stochastically dominates b.
    Uses binary-search path when len(a)*len(b) > 5_000_000 to save memory.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0
    if len(arr_a) * len(arr_b) <= _BROADCAST_THRESHOLD:
        return _cliffs_delta_broadcast(arr_a, arr_b)
    return _cliffs_delta_binsearch(arr_a, arr_b)


# ---------------------------------------------------------------------------
# Cohen's d (paired) — secondary/parametric, not robust to skew
# ---------------------------------------------------------------------------

def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean of paired differences.

    WARNING: Not robust to skew; use as a secondary statistic only.
    Returns ±inf when std of differences is zero and mean != 0.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    if std_d == 0.0:
        if mean_d == 0.0:
            return 0.0
        return math.copysign(math.inf, mean_d)
    return mean_d / std_d


# ---------------------------------------------------------------------------
# Hodges–Lehmann estimator
# ---------------------------------------------------------------------------

def hodges_lehmann(deltas: Sequence[float], max_pairs: int = 200_000) -> float:
    """Median of all pairwise means (Walsh averages) of *deltas*.

    When n > 1000, sub-samples *max_pairs* random pairs to avoid O(n^2) cost.
    """
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(arr[0])

    if n > 1000:
        rng = np.random.default_rng(0)
        idx1 = rng.integers(0, n, size=max_pairs)
        idx2 = rng.integers(0, n, size=max_pairs)
        walsh = (arr[idx1] + arr[idx2]) / 2.0
    else:
        # All n*(n+1)/2 Walsh averages (including self-pairs)
        i, j = np.triu_indices(n, k=0)
        walsh = (arr[i] + arr[j]) / 2.0

    return float(np.median(walsh))


# ---------------------------------------------------------------------------
# Sign-flip permutation test (paired, two-sided)
# ---------------------------------------------------------------------------

def paired_permutation_p(
    deltas: Sequence[float],
    reps: int = 1000,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value via sign-flip resampling.

    Under H0 the sign of each delta is exchangeable.
    Observed statistic: |mean(deltas)|.
    """
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(arr)))
    rng = np.random.default_rng(seed)
    count_geq = 0
    for _ in range(reps):
        signs = rng.choice([-1.0, 1.0], size=n)
        stat = abs(float(np.mean(signs * arr)))
        if stat >= observed:
            count_geq += 1
    return count_geq / reps


# ---------------------------------------------------------------------------
# Percentile bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    reps: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for *statistic* applied to *values*.

    Returns (lower, upper) at confidence level (1 - alpha).
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = arr[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Practical significance label  —  Romano et al. 2006 thresholds
# PRE-REGISTERED: do not change thresholds without an amendment comment.
# ---------------------------------------------------------------------------

def practical_significance_label(cliffs_d: float) -> str:
    """Map |Cliff's δ| to a verbal label using Romano et al. 2006 thresholds.

    |δ| < 0.147  → "none"
    |δ| < 0.33   → "small"
    |δ| < 0.474  → "medium"
    |δ| ≥ 0.474  → "large"
    """
    d = abs(cliffs_d)
    if d < 0.147:
        return "none"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"
