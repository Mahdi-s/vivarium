"""Tests for robust_stats.py — pre-registered metrics for Dolci data audit.

These tests were written BEFORE the implementation (TDD).
Do not alter thresholds after Task 4 runs on the full corpus.
"""
import math
import numpy as np
import pytest
from robust_stats import (
    robust_summary, cliffs_delta, cohens_d_paired,
    hodges_lehmann, paired_permutation_p, bootstrap_ci,
    practical_significance_label,
)


def test_robust_summary_basic():
    s = robust_summary([0]*90 + [1]*5 + [10]*4 + [100]*1)
    assert s["n"] == 100
    assert s["zero_inflation_rate"] == pytest.approx(0.90)
    assert s["median"] == 0
    assert s["P95"] >= 10
    assert s["P99"] >= 100
    assert s["mean"] > s["median"]  # right-skewed


def test_robust_summary_empty():
    s = robust_summary([])
    assert s["n"] == 0


def test_robust_summary_single():
    s = robust_summary([42.0])
    assert s["n"] == 1
    assert s["mean"] == pytest.approx(42.0)
    assert s["median"] == pytest.approx(42.0)


def test_robust_summary_keys():
    s = robust_summary([1, 2, 3])
    required = {"n", "n_zero", "zero_inflation_rate", "mean", "stdev",
                "P5", "P25", "median", "P75", "P95", "P99", "max", "iqr"}
    assert required.issubset(s.keys())


def test_cliffs_delta_extremes():
    a = [1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0]
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])) < 0.01


def test_cliffs_delta_antisymmetry():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 50).tolist()
    b = rng.normal(0.5, 1, 50).tolist()
    assert cliffs_delta(a, b) == pytest.approx(-cliffs_delta(b, a), abs=1e-9)


def test_cliffs_delta_paths_agree():
    """Broadcast path and binary-search path must agree on small randomized inputs.
    We test at least 3 distinct random cases (seeds 0, 1, 2)."""
    import robust_stats as rs
    for seed in range(3):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 10, size=30).tolist()
        b = rng.integers(0, 10, size=25).tolist()
        # Force broadcast path (small arrays)
        broadcast = rs._cliffs_delta_broadcast(np.array(a, dtype=float),
                                                np.array(b, dtype=float))
        # Force binary-search path
        binsearch = rs._cliffs_delta_binsearch(np.array(a, dtype=float),
                                                np.array(b, dtype=float))
        assert broadcast == pytest.approx(binsearch, abs=1e-9), (
            f"seed={seed}: broadcast={broadcast} vs binsearch={binsearch}"
        )


def test_hodges_lehmann_centers_zero():
    deltas = [-2, -1, 0, 0, 0, 1, 2]
    assert abs(hodges_lehmann(deltas)) < 0.01


def test_hodges_lehmann_positive():
    deltas = [1, 2, 3, 4, 5]
    assert hodges_lehmann(deltas) > 0


def test_paired_permutation_p_calibration():
    rng = np.random.default_rng(42)
    deltas = rng.normal(0, 1, 100).tolist()
    p = paired_permutation_p(deltas, reps=1000, seed=0)
    assert 0.05 <= p <= 0.95


def test_paired_permutation_p_significant():
    """Large consistent positive shift should yield small p-value."""
    deltas = [5.0] * 100
    p = paired_permutation_p(deltas, reps=1000, seed=0)
    assert p < 0.05


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(5, 1, 200).tolist()
    lo, hi = bootstrap_ci(vals, np.mean, reps=2000, seed=0)
    assert lo < np.mean(vals) < hi


def test_bootstrap_ci_width():
    """Larger sample → narrower CI."""
    rng = np.random.default_rng(1)
    small = rng.normal(0, 1, 30).tolist()
    large = rng.normal(0, 1, 300).tolist()
    lo_s, hi_s = bootstrap_ci(small, np.mean, reps=1000, seed=0)
    lo_l, hi_l = bootstrap_ci(large, np.mean, reps=1000, seed=0)
    assert (hi_s - lo_s) > (hi_l - lo_l)


def test_practical_significance_label():
    assert practical_significance_label(0.05) == "none"
    assert practical_significance_label(0.20) == "small"
    assert practical_significance_label(0.40) == "medium"
    assert practical_significance_label(0.60) == "large"
    # boundary values
    assert practical_significance_label(0.147) == "small"    # exactly at boundary → next tier
    assert practical_significance_label(0.146) == "none"
    assert practical_significance_label(0.33) == "medium"
    assert practical_significance_label(0.329) == "small"
    assert practical_significance_label(0.474) == "large"
    assert practical_significance_label(0.473) == "medium"
    # negative deltas use abs
    assert practical_significance_label(-0.5) == "large"


def test_paired_permutation_p_floor():
    # Constant non-zero deltas → p must be small but bounded below by 1/(reps+1)
    # This locks in the Phipson-Smyth smoothing convention so a future regression
    # to unsmoothed counts/reps would fail this test.
    p = paired_permutation_p([5.0]*50, reps=100, seed=0)
    assert p >= 1.0 / 101
    assert p <= 5.0 / 101  # very small but not zero


def test_cohens_d_paired_zero_diff():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert abs(cohens_d_paired(a, b)) < 1e-9


def test_cohens_d_paired_known():
    """diffs all = 1.0 → std = 0 → d = inf (or very large)."""
    a = [1.0] * 10
    b = [2.0] * 10
    # diff = -1 always, std_diff = 0 → d = -inf
    d = cohens_d_paired(a, b)
    assert not math.isfinite(d) or abs(d) > 100


# ---------------------------------------------------------------------------
# Tests for null-distribution z-score logic (Fix 1 — review #7, #12)
# These test the sign-flip null-distribution pattern used in phase6
# _build_delta_stats to produce null_cliffs_delta_{mean,sd,z}.
# The logic is inlined in phase6 but the underlying primitives are here.
# ---------------------------------------------------------------------------

def _compute_null_cd_distribution(arr_l, arr_w, reps=2000, seed=43):
    """Replicate the sign-flip null distribution from phase6._build_delta_stats."""
    n = len(arr_l)
    rng = np.random.default_rng(seed)
    null_cd_vals = np.empty(reps, dtype=float)
    for i in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=n)
        pos_mask = signs > 0
        null_l = np.where(pos_mask, arr_l, arr_w)
        null_w = np.where(pos_mask, arr_w, arr_l)
        null_cd_vals[i] = cliffs_delta(null_l, null_w)
    return null_cd_vals


def test_null_cliffs_delta_centers_at_zero_for_symmetric_data():
    """For symmetric (exchangeable) data, sign-flip null Cliff's δ should center ≈ 0.

    When arr_l and arr_w are drawn from the same distribution, every sign-flip
    permutation is equally valid and the null distribution of Cliff's δ should
    be centered at 0 (within Monte Carlo noise).
    """
    rng = np.random.default_rng(99)
    n = 500
    # Both sides from the same distribution → exchangeable pairs
    arr_l = rng.normal(0.0, 1.0, n).astype(float)
    arr_w = rng.normal(0.0, 1.0, n).astype(float)

    null_vals = _compute_null_cd_distribution(arr_l, arr_w, reps=2000, seed=43)
    null_mean = float(np.mean(null_vals))

    # Null mean should be very close to 0; allow ±0.05 for Monte Carlo variation
    assert abs(null_mean) < 0.05, (
        f"null_cliffs_delta_mean={null_mean:.4f} is too far from 0 for symmetric data"
    )


def test_null_cliffs_delta_z_positive_for_positive_effect():
    """Positive observed Cliff's δ should yield positive z relative to the null.

    When arr_l consistently exceeds arr_w (clear DPO penalty signal),
    the observed Cliff's δ should be positive and well above the null mean.
    """
    rng = np.random.default_rng(7)
    n = 300
    # arr_l (rejected) is clearly higher than arr_w (chosen) — strong positive effect
    arr_l = rng.normal(1.0, 1.0, n).astype(float)
    arr_w = rng.normal(0.0, 1.0, n).astype(float)

    observed_cd = cliffs_delta(arr_l, arr_w)
    assert observed_cd > 0.1, f"Expected positive Cliff's δ, got {observed_cd:.4f}"

    null_vals = _compute_null_cd_distribution(arr_l, arr_w, reps=2000, seed=43)
    null_mean = float(np.mean(null_vals))
    null_sd   = float(np.std(null_vals, ddof=1))
    null_z    = (observed_cd - null_mean) / null_sd

    assert null_z > 3.0, (
        f"Expected null_cliffs_delta_z > 3 for strong positive effect, "
        f"got z={null_z:.2f} (cd={observed_cd:.4f}, null_mean={null_mean:.4f}, "
        f"null_sd={null_sd:.4f})"
    )
