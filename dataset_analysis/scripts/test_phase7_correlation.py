"""Tests for phase7_correlation.py.

Run with: python -m pytest test_phase7_correlation.py -v
      or: python test_phase7_correlation.py

All tests use real (smoke or full) phase5 outputs that exist under
dataset_analysis/results/.  The domain_breakdown.csv is at its canonical
location relative to the repo root.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from the scripts/ directory when invoked directly
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phase7_correlation import (
    build_three_pillars_table,
    per_domain_pillar3_correlation,
    per_source_dataset_pillar3_correlation,
)

# ---------------------------------------------------------------------------
# Canonical file paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS   = _SCRIPT_DIR.parent / "results"

# Use full data when available; fall back to smoke outputs for testing.
def _pick_phase5(preferred_tag: str = "instruct-sft") -> tuple[Path, Path, Path]:
    """Return (per_example, summary, by_source) paths for the best available tag."""
    for tag in [preferred_tag, "review_smoke", "smoke"]:
        pe  = _RESULTS / f"phase5_{tag}_per_example.csv"
        sm  = _RESULTS / f"phase5_{tag}_summary.json"
        bys = _RESULTS / f"phase5_{tag}_by_source.csv"
        if pe.exists() and sm.exists():
            return pe, sm, bys
    raise FileNotFoundError(
        "No phase5 output files found. Run phase5_consensus_audit.py first."
    )


def _pick_phase6(preferred_tag: str = "instruct-dpo") -> Path:
    """Return path to a phase6 summary JSON (first existing tag)."""
    for tag in [preferred_tag, "review_smoke", "smoke"]:
        p = _RESULTS / f"phase6_{tag}_summary.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        "No phase6 output files found. Run phase6_dpo_consensus_audit.py first."
    )


_DOMAIN_BREAKDOWN = (
    _REPO_ROOT / "Comparing_Experiments/April_analysis/tables/behavioral/domain_breakdown.csv"
)


# ---------------------------------------------------------------------------
# Plan-mandated test (must be verbatim-equivalent to plan spec)
# ---------------------------------------------------------------------------

def test_phase7_pillar3_returns_finite():
    """Plan-mandated test: per_domain_pillar3_correlation returns sensible output.

    Updated (Fix 2 / review #4): when n_domains < 5, the function now returns
    status='underpowered' with spearman_rho=None (not a misleadingly precise float).
    When n_domains >= 5 (enough rank permutations for a meaningful test), it returns
    a finite float. The test validates both branches.
    """
    pe, _, _ = _pick_phase5("instruct-sft")
    out = per_domain_pillar3_correlation(
        phase5_per_example=str(pe),
        domain_breakdown_csv=str(_DOMAIN_BREAKDOWN),
        metric_col="has_run_5_response",
        boot_reps=50,
        seed=0,
    )
    assert "spearman_rho" in out, "output must have 'spearman_rho' key"
    assert "n_domains" in out, "output must have 'n_domains' key"

    n = out["n_domains"]
    rho = out["spearman_rho"]

    if n < 5:
        # Fix 2: underpowered — rho must be None and status must reflect it
        assert rho is None, (
            f"spearman_rho should be None for underpowered n_domains={n}, got {rho!r}"
        )
        assert out.get("status") == "underpowered", (
            f"status should be 'underpowered' for n_domains={n}, got {out.get('status')!r}"
        )
        assert "reason" in out, "underpowered result must include 'reason' key"
    elif n >= 5:
        # Enough domains — rho should be a finite float
        assert isinstance(rho, float), f"spearman_rho must be float, got {type(rho)}"
        assert not math.isnan(rho), (
            f"spearman_rho is NaN with {n} domains — check domain canonicalization"
        )
    else:
        pytest.skip(f"Only {n} domain(s) in data — skipping finiteness check")


# ---------------------------------------------------------------------------
# Test 2: build_three_pillars_table returns a DataFrame with the right columns
# ---------------------------------------------------------------------------

def test_build_three_pillars_table_columns():
    """build_three_pillars_table must return a DataFrame with exactly the right columns."""
    _, sm, _ = _pick_phase5("instruct-sft")
    p6_sm = _pick_phase6()

    df = build_three_pillars_table(
        phase5_summary=str(sm),
        phase6_summary=str(p6_sm),
        domain_breakdown_csv=str(_DOMAIN_BREAKDOWN),
    )

    assert isinstance(df, pd.DataFrame), "must return a pandas DataFrame"

    expected_cols = {
        "pillar", "metric", "stage", "value", "ci_lo", "ci_hi",
        "effect_size_label", "interpretation",
    }
    assert set(df.columns) == expected_cols, (
        f"column mismatch. got={set(df.columns)}, expected={expected_cols}"
    )

    # Pillar I rows must be present
    assert "I" in df["pillar"].values, "Pillar I rows missing"

    # Pillar II rows must be present when phase6 summary exists
    assert "II" in df["pillar"].values, "Pillar II rows missing"

    # All values in 'value' column must be numeric
    assert df["value"].apply(lambda v: isinstance(v, (int, float))).all(), (
        "all 'value' entries must be numeric"
    )


# ---------------------------------------------------------------------------
# Test 3: per_source_dataset_pillar3_correlation gracefully handles empty dict
# ---------------------------------------------------------------------------

def test_per_source_gracefully_handles_empty_source_to_domain():
    """per_source_dataset_pillar3_correlation must warn + return placeholder when
    source_to_domain is empty (Task 4 not yet complete)."""
    _, _, bys = _pick_phase5("instruct-sft")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = per_source_dataset_pillar3_correlation(
            phase5_by_source=str(bys),
            domain_breakdown_csv=str(_DOMAIN_BREAKDOWN),
            source_to_domain={},
            metric_col="max_run_geq_5_rate",
            boot_reps=10,
            seed=0,
        )
        warning_msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]

    assert out.get("skipped") is True, "result must have skipped=True"
    assert "reason" in out, "result must include 'reason' key"
    assert math.isnan(out["mean_within_domain_spearman_rho"]), (
        "mean rho must be NaN when skipped"
    )
    assert any("SOURCE_DATASET_TO_BER_DOMAIN" in m or "empty" in m.lower()
               for m in warning_msgs), (
        f"expected a warning about empty source_to_domain, got: {warning_msgs}"
    )


# ---------------------------------------------------------------------------
# Test 4: per_domain BER delta loading produces expected sign for math domain
# ---------------------------------------------------------------------------

def test_ber_delta_math_is_positive_for_instruct_sft():
    """instruct_sft should show higher BER than base in math under asch_ T=0 conditions.

    This is a directional sanity check based on the paper's core finding.
    """
    from phase7_correlation import _load_ber_delta

    delta = _load_ber_delta(
        domain_breakdown_csv=str(_DOMAIN_BREAKDOWN),
        base_variant="base",
        sft_variant="instruct_sft",
        temperature=0.0,
    )

    assert "math" in delta, "math domain must appear in BER delta dict"
    assert delta["math"] > 0, (
        f"instruct_sft should have higher BER than base in math "
        f"(delta={delta['math']:.4f}). If this fails, check variant strings "
        "in domain_breakdown.csv."
    )


# ---------------------------------------------------------------------------
# Test 5: consensus_hits_response ρ returns a value (may be nan for small data)
# ---------------------------------------------------------------------------

def test_pillar3_consensus_hits_returns_output():
    """per_domain_pillar3_correlation for consensus_hits_response must return a dict."""
    pe, _, _ = _pick_phase5("instruct-sft")
    out = per_domain_pillar3_correlation(
        phase5_per_example=str(pe),
        domain_breakdown_csv=str(_DOMAIN_BREAKDOWN),
        metric_col="consensus_hits_response",
        boot_reps=50,
        seed=42,
    )
    assert isinstance(out, dict)
    assert "spearman_rho" in out
    assert "n_domains" in out
    assert "ci_lo" in out and "ci_hi" in out


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        test_phase7_pillar3_returns_finite,
        test_build_three_pillars_table_columns,
        test_per_source_gracefully_handles_empty_source_to_domain,
        test_ber_delta_math_is_positive_for_instruct_sft,
        test_pillar3_consensus_hits_returns_output,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
