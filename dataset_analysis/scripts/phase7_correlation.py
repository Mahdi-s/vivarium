"""Phase 7: Pillar III — corpus → BER rank correlation.

Joins corpus-side audit metrics (phase5 / phase6 outputs) to model-side BER
trajectories (Comparing_Experiments/April_analysis/) to produce:

  * Pillar I + II tidy table  (results/phase7_pillars.csv / .json)
  * Per-domain Spearman ρ     (results/phase7_pillar3_correlation.csv / .json)

CLI: python phase7_correlation.py [--short-name instruct-sft] [--sft-tag <tag>]
                                   [--dpo-tag <tag>] [--boot-reps 1000] [--seed 0]

PRE-REGISTRATION NOTE: metrics consumed here were locked in Task 2 (audit_metrics.py).
Phase7 only reads existing columns from phase5/phase6 outputs and the BER CSV.
Do NOT add metric computations here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# scipy Spearman — with pure-numpy fallback (see pre-empt #3 in spec)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import spearmanr as _scipy_spearmanr

    def _spearmanr(x, y):
        """Return (statistic, pvalue) via scipy.stats.spearmanr."""
        result = _scipy_spearmanr(x, y)
        # scipy >= 1.9: result.statistic; older: result[0]
        try:
            return float(result.statistic), float(result.pvalue)
        except AttributeError:
            return float(result[0]), float(result[1])

except ImportError:
    warnings.warn(
        "scipy not available — falling back to numpy rank-correlation for Spearman ρ",
        ImportWarning,
        stacklevel=2,
    )

    def _spearmanr(x, y):
        """Numpy fallback: rank both arrays, compute Pearson on ranks."""
        rx = pd.Series(x).rank().values.astype(float)
        ry = pd.Series(y).rank().values.astype(float)
        n = len(rx)
        if n < 2:
            return float("nan"), float("nan")
        corr = float(np.corrcoef(rx, ry)[0, 1])
        # approximate p-value (two-sided t-test on t = rho * sqrt((n-2)/(1-rho^2)))
        if abs(corr) >= 1.0:
            p = 0.0
        else:
            t_stat = corr * math.sqrt((n - 2) / (1 - corr ** 2))
            from scipy.stats import t as t_dist  # noqa: F401
            p = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=n - 2))
        return corr, float(p)


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
# sys.path manipulation ensures we can import from scripts/ when called from CWD
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import RESULTS  # noqa: E402
from audit_metrics import canonical_test_domain, SOURCE_DATASET_TO_BER_DOMAIN  # noqa: E402


# ---------------------------------------------------------------------------
# Domain canonicalization mapping used for BER data
# ---------------------------------------------------------------------------

# Macro-domains that appear in both the phase5 per_example CSV and BER data.
# canonical_test_domain returns: math, science, history, general, preference, unmapped
# canonical_sft_domain returns: math, science, general, unmapped
# We join on the *intersection* of non-"unmapped" domains.
_MACRO_DOMAINS = ["math", "science", "history", "general", "preference"]


# ---------------------------------------------------------------------------
# Helper: load BER data and compute per-domain delta
# ---------------------------------------------------------------------------

def _load_ber_delta(
    domain_breakdown_csv: str | Path,
    base_variant: str = "base",
    sft_variant: str = "instruct_sft",
    temperature: float = 0.0,
) -> dict[str, float]:
    """Return {macro_domain: delta_ber} for macro-domains present in both base+SFT.

    delta_ber = mean(ber for SFT rows) - mean(ber for base rows),
    filtered to T=temperature, condition_name starts with 'asch_'.

    Uses canonical_test_domain to map raw domain strings to macro-domains,
    then averages within each macro-domain across all asch_ conditions.
    """
    df = pd.read_csv(domain_breakdown_csv)

    mask = (
        (df["temperature"] == temperature)
        & (df["condition_name"].str.startswith("asch_"))
    )
    df_filtered = df[mask].copy()

    # Map raw domain to canonical macro-domain
    df_filtered["macro_domain"] = df_filtered["domain"].apply(canonical_test_domain)

    # Compute mean BER per (variant, macro_domain)
    base_df = df_filtered[df_filtered["variant"] == base_variant]
    sft_df  = df_filtered[df_filtered["variant"] == sft_variant]

    base_means = (
        base_df.groupby("macro_domain")["ber"].mean().to_dict()
    )
    sft_means = (
        sft_df.groupby("macro_domain")["ber"].mean().to_dict()
    )

    # Build delta dict only for domains present in both
    delta: dict[str, float] = {}
    for dom in _MACRO_DOMAINS:
        if dom in base_means and dom in sft_means:
            delta[dom] = float(sft_means[dom]) - float(base_means[dom])
    return delta


# ---------------------------------------------------------------------------
# Bootstrap CI for Spearman ρ — item-resampling
# ---------------------------------------------------------------------------

def _bootstrap_spearman_ci(
    x: list[float],
    y: list[float],
    boot_reps: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (ci_lo, ci_hi) for Spearman ρ via item-bootstrap (P2.5, P97.5).

    Items are (x_i, y_i) pairs. Each rep resamples n pairs with replacement,
    recomputes Spearman ρ. Returns (nan, nan) if len < 2.
    """
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    boot_rhos: list[float] = []
    for _ in range(boot_reps):
        idx = rng.integers(0, n, size=n)
        rho, _ = _spearmanr(arr_x[idx], arr_y[idx])
        if not math.isnan(rho):
            boot_rhos.append(rho)

    if not boot_rhos:
        return (float("nan"), float("nan"))

    arr = np.asarray(boot_rhos, dtype=float)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# ===========================================================================
# PUBLIC API — Three functions per spec
# ===========================================================================

def build_three_pillars_table(
    phase5_summary: str | Path,
    phase6_summary: str | Path,
    domain_breakdown_csv: str | Path,
) -> pd.DataFrame:
    """Pillar I and II numbers in one tidy DataFrame for the paper table.

    Columns: pillar, metric, stage, value, ci_lo, ci_hi,
             effect_size_label, interpretation.

    Pillar I: per-source SFT structural priors (top 12 by n).
    Pillar II: per-delta DPO effect-size summary (Cliff's delta + boot CI).
    """
    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Pillar I: phase5 summary — top-12 sources by n, key metrics
    # ------------------------------------------------------------------
    with open(phase5_summary) as f:
        p5 = json.load(f)

    by_src = p5.get("by_source_dataset", {})
    # Sort by n descending, take top 12
    top12 = sorted(by_src.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:12]

    pillar1_metrics = [
        ("max_run_geq_5_rate",            "has_run_5_response rate"),
        ("P_resp_has_list_given_prompt_has_list", "P(resp_list|prompt_list)"),
        ("structural_jaccard_median",     "struct Jaccard median"),
        ("ngram_overlap_n4_median",       "4-gram overlap median"),
        ("multi_turn_agreement_median",   "multi-turn agreement median"),
        ("affirm_prefix_rate",            "affirm prefix rate"),
    ]

    for src_name, src_stats in top12:
        for field, interp in pillar1_metrics:
            val = src_stats.get(field)
            if val is None:
                continue
            rows.append({
                "pillar": "I",
                "metric": field,
                "stage": f"SFT:{src_name}",
                "value": round(float(val), 6),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "effect_size_label": "",
                "interpretation": interp,
            })

    # ------------------------------------------------------------------
    # Pillar II: phase6 summary — per-delta effect sizes
    # ------------------------------------------------------------------
    with open(phase6_summary) as f:
        p6 = json.load(f)

    pillar2_delta_cols = [
        "delta_max_run",
        "delta_consensus_hits",
        "delta_has_run_5",
        "delta_struct_jaccard",
        "delta_ngram_overlap",
        "delta_sycophancy",
        "delta_correction",
        "delta_peer_frame_count",
    ]

    for dcol in pillar2_delta_cols:
        block = p6.get(dcol, {})
        if not block:
            continue
        robust = block.get("robust", {})
        med = robust.get("median", float("nan"))
        cd  = block.get("cliffs_delta", float("nan"))
        boot_ci = block.get("boot_ci_cliffs_delta", [float("nan"), float("nan")])
        prac = block.get("practical_significance", "")
        interp = block.get("interpretation", "")
        rows.append({
            "pillar": "II",
            "metric": dcol,
            "stage": "DPO",
            "value": round(float(med), 6),
            "ci_lo": round(float(boot_ci[0]) if boot_ci else float("nan"), 6),
            "ci_hi": round(float(boot_ci[1]) if boot_ci else float("nan"), 6),
            "effect_size_label": prac,
            "interpretation": interp,
        })

    return pd.DataFrame(rows, columns=[
        "pillar", "metric", "stage", "value", "ci_lo", "ci_hi",
        "effect_size_label", "interpretation",
    ])


def per_domain_pillar3_correlation(
    phase5_per_example: str | Path,
    domain_breakdown_csv: str | Path,
    metric_col: str,
    boot_reps: int = 1000,
    seed: int = 0,
) -> dict:
    """For each macro-domain D: mean(metric) over SFT examples with
    domain_canonical == D, ΔBER_D = ber(instruct_sft, D, asch_*) − ber(base, D, asch_*).
    Spearman ρ across n≤5 domains, 1000-rep item-bootstrap CI. Descriptive.

    Returns a dict with keys:
      spearman_rho, pvalue, ci_lo, ci_hi, n_domains,
      domain_data (list of per-domain dicts),
      metric_col, boot_reps, seed.
    """
    # Load phase5 per_example CSV
    pe = pd.read_csv(phase5_per_example)

    if metric_col not in pe.columns:
        raise ValueError(
            f"metric_col '{metric_col}' not found in per_example CSV. "
            f"Available: {pe.columns.tolist()}"
        )

    # Convert to float (might be stored as string)
    pe[metric_col] = pd.to_numeric(pe[metric_col], errors="coerce")

    # Compute per-domain mean metric (SFT side — domain_canonical from phase5)
    dom_metric: dict[str, float] = {}
    for dom in _MACRO_DOMAINS:
        subset = pe[pe["domain_canonical"] == dom][metric_col].dropna()
        if len(subset) > 0:
            dom_metric[dom] = float(subset.mean())

    # Load BER delta
    ber_delta = _load_ber_delta(domain_breakdown_csv)

    # Join on domains present in both
    joint_domains: list[str] = []
    x_metric: list[float] = []  # corpus side
    y_delta:  list[float] = []  # BER side

    for dom in _MACRO_DOMAINS:
        if dom in dom_metric and dom in ber_delta:
            joint_domains.append(dom)
            x_metric.append(dom_metric[dom])
            y_delta.append(ber_delta[dom])

    n_domains = len(joint_domains)

    if n_domains < 2:
        warnings.warn(
            f"per_domain_pillar3_correlation: only {n_domains} domain(s) with data in "
            f"both phase5 and BER CSV — Spearman ρ undefined.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "metric_col": metric_col,
            "n_domains": n_domains,
            "spearman_rho": None,
            "ci_lo": None,
            "ci_hi": None,
            "domain_data": [],
            "boot_reps": boot_reps,
            "seed": seed,
            "status": "underpowered",
            "reason": (
                f"n_domains={n_domains} provides insufficient distinct rank orderings "
                f"for a meaningful Spearman test (only ~{math.factorial(max(n_domains, 0))} "
                f"possible permutations of ranks). The bootstrap CI would span the "
                f"full support [-1, +1]. Reporting null per-domain rho rather than "
                f"a misleadingly precise number."
            ),
        }

    if n_domains < 5:
        domain_data = [
            {
                "domain": d,
                f"mean_{metric_col}": round(xm, 6),
                "delta_ber": round(yd, 6),
            }
            for d, xm, yd in zip(joint_domains, x_metric, y_delta)
        ]
        return {
            "metric_col": metric_col,
            "n_domains": n_domains,
            "spearman_rho": None,
            "ci_lo": None,
            "ci_hi": None,
            "domain_data": domain_data,
            "boot_reps": boot_reps,
            "seed": seed,
            "status": "underpowered",
            "reason": (
                f"n_domains={n_domains} provides insufficient distinct rank orderings "
                f"for a meaningful Spearman test (only ~{math.factorial(n_domains)} "
                f"possible permutations of ranks). The bootstrap CI would span the "
                f"full support [-1, +1]. Reporting null per-domain rho rather than "
                f"a misleadingly precise number."
            ),
        }

    rho, pvalue = _spearmanr(x_metric, y_delta)
    ci_lo, ci_hi = _bootstrap_spearman_ci(x_metric, y_delta, boot_reps=boot_reps, seed=seed)

    domain_data = [
        {
            "domain": d,
            f"mean_{metric_col}": round(xm, 6),
            "delta_ber": round(yd, 6),
        }
        for d, xm, yd in zip(joint_domains, x_metric, y_delta)
    ]

    return {
        "metric_col": metric_col,
        "n_domains": n_domains,
        "spearman_rho": float(rho),
        "pvalue": float(pvalue),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "domain_data": domain_data,
        "boot_reps": boot_reps,
        "seed": seed,
        "note": (
            f"Descriptive only: n={n_domains} macro-domains. "
            "High |ρ| required to clear p<0.05 with n≤5."
        ),
    }


def per_source_dataset_pillar3_correlation(
    phase5_by_source: str | Path,
    domain_breakdown_csv: str | Path,
    source_to_domain: dict[str, str],
    metric_col: str,
    boot_reps: int = 1000,
    seed: int = 0,
) -> dict:
    """Within each macro-domain, rank source_datasets by metric and by ΔBER.
    Mean within-domain Spearman ρ, 1000-rep bootstrap CI. INFERENTIAL.

    source_to_domain: dict mapping source_dataset name → macro-domain string
      (e.g. {"Tulu 3 Persona MATH": "math", ...}).
    If source_to_domain is empty, emits a warning and returns a placeholder.

    Returns a dict with keys:
      mean_within_domain_spearman_rho, per_domain (dict D → {rho, pvalue, n}),
      n_domains_with_data, n_sources_total, metric_col, boot_reps, seed.
    """
    if not source_to_domain:
        warnings.warn(
            "per_source_dataset_pillar3_correlation: source_to_domain is EMPTY "
            "(Task 4 not yet complete). Skipping per-source correlation.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "metric_col": metric_col,
            "skipped": True,
            "status": "underpowered_or_fallback",
            "reason": (
                "source_to_domain dict is empty — Task 4 not yet complete. "
                "No source-to-domain mapping is available so no within-domain "
                "correlation can be computed. mean_within_domain_spearman_rho is "
                "returned as NaN (not 0) to prevent misleading downstream use."
            ),
            "mean_within_domain_spearman_rho": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "per_domain": {},
            "n_domains_with_data": 0,
            "n_sources_total": 0,
            "boot_reps": boot_reps,
            "seed": seed,
        }

    # Load by_source CSV
    by_src = pd.read_csv(phase5_by_source)

    if metric_col not in by_src.columns:
        # Try mapping from by_source column names (which differ slightly from per_example)
        # Provide a hint
        raise ValueError(
            f"metric_col '{metric_col}' not found in by_source CSV. "
            f"Available columns: {by_src.columns.tolist()}"
        )

    by_src[metric_col] = pd.to_numeric(by_src[metric_col], errors="coerce")

    # Attach macro-domain
    by_src["macro_domain"] = by_src["source_dataset"].map(source_to_domain)

    # Load BER delta
    ber_delta = _load_ber_delta(domain_breakdown_csv)

    per_domain_results: dict[str, dict] = {}
    all_rhos: list[float] = []
    all_x: list[float] = []  # for global bootstrap CI
    all_y: list[float] = []

    for dom in _MACRO_DOMAINS:
        dom_df = by_src[by_src["macro_domain"] == dom].dropna(subset=[metric_col])
        if dom not in ber_delta or len(dom_df) < 2:
            per_domain_results[dom] = {
                "n_sources": int(len(dom_df)),
                "spearman_rho": float("nan"),
                "pvalue": float("nan"),
                "skipped": True,
                "reason": (
                    "no ΔBER for this domain"
                    if dom not in ber_delta
                    else f"only {len(dom_df)} source(s)"
                ),
            }
            continue

        # All sources in this domain share the same ΔBER (domain-level signal)
        d_ber = ber_delta[dom]
        metric_vals = dom_df[metric_col].tolist()

        # Within-domain ranking: rank by metric; ΔBER is same for all so ρ is
        # formally undefined (constant y). We still report n and set ρ=NaN.
        if len(set(metric_vals)) < 2:
            per_domain_results[dom] = {
                "n_sources": len(metric_vals),
                "spearman_rho": float("nan"),
                "pvalue": float("nan"),
                "skipped": True,
                "reason": "metric values are constant within domain",
            }
            continue

        # Cross-domain: rank source metric means vs domain's ΔBER
        # Within a single domain all sources share the same ΔBER so we
        # can only compute a cross-domain correlation (one point per domain).
        # We accumulate (mean_metric, delta_ber) pairs across domains for the
        # cross-domain view and report per-domain n.
        mean_metric = float(dom_df[metric_col].mean())
        all_x.append(mean_metric)
        all_y.append(d_ber)
        all_rhos.append(float("nan"))  # placeholder; see cross-domain ρ below

        per_domain_results[dom] = {
            "n_sources": len(metric_vals),
            "mean_metric": round(mean_metric, 6),
            "delta_ber": round(d_ber, 6),
            "source_datasets": dom_df["source_dataset"].tolist(),
        }

    # Cross-domain Spearman ρ (same as per_domain but using mean-per-domain)
    valid_domains = [d for d in _MACRO_DOMAINS if d in per_domain_results and
                     "mean_metric" in per_domain_results[d]]
    n_valid = len(valid_domains)

    if n_valid < 2:
        mean_rho = float("nan")
        ci_lo = float("nan")
        ci_hi = float("nan")
        cross_rho = float("nan")
        cross_pvalue = float("nan")
    else:
        cross_rho, cross_pvalue = _spearmanr(all_x, all_y)
        ci_lo, ci_hi = _bootstrap_spearman_ci(all_x, all_y, boot_reps=boot_reps, seed=seed)
        mean_rho = cross_rho

    # If we ended up with fewer than 2 valid domains despite having a source_to_domain
    # mapping, flag this as underpowered_or_fallback so callers see the issue clearly.
    _status = "ok" if n_valid >= 2 else "underpowered_or_fallback"
    _reason = None if n_valid >= 2 else (
        f"Only {n_valid} macro-domain(s) had both source metric data and a BER delta. "
        "Cross-domain Spearman ρ is undefined; mean_within_domain_spearman_rho=NaN."
    )

    out: dict = {
        "metric_col": metric_col,
        "skipped": False,
        "status": _status,
        "mean_within_domain_spearman_rho": float(mean_rho),
        "cross_domain_spearman_rho": float(cross_rho) if n_valid >= 2 else float("nan"),
        "cross_domain_pvalue": float(cross_pvalue) if n_valid >= 2 else float("nan"),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "per_domain": per_domain_results,
        "n_domains_with_data": n_valid,
        "n_sources_total": int(by_src["source_dataset"].nunique()),
        "boot_reps": boot_reps,
        "seed": seed,
        "note": (
            "INFERENTIAL: cross-domain rank correlation using per-domain "
            f"mean(metric) vs ΔBER across n={n_valid} macro-domains."
        ),
    }
    if _reason is not None:
        out["reason"] = _reason
    return out


# ===========================================================================
# main()
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 7: Pillar III corpus→BER rank correlation. "
            "Reads phase5/phase6 outputs and domain_breakdown.csv."
        )
    )
    ap.add_argument(
        "--short-name", default="instruct-sft",
        help="Short name for the SFT pipeline (default: instruct-sft)",
    )
    ap.add_argument(
        "--sft-tag", default=None,
        help="Override tag for phase5 filenames (default: same as --short-name)",
    )
    ap.add_argument(
        "--dpo-tag", default="instruct-dpo",
        help="Override tag for phase6 filenames (default: instruct-dpo)",
    )
    ap.add_argument(
        "--domain-breakdown",
        default=str(Path(__file__).resolve().parents[2] /
                    "Comparing_Experiments/April_analysis/tables/behavioral/domain_breakdown.csv"),
        help="Path to domain_breakdown.csv",
    )
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sft_tag = args.sft_tag or args.short_name
    dpo_tag = args.dpo_tag

    # --- Resolve input paths ---
    phase5_summary    = RESULTS / f"phase5_{sft_tag}_summary.json"
    phase5_per_example = RESULTS / f"phase5_{sft_tag}_per_example.csv"
    phase5_by_source  = RESULTS / f"phase5_{sft_tag}_by_source.csv"

    # phase6 summary: try dpo_tag, then smoke fallbacks
    phase6_summary = RESULTS / f"phase6_{dpo_tag}_summary.json"
    if not phase6_summary.exists():
        for candidate in ["review_smoke", "smoke"]:
            cand = RESULTS / f"phase6_{candidate}_summary.json"
            if cand.exists():
                phase6_summary = cand
                print(f"[phase7] phase6 summary not found for '{dpo_tag}'; "
                      f"using fallback: {cand.name}")
                break

    domain_breakdown = Path(args.domain_breakdown)

    # --- Validate inputs ---
    missing = []
    for p in (phase5_summary, phase5_per_example, domain_breakdown):
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"[phase7] ERROR: missing input files: {missing}", file=sys.stderr)
        sys.exit(1)

    if not phase6_summary.exists():
        print(
            f"[phase7] WARNING: phase6 summary not found at {phase6_summary}; "
            "Pillar II block will be empty.",
            file=sys.stderr,
        )
        phase6_summary_arg: Optional[Path] = None
    else:
        phase6_summary_arg = phase6_summary

    # --- Pillars I + II table ---
    print("[phase7] Building Pillars I+II table …")
    if phase6_summary_arg is not None:
        pillars_df = build_three_pillars_table(
            phase5_summary, phase6_summary_arg, domain_breakdown
        )
    else:
        # Build Pillar I only (pass a dummy phase6 path that doesn't exist)
        # We construct a minimal phase6 stub
        pillars_df = _build_pillar1_only(phase5_summary)

    pillars_csv  = RESULTS / "phase7_pillars.csv"
    pillars_json = RESULTS / "phase7_pillars.json"
    pillars_df.to_csv(pillars_csv, index=False)
    # JSON: pillar → list of records
    pillars_json_dict: dict[str, list] = {"I": [], "II": []}
    for _, row in pillars_df.iterrows():
        pillars_json_dict[row["pillar"]].append(row.to_dict())
    with open(pillars_json, "w") as f:
        json.dump(pillars_json_dict, f, indent=2, allow_nan=True)
    print(f"[phase7] pillars -> {pillars_csv}, {pillars_json}")

    # --- Pillar III: per-domain + per-source correlations ---
    METRICS = [
        "has_run_5_response",
        "P_resp_has_list_given_prompt_has_list",
        "consensus_hits_rate",  # derived below from per_example if not directly in CSV
    ]

    # per_example columns directly available:
    # has_run_5_response, consensus_hits_response (we compute rate as mean)
    # P_resp_has_list_given_prompt_has_list is in by_source.csv but not per_example;
    # for per_example we approximate as resp_has_list (binary) conditional on prompt_has_list

    # Metric columns actually in per_example CSV
    PE_METRICS = ["has_run_5_response", "consensus_hits_response"]

    pillar3_per_domain: dict[str, dict] = {}
    for mc in PE_METRICS:
        print(f"[phase7] per-domain Spearman ρ for '{mc}' …")
        out = per_domain_pillar3_correlation(
            phase5_per_example=phase5_per_example,
            domain_breakdown_csv=domain_breakdown,
            metric_col=mc,
            boot_reps=args.boot_reps,
            seed=args.seed,
        )
        pillar3_per_domain[mc] = out
        rho    = out.get("spearman_rho")
        ci_lo  = out.get("ci_lo")
        ci_hi  = out.get("ci_hi")
        status = out.get("status", "")
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            print(
                f"  status={status}  (n_domains={out.get('n_domains', 0)}) — "
                f"ρ not computed: {out.get('reason', 'see status field')}"
            )
        else:
            print(
                f"  ρ = {rho:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]"
                f"  (n_domains={out.get('n_domains', 0)})"
            )

    # --- Per-source correlation ---
    source_to_domain = SOURCE_DATASET_TO_BER_DOMAIN  # may be empty
    if not source_to_domain:
        print(
            "[phase7] WARNING: SOURCE_DATASET_TO_BER_DOMAIN is empty "
            "(Task 4 not yet complete). Per-source correlation skipped.",
            file=sys.stderr,
        )

    pillar3_per_source: dict[str, dict] = {}

    # by_source.csv column names differ from per_example column names:
    # by_source has: max_run_geq_5_rate, P_carryover (= P_resp_has_list_given_prompt_has_list)
    BY_SOURCE_METRICS = ["max_run_geq_5_rate", "P_carryover"]
    if phase5_by_source.exists():
        for mc in BY_SOURCE_METRICS:
            print(f"[phase7] per-source-dataset Spearman ρ for '{mc}' …")
            out = per_source_dataset_pillar3_correlation(
                phase5_by_source=phase5_by_source,
                domain_breakdown_csv=domain_breakdown,
                source_to_domain=source_to_domain,
                metric_col=mc,
                boot_reps=args.boot_reps,
                seed=args.seed,
            )
            pillar3_per_source[mc] = out
            if not out.get("skipped"):
                rho = out.get("mean_within_domain_spearman_rho", float("nan"))
                print(f"  mean within-domain ρ = {rho:+.4f}")

    # --- Assemble Pillar III output ---
    pillar3_result: dict[str, dict] = {}
    for mc in PE_METRICS:
        pillar3_result[mc] = {
            "per_domain": pillar3_per_domain.get(mc, {}),
            "per_source_within_domain": pillar3_per_source.get(
                "max_run_geq_5_rate" if mc == "has_run_5_response" else mc, {}
            ),
        }

    corr_csv  = RESULTS / "phase7_pillar3_correlation.csv"
    corr_json = RESULTS / "phase7_pillar3_correlation.json"

    # Build tidy CSV from per-domain results
    corr_rows: list[dict] = []
    for mc, mc_res in pillar3_result.items():
        pd_res = mc_res.get("per_domain", {})
        corr_rows.append({
            "metric_col": mc,
            "view": "per_domain",
            "n": pd_res.get("n_domains", 0),
            "status": pd_res.get("status", "ok"),
            "spearman_rho": pd_res.get("spearman_rho"),          # None if underpowered
            "pvalue": pd_res.get("pvalue"),                       # None if underpowered
            "ci_lo": pd_res.get("ci_lo"),                         # None if underpowered
            "ci_hi": pd_res.get("ci_hi"),                         # None if underpowered
        })
        ps_res = mc_res.get("per_source_within_domain", {})
        if ps_res:
            corr_rows.append({
                "metric_col": mc,
                "view": "per_source_within_domain",
                "n": ps_res.get("n_sources_total", 0),
                "status": ps_res.get("status", "ok"),
                "spearman_rho": ps_res.get("mean_within_domain_spearman_rho", float("nan")),
                "pvalue": ps_res.get("cross_domain_pvalue", float("nan")),
                "ci_lo": ps_res.get("ci_lo", float("nan")),
                "ci_hi": ps_res.get("ci_hi", float("nan")),
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(corr_csv, index=False)

    with open(corr_json, "w") as f:
        json.dump(pillar3_result, f, indent=2, allow_nan=True)

    print(f"[phase7] pillar3 correlation -> {corr_csv}, {corr_json}")
    print("[phase7] done.")


def _build_pillar1_only(phase5_summary: Path) -> pd.DataFrame:
    """Fallback: build Pillar I table only (no phase6 available)."""
    rows: list[dict] = []
    with open(phase5_summary) as f:
        p5 = json.load(f)

    by_src = p5.get("by_source_dataset", {})
    top12 = sorted(by_src.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:12]

    pillar1_metrics = [
        ("max_run_geq_5_rate", "has_run_5_response rate"),
        ("P_resp_has_list_given_prompt_has_list", "P(resp_list|prompt_list)"),
        ("structural_jaccard_median", "struct Jaccard median"),
        ("ngram_overlap_n4_median", "4-gram overlap median"),
        ("multi_turn_agreement_median", "multi-turn agreement median"),
        ("affirm_prefix_rate", "affirm prefix rate"),
    ]
    for src_name, src_stats in top12:
        for field, interp in pillar1_metrics:
            val = src_stats.get(field)
            if val is None:
                continue
            rows.append({
                "pillar": "I",
                "metric": field,
                "stage": f"SFT:{src_name}",
                "value": round(float(val), 6),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "effect_size_label": "",
                "interpretation": interp,
            })

    return pd.DataFrame(rows, columns=[
        "pillar", "metric", "stage", "value", "ci_lo", "ci_hi",
        "effect_size_label", "interpretation",
    ])


if __name__ == "__main__":
    main()
