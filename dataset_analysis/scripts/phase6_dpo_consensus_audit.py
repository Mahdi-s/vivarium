"""Phase 6: Pillar II — effect-size-corrected DPO chosen-vs-rejected audit.

Extends phase2_dpo_audit with:
  - Run-length metrics (max_run, has_run_5)
  - Consensus framing hits (CONSENSUS_RE)
  - Peer-frame counts (PEER_FRAME_RE)
  - Effect-size statistics replacing mean+sign-test (Task 0 identified as misleading):
      Cliff's delta, Hodges-Lehmann, paired permutation p, bootstrap CI
  - Length-normalized delta_correction_per_1k_chars (correction hits per 1000 chars)
  - Stratification by preference_type and chosen_model

PRE-REGISTRATION NOTE: All regex patterns and thresholds are imported from
audit_metrics.py which was registered before any full-corpus analysis ran
(Task 2 of the audit plan). Do not add metrics here; add them in audit_metrics.

Outputs:
  results/phase6_<tag>_per_pair.csv          (row-level, 36 columns)
  results/phase6_<tag>_summary.json          (aggregate stats with effect sizes)
  results/phase6_<tag>_by_preference_type.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    RESULTS, iter_rows,
    structural_jaccard, count_hits, affirm_prefix,
    SYCOPHANCY_RE, CORRECTION_RE,
    prompt_response_ngram_overlap, write_json,
)
from audit_metrics import (
    CONSENSUS_RE, PEER_FRAME_RE,
    max_run, repeat_run_geq_k,
    extract_triplet_with_meta,
)
from robust_stats import (
    robust_summary, cliffs_delta, cohens_d_paired,
    hodges_lehmann, paired_permutation_p, bootstrap_ci,
    practical_significance_label,
)


# ---------------------------------------------------------------------------
# CSV column orders — must match spec exactly
# ---------------------------------------------------------------------------

PER_PAIR_COLS = [
    # identity / provenance
    "idx", "source_dataset", "domain_canonical", "prompt_id",
    # phase2 reproductions (parity block):
    "w_struct_jaccard", "l_struct_jaccard", "delta_struct_jaccard",
    "w_ngram_overlap", "l_ngram_overlap", "delta_ngram_overlap",
    "w_sycophancy", "l_sycophancy", "delta_sycophancy",
    "w_correction", "l_correction", "delta_correction",
    "w_affirm_prefix", "l_affirm_prefix",
    # new Pillar II targets:
    "w_max_run", "l_max_run", "delta_max_run",
    "w_consensus_hits", "l_consensus_hits", "delta_consensus_hits",
    "w_has_run_5", "l_has_run_5", "delta_has_run_5",
    "w_peer_frame_count", "l_peer_frame_count", "delta_peer_frame_count",
    # length-normalized correction (Task spec gotcha #3):
    "w_correction_per_1k", "l_correction_per_1k", "delta_correction_per_1k",
    # DPO-specific stratification axes:
    "chosen_model", "rejected_model", "preference_type",
]

# Delta columns that get the full effect-size treatment
DELTA_COLS = [
    "delta_struct_jaccard",
    "delta_ngram_overlap",
    "delta_sycophancy",
    "delta_correction",
    "delta_correction_per_1k",
    "delta_max_run",
    "delta_consensus_hits",
    "delta_has_run_5",
    "delta_peer_frame_count",
]

# For Cliff's delta we need the raw chosen/rejected arrays for each metric
# (cliffs_delta is between the two distributions, NOT on the delta column)
CHOSEN_REJECTED_PAIRS = [
    ("w_struct_jaccard",   "l_struct_jaccard"),
    ("w_ngram_overlap",    "l_ngram_overlap"),
    ("w_sycophancy",       "l_sycophancy"),
    ("w_correction",       "l_correction"),
    ("w_correction_per_1k", "l_correction_per_1k"),
    ("w_max_run",          "l_max_run"),
    ("w_consensus_hits",   "l_consensus_hits"),
    ("w_has_run_5",        "l_has_run_5"),
    ("w_peer_frame_count", "l_peer_frame_count"),
]

BY_PREFERENCE_TYPE_COLS = [
    "preference_type", "n",
    "delta_max_run_median",      "delta_max_run_cliffs",
    "delta_struct_jaccard_median", "delta_struct_jaccard_cliffs",
    "delta_ngram_overlap_median",  "delta_ngram_overlap_cliffs",
    "delta_consensus_hits_median", "delta_consensus_hits_cliffs",
]

# Interpretation strings for all delta columns
_INTERPRETATIONS = {
    "delta_struct_jaccard":       "positive -> DPO rejected responses mirror structural formatting more than chosen",
    "delta_ngram_overlap":        "positive -> DPO rejected responses copy more n-grams verbatim from the prompt",
    "delta_sycophancy":           "positive -> DPO rejected responses use more sycophantic agreement phrases",
    "delta_correction":           "positive -> DPO chosen responses use more factual correction markers",
    "delta_correction_per_1k":    "positive -> DPO chosen responses have higher correction density (per 1k chars); controls for response length",
    "delta_max_run":              "positive -> DPO rejected responses contain longer literal-repetition runs",
    "delta_consensus_hits":       "positive -> DPO rejected responses use more consensus framing (English-only regex; lower bound)",
    "delta_has_run_5":            "positive -> DPO rejected responses more frequently contain runs of 5+ identical consecutive tokens",
    "delta_peer_frame_count":     "positive -> DPO rejected responses contain more 'Participant N:' style framing",
}


# ---------------------------------------------------------------------------
# Per-pair scoring
# ---------------------------------------------------------------------------

def score_response_extended(prompt: str, resp: str) -> dict:
    """Score a single (prompt, response) pair for all Pillar II metrics.

    Returns a flat dict of metric values. Deliberately separated from
    phase2's score_response() to avoid coupling that module.
    """
    resp_len = len(resp)
    sj  = structural_jaccard(prompt, resp)
    ng  = prompt_response_ngram_overlap(prompt, resp, n=4)
    ap  = int(affirm_prefix(resp))
    syc = count_hits(SYCOPHANCY_RE, resp)
    cor = count_hits(CORRECTION_RE, resp)
    mr  = max_run(resp)
    hr5 = int(mr >= 5)
    cons = count_hits(CONSENSUS_RE, resp)
    peer = count_hits(PEER_FRAME_RE, resp)
    # Length-normalized correction: hits per 1000 chars of response
    cor_per_1k = (cor / resp_len * 1000.0) if resp_len > 0 else 0.0
    return {
        "struct_jaccard":  sj,
        "ngram_overlap":   ng,
        "affirm_prefix":   ap,
        "sycophancy":      syc,
        "correction":      cor,
        "correction_per_1k": cor_per_1k,
        "max_run":         mr,
        "has_run_5":       hr5,
        "consensus_hits":  cons,
        "peer_frame_count": peer,
    }


# ---------------------------------------------------------------------------
# Summary statistics builder for one delta column
# ---------------------------------------------------------------------------

def _build_delta_stats(
    delta_col: str,
    delta_vals: list[float],
    chosen_vals: list[float],
    rejected_vals: list[float],
    interpretation: str,
) -> dict:
    """Compute the full effect-size stats block for one delta column.

    cliffs_delta is called as cliffs_delta(rejected, chosen) so that
    positive = rejected > chosen = DPO penalty signal.

    Boot CI on cliffs_delta uses 200 reps (not 1000) because cliffs_delta is
    O(N log N) via binary search for large N, but 200 reps on N=260k still takes
    ~10s; 1000 reps would exceed the ~30s budget for this block alone.
    boot_ci_median uses 1000 reps (cheap on a delta array).
    """
    n = len(delta_vals)
    if n == 0:
        return {"n": 0, "interpretation": interpretation}

    arr = np.asarray(delta_vals, dtype=float)
    arr_w = np.asarray(chosen_vals, dtype=float)
    arr_l = np.asarray(rejected_vals, dtype=float)

    rs = robust_summary(delta_vals)
    # Repack into the spec's "robust" sub-dict
    robust = {
        "n":       rs["n"],
        "median":  rs["median"],
        "P25":     rs["P25"],
        "P75":     rs["P75"],
        "iqr":     rs["iqr"],
        "P5":      rs["P5"],
        "P95":     rs["P95"],
        "mean":    rs["mean"],
        "stdev":   rs["stdev"] if "stdev" in rs else float("nan"),
    }

    hl    = hodges_lehmann(delta_vals)
    cd    = cliffs_delta(arr_l, arr_w)   # (rejected, chosen): positive = rejected higher
    cod   = cohens_d_paired(arr_l, arr_w)
    perm_p = paired_permutation_p(delta_vals, reps=1000, seed=42)

    # boot_ci_median: 1000 reps (cheap)
    boot_med = bootstrap_ci(delta_vals, lambda a: float(np.median(a)), reps=1000, seed=42)
    # boot_ci_cliffs_delta: 200 reps (expensive at large N — see docstring above)
    # NOTE: bootstrap here re-samples from the PAIRED arrays; we zip them to
    # keep the pairing intact.  We sample both arrays with the same index set.
    rng = np.random.default_rng(42)
    boot_cd_vals = np.empty(200, dtype=float)
    for i in range(200):
        idx = rng.integers(0, n, size=n)
        boot_cd_vals[i] = cliffs_delta(arr_l[idx], arr_w[idx])
    boot_cd = (
        float(np.percentile(boot_cd_vals, 2.5)),
        float(np.percentile(boot_cd_vals, 97.5)),
    )

    pct_pos = float(np.sum(arr > 0)) / n

    prac = practical_significance_label(cd)

    return {
        "robust":                 robust,
        "hodges_lehmann":         float(hl),
        "cliffs_delta":           float(cd),
        "cohens_d_paired":        float(cod),
        "permutation_p_two_sided": float(perm_p),
        "boot_ci_median":         [float(boot_med[0]), float(boot_med[1])],
        "boot_ci_cliffs_delta":   [float(boot_cd[0]), float(boot_cd[1])],
        "practical_significance": prac,
        "pct_positive":           pct_pos,
        "interpretation":         interpretation,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase 6: Pillar II effect-size-corrected DPO chosen-vs-rejected audit"
    )
    ap.add_argument("--short-name", default="instruct-dpo",
                    help="parquet dir under data/raw/ (instruct-dpo, think-dpo)")
    ap.add_argument("--jsonl-fallback", default="dpo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or args.short_name

    per_pair_path    = RESULTS / f"phase6_{tag}_per_pair.csv"
    summary_path     = RESULTS / f"phase6_{tag}_summary.json"
    by_ptype_path    = RESULTS / f"phase6_{tag}_by_preference_type.csv"

    # -----------------------------------------------------------------------
    # Aggregation structures
    # -----------------------------------------------------------------------
    # raw metric arrays per side (chosen=w, rejected=l)
    chosen_raw:  dict[str, list[float]] = {k: [] for k in [
        "struct_jaccard", "ngram_overlap", "sycophancy", "correction",
        "correction_per_1k", "max_run", "has_run_5", "consensus_hits", "peer_frame_count",
    ]}
    rejected_raw: dict[str, list[float]] = {k: [] for k in chosen_raw}

    # delta columns
    delta_agg: dict[str, list[float]] = {k: [] for k in DELTA_COLS}

    # affirm arrays (binary, tracked separately — no delta col in spec)
    w_aff: list[int] = []
    l_aff: list[int] = []

    # Per-preference-type accumulation
    # ptype -> delta_col -> list[float]
    ptype_delta_agg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in DELTA_COLS}
    )
    # ptype -> (w_arr, l_arr) for cliffs
    ptype_chosen_raw: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in chosen_raw}
    )
    ptype_rejected_raw: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in chosen_raw}
    )

    # Per-chosen-model accumulation (headline stats only)
    model_delta_agg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in DELTA_COLS}
    )

    # Top-1% outlier capture: delta_col -> list of (idx, value)
    # We keep all values, then trim to top-1% after the loop.
    outlier_pairs: dict[str, list[tuple[int, float]]] = {k: [] for k in DELTA_COLS}

    n_ok   = 0
    n_skip = 0

    with per_pair_path.open("w", newline="") as fout:
        wcsv = csv.writer(fout)
        wcsv.writerow(PER_PAIR_COLS)

        for idx, row in enumerate(iter_rows(
            args.short_name,
            jsonl_fallback=args.jsonl_fallback,
            limit=args.limit,
        )):
            result = extract_triplet_with_meta(row)
            if result is None:
                n_skip += 1
                continue
            prompt, chosen, rejected, meta = result

            # Provenance — DPO rows have no source_dataset/domain
            src     = meta["source_dataset"] or "unknown"
            dom_can = meta["domain_canonical"]   # always "unmapped" per spec
            prompt_id     = meta.get("prompt_id") or ""
            chosen_model  = meta.get("chosen_model") or ""
            rejected_model = meta.get("rejected_model") or ""
            ptype  = meta.get("preference_type") or "unknown"

            # Score both sides
            sw = score_response_extended(prompt, chosen)
            sl = score_response_extended(prompt, rejected)

            # --- Compute deltas (positive = rejected > chosen = DPO penalty signal) ---
            d_struct    = sl["struct_jaccard"]    - sw["struct_jaccard"]
            d_ngram     = sl["ngram_overlap"]     - sw["ngram_overlap"]
            d_syc       = sl["sycophancy"]        - sw["sycophancy"]
            # delta_correction: phase2 definition = sw - sl (positive = chosen has more corrections)
            d_cor       = sw["correction"]        - sl["correction"]
            d_cor_1k    = sw["correction_per_1k"] - sl["correction_per_1k"]
            d_maxrun    = sl["max_run"]            - sw["max_run"]
            d_cons      = sl["consensus_hits"]     - sw["consensus_hits"]
            d_hr5       = sl["has_run_5"]          - sw["has_run_5"]
            d_peer      = sl["peer_frame_count"]   - sw["peer_frame_count"]

            # --- Write per-pair CSV row ---
            wcsv.writerow([
                idx,
                src,
                dom_can,
                prompt_id,
                # phase2 parity block (chosen=w, rejected=l)
                f"{sw['struct_jaccard']:.4f}",    f"{sl['struct_jaccard']:.4f}",    f"{d_struct:.4f}",
                f"{sw['ngram_overlap']:.4f}",     f"{sl['ngram_overlap']:.4f}",     f"{d_ngram:.4f}",
                sw["sycophancy"],                  sl["sycophancy"],                  d_syc,
                sw["correction"],                  sl["correction"],                  d_cor,
                sw["affirm_prefix"],               sl["affirm_prefix"],
                # new Pillar II
                sw["max_run"],       sl["max_run"],       d_maxrun,
                sw["consensus_hits"], sl["consensus_hits"], d_cons,
                sw["has_run_5"],     sl["has_run_5"],     d_hr5,
                sw["peer_frame_count"], sl["peer_frame_count"], d_peer,
                # length-normalized correction
                f"{sw['correction_per_1k']:.4f}", f"{sl['correction_per_1k']:.4f}", f"{d_cor_1k:.4f}",
                # stratification axes
                chosen_model, rejected_model, ptype,
            ])

            # --- Global aggregation ---
            chosen_raw["struct_jaccard"].append(sw["struct_jaccard"])
            chosen_raw["ngram_overlap"].append(sw["ngram_overlap"])
            chosen_raw["sycophancy"].append(float(sw["sycophancy"]))
            chosen_raw["correction"].append(float(sw["correction"]))
            chosen_raw["correction_per_1k"].append(sw["correction_per_1k"])
            chosen_raw["max_run"].append(float(sw["max_run"]))
            chosen_raw["has_run_5"].append(float(sw["has_run_5"]))
            chosen_raw["consensus_hits"].append(float(sw["consensus_hits"]))
            chosen_raw["peer_frame_count"].append(float(sw["peer_frame_count"]))

            rejected_raw["struct_jaccard"].append(sl["struct_jaccard"])
            rejected_raw["ngram_overlap"].append(sl["ngram_overlap"])
            rejected_raw["sycophancy"].append(float(sl["sycophancy"]))
            rejected_raw["correction"].append(float(sl["correction"]))
            rejected_raw["correction_per_1k"].append(sl["correction_per_1k"])
            rejected_raw["max_run"].append(float(sl["max_run"]))
            rejected_raw["has_run_5"].append(float(sl["has_run_5"]))
            rejected_raw["consensus_hits"].append(float(sl["consensus_hits"]))
            rejected_raw["peer_frame_count"].append(float(sl["peer_frame_count"]))

            delta_agg["delta_struct_jaccard"].append(d_struct)
            delta_agg["delta_ngram_overlap"].append(d_ngram)
            delta_agg["delta_sycophancy"].append(float(d_syc))
            delta_agg["delta_correction"].append(float(d_cor))
            delta_agg["delta_correction_per_1k"].append(d_cor_1k)
            delta_agg["delta_max_run"].append(float(d_maxrun))
            delta_agg["delta_consensus_hits"].append(float(d_cons))
            delta_agg["delta_has_run_5"].append(float(d_hr5))
            delta_agg["delta_peer_frame_count"].append(float(d_peer))

            w_aff.append(sw["affirm_prefix"])
            l_aff.append(sl["affirm_prefix"])

            # Outlier capture — top-1% = largest positive deltas
            outlier_pairs["delta_struct_jaccard"].append((idx, d_struct))
            outlier_pairs["delta_ngram_overlap"].append((idx, d_ngram))
            outlier_pairs["delta_sycophancy"].append((idx, float(d_syc)))
            outlier_pairs["delta_correction"].append((idx, float(d_cor)))
            outlier_pairs["delta_correction_per_1k"].append((idx, d_cor_1k))
            outlier_pairs["delta_max_run"].append((idx, float(d_maxrun)))
            outlier_pairs["delta_consensus_hits"].append((idx, float(d_cons)))
            outlier_pairs["delta_has_run_5"].append((idx, float(d_hr5)))
            outlier_pairs["delta_peer_frame_count"].append((idx, float(d_peer)))

            # --- Per-preference-type accumulation ---
            ptype_delta_agg[ptype]["delta_struct_jaccard"].append(d_struct)
            ptype_delta_agg[ptype]["delta_ngram_overlap"].append(d_ngram)
            ptype_delta_agg[ptype]["delta_sycophancy"].append(float(d_syc))
            ptype_delta_agg[ptype]["delta_correction"].append(float(d_cor))
            ptype_delta_agg[ptype]["delta_correction_per_1k"].append(d_cor_1k)
            ptype_delta_agg[ptype]["delta_max_run"].append(float(d_maxrun))
            ptype_delta_agg[ptype]["delta_consensus_hits"].append(float(d_cons))
            ptype_delta_agg[ptype]["delta_has_run_5"].append(float(d_hr5))
            ptype_delta_agg[ptype]["delta_peer_frame_count"].append(float(d_peer))

            for mk in chosen_raw:
                ptype_chosen_raw[ptype][mk].append(chosen_raw[mk][-1])
                ptype_rejected_raw[ptype][mk].append(rejected_raw[mk][-1])

            # --- Per-chosen-model accumulation ---
            for k in DELTA_COLS:
                model_delta_agg[chosen_model][k].append(delta_agg[k][-1])

            n_ok += 1

    print(f"[phase6] processed {n_ok} pairs, skipped {n_skip}")

    # -----------------------------------------------------------------------
    # Build per-delta summary blocks
    # -----------------------------------------------------------------------
    # Map delta col name -> (chosen metric key, rejected metric key)
    _metric_key_map = {
        "delta_struct_jaccard":    ("struct_jaccard",    "struct_jaccard"),
        "delta_ngram_overlap":     ("ngram_overlap",     "ngram_overlap"),
        "delta_sycophancy":        ("sycophancy",        "sycophancy"),
        "delta_correction":        ("correction",        "correction"),
        "delta_correction_per_1k": ("correction_per_1k","correction_per_1k"),
        "delta_max_run":           ("max_run",           "max_run"),
        "delta_consensus_hits":    ("consensus_hits",    "consensus_hits"),
        "delta_has_run_5":         ("has_run_5",         "has_run_5"),
        "delta_peer_frame_count":  ("peer_frame_count",  "peer_frame_count"),
    }

    delta_summary: dict[str, dict] = {}
    for dcol in DELTA_COLS:
        w_key, l_key = _metric_key_map[dcol]
        delta_summary[dcol] = _build_delta_stats(
            delta_col=dcol,
            delta_vals=delta_agg[dcol],
            chosen_vals=chosen_raw[w_key],
            rejected_vals=rejected_raw[l_key],
            interpretation=_INTERPRETATIONS[dcol],
        )
        print(f"[phase6] {dcol}: n={delta_summary[dcol].get('robust', {}).get('n', 0)}, "
              f"median={delta_summary[dcol].get('robust', {}).get('median', float('nan')):.4f}, "
              f"cliffs={delta_summary[dcol].get('cliffs_delta', float('nan')):.4f}, "
              f"pct_pos={delta_summary[dcol].get('pct_positive', float('nan')):.4f}")

    # -----------------------------------------------------------------------
    # Top-1% outlier indices (capped at 5000 per column)
    # -----------------------------------------------------------------------
    top_1pct_outlier_indices: dict[str, list[int]] = {}
    for dcol, pairs in outlier_pairs.items():
        if not pairs:
            top_1pct_outlier_indices[dcol] = []
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        cutoff = max(1, int(len(pairs_sorted) * 0.01))
        top_1pct_outlier_indices[dcol] = [
            int(p[0]) for p in pairs_sorted[:min(cutoff, 5000)]
        ]

    # -----------------------------------------------------------------------
    # by_preference_type block (summary JSON sub-dict + CSV)
    # -----------------------------------------------------------------------
    by_preference_type: dict[str, dict] = {}
    for pt, dagg in sorted(ptype_delta_agg.items()):
        n_pt = len(dagg["delta_struct_jaccard"])
        if n_pt == 0:
            continue

        def _median_safe(vals: list[float]) -> float:
            return float(np.median(np.asarray(vals, dtype=float))) if vals else float("nan")

        # args: (rejected_vals, chosen_vals) — positive δ = rejected greater = DPO penalty signal (matches global path at L202).
        def _cliffs_safe(rejected_vals: list[float], chosen_vals: list[float]) -> float:
            if not rejected_vals or not chosen_vals:
                return float("nan")
            return float(cliffs_delta(rejected_vals, chosen_vals))

        by_preference_type[pt] = {
            "n": n_pt,
            "delta_max_run": {
                "median": _median_safe(dagg["delta_max_run"]),
                "cliffs_delta": _cliffs_safe(
                    ptype_rejected_raw[pt]["max_run"], ptype_chosen_raw[pt]["max_run"]
                ),
            },
            "delta_struct_jaccard": {
                "median": _median_safe(dagg["delta_struct_jaccard"]),
                "cliffs_delta": _cliffs_safe(
                    ptype_rejected_raw[pt]["struct_jaccard"], ptype_chosen_raw[pt]["struct_jaccard"]
                ),
            },
            "delta_ngram_overlap": {
                "median": _median_safe(dagg["delta_ngram_overlap"]),
                "cliffs_delta": _cliffs_safe(
                    ptype_rejected_raw[pt]["ngram_overlap"], ptype_chosen_raw[pt]["ngram_overlap"]
                ),
            },
            "delta_consensus_hits": {
                "median": _median_safe(dagg["delta_consensus_hits"]),
                "cliffs_delta": _cliffs_safe(
                    ptype_rejected_raw[pt]["consensus_hits"], ptype_chosen_raw[pt]["consensus_hits"]
                ),
            },
        }

    # -----------------------------------------------------------------------
    # by_chosen_model headline block (summary JSON sub-dict)
    # -----------------------------------------------------------------------
    by_chosen_model: dict[str, dict] = {}
    for model, magg in sorted(model_delta_agg.items()):
        n_m = len(magg["delta_struct_jaccard"])
        if n_m == 0:
            continue
        by_chosen_model[model] = {
            "n": n_m,
            "delta_struct_jaccard_median": (
                float(np.median(np.asarray(magg["delta_struct_jaccard"], dtype=float)))
                if magg["delta_struct_jaccard"] else float("nan")
            ),
            "delta_ngram_overlap_median": (
                float(np.median(np.asarray(magg["delta_ngram_overlap"], dtype=float)))
                if magg["delta_ngram_overlap"] else float("nan")
            ),
            "delta_max_run_median": (
                float(np.median(np.asarray(magg["delta_max_run"], dtype=float)))
                if magg["delta_max_run"] else float("nan")
            ),
        }

    # -----------------------------------------------------------------------
    # Global affirm prefix rates
    # -----------------------------------------------------------------------
    affirm_rate_chosen   = sum(w_aff) / max(1, len(w_aff))
    affirm_rate_rejected = sum(l_aff) / max(1, len(l_aff))

    # -----------------------------------------------------------------------
    # Assemble summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "dataset_short":   args.short_name,
        "pairs_analyzed":  n_ok,
        "pairs_skipped":   n_skip,
        "affirm_prefix_rate_chosen":   affirm_rate_chosen,
        "affirm_prefix_rate_rejected": affirm_rate_rejected,
        **{dcol: delta_summary[dcol] for dcol in DELTA_COLS},
        "by_preference_type": by_preference_type,
        "by_chosen_model":    by_chosen_model,
        "top_1pct_outlier_indices": top_1pct_outlier_indices,
        "regex_lower_bound_disclaimer": (
            "consensus_hits and sycophancy_hits are reported as English-only "
            "regex lower bounds; recall is bounded by phase10 LLM-judge."
        ),
        "boot_ci_note": (
            "boot_ci_median uses 1000 reps; boot_ci_cliffs_delta uses 200 reps "
            "(cliffs_delta is O(N log N) per rep at N=260k; 200 reps is the "
            "plan-mandated cap to stay under ~30s total for this block)."
        ),
    }
    write_json(summary_path, summary)
    print(f"[phase6] summary -> {summary_path}")

    # -----------------------------------------------------------------------
    # by_preference_type CSV
    # -----------------------------------------------------------------------
    with by_ptype_path.open("w", newline="") as f:
        wcsv2 = csv.writer(f)
        wcsv2.writerow(BY_PREFERENCE_TYPE_COLS)
        for pt in sorted(by_preference_type.keys()):
            bpt = by_preference_type[pt]
            wcsv2.writerow([
                pt,
                bpt["n"],
                f"{bpt['delta_max_run']['median']:.4f}",
                f"{bpt['delta_max_run']['cliffs_delta']:.4f}",
                f"{bpt['delta_struct_jaccard']['median']:.4f}",
                f"{bpt['delta_struct_jaccard']['cliffs_delta']:.4f}",
                f"{bpt['delta_ngram_overlap']['median']:.4f}",
                f"{bpt['delta_ngram_overlap']['cliffs_delta']:.4f}",
                f"{bpt['delta_consensus_hits']['median']:.4f}",
                f"{bpt['delta_consensus_hits']['cliffs_delta']:.4f}",
            ])
    print(f"[phase6] by_preference_type CSV -> {by_ptype_path}")
    print(f"[phase6] wrote {n_ok} pairs -> {per_pair_path}")

    # -----------------------------------------------------------------------
    # Quick parity check printout (for cross-validation vs phase2)
    # -----------------------------------------------------------------------
    print("\n[phase6] PARITY CHECK (compare to phase2 sign_tests.pct_positive):")
    for k in ("delta_struct_jaccard", "delta_ngram_overlap"):
        pct = delta_summary[k].get("pct_positive", float("nan"))
        print(f"  phase6 {k} pct_positive = {pct:.6f}")


if __name__ == "__main__":
    main()
