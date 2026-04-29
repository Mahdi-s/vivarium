"""Phase 5: Pillar I — SFT structural priors per source_dataset (consensus audit).

Extends phase1_sft_audit with:
  - Stratification by source_dataset and domain_canonical
  - Consensus regex hits (CONSENSUS_RE)
  - Peer-frame counts (PEER_FRAME_RE)
  - Run-length metrics (max_run, repeat_run_geq_k)
  - Multi-turn agreement score
  - Correction hits (CORRECTION_RE)

PRE-REGISTRATION NOTE: All regex patterns and thresholds are imported from
audit_metrics.py which was registered before any full-corpus analysis ran
(Task 2 of the audit plan). Do not add metrics here; add them in audit_metrics.

Outputs:
  results/phase5_<short>_per_example.csv   (row-level, 24 columns)
  results/phase5_<short>_summary.json      (aggregate stats with by_source_dataset)
  results/phase5_<short>_by_source.csv     (one row per source_dataset, Pillar III handoff)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from common import (
    RESULTS, iter_rows,
    structural_jaccard, structural_fingerprint,
    affirm_prefix, count_hits,
    SYCOPHANCY_RE, CORRECTION_RE,
    prompt_response_ngram_overlap,
    summarise, write_json,
)
from audit_metrics import (
    CONSENSUS_RE, PEER_FRAME_RE,
    max_run, repeat_run_geq_k,
    multi_turn_agreement_score,
    canonical_sft_domain,
    extract_pair_with_meta,
)
from robust_stats import robust_summary


# ---------------------------------------------------------------------------
# CSV column order — must match exactly (plan spec)
# ---------------------------------------------------------------------------

COLS = [
    "idx", "source_dataset", "domain_raw", "domain_canonical",
    "prompt_len", "resp_len",
    # phase1 reproductions (parity check):
    "structural_jaccard", "ngram_overlap_n4", "affirm_prefix", "sycophancy_hits",
    "prompt_has_list", "resp_has_list",
    # new (Pillar I targets):
    "consensus_hits_response", "has_consensus_response",
    "peer_frame_count_prompt", "peer_frame_count_response",
    "max_run_response", "max_run_prompt",
    "has_run_3_response", "has_run_5_response", "has_run_7_response",
    "n_messages", "multi_turn_agreement",
    "correction_hits_response", "has_correction_response",
]

# Aggregation keys (everything except idx and domain_raw which are non-numeric)
_AGG_KEYS = [c for c in COLS if c not in ("idx", "source_dataset", "domain_raw", "domain_canonical")]

# By-source CSV columns (Pillar III handoff) — must be exactly these names
BY_SOURCE_COLS = [
    "source_dataset", "n",
    "P_carryover", "affirm_rate",
    "struct_jaccard_median", "struct_jaccard_iqr",
    "max_run_geq_5_rate", "max_run_geq_5_n",
    "ngram_overlap_n4_median", "ngram_overlap_n4_P95",
    "multi_turn_agreement_median",
]


def _get_n_messages(row: dict) -> int:
    """Return the count of messages in the row's message list, or 0."""
    msgs = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(msgs, list):
        return len(msgs)
    return 0


def _get_messages_list(row: dict) -> list[dict]:
    """Return the messages list for multi_turn_agreement_score, or []."""
    msgs = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(msgs, list):
        return msgs
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Phase 5: Pillar I SFT structural-prior audit per source_dataset"
    )
    ap.add_argument("--short-name", default="instruct-sft",
                    help="parquet dir under data/raw/ (instruct-sft, think-sft, instruct-rl)")
    ap.add_argument("--jsonl-fallback", default="sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or args.short_name

    per_example_path = RESULTS / f"phase5_{tag}_per_example.csv"
    summary_path     = RESULTS / f"phase5_{tag}_summary.json"
    by_source_path   = RESULTS / f"phase5_{tag}_by_source.csv"

    # Global aggregators — lists of per-row values
    agg: dict[str, list] = {k: [] for k in _AGG_KEYS}

    # Per-source aggregators: source_dataset -> metric_key -> list of values
    src_agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # Per-domain aggregators: domain_canonical -> metric_key -> list of values
    dom_agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # List-carryover counts (global and per-source)
    list_carry_num: int = 0
    list_carry_den: int = 0
    src_carry_num: dict[str, int] = defaultdict(int)
    src_carry_den: dict[str, int] = defaultdict(int)

    # source -> set of all domain_canonical values seen for that source
    source_domain_canonical_set: dict[str, set[str]] = defaultdict(set)
    unmapped_domains: set[str] = set()

    rows_written: int = 0
    skipped: int = 0

    with per_example_path.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(COLS)

        for idx, row in enumerate(iter_rows(args.short_name,
                                            jsonl_fallback=args.jsonl_fallback,
                                            limit=args.limit)):

            result = extract_pair_with_meta(row)
            if result is None:
                skipped += 1
                continue

            user, asst, meta = result
            src = meta["source_dataset"] or "unknown"
            dom_raw = meta["domain_raw"] or ""
            dom_can = meta["domain_canonical"]

            # Track unmapped domains and per-source canonical sets
            if dom_raw and dom_can == "unmapped":
                unmapped_domains.add(dom_raw)
            source_domain_canonical_set[src].add(dom_can)

            # ---- phase1 metrics (parity block) ----
            sj  = structural_jaccard(user, asst)
            ng  = prompt_response_ngram_overlap(user, asst, n=4)
            ap_ = int(affirm_prefix(asst))
            sh  = count_hits(SYCOPHANCY_RE, asst)

            pf = structural_fingerprint(user)
            rf = structural_fingerprint(asst)
            p_list = int((pf["bullet"] + pf["numbered"]) > 0)
            r_list = int((rf["bullet"] + rf["numbered"]) > 0)

            # ---- new Pillar I metrics ----
            cons_hits = count_hits(CONSENSUS_RE, asst)
            has_cons  = int(cons_hits > 0)

            peer_prompt = count_hits(PEER_FRAME_RE, user)
            peer_resp   = count_hits(PEER_FRAME_RE, asst)

            # Compute max_run once per text
            mr_resp = max_run(asst)
            mr_prompt = max_run(user)

            has_run3 = int(mr_resp >= 3)
            has_run5 = int(mr_resp >= 5)
            has_run7 = int(mr_resp >= 7)

            # n_messages from the raw row (full message list)
            n_msgs = _get_n_messages(row)

            # multi_turn_agreement requires full messages list
            msgs_list = _get_messages_list(row)
            mta = multi_turn_agreement_score(msgs_list)

            corr_hits = count_hits(CORRECTION_RE, asst)
            has_corr  = int(corr_hits > 0)

            # ---- CSV row ----
            w.writerow([
                idx,
                src,
                dom_raw,
                dom_can,
                pf["len"],
                rf["len"],
                f"{sj:.4f}",
                f"{ng:.4f}",
                ap_,
                sh,
                p_list,
                r_list,
                cons_hits,
                has_cons,
                peer_prompt,
                peer_resp,
                mr_resp,
                mr_prompt,
                has_run3,
                has_run5,
                has_run7,
                n_msgs,
                f"{mta:.4f}",
                corr_hits,
                has_corr,
            ])
            rows_written += 1

            # ---- Global aggregation ----
            vals = {
                "prompt_len":               pf["len"],
                "resp_len":                 rf["len"],
                "structural_jaccard":       sj,
                "ngram_overlap_n4":         ng,
                "affirm_prefix":            ap_,
                "sycophancy_hits":          sh,
                "prompt_has_list":          p_list,
                "resp_has_list":            r_list,
                "consensus_hits_response":  cons_hits,
                "has_consensus_response":   has_cons,
                "peer_frame_count_prompt":  peer_prompt,
                "peer_frame_count_response": peer_resp,
                "max_run_response":         mr_resp,
                "max_run_prompt":           mr_prompt,
                "has_run_3_response":       has_run3,
                "has_run_5_response":       has_run5,
                "has_run_7_response":       has_run7,
                "n_messages":               n_msgs,
                "multi_turn_agreement":     mta,
                "correction_hits_response": corr_hits,
                "has_correction_response":  has_corr,
            }
            for k, v in vals.items():
                agg[k].append(v)

            # Global list-carryover
            if p_list:
                list_carry_den += 1
                if r_list:
                    list_carry_num += 1

            # ---- Per-source aggregation ----
            for k, v in vals.items():
                src_agg[src][k].append(v)
            if p_list:
                src_carry_den[src] += 1
                if r_list:
                    src_carry_num[src] += 1

            # ---- Per-domain aggregation ----
            for k, v in vals.items():
                dom_agg[dom_can][k].append(v)

    # A source is "unmapped" if every row from it had domain_canonical == "unmapped"
    unmapped_sources: set[str] = {
        s for s, ds in source_domain_canonical_set.items() if ds == {"unmapped"}
    }

    # ---------------------------------------------------------------------------
    # Build summary JSON
    # ---------------------------------------------------------------------------

    # Continuous metrics that benefit from robust_summary
    ROBUST_KEYS = [
        "structural_jaccard", "ngram_overlap_n4",
        "consensus_hits_response", "peer_frame_count_prompt", "peer_frame_count_response",
        "max_run_response", "max_run_prompt",
        "sycophancy_hits", "correction_hits_response",
        "prompt_len", "resp_len", "n_messages", "multi_turn_agreement",
    ]

    metrics_robust = {k: robust_summary(agg[k]) for k in ROBUST_KEYS}

    # Also emit legacy metrics block for parity-check purposes
    metrics_legacy = {k: summarise(agg[k]) for k in _AGG_KEYS}

    # Per-source summary
    def _source_summary(src: str) -> dict:
        sa = src_agg[src]
        n = len(sa.get("structural_jaccard", []))
        sj_vals = sa.get("structural_jaccard", [])
        ng_vals  = sa.get("ngram_overlap_n4", [])
        mr5_vals = sa.get("has_run_5_response", [])
        mta_vals = sa.get("multi_turn_agreement", [])
        affirm_vals = sa.get("affirm_prefix", [])

        sj_rs  = robust_summary(sj_vals)  if sj_vals  else {}
        ng_rs  = robust_summary(ng_vals)  if ng_vals  else {}
        mr5_n  = int(sum(mr5_vals))
        mta_rs = robust_summary(mta_vals) if mta_vals else {}

        p_carry = (
            src_carry_num[src] / src_carry_den[src]
            if src_carry_den[src] else None
        )
        affirm_rate = (
            sum(affirm_vals) / len(affirm_vals) if affirm_vals else None
        )
        mr5_rate = mr5_n / n if n else None

        return {
            "n": n,
            "P_resp_has_list_given_prompt_has_list": p_carry,
            "affirm_prefix_rate":       affirm_rate,
            "structural_jaccard_median": sj_rs.get("median"),
            "structural_jaccard_iqr":    sj_rs.get("iqr"),
            "max_run_geq_5_rate":        mr5_rate,
            "max_run_geq_5_n":           mr5_n,
            "ngram_overlap_n4_median":   ng_rs.get("median"),
            "ngram_overlap_n4_P95":      ng_rs.get("P95"),
            "multi_turn_agreement_median": mta_rs.get("median"),
        }

    by_source_dataset: dict[str, dict] = {
        src: _source_summary(src) for src in sorted(src_agg.keys())
    }

    # Per-domain summary (same structure)
    def _domain_summary(dom: str) -> dict:
        da = dom_agg[dom]
        n = len(da.get("structural_jaccard", []))
        sj_vals = da.get("structural_jaccard", [])
        ng_vals  = da.get("ngram_overlap_n4", [])
        mr5_vals = da.get("has_run_5_response", [])
        mta_vals = da.get("multi_turn_agreement", [])
        affirm_vals = da.get("affirm_prefix", [])

        sj_rs  = robust_summary(sj_vals)  if sj_vals  else {}
        ng_rs  = robust_summary(ng_vals)  if ng_vals  else {}
        mta_rs = robust_summary(mta_vals) if mta_vals else {}
        mr5_n  = int(sum(mr5_vals))
        mr5_rate = mr5_n / n if n else None
        affirm_rate = sum(affirm_vals) / len(affirm_vals) if affirm_vals else None

        return {
            "n": n,
            "affirm_prefix_rate":       affirm_rate,
            "structural_jaccard_median": sj_rs.get("median"),
            "structural_jaccard_iqr":    sj_rs.get("iqr"),
            "max_run_geq_5_rate":        mr5_rate,
            "ngram_overlap_n4_median":   ng_rs.get("median"),
            "multi_turn_agreement_median": mta_rs.get("median"),
        }

    by_domain_canonical: dict[str, dict] = {
        dom: _domain_summary(dom) for dom in sorted(dom_agg.keys())
    }

    global_p_carry = (
        list_carry_num / list_carry_den if list_carry_den else None
    )
    affirm_list = agg["affirm_prefix"]
    global_affirm_rate = sum(affirm_list) / len(affirm_list) if affirm_list else None

    summary = {
        "dataset_short": args.short_name,
        "rows_analyzed": rows_written,
        "rows_skipped": skipped,
        "metrics_robust": metrics_robust,
        "metrics": metrics_legacy,           # parity-check: matches phase1 schema
        "P(resp_has_list | prompt_has_list)": global_p_carry,
        "prompt_has_list_rate": (
            sum(agg["prompt_has_list"]) / max(1, len(agg["prompt_has_list"]))
        ),
        "affirm_prefix_rate": global_affirm_rate,
        "by_source_dataset": by_source_dataset,
        "by_domain_canonical": by_domain_canonical,
        "unmapped_source_datasets": sorted(unmapped_sources),
        "unmapped_domains": sorted(unmapped_domains),
        "regex_lower_bound_disclaimer": (
            "consensus_hits and sycophancy_hits are reported as English-only regex "
            "lower bounds; recall is bounded by phase10 LLM-judge."
        ),
    }
    write_json(summary_path, summary)

    # ---------------------------------------------------------------------------
    # Write by-source CSV (Pillar III handoff — schema is stable, phase7 joins on it)
    # ---------------------------------------------------------------------------
    has_real_sources = any(s != "unknown" for s in src_agg.keys())
    if has_real_sources:
        with by_source_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(BY_SOURCE_COLS)
            for src in sorted(src_agg.keys()):
                s = by_source_dataset[src]
                n = s["n"]
                w.writerow([
                    src,
                    n,
                    f"{s['P_resp_has_list_given_prompt_has_list']:.4f}" if s["P_resp_has_list_given_prompt_has_list"] is not None else "",
                    f"{s['affirm_prefix_rate']:.4f}" if s["affirm_prefix_rate"] is not None else "",
                    f"{s['structural_jaccard_median']:.4f}" if s["structural_jaccard_median"] is not None else "",
                    f"{s['structural_jaccard_iqr']:.4f}"    if s["structural_jaccard_iqr"]    is not None else "",
                    f"{s['max_run_geq_5_rate']:.4f}"        if s["max_run_geq_5_rate"]        is not None else "",
                    s["max_run_geq_5_n"],
                    f"{s['ngram_overlap_n4_median']:.4f}"   if s["ngram_overlap_n4_median"]   is not None else "",
                    f"{s['ngram_overlap_n4_P95']:.4f}"      if s["ngram_overlap_n4_P95"]      is not None else "",
                    f"{s['multi_turn_agreement_median']:.4f}" if s["multi_turn_agreement_median"] is not None else "",
                ])
        print(f"[phase5] by-source CSV -> {by_source_path}")
    else:
        print(f"[phase5] no real source_dataset values found; skipping by_source CSV emission")

    print(f"[phase5] wrote {rows_written} rows -> {per_example_path}")
    print(f"[phase5] skipped {skipped} malformed rows")
    print(f"[phase5] summary -> {summary_path}")
    print(f"[phase5] sources found: {sorted(src_agg.keys())}")
    print(json.dumps({
        "rows_analyzed": rows_written,
        "P(resp_has_list | prompt_has_list)": global_p_carry,
        "affirm_prefix_rate": global_affirm_rate,
        "structural_jaccard_median": metrics_robust["structural_jaccard"].get("median"),
        "ngram_overlap_n4_median": metrics_robust["ngram_overlap_n4"].get("median"),
        "n_sources": len(src_agg),
    }, indent=2))


if __name__ == "__main__":
    main()
