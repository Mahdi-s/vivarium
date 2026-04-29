"""Phase 2: audit `allenai/Dolci-Instruct-DPO` to show *what* DPO penalises.

Hypothesis: DPO partially reverses the SFT formatting trap by rewarding
factually-corrective divergence and penalising agreement-by-mirroring.

For each preference pair (prompt, chosen=y_w, rejected=y_l) we compute the
three audits in parallel on both y_w and y_l, then report paired deltas:
    delta_sycophancy  = y_l_syc - y_w_syc     (expected > 0)
    delta_correction  = y_w_cor - y_l_cor     (expected > 0)
    delta_mirror      = y_l_mirror - y_w_mirror  (expected > 0)

Outputs:
  results/phase2_dpo_per_pair.csv
  results/phase2_dpo_summary.json
"""
from __future__ import annotations

import csv
import json
import statistics as st

from common import (
    DATA, RESULTS, iter_rows,
    structural_jaccard, count_hits, affirm_prefix,
    SYCOPHANCY_RE, CORRECTION_RE, AFFIRM_RE,
    prompt_response_ngram_overlap, summarise, write_json,
)


def extract_triplet(row: dict) -> tuple[str, str, str] | None:
    # Common DPO schemas: (prompt, chosen, rejected) or
    # (chosen=[msgs], rejected=[msgs])
    prompt = row.get("prompt") or row.get("instruction") or row.get("input") or ""
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, list):
        # take last assistant message
        c = next((m.get("content", "") for m in reversed(chosen)
                  if (m.get("role") or m.get("from")) in ("assistant", "gpt")), "")
        r = next((m.get("content", "") for m in reversed(rejected or [])
                  if (m.get("role") or m.get("from")) in ("assistant", "gpt")), "")
        if not prompt:
            # reconstruct prompt as concat of user turns in chosen
            prompt = "\n".join(
                m.get("content", "") for m in chosen
                if (m.get("role") or m.get("from")) in ("user", "human")
            )
        chosen, rejected = c, r
    if not (prompt and chosen and rejected):
        return None
    return str(prompt), str(chosen), str(rejected)


def score_response(prompt: str, resp: str) -> dict:
    return {
        "len": len(resp),
        "struct_jaccard": structural_jaccard(prompt, resp),
        "ngram_overlap_n4": prompt_response_ngram_overlap(prompt, resp, n=4),
        "affirm_prefix": int(affirm_prefix(resp)),
        "sycophancy_hits": count_hits(SYCOPHANCY_RE, resp),
        "correction_hits": count_hits(CORRECTION_RE, resp),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-name", default="instruct-dpo",
                    help="parquet dir under data/raw/ (instruct-dpo, think-dpo)")
    ap.add_argument("--jsonl-fallback", default="dpo")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = args.tag or args.short_name
    per_pair = RESULTS / f"phase2_{tag}_per_pair.csv"
    summary_path = RESULTS / f"phase2_{tag}_summary.json"

    header = [
        "idx",
        "w_struct_jaccard", "l_struct_jaccard", "delta_struct_jaccard",
        "w_ngram_overlap", "l_ngram_overlap", "delta_ngram_overlap",
        "w_sycophancy", "l_sycophancy", "delta_sycophancy",
        "w_correction", "l_correction", "delta_correction",
        "w_affirm_prefix", "l_affirm_prefix",
    ]
    deltas = {k: [] for k in (
        "delta_struct_jaccard", "delta_ngram_overlap",
        "delta_sycophancy", "delta_correction",
    )}
    w_aff = []
    l_aff = []
    n_ok = 0
    n_skip = 0

    with per_pair.open("w", newline="") as fout:
        wcsv = csv.writer(fout)
        wcsv.writerow(header)
        for idx, row in enumerate(iter_rows(args.short_name,
                                            jsonl_fallback=args.jsonl_fallback,
                                            limit=args.limit)):
            t = extract_triplet(row)
            if not t:
                n_skip += 1
                continue
            prompt, yw, yl = t
            sw = score_response(prompt, yw)
            sl = score_response(prompt, yl)
            d_struct = sl["struct_jaccard"] - sw["struct_jaccard"]
            d_ngram = sl["ngram_overlap_n4"] - sw["ngram_overlap_n4"]
            d_syc = sl["sycophancy_hits"] - sw["sycophancy_hits"]
            d_cor = sw["correction_hits"] - sl["correction_hits"]
            wcsv.writerow([
                idx,
                f"{sw['struct_jaccard']:.4f}", f"{sl['struct_jaccard']:.4f}", f"{d_struct:.4f}",
                f"{sw['ngram_overlap_n4']:.4f}", f"{sl['ngram_overlap_n4']:.4f}", f"{d_ngram:.4f}",
                sw["sycophancy_hits"], sl["sycophancy_hits"], d_syc,
                sw["correction_hits"], sl["correction_hits"], d_cor,
                sw["affirm_prefix"], sl["affirm_prefix"],
            ])
            deltas["delta_struct_jaccard"].append(d_struct)
            deltas["delta_ngram_overlap"].append(d_ngram)
            deltas["delta_sycophancy"].append(d_syc)
            deltas["delta_correction"].append(d_cor)
            w_aff.append(sw["affirm_prefix"])
            l_aff.append(sl["affirm_prefix"])
            n_ok += 1

    def sign_test(xs):
        pos = sum(1 for v in xs if v > 0)
        neg = sum(1 for v in xs if v < 0)
        return {"n": len(xs), "positive": pos, "negative": neg,
                "mean": st.fmean(xs) if xs else 0.0,
                "pct_positive": pos / max(1, len(xs))}

    summary = {
        "dataset_short": args.short_name,
        "pairs_analyzed": n_ok,
        "pairs_skipped": n_skip,
        "delta_stats": {k: summarise(v) for k, v in deltas.items()},
        "sign_tests": {k: sign_test(v) for k, v in deltas.items()},
        "affirm_prefix_rate_chosen":   sum(w_aff) / max(1, len(w_aff)),
        "affirm_prefix_rate_rejected": sum(l_aff) / max(1, len(l_aff)),
        "interpretation": {
            "delta_struct_jaccard": "positive -> DPO penalises structural mirroring",
            "delta_ngram_overlap": "positive -> DPO penalises direct prompt copying",
            "delta_sycophancy": "positive -> DPO penalises sycophantic agreement",
            "delta_correction": "positive -> DPO rewards factual correction markers",
        },
    }
    write_json(summary_path, summary)
    print(f"[phase2] wrote {n_ok} pairs -> {per_pair}")
    print(f"[phase2] summary -> {summary_path}")
    print(json.dumps(summary["sign_tests"], indent=2))


if __name__ == "__main__":
    main()
