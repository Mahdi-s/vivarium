"""Phase 1: audit `allenai/Dolci-Instruct-SFT` for the formatting trap.

Hypothesis: SFT teaches "continue the prompt's structural format" and
"open with affirmation". That conditioning is exactly what the Asch prompt
exploits at inference time.

For each (user prompt, assistant response) pair we compute:
  - structural_jaccard:       set-overlap of {bullets, numbered, headings,
                              code fences, colon-lines, newlines} between
                              prompt and response
  - ngram_overlap_n4:         fraction of 4-grams in the response that occur
                              verbatim in the prompt (direct induction proxy)
  - affirm_prefix:            does the response open with "Yes,"/"Sure,"/
                              "Here is..."/etc.
  - sycophancy_hits:          count of "you're right"-style phrases
  - resp_has_list:            response contains any bullet/numbered list
  - prompt_has_list:          prompt contains any bullet/numbered list
  - list_carry_over:          prompt_has_list AND resp_has_list (conditional
                              on prompt_has_list gives P(continue format))

Outputs:
  results/phase1_sft_per_example.csv   (row-level metrics, limited to 20k)
  results/phase1_sft_summary.json      (aggregate stats + conditional probs)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (
    DATA, RESULTS, LOGS, iter_rows,
    structural_jaccard, structural_fingerprint,
    affirm_prefix, count_hits, SYCOPHANCY_RE,
    prompt_response_ngram_overlap, summarise, write_json,
)


def extract_pair(row: dict) -> tuple[str, str] | None:
    """Dolci-SFT rows typically have a `messages` list of chat turns.
    Return the first (user, assistant) pair; return None if malformed."""
    msgs = row.get("messages") or row.get("conversation") or row.get("conversations")
    if not msgs:
        # sometimes flat prompt/response fields
        u = row.get("prompt") or row.get("instruction") or row.get("input")
        a = row.get("response") or row.get("output") or row.get("completion")
        if u and a:
            return str(u), str(a)
        return None
    user, asst = None, None
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or ""
        if role in ("user", "human") and user is None:
            user = str(content)
        elif role in ("assistant", "gpt", "model") and user is not None and asst is None:
            asst = str(content)
            break
    if user and asst:
        return user, asst
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-name", default="instruct-sft",
                    help="parquet dir under data/raw/ (instruct-sft, think-sft)")
    ap.add_argument("--jsonl-fallback", default="sft")
    ap.add_argument("--tag", default=None,
                    help="suffix for output files, defaults to --short-name")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = args.tag or args.short_name

    per_example_path = RESULTS / f"phase1_{tag}_per_example.csv"
    summary_path = RESULTS / f"phase1_{tag}_summary.json"

    cols = [
        "idx", "prompt_len", "resp_len",
        "structural_jaccard", "ngram_overlap_n4",
        "affirm_prefix", "sycophancy_hits",
        "prompt_has_list", "resp_has_list",
    ]
    rows_written = 0
    skipped = 0
    agg = {c: [] for c in cols if c not in ("idx",)}
    list_carry_num = 0
    list_carry_den = 0

    with per_example_path.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(cols)
        for idx, row in enumerate(iter_rows(args.short_name,
                                            jsonl_fallback=args.jsonl_fallback,
                                            limit=args.limit)):
            pair = extract_pair(row)
            if not pair:
                skipped += 1
                continue
            user, asst = pair
            sj = structural_jaccard(user, asst)
            ng = prompt_response_ngram_overlap(user, asst, n=4)
            ap = int(affirm_prefix(asst))
            sh = count_hits(SYCOPHANCY_RE, asst)
            pf = structural_fingerprint(user)
            rf = structural_fingerprint(asst)
            p_list = int((pf["bullet"] + pf["numbered"]) > 0)
            r_list = int((rf["bullet"] + rf["numbered"]) > 0)

            w.writerow([idx, pf["len"], rf["len"], f"{sj:.4f}",
                        f"{ng:.4f}", ap, sh, p_list, r_list])
            rows_written += 1
            agg["prompt_len"].append(pf["len"])
            agg["resp_len"].append(rf["len"])
            agg["structural_jaccard"].append(sj)
            agg["ngram_overlap_n4"].append(ng)
            agg["affirm_prefix"].append(ap)
            agg["sycophancy_hits"].append(sh)
            agg["prompt_has_list"].append(p_list)
            agg["resp_has_list"].append(r_list)
            if p_list:
                list_carry_den += 1
                if r_list:
                    list_carry_num += 1

    summary = {
        "dataset_short": args.short_name,
        "rows_analyzed": rows_written,
        "rows_skipped": skipped,
        "metrics": {k: summarise(v) for k, v in agg.items()},
        "P(resp_has_list | prompt_has_list)":
            (list_carry_num / list_carry_den) if list_carry_den else None,
        "prompt_has_list_rate":
            sum(agg["prompt_has_list"]) / max(1, len(agg["prompt_has_list"])),
        "affirm_prefix_rate":
            sum(agg["affirm_prefix"]) / max(1, len(agg["affirm_prefix"])),
    }
    write_json(summary_path, summary)
    print(f"[phase1] wrote {rows_written} rows -> {per_example_path}")
    print(f"[phase1] summary -> {summary_path}")
    print(json.dumps({k: summary[k] for k in
                      ("rows_analyzed", "P(resp_has_list | prompt_has_list)",
                       "affirm_prefix_rate")}, indent=2))


if __name__ == "__main__":
    main()
