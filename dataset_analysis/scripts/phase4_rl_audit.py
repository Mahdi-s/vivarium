"""Phase 4: audit the RL stages (Dolci-Instruct-RL, Dolci-Think-RL-7B).

RL corpora for OLMo 3 contain the *prompts* (and in some variants the
reference/preferred completions) used for RLVR/GRPO. Schema is typically
{prompt, ground_truth, verifier, ...} or {prompt, chosen, rejected}
depending on subset.

We audit two things on the prompts themselves (not the model outputs):
  1. How many RL prompts contain contiguously-repeated structural frames
     (the exact pattern your Asch test exploits)? A high rate here means
     the model gets graded on resisting repetition — good.
  2. Where completions are present, do they show the DPO-style divergence
     (correction markers, low n-gram mirroring) or SFT-style compliance?

Usage:
    python scripts/phase4_rl_audit.py --short-name instruct-rl
    python scripts/phase4_rl_audit.py --short-name think-rl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st

from common import (
    RESULTS, iter_rows,
    structural_fingerprint, count_hits,
    SYCOPHANCY_RE, CORRECTION_RE, AFFIRM_RE,
    prompt_response_ngram_overlap, summarise, write_json,
)

REPEAT_FRAME_RE = re.compile(
    r"(?:(?:participant|string|person|user|voter|respondent|source|"
    r"reviewer|choice|option|item)\s*\d+\s*[:.\)])",
    re.IGNORECASE,
)


def extract_prompt(row: dict) -> str:
    for k in ("prompt", "input", "question", "instruction"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list):
            # list of chat turns
            return "\n".join(m.get("content", "") for m in v if isinstance(m, dict))
    msgs = row.get("messages")
    if isinstance(msgs, list):
        return "\n".join(
            m.get("content", "") for m in msgs
            if isinstance(m, dict) and (m.get("role") or "") in ("user", "human", "system")
        )
    return ""


def extract_completion(row: dict) -> str:
    for k in ("completion", "response", "output", "answer", "chosen"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list):
            for m in reversed(v):
                if isinstance(m, dict) and (m.get("role") or m.get("from")) in ("assistant", "gpt"):
                    return m.get("content", "") or ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-name", required=True,
                    choices=("instruct-rl", "think-rl"))
    ap.add_argument("--jsonl-fallback", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    per_row = RESULTS / f"phase4_{args.short_name}_per_row.csv"
    summary_path = RESULTS / f"phase4_{args.short_name}_summary.json"

    header = [
        "idx", "prompt_len", "repeat_frames", "has_repeat_frame",
        "prompt_has_list", "comp_len", "comp_ngram_overlap",
        "comp_correction", "comp_sycophancy", "comp_affirm_prefix",
    ]
    n = 0
    with_comp = 0
    with_frame = 0
    acc = {k: [] for k in (
        "repeat_frames", "has_repeat_frame",
        "prompt_has_list", "comp_ngram_overlap",
        "comp_correction", "comp_sycophancy", "comp_affirm_prefix",
    )}

    with per_row.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        for idx, row in enumerate(iter_rows(args.short_name,
                                            jsonl_fallback=args.jsonl_fallback,
                                            limit=args.limit)):
            p = extract_prompt(row)
            c = extract_completion(row)
            if not p:
                continue
            pf = structural_fingerprint(p)
            frames = len(REPEAT_FRAME_RE.findall(p))
            has_frame = int(frames >= 3)  # "Participant 1:X ... Participant 3:X" minimum
            p_list = int((pf["bullet"] + pf["numbered"]) > 0)
            if c:
                with_comp += 1
                ng = prompt_response_ngram_overlap(p, c, n=4)
                cor = count_hits(CORRECTION_RE, c)
                syc = count_hits(SYCOPHANCY_RE, c)
                ap_pref = int(bool(c) and bool(AFFIRM_RE.search(c[:80])))
            else:
                ng, cor, syc, ap_pref = 0.0, 0, 0, 0
            w.writerow([idx, pf["len"], frames, has_frame, p_list,
                        len(c), f"{ng:.4f}", cor, syc, ap_pref])
            acc["repeat_frames"].append(frames)
            acc["has_repeat_frame"].append(has_frame)
            acc["prompt_has_list"].append(p_list)
            if c:
                acc["comp_ngram_overlap"].append(ng)
                acc["comp_correction"].append(cor)
                acc["comp_sycophancy"].append(syc)
                acc["comp_affirm_prefix"].append(ap_pref)
            with_frame += has_frame
            n += 1

    summary = {
        "dataset_short": args.short_name,
        "rows_analyzed": n,
        "rows_with_completion": with_comp,
        "prompts_with_repeat_frame": with_frame,
        "repeat_frame_rate": with_frame / max(1, n),
        "metrics": {k: summarise(v) for k, v in acc.items() if v},
    }
    write_json(summary_path, summary)
    print(f"[phase4:{args.short_name}] n={n}  repeat_frame_rate={summary['repeat_frame_rate']:.4f}")
    print(f"[phase4:{args.short_name}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
