#!/usr/bin/env python3
"""
Phase C2 of April_analysis: 12-condition pattern-match gradient.

Pre-registered pattern-match score: **target-answer repetition count** in the
rendered peer/authority block. See
`Comparing_Experiments/April_analysis/pattern_match/pattern_score_definition.md`
for the formal definition.

Hypothesis under the autoregressive-pattern-completion reframe: BER should
be monotonically non-decreasing in pattern-match repetition count. A
social-deference hypothesis predicts BER depends on tone/authority rather
than repetition count, so confident-tone conditions (reps=5) should cluster
with QD (reps=1) rather than with plain/neutral unanimous variants.

Outputs:
    pattern_match/pattern_score_definition.md   Written on first run
    pattern_match/per_condition_pattern_scores.csv
    pattern_match/er_vs_pattern_correlation.csv Spearman rho per variant
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    CONDITION_GRADIENT_ORDER,
    PATTERN_MATCH_REPS,
    VARIANT_ORDER,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)

# Spearman rank correlation without scipy dependency (avoid pulling in scipy).
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    xs = pd.Series(x).rank(method="average").to_numpy()
    ys = pd.Series(y).rank(method="average").to_numpy()
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


PATTERN_SCORE_DEFINITION = """\
# Pattern-Match Score Definition (pre-registered)

**Score:** `target-answer repetition count` — the number of times the exact
wrong-answer string appears in the peer/authority block of the rendered
prompt.

This is the simplest principled pattern-completion score and maps directly
to the autoregressive-pattern-completion theory: more repetitions of the
target string in the context increase the argmax-continuation probability
of that string under an induction-head attention pattern.

| Condition | Repetitions |
|---|---|
| control | 0 (no peer/authority block) |
| asch_history_5 | 5 (legacy 5-peer unanimous) |
| asch_zhu_unbiased_unanimous_plain | 5 |
| asch_zhu_unbiased_unanimous_confident | 5 |
| asch_zhu_unbiased_unanimous_uncertain | 5 |
| asch_zhu_unbiased_unanimous_neutral | 5 |
| asch_zhu_unbiased_da | 4 (devil's advocate: 4 of 5 repeat the target) |
| asch_zhu_unbiased_qd | 1 (question distillation rewrites once) |
| asch_zhu_unbiased_diverse_plain | 1 (diverse peers: only 1 target mention) |
| authoritative_bias | 1 (single authority statement) |
| authority_zhu_unbiased_trust | 1 |
| authority_zhu_unbiased_trust_da | 1 |

**Test:** per-variant Spearman rank correlation between condition-level BER
and repetition count at T=0. A correlation above +0.5 is evidence for the
pattern-completion account; a correlation near zero or negative would
falsify it.

**Pre-registration timestamp:** recorded at the first write of this file
under `Comparing_Experiments/April_analysis/pattern_match/`. This file is
git-tracked alongside the analysis output to establish precedence against
any post-hoc re-parameterization.
"""


def emit_definition_file(out_dir: str) -> str:
    pm_dir = ensure_dir(os.path.join(out_dir, "pattern_match"))
    path = os.path.join(pm_dir, "pattern_score_definition.md")
    if not os.path.exists(path):
        Path(path).write_text(PATTERN_SCORE_DEFINITION)
    return path


def main() -> int:
    args = build_argparser(
        "Phase C2: 12-condition pattern-match gradient"
    ).parse_args()
    df = load_trials_from_args(args)
    cells = april_cell_metrics(df)

    out_dir = args.out_dir
    pm_dir = ensure_dir(os.path.join(out_dir, "pattern_match"))
    emit_definition_file(out_dir)

    # Per-condition pattern-match score table, joined with T=0 BER for each
    # of the 7 variants.
    cells_t0 = cells[cells["temperature"] == 0.0].copy()
    cells_t0["pattern_reps"] = cells_t0["condition_name"].map(PATTERN_MATCH_REPS)
    if cells_t0["pattern_reps"].isna().any():
        missing = sorted(cells_t0[cells_t0["pattern_reps"].isna()]["condition_name"].unique())
        raise SystemExit(f"Unknown conditions (add to PATTERN_MATCH_REPS): {missing}")

    per_cond = cells_t0[[
        "variant", "condition_name", "pattern_reps",
        "state_B_n", "n_denominator", "ber",
    ]].copy()
    per_cond = per_cond.rename(columns={"state_B_n": "n_wrong_endorsed"})

    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
    cond_rank = {c: i for i, c in enumerate(CONDITION_GRADIENT_ORDER)}
    per_cond["__v"] = per_cond["variant"].map(variant_rank).fillna(99)
    per_cond["__c"] = per_cond["condition_name"].map(cond_rank).fillna(99)
    per_cond = per_cond.sort_values(["__v", "__c"]).drop(columns=["__v", "__c"])

    per_cond_path = os.path.join(pm_dir, "per_condition_pattern_scores.csv")
    per_cond.to_csv(per_cond_path, index=False)
    print_summary(f"per_condition_pattern_scores -> {per_cond_path}", per_cond)

    # Spearman rho per variant. The Think family only has 4 conditions
    # (control, unanimous_confident, authoritative_bias, authority_trust)
    # which collapses to 3 distinct repetition counts {0, 1, 5}. That is
    # enough for Spearman to return a value but the power is much lower
    # than on the Instruct family's 12 conditions. We additionally emit a
    # delta_ber_reps5_minus_reps1 column as the simpler pattern-completion
    # gradient metric that works symmetrically across Instruct and Think.
    rows = []
    for variant in VARIANT_ORDER:
        sub = per_cond[per_cond["variant"] == variant]
        if sub.empty:
            continue
        reps = sub["pattern_reps"].to_numpy(dtype=float)
        ber = sub["ber"].to_numpy(dtype=float)
        rho = _spearman(reps, ber)
        ber_r5 = float(sub[sub["pattern_reps"] == 5]["ber"].mean()) if (sub["pattern_reps"] == 5).any() else float("nan")
        ber_r4 = float(sub[sub["pattern_reps"] == 4]["ber"].mean()) if (sub["pattern_reps"] == 4).any() else float("nan")
        ber_r1 = float(sub[sub["pattern_reps"] == 1]["ber"].mean()) if (sub["pattern_reps"] == 1).any() else float("nan")
        ber_r0 = float(sub[sub["pattern_reps"] == 0]["ber"].mean()) if (sub["pattern_reps"] == 0).any() else float("nan")
        delta_5_minus_1 = (ber_r5 - ber_r1) if not (np.isnan(ber_r5) or np.isnan(ber_r1)) else float("nan")
        rows.append({
            "variant": variant,
            "n_conditions": int(len(sub)),
            "n_distinct_reps": int(sub["pattern_reps"].nunique()),
            "spearman_rho": rho,
            "delta_ber_reps5_minus_reps1": delta_5_minus_1,
            "ber_at_reps_0": ber_r0,
            "ber_at_reps_1": ber_r1,
            "ber_at_reps_4": ber_r4,
            "ber_at_reps_5": ber_r5,
        })
    corr = pd.DataFrame(rows)
    corr_path = os.path.join(pm_dir, "er_vs_pattern_correlation.csv")
    corr.to_csv(corr_path, index=False)
    print_summary(f"er_vs_pattern_correlation -> {corr_path}", corr)

    print("\n=== Pattern-match gradient (T=0) ===")
    print(corr.round(3).to_string(index=False))

    print("\n=== BER ladder at reps 0/1/4/5 (T=0), averaged across 5-rep variants ===")
    ladder = (
        per_cond.groupby(["variant", "pattern_reps"])["ber"]
        .mean()
        .unstack()
        .reindex(VARIANT_ORDER)
    )
    print(ladder.round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
