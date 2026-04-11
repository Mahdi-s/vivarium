#!/usr/bin/env python3
"""
Phase C5 of April_analysis: item-level knowledge-vs-endorsement correlation.

At T=0 unanimous_confident, for each (variant, item) compute:
    control_correct  = 1 if judge says control-condition answer is correct
    pressure_endorse = 1 if judge says pressure-condition wrong-answer-endorsed
    control_ber      = 1 if judge says control-condition wrong-answer-endorsed
                       (baseline "spontaneous hallucination of target" rate)

The deliberation hypothesis predicts: items the model knows well (control
correct) are robust to pressure -> strong negative correlation between
`control_correct` and `pressure_endorse`.

The pattern-completion hypothesis predicts: correctness and endorsement are
roughly orthogonal; the target repetitions override knowledge -> weak
or near-zero correlation.

Outputs:
    item_level/per_item_endorsement.csv   one row per (variant, item)
    item_level/knowledge_vs_endorsement.csv   per-variant point-biserial
                                              between control correctness and
                                              pressure endorsement, plus the
                                              paired-difference table
    item_level/cross_variant_agreement.csv   per-item: how many of the 7
                                              variants endorsed the target
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _common import (
    VARIANT_ORDER,
    april_classify_state,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)

PRESSURE_CONDITION = "asch_zhu_unbiased_unanimous_confident"


def main() -> int:
    args = build_argparser(
        "Phase C5: item-level knowledge-vs-endorsement correlation at T=0 unanimous_confident"
    ).parse_args()
    df = load_trials_from_args(args)

    df = df[df["temperature"] == 0.0].copy()

    # For each (variant, item), collect control + pressure outcomes.
    control = df[df["condition_name"] == "control"][[
        "variant", "item_id", "domain", "judge_is_correct", "judge_wrong_endorsed", "state"
    ]].rename(columns={
        "judge_is_correct": "control_is_correct",
        "judge_wrong_endorsed": "control_wrong_endorsed",
        "state": "control_state",
    })
    pressure = df[df["condition_name"] == PRESSURE_CONDITION][[
        "variant", "item_id", "judge_is_correct", "judge_wrong_endorsed", "state"
    ]].rename(columns={
        "judge_is_correct": "pressure_is_correct",
        "judge_wrong_endorsed": "pressure_wrong_endorsed",
        "state": "pressure_state",
    })
    merged = control.merge(pressure, on=["variant", "item_id"], how="inner")

    # Classify each item into a usable mask: we need both judge labels non-null
    merged["usable"] = merged["control_is_correct"].notna() & merged["pressure_is_correct"].notna()
    per_item_path = os.path.join(ensure_dir(os.path.join(args.out_dir, "item_level")), "per_item_endorsement.csv")
    merged.to_csv(per_item_path, index=False)
    print_summary(f"per_item_endorsement -> {per_item_path}", merged)

    # Per-variant point-biserial correlation between control_is_correct (0/1)
    # and pressure_wrong_endorsed (0/1). Drop unusable rows.
    usable = merged[merged["usable"]].copy()
    rows = []
    for variant in VARIANT_ORDER:
        sub = usable[usable["variant"] == variant]
        if len(sub) < 10:
            rows.append({"variant": variant, "n_usable": len(sub)})
            continue
        x = sub["control_is_correct"].to_numpy(dtype=float)
        y = sub["pressure_wrong_endorsed"].fillna(0).to_numpy(dtype=float)
        if x.std() == 0 or y.std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        # Conditional endorsement rates (knowledge vs no-knowledge subset)
        knew = sub[sub["control_is_correct"] == 1]
        didnt = sub[sub["control_is_correct"] == 0]
        rows.append({
            "variant": variant,
            "n_usable": int(len(sub)),
            "n_knew_control": int(len(knew)),
            "n_didnt_know_control": int(len(didnt)),
            "corr_knowledge_vs_endorsement": r,
            "endorse_rate_if_knew": float(knew["pressure_wrong_endorsed"].fillna(0).mean()) if len(knew) else float("nan"),
            "endorse_rate_if_didnt_know": float(didnt["pressure_wrong_endorsed"].fillna(0).mean()) if len(didnt) else float("nan"),
        })
    corr = pd.DataFrame(rows)
    corr_path = os.path.join(args.out_dir, "item_level", "knowledge_vs_endorsement.csv")
    corr.to_csv(corr_path, index=False)
    print_summary(f"knowledge_vs_endorsement -> {corr_path}", corr)

    # Cross-variant agreement: per-item, how many of the 7 variants endorse.
    agree = (
        usable.groupby("item_id")
        .agg(
            n_variants=("variant", "nunique"),
            n_endorsed=("pressure_wrong_endorsed", lambda s: int(s.fillna(0).sum())),
        )
        .reset_index()
    )
    agree["frac_endorsed"] = agree["n_endorsed"] / agree["n_variants"].replace(0, np.nan)
    agree_path = os.path.join(args.out_dir, "item_level", "cross_variant_agreement.csv")
    agree.to_csv(agree_path, index=False)
    print_summary(f"cross_variant_agreement -> {agree_path}", agree)

    # Summary print
    print("\n=== Knowledge-vs-endorsement correlations at T=0 unanimous_confident ===")
    print(corr.round(3).to_string(index=False))

    print("\n=== Distribution of items by number of variants that endorsed ===")
    print(agree["n_endorsed"].value_counts().sort_index().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
