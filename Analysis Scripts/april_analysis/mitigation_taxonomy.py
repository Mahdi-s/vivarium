#!/usr/bin/env python3
"""
Phase C6 of April_analysis: mitigation taxonomy table.

Mitigations collected on the Instruct family (base + 3 Instruct stages, 12
conditions × 6 temperatures). Each row computes

    Delta = BER(mitigation) - BER(unanimous_plain_anchor)

at T=0.

The Think family has **only the 4 shared conditions** (control,
unanimous_confident, authoritative_bias, authority_zhu_unbiased_trust).
The `unanimous_plain` anchor, DA, QD, diverse peers, uncertain tone, and
authority_trust_da conditions were never run on the HPC Think pipeline.
For those cells we emit explicit N/A rows with
`status = not_collected_for_think_path` so the table is transparent
about the asymmetric coverage.

Because we cannot compute DA/QD/Diverse for the Think path directly, we
add a weak proxy row:

    "Reasoning prefix (<think>)": for each stage s ∈ {sft, dpo, rl}, this
    row reports BER on `unanimous_confident` at T=0 for the Think variant
    minus the matched Instruct variant. Negative delta means the `<think>`
    prefix itself already functions as a pattern-break mitigation, without
    any of the explicit prompt-level mitigations needing to be applied.
    The base row repeats the base variant for both sides (delta = 0).

Output:
    tables/mitigation_taxonomy/taxonomy_table.csv
    tables/mitigation_taxonomy/pattern_match_scores.csv
    tables/mitigation_taxonomy/think_prefix_proxy.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _common import (
    PATTERN_MATCH_REPS,
    SHARED_4_CONDITIONS,
    VARIANT_ORDER,
    add_wilson_ci_columns,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)

ANCHOR = "asch_zhu_unbiased_unanimous_plain"

# Which conditions the Think family actually has in the HPC / runs-think
# sources. Every other condition must be marked N/A for Think variants.
THINK_COLLECTED_CONDITIONS = set(SHARED_4_CONDITIONS)
THINK_VARIANTS = ("think_sft", "think_dpo", "think")

MITIGATIONS = [
    # (label, condition_name, predicted_direction, mechanism_hypothesis)
    ("Devil's Advocate (4+1)",
     "asch_zhu_unbiased_da", "partial_break",
     "1 peer disagrees: reps=4 vs 5; pattern completion slightly degraded"),
    ("Question Distillation",
     "asch_zhu_unbiased_qd", "strong_break",
     "Peer block rewritten away from verbatim target: reps=1"),
    ("Diverse peers (no majority)",
     "asch_zhu_unbiased_diverse_plain", "complete_break",
     "No repeated target; reps=1; full pattern disruption"),
    ("Authority trust + DA",
     "authority_zhu_unbiased_trust_da", "authority_bound",
     "Authority prompt with DA: tests whether authority context keeps endorsement up"),
    ("Unanimous confident tone",
     "asch_zhu_unbiased_unanimous_confident", "no_break",
     "Max pressure with tone cue; reps=5; baseline vs anchor"),
    ("Unanimous uncertain tone",
     "asch_zhu_unbiased_unanimous_uncertain", "no_break",
     "Softened tone on identical reps=5 anchor"),
]

# Every mitigation row here depends on one of these two conditions being
# present. If the variant's family did not collect either the anchor or
# the mitigation condition, emit N/A instead of silently dropping.
REQUIRED_CONDITIONS = {ANCHOR} | {cond for (_, cond, _, _) in MITIGATIONS}


def _mech_consistent(pred: str, delta: float) -> "bool | None":
    if pred == "partial_break":
        return delta < -0.02
    if pred in ("strong_break", "complete_break"):
        return delta < -0.10
    if pred == "no_break":
        return abs(delta) < 0.10
    # authority_bound or anything else
    return None


def main() -> int:
    args = build_argparser(
        "Phase C6: mitigation taxonomy table"
    ).parse_args()
    df = load_trials_from_args(args)
    cells = april_cell_metrics(df)

    out_dir = args.out_dir
    tax_dir = ensure_dir(os.path.join(out_dir, "tables", "mitigation_taxonomy"))

    # Pattern-match score table for reference
    scores = pd.DataFrame([
        {"condition_name": c, "pattern_reps": r}
        for c, r in sorted(PATTERN_MATCH_REPS.items(), key=lambda kv: (kv[1], kv[0]))
    ])
    scores_path = os.path.join(tax_dir, "pattern_match_scores.csv")
    scores.to_csv(scores_path, index=False)
    print_summary(f"pattern_match_scores -> {scores_path}", scores)

    # ---- Main taxonomy table: mitigations x variants at T=0 -----------
    t0 = cells[cells["temperature"] == 0.0].copy()

    # Set of (variant, condition) pairs actually present in the cells table.
    present = set(zip(t0["variant"], t0["condition_name"]))

    anchor_ber = (
        t0[t0["condition_name"] == ANCHOR].set_index("variant")["ber"]
    )

    rows = []
    for label, cond, pred, mech in MITIGATIONS:
        sub = t0[t0["condition_name"] == cond].set_index("variant")
        for variant in VARIANT_ORDER:
            think_family = variant in THINK_VARIANTS
            cond_missing = (variant, cond) not in present
            anchor_missing = (variant, ANCHOR) not in present

            if think_family and (cond_missing or anchor_missing):
                # Think-family N/A row: transparent about coverage gap.
                rows.append({
                    "mitigation": label,
                    "condition_name": cond,
                    "variant": variant,
                    "pattern_reps": PATTERN_MATCH_REPS.get(cond, "NA"),
                    "ber_mitigation": float("nan"),
                    "ber_anchor_unanimous_plain": float("nan"),
                    "delta_from_anchor": float("nan"),
                    "predicted_direction": pred,
                    "mechanism_hypothesis": mech,
                    "mechanism_consistent": None,
                    "status": "not_collected_for_think_path",
                })
                continue

            if cond_missing or anchor_missing:
                # Non-Think variant with a coverage gap — shouldn't happen
                # in practice, but be explicit.
                rows.append({
                    "mitigation": label,
                    "condition_name": cond,
                    "variant": variant,
                    "pattern_reps": PATTERN_MATCH_REPS.get(cond, "NA"),
                    "ber_mitigation": float("nan"),
                    "ber_anchor_unanimous_plain": float("nan"),
                    "delta_from_anchor": float("nan"),
                    "predicted_direction": pred,
                    "mechanism_hypothesis": mech,
                    "mechanism_consistent": None,
                    "status": "missing_data",
                })
                continue

            ber = float(sub.loc[variant, "ber"])
            ber_anchor = float(anchor_ber.loc[variant])
            delta = ber - ber_anchor
            rows.append({
                "mitigation": label,
                "condition_name": cond,
                "variant": variant,
                "pattern_reps": PATTERN_MATCH_REPS.get(cond, "NA"),
                "ber_mitigation": ber,
                "ber_anchor_unanimous_plain": ber_anchor,
                "delta_from_anchor": delta,
                "predicted_direction": pred,
                "mechanism_hypothesis": mech,
                "mechanism_consistent": _mech_consistent(pred, delta),
                "status": "computed",
            })

    tax = pd.DataFrame(rows)

    # Canonical ordering: mitigation label, then VARIANT_ORDER.
    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
    tax["__v"] = tax["variant"].map(variant_rank).fillna(99)
    tax = tax.sort_values(["mitigation", "__v"]).drop(columns="__v")

    tax_path = os.path.join(tax_dir, "taxonomy_table.csv")
    tax.to_csv(tax_path, index=False)
    print_summary(f"taxonomy_table -> {tax_path}", tax)

    # ---- <think> prefix proxy (Q3 decision) ----------------------------
    # Cross-path comparison at T=0 on unanimous_confident: for each
    # post-training stage, does the Think-path variant endorse less than
    # the matched Instruct-path variant? This is the only place in the
    # table where we can make a Think-vs-Instruct mitigation claim because
    # the shared 4 conditions include unanimous_confident on every stage.
    proxy_rows = []
    stage_pairs = [
        ("base",         "base",        "base"),
        ("instruct_sft", "think_sft",   "sft"),
        ("instruct_dpo", "think_dpo",   "dpo"),
        ("instruct",     "think",       "rl"),
    ]
    confident = "asch_zhu_unbiased_unanimous_confident"
    conf_lookup = (
        t0[t0["condition_name"] == confident]
        .set_index("variant")[["ber", "state_B_n", "n_denominator"]]
    )
    ctrl_lookup = (
        t0[t0["condition_name"] == "control"]
        .set_index("variant")[["ber", "state_B_n", "n_denominator"]]
    )
    for instruct_var, think_var, stage in stage_pairs:
        if instruct_var not in conf_lookup.index or think_var not in conf_lookup.index:
            continue
        instr_ber = float(conf_lookup.loc[instruct_var, "ber"])
        think_ber = float(conf_lookup.loc[think_var, "ber"])
        instr_ctrl = float(ctrl_lookup.loc[instruct_var, "ber"]) if instruct_var in ctrl_lookup.index else float("nan")
        think_ctrl = float(ctrl_lookup.loc[think_var, "ber"]) if think_var in ctrl_lookup.index else float("nan")
        delta_prefix = think_ber - instr_ber  # negative = <think> prefix helps
        delta_ce_instr = instr_ber - instr_ctrl  # conformity effect on Instruct side
        delta_ce_think = think_ber - think_ctrl  # conformity effect on Think side
        proxy_rows.append({
            "stage": stage,
            "instruct_variant": instruct_var,
            "think_variant": think_var,
            "condition_name": confident,
            "ber_instruct_unanimous_confident": instr_ber,
            "ber_think_unanimous_confident": think_ber,
            "ber_instruct_control": instr_ctrl,
            "ber_think_control": think_ctrl,
            "delta_think_minus_instruct_pressure": delta_prefix,
            "conformity_effect_instruct": delta_ce_instr,
            "conformity_effect_think": delta_ce_think,
            "think_prefix_breaks_pattern": delta_prefix < -0.10,
            "notes": "Cross-path proxy for pattern-break effect of reasoning prefix. Think family does not collect DA/QD/Diverse conditions; this is the only mitigation-style claim achievable from the 4 shared conditions.",
        })
    proxy = pd.DataFrame(proxy_rows)
    proxy_path = os.path.join(tax_dir, "think_prefix_proxy.csv")
    proxy.to_csv(proxy_path, index=False)
    print_summary(f"think_prefix_proxy -> {proxy_path}", proxy)

    # ---- Summary: Instruct-only mean delta per mitigation -------------
    print("\n=== Mean Delta from anchor (T=0, Instruct-path rows only) ===")
    instruct_rows = tax[
        tax["status"] == "computed"
    ].copy()
    instruct_rows["path"] = instruct_rows["variant"].apply(
        lambda v: "instruct"
        if v.startswith("instruct")
        else ("shared" if v == "base" else "think")
    )
    summary = (
        instruct_rows.groupby(["mitigation", "path"])["delta_from_anchor"]
        .mean()
        .unstack()
        .round(3)
    )
    print(summary.to_string())
    print("\n(Negative delta = mitigation lowered BER. Think-path cells are "
          "intentionally absent — conditions not collected on HPC Think runs.)")

    print("\n=== <think> prefix proxy (BER on unanimous_confident, T=0) ===")
    if not proxy.empty:
        cols = [
            "stage",
            "ber_instruct_unanimous_confident",
            "ber_think_unanimous_confident",
            "delta_think_minus_instruct_pressure",
        ]
        print(proxy[cols].round(3).to_string(index=False))
        print("\n(Negative delta = <think> prefix suppresses pattern completion "
              "relative to matched Instruct stage.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
