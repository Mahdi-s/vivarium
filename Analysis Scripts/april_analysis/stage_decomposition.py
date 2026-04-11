#!/usr/bin/env python3
"""
Phase C1 of April_analysis: 2-axis stage decomposition.

Produces:
    tables/stage_decomposition/instruct_path_by_temperature.csv
    tables/stage_decomposition/think_path_by_temperature.csv
    tables/stage_decomposition/instruct_vs_think_t0.csv
    tables/stage_decomposition/instruct_vs_think_t06.csv

Each table reports, per (variant, temperature, condition): BER (wrong-answer
endorsement rate), CE (= BER_pressure - BER_control), and Wilson 95% CIs.

The cross-path tables (instruct_vs_think_t{0,06}) are restricted to the four
conditions that are shared between the Instruct 12-condition suite and the
Think-RL 4-condition suite, so that Instruct vs Think magnitudes are directly
comparable.
"""
from __future__ import annotations

import os
from typing import List

import pandas as pd

from _common import (
    SHARED_4_CONDITIONS,
    VARIANT_ORDER,
    add_wilson_ci_columns,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)
from vivarium.analytics.behavioral import APRIL_PATH_OF, APRIL_STAGE_OF

INSTRUCT_VARIANTS: List[str] = ["base", "instruct_sft", "instruct_dpo", "instruct"]
THINK_VARIANTS: List[str] = ["base", "think_sft", "think_dpo", "think"]


def _stage_table(cells: pd.DataFrame, variants: List[str]) -> pd.DataFrame:
    out = cells[cells["variant"].isin(variants)].copy()
    out["path"] = out["variant"].map(APRIL_PATH_OF)
    out["stage"] = out["variant"].map(APRIL_STAGE_OF)
    out = add_wilson_ci_columns(out, k_col="state_B_n", n_col="n_denominator", out_prefix="ber")
    out = add_wilson_ci_columns(out, k_col="state_A_n", n_col="n_denominator", out_prefix="er")
    out = add_wilson_ci_columns(out, k_col="state_C_n", n_col="n_denominator", out_prefix="rr")

    # Attach the control-cell BER as "ber_control" so each pressure row has
    # its own CE = BER_pressure - BER_control.
    control = out[out["condition_name"] == "control"][
        ["variant", "temperature", "ber"]
    ].rename(columns={"ber": "ber_control"})
    out = out.merge(control, on=["variant", "temperature"], how="left")
    out["conformity_effect"] = out["ber"] - out["ber_control"]

    # Canonical ordering
    variant_rank = {v: i for i, v in enumerate(variants)}
    out["__rank"] = out["variant"].map(variant_rank)
    out = out.sort_values(["__rank", "temperature", "condition_name"]).drop(columns="__rank")
    return out


def _cross_path_table(cells: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """
    7 variants × 4 shared conditions at a single temperature, tagged with
    path ("instruct"/"think"/"shared") and stage for direct comparison.
    """
    mask = (
        cells["temperature"] == temperature
    ) & (
        cells["condition_name"].isin(SHARED_4_CONDITIONS)
    )
    out = cells[mask].copy()
    out["path"] = out["variant"].map(APRIL_PATH_OF)
    out["stage"] = out["variant"].map(APRIL_STAGE_OF)
    out = add_wilson_ci_columns(out, k_col="state_B_n", n_col="n_denominator", out_prefix="ber")

    # Attach control BER per variant at this temperature
    control = out[out["condition_name"] == "control"][
        ["variant", "ber"]
    ].rename(columns={"ber": "ber_control"})
    out = out.merge(control, on="variant", how="left")
    out["conformity_effect"] = out["ber"] - out["ber_control"]

    # Path and stage come from APRIL_PATH_OF / APRIL_STAGE_OF via the loader.
    # Canonical ordering: base first, then instruct_*, then think_*
    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
    condition_rank = {c: i for i, c in enumerate(SHARED_4_CONDITIONS)}
    out["__v_rank"] = out["variant"].map(variant_rank).fillna(99)
    out["__c_rank"] = out["condition_name"].map(condition_rank).fillna(99)
    out = out.sort_values(["__v_rank", "__c_rank"]).drop(columns=["__v_rank", "__c_rank"])
    return out


def main() -> int:
    args = build_argparser(
        "Phase C1: 2-axis stage decomposition (Instruct vs Think x Base-SFT-DPO-RL)"
    ).parse_args()
    df = load_trials_from_args(args)
    cells = april_cell_metrics(df)

    out_dir = args.out_dir
    out_tables = ensure_dir(os.path.join(out_dir, "tables", "stage_decomposition"))

    instruct = _stage_table(cells, INSTRUCT_VARIANTS)
    think = _stage_table(cells, THINK_VARIANTS)
    cross_t0 = _cross_path_table(cells, 0.0)
    cross_t06 = _cross_path_table(cells, 0.6)

    instruct.to_csv(os.path.join(out_tables, "instruct_path_by_temperature.csv"), index=False)
    think.to_csv(os.path.join(out_tables, "think_path_by_temperature.csv"), index=False)
    cross_t0.to_csv(os.path.join(out_tables, "instruct_vs_think_t0.csv"), index=False)
    cross_t06.to_csv(os.path.join(out_tables, "instruct_vs_think_t06.csv"), index=False)

    print_summary("instruct_path_by_temperature", instruct)
    print_summary("think_path_by_temperature", think)
    print_summary("instruct_vs_think_t0", cross_t0)
    print_summary("instruct_vs_think_t06", cross_t06)

    # Quick sanity print so the user sees the headline numbers during the run
    print("\n=== 2-axis headline (T=0, unanimous_confident) ===")
    headline = cross_t0[cross_t0["condition_name"] == "asch_zhu_unbiased_unanimous_confident"]
    print(headline[[
        "variant", "path", "stage", "ber", "ber_lo", "ber_hi",
        "ber_control", "conformity_effect"
    ]].round(3).to_string(index=False))

    print(f"\n[stage_decomposition] Wrote 4 tables to {out_tables}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
