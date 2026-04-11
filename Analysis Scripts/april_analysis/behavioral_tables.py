#!/usr/bin/env python3
"""
Emit the core behavioral tables for Comparing_Experiments/April_analysis/.

Outputs (all under --out-dir / Comparing_Experiments/April_analysis):

    metadata/trial_counts.csv           raw per-(variant, T, condition) trial counts
    metadata/judge_coverage.csv         _llm_judge presence rates per cell
    tables/behavioral/error_rates.csv           strict-correct rate (state A)
    tables/behavioral/endorsement_rates.csv     wrong-answer endorsement (state B, the key metric)
                                                with Wilson 95% CI
    tables/behavioral/refusal_rates.csv         refusal rate (state C, includes implicit refusals)
    tables/behavioral/state_decomposition.csv   full 4-state decomposition (A/B/C/D) per cell,
                                                fixed N=400 denominator
    tables/behavioral/conformity_effect.csv     CE = BER(pressure) - BER(control) per
                                                (variant, T, pressure condition)
    tables/behavioral/domain_breakdown.csv      per-(variant, T, condition, domain) state counts

Every metric is derived from parsed_answer_json top-level judge labels via the
canonical loader in vivarium.analytics.behavioral.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    SHARED_4_CONDITIONS,
    VARIANT_ORDER,
    add_wilson_ci_columns,
    april_cell_denominator,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)


def emit_trial_counts(df: pd.DataFrame, out_dir: str) -> None:
    counts = (
        df.groupby(["variant", "temperature", "condition_name"], dropna=False)
        .size()
        .reset_index(name="n_trials")
    )
    path = os.path.join(out_dir, "metadata", "trial_counts.csv")
    counts.to_csv(path, index=False)
    print_summary(f"trial_counts -> {path}", counts)


def emit_judge_coverage(df: pd.DataFrame, out_dir: str) -> None:
    # load_april_trials already filtered to has_judge=True, so coverage is 1.0
    # by construction. This table reports the breakdown by whether the judge's
    # final labels were usable (non-null is_correct / wrong_endorsed).
    work = df.copy()
    work["has_judge"] = True  # already filtered
    work["judge_label_ok"] = work["judge_is_correct"].notna() & work["judge_wrong_endorsed"].notna()
    coverage = (
        work.groupby(["variant", "temperature", "condition_name"], dropna=False)
        .agg(
            n_trials=("trial_id", "size"),
            n_with_judge=("has_judge", "sum"),
            n_judge_label_ok=("judge_label_ok", "sum"),
        )
        .reset_index()
    )
    coverage["judge_presence_rate"] = coverage["n_with_judge"] / coverage["n_trials"]
    coverage["judge_usable_rate"] = coverage["n_judge_label_ok"] / coverage["n_trials"]
    path = os.path.join(out_dir, "metadata", "judge_coverage.csv")
    coverage.to_csv(path, index=False)
    print_summary(f"judge_coverage -> {path}", coverage)


def emit_state_decomposition(cells: pd.DataFrame, out_dir: str) -> None:
    """Full 4-state decomposition with Wilson CIs on each state rate."""
    out = cells.copy()
    for state_col, prefix in [
        ("state_A_n", "A_correct"),
        ("state_B_n", "B_wrong_endorsed"),
        ("state_C_n", "C_refusal"),
        ("state_D_n", "D_unclassified"),
    ]:
        out = add_wilson_ci_columns(out, k_col=state_col, n_col="n_denominator", out_prefix=prefix)
    path = os.path.join(out_dir, "tables", "behavioral", "state_decomposition.csv")
    out.to_csv(path, index=False)
    print_summary(f"state_decomposition -> {path}", out)


def emit_error_rates(cells: pd.DataFrame, out_dir: str) -> None:
    """Strict-correct rate (state A) with Wilson 95% CIs."""
    out = cells[["variant", "temperature", "condition_name",
                 "n_denominator", "state_A_n"]].copy()
    out = add_wilson_ci_columns(out, k_col="state_A_n", n_col="n_denominator", out_prefix="correct_rate")
    out = out.rename(columns={"state_A_n": "n_correct"})
    path = os.path.join(out_dir, "tables", "behavioral", "error_rates.csv")
    out.to_csv(path, index=False)
    print_summary(f"error_rates -> {path}", out)


def emit_endorsement_rates(cells: pd.DataFrame, out_dir: str) -> None:
    """BER (wrong-answer endorsement rate) with Wilson 95% CIs. This is THE metric."""
    out = cells[["variant", "temperature", "condition_name",
                 "n_denominator", "state_B_n"]].copy()
    out = add_wilson_ci_columns(out, k_col="state_B_n", n_col="n_denominator", out_prefix="ber")
    out = out.rename(columns={"state_B_n": "n_wrong_endorsed"})
    path = os.path.join(out_dir, "tables", "behavioral", "endorsement_rates.csv")
    out.to_csv(path, index=False)
    print_summary(f"endorsement_rates -> {path}", out)


def emit_refusal_rates(cells: pd.DataFrame, out_dir: str) -> None:
    """Refusal rate (explicit + implicit refusals) with Wilson 95% CIs."""
    out = cells[["variant", "temperature", "condition_name",
                 "n_denominator", "state_C_n"]].copy()
    out = add_wilson_ci_columns(out, k_col="state_C_n", n_col="n_denominator", out_prefix="rr")
    out = out.rename(columns={"state_C_n": "n_refusal"})
    path = os.path.join(out_dir, "tables", "behavioral", "refusal_rates.csv")
    out.to_csv(path, index=False)
    print_summary(f"refusal_rates -> {path}", out)


def emit_conformity_effect(cells: pd.DataFrame, out_dir: str) -> None:
    """
    CE = BER(pressure) - BER(control) per (variant, T, pressure_condition).

    BCa bootstrap is overkill for a simple rate difference; we report the
    Wilson CIs for the component rates plus a naive sum-of-variances CI for
    the CE itself. A proper paired bootstrap lives in a separate stats script.
    """
    work = cells.copy()
    control = work[work["condition_name"] == "control"].copy()
    control = control.rename(columns={
        "state_B_n": "control_B_n",
        "ber": "ber_control",
    })[["variant", "temperature", "control_B_n", "ber_control"]]

    pressure = work[work["condition_name"] != "control"].copy()
    merged = pressure.merge(control, on=["variant", "temperature"], how="left")
    merged["n_denominator_control"] = merged["n_denominator"]
    merged["conformity_effect"] = merged["ber"] - merged["ber_control"]

    # Wilson CIs for the pressure BER and the control BER
    merged = add_wilson_ci_columns(merged, k_col="state_B_n", n_col="n_denominator", out_prefix="ber_pressure")
    merged = add_wilson_ci_columns(
        merged, k_col="control_B_n", n_col="n_denominator_control", out_prefix="ber_ctrl"
    )
    # Naive CE CI from the two independent-rate Wilson halves (approximate).
    merged["ce_lo"] = merged["ber_pressure_lo"] - merged["ber_ctrl_hi"]
    merged["ce_hi"] = merged["ber_pressure_hi"] - merged["ber_ctrl_lo"]

    keep = [
        "variant", "temperature", "condition_name",
        "n_denominator",
        "ber", "ber_pressure_lo", "ber_pressure_hi",
        "ber_control", "ber_ctrl_lo", "ber_ctrl_hi",
        "conformity_effect", "ce_lo", "ce_hi",
    ]
    out = merged[keep].rename(columns={
        "ber": "ber_pressure",
    })
    path = os.path.join(out_dir, "tables", "behavioral", "conformity_effect.csv")
    out.to_csv(path, index=False)
    print_summary(f"conformity_effect -> {path}", out)


def emit_domain_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    work = df.copy()
    work["state"] = work["state"].astype(str)
    counts = (
        work.groupby(["variant", "temperature", "condition_name", "domain", "state"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    pivot = counts.pivot_table(
        index=["variant", "temperature", "condition_name", "domain"],
        columns="state",
        values="n",
        fill_value=0,
    ).reset_index()
    for c in ("A_correct", "B_wrong_endorsed", "C_refusal", "D_unclassified"):
        if c not in pivot.columns:
            pivot[c] = 0
        pivot = pivot.rename(columns={c: f"state_{c}_n"})
    pivot["n_observed"] = pivot[[
        "state_A_correct_n", "state_B_wrong_endorsed_n",
        "state_C_refusal_n", "state_D_unclassified_n",
    ]].sum(axis=1)
    pivot["ber"] = pivot["state_B_wrong_endorsed_n"] / pivot["n_observed"].replace(0, np.nan)
    path = os.path.join(out_dir, "tables", "behavioral", "domain_breakdown.csv")
    pivot.to_csv(path, index=False)
    print_summary(f"domain_breakdown -> {path}", pivot)


def main() -> int:
    args = build_argparser(
        "Emit core behavioral tables (state decomposition, BER, CE) into April_analysis/"
    ).parse_args()
    df = load_trials_from_args(args)

    out_dir = args.out_dir
    ensure_dir(os.path.join(out_dir, "metadata"))
    ensure_dir(os.path.join(out_dir, "tables", "behavioral"))

    emit_trial_counts(df, out_dir)
    emit_judge_coverage(df, out_dir)

    cells = april_cell_metrics(df)

    emit_state_decomposition(cells, out_dir)
    emit_error_rates(cells, out_dir)
    emit_endorsement_rates(cells, out_dir)
    emit_refusal_rates(cells, out_dir)
    emit_conformity_effect(cells, out_dir)
    emit_domain_breakdown(df, out_dir)

    print(f"[behavioral_tables] Wrote tables to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
