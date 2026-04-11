#!/usr/bin/env python3
"""
Phase C4 of April_analysis: temperature concentration analysis.

Pattern completion predicts argmax-continuation behavior: at T=0 the model
deterministically picks the token sequence most likely to continue the
context (including the target anchor). Higher sampling temperatures mix in
lower-probability continuations, so pure pattern-completion events should
decay with T. Social-deference predictions are the opposite (deliberation
should be robust to sampling noise).

Output:
    tables/behavioral/ber_by_temperature.csv   (variant, condition, T) -> BER
                                               + Wilson CIs per cell
    tables/behavioral/temperature_slopes.csv   Per (variant, condition) linear
                                               slope of BER with respect to T,
                                               plus Spearman rho between BER and T.

The slopes table is the main artifact: a negative slope means "cooler
argmax concentrates wrong-answer endorsement", which is the pattern-
completion prediction.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _common import (
    VARIANT_ORDER,
    add_wilson_ci_columns,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
    print_summary,
)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    xs = pd.Series(x).rank(method="average").to_numpy()
    ys = pd.Series(y).rank(method="average").to_numpy()
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> int:
    args = build_argparser(
        "Phase C4: temperature concentration (does BER peak at T=0?)"
    ).parse_args()
    df = load_trials_from_args(args)
    cells = april_cell_metrics(df)

    out_dir = args.out_dir
    ensure_dir(os.path.join(out_dir, "tables", "behavioral"))

    # BER by temperature table with Wilson CIs.
    tbl = cells[[
        "variant", "temperature", "condition_name",
        "n_denominator", "state_B_n", "ber"
    ]].copy()
    tbl = add_wilson_ci_columns(tbl, k_col="state_B_n", n_col="n_denominator", out_prefix="ber")
    tbl_path = os.path.join(out_dir, "tables", "behavioral", "ber_by_temperature.csv")
    tbl.to_csv(tbl_path, index=False)
    print_summary(f"ber_by_temperature -> {tbl_path}", tbl)

    # Per (variant, condition) linear fit of BER vs T, plus Spearman rho.
    # Handles asymmetric temperature coverage: Instruct family has 6 T
    # points, Think SFT/DPO have 2, Think-RL has only 1. Slopes and
    # Spearman require >= 3 points; we emit NaN with a reason code for
    # the cells that fall short.
    rows = []
    for (variant, cond), group in cells.groupby(["variant", "condition_name"]):
        g = group.sort_values("temperature")
        ts = g["temperature"].to_numpy(dtype=float)
        bers = g["ber"].to_numpy(dtype=float)
        n_temps = int(len(g))
        if n_temps < 3:
            slope = intercept = float("nan")
            rho = float("nan")
            reason = f"only_{n_temps}_temps"
        else:
            try:
                slope, intercept = np.polyfit(ts, bers, deg=1)
            except Exception:
                slope, intercept = float("nan"), float("nan")
            rho = _spearman(ts, bers)
            reason = ""
        delta_t06_minus_t0 = float("nan")
        if (g["temperature"] == 0.0).any() and (g["temperature"] == 0.6).any():
            ber_t0 = float(g[g["temperature"] == 0.0]["ber"].mean())
            ber_t06 = float(g[g["temperature"] == 0.6]["ber"].mean())
            delta_t06_minus_t0 = ber_t06 - ber_t0
        rows.append({
            "variant": variant,
            "condition_name": cond,
            "n_temperatures": n_temps,
            "ber_at_t0": float(g[g["temperature"] == 0.0]["ber"].mean()) if (g["temperature"] == 0.0).any() else float("nan"),
            "ber_at_t06": float(g[g["temperature"] == 0.6]["ber"].mean()) if (g["temperature"] == 0.6).any() else float("nan"),
            "ber_at_t1": float(g[g["temperature"] == 1.0]["ber"].mean()) if (g["temperature"] == 1.0).any() else float("nan"),
            "delta_t06_minus_t0": delta_t06_minus_t0,
            "slope_ber_per_T": float(slope),
            "intercept": float(intercept),
            "spearman_rho_T_vs_ber": rho,
            "reason": reason,
        })
    slopes = pd.DataFrame(rows)
    # Canonical ordering
    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
    slopes["__v"] = slopes["variant"].map(variant_rank).fillna(99)
    slopes = slopes.sort_values(["__v", "condition_name"]).drop(columns="__v")

    slopes_path = os.path.join(out_dir, "tables", "behavioral", "temperature_slopes.csv")
    slopes.to_csv(slopes_path, index=False)
    print_summary(f"temperature_slopes -> {slopes_path}", slopes)

    # Summary: mean slope per variant across 5 confident-rep conditions
    print("\n=== Mean BER slope per variant across the 5-rep unanimous conditions ===")
    five_rep_conds = [
        "asch_history_5",
        "asch_zhu_unbiased_unanimous_plain",
        "asch_zhu_unbiased_unanimous_confident",
        "asch_zhu_unbiased_unanimous_uncertain",
        "asch_zhu_unbiased_unanimous_neutral",
    ]
    five_rep = slopes[slopes["condition_name"].isin(five_rep_conds)]
    summary = five_rep.groupby("variant").agg(
        mean_slope=("slope_ber_per_T", "mean"),
        mean_ber_t0=("ber_at_t0", "mean"),
        mean_ber_t1=("ber_at_t1", "mean"),
    ).reindex(VARIANT_ORDER)
    print(summary.round(3).to_string())
    print("\n(Pattern completion predicts negative slopes: BER(T=0) > BER(T=1).)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
