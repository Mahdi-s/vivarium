#!/usr/bin/env python3
"""
Emit the 2026-04-09 ablation-probes tables for the April_analysis expansion.

Two ablation DBs were run for the April_analysis expansion:

  - meta-llama/llama-3.1-70b-instruct with system_style:none
  - allenai/olmo-3.1-32b-instruct   with system_style:none

Both at T=0.0 on two conditions:

  1. asch_zhu_naked_unanimous_confident — 5 unanimous confident peers voting
     for the wrong answer, but the "be truthful" system prompt is STRIPPED.
     Paired against the main cross-family DB's
     asch_zhu_unbiased_unanimous_confident condition (same 400 item_ids,
     same wrong answers, only the system prompt differs).

  2. ngram_sequence_baseline — a pure abstract-pattern probe. The prompt
     consists of an abstract A→B→C pattern with no question or social
     framing at all; the "wrong answer" is the pattern-completion target.
     Any BER > 0 on this condition is evidence that the model is doing
     autoregressive pattern completion on SOMETHING, independent of social
     pressure.

Outputs (under --out-dir / tables/ablation_probes/):

    ngram_baseline_ber.csv
        BER + Wilson CI on ngram_sequence_baseline for each of the 2
        ablation models at T=0. This is the "pure pattern-completion
        without social framing" probe.

    system_prompt_ablation_delta.csv
        Paired 400-item comparison: BER(naked) − BER(unbiased) per
        ablation model, with McNemar test on the shared item_ids.

    pattern_completion_ratio.csv
        BER(ngram_baseline) / BER(unanimous_confident) per ablation
        model. This is the H2 load-bearing number: how much of the
        confident-peer BER survives when the entire social framing is
        stripped.

    combined_ablation_scorecard.csv
        One-row-per-model summary: model | BER(unanimous) |
        BER(naked_unanimous) | BER(ngram_baseline) | pattern_ratio |
        system_prompt_delta | McNemar_p.

Requires that cross_family_tables.py has already run (reads
per_model_condition_metrics.csv for the paired
asch_zhu_unbiased_unanimous_confident baseline).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    ABLATION_CONDITIONS,
    ABLATION_MODELS,
    add_wilson_ci_columns,
    build_cross_family_argparser,
    ensure_dir,
    load_cross_family_trials_from_args,
    print_summary,
    wilson_ci,
)


_NAKED = "asch_zhu_naked_unanimous_confident"
_NGRAM = "ngram_sequence_baseline"
_UNBIASED = "asch_zhu_unbiased_unanimous_confident"


def mcnemar_paired(ctrl_endorsed: np.ndarray, pres_endorsed: np.ndarray) -> Dict[str, float]:
    n = min(len(ctrl_endorsed), len(pres_endorsed))
    if n == 0:
        return {"chi2": float("nan"), "p": float("nan"), "OR": float("nan"),
                "n": 0, "b": 0, "c": 0}
    c_arr = ctrl_endorsed[:n].astype(int)
    p_arr = pres_endorsed[:n].astype(int)
    b = int(np.sum((c_arr == 0) & (p_arr == 1)))
    c_val = int(np.sum((c_arr == 1) & (p_arr == 0)))
    if b + c_val == 0:
        return {"chi2": 0.0, "p": 1.0, "OR": 1.0, "n": n, "b": b, "c": c_val}
    chi2 = (abs(b - c_val) - 1) ** 2 / (b + c_val)
    p_val = float(1 - sp_stats.chi2.cdf(chi2, df=1))
    odds_ratio = (b / c_val) if c_val > 0 else float("inf")
    return {"chi2": float(chi2), "p": p_val, "OR": float(odds_ratio),
            "n": int(n), "b": b, "c": c_val}


def build_ngram_baseline_ber(df_ablation: pd.DataFrame) -> pd.DataFrame:
    """BER + Wilson CI on ngram_sequence_baseline per ablation model."""
    rows: List[Dict] = []
    sub = df_ablation[df_ablation["condition_name"] == _NGRAM]
    for (model_id, short_name, architecture), gdf in sub.groupby(
        ["model_id", "short_name", "architecture"], dropna=False
    ):
        b = int((gdf["state"] == "B_wrong_endorsed").sum())
        a = int((gdf["state"] == "A_correct").sum())
        c = int((gdf["state"] == "C_refusal").sum())
        d = int((gdf["state"] == "D_unclassified").sum())
        n = 400  # asserted by CF1
        p, lo, hi = wilson_ci(b, n)
        rows.append({
            "model_id": model_id,
            "short_name": short_name,
            "architecture": architecture,
            "condition": _NGRAM,
            "n_denominator": n,
            "state_A_n": a,
            "state_B_n": b,
            "state_C_n": c,
            "state_D_n": d,
            "ber": p,
            "ber_lo": lo,
            "ber_hi": hi,
        })
    return pd.DataFrame(rows)


def build_system_prompt_delta(
    df_ablation: pd.DataFrame,
    df_main: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paired McNemar on "wrong-answer endorsed?" between
    asch_zhu_unbiased_unanimous_confident (with system prompt, from
    cross_family_main) and asch_zhu_naked_unanimous_confident (system prompt
    stripped, from ablation DB). Shared 400 item_ids per model.
    """
    rows: List[Dict] = []
    for model_id in ABLATION_MODELS:
        # With system prompt (from cross_family_main at T=0)
        with_sys = df_main[
            (df_main["model_id"] == model_id)
            & (df_main["temperature"] == 0.0)
            & (df_main["condition_name"] == _UNBIASED)
        ].set_index("item_id")["judge_wrong_endorsed"].fillna(False).astype(int)

        # Without system prompt (from ablation DB)
        without_sys = df_ablation[
            (df_ablation["model_id"] == model_id)
            & (df_ablation["condition_name"] == _NAKED)
        ].set_index("item_id")["judge_wrong_endorsed"].fillna(False).astype(int)

        shared = with_sys.index.intersection(without_sys.index)
        if len(shared) == 0:
            continue

        a = with_sys.loc[shared].to_numpy()
        b = without_sys.loc[shared].to_numpy()
        ber_with = float(a.mean())
        ber_without = float(b.mean())
        _, lo_with, hi_with = wilson_ci(int(a.sum()), len(shared))
        _, lo_without, hi_without = wilson_ci(int(b.sum()), len(shared))
        res = mcnemar_paired(a, b)

        # Short_name + architecture from the main DF (must exist by CF5)
        meta = df_main[df_main["model_id"] == model_id].iloc[0]
        rows.append({
            "model_id": model_id,
            "short_name": meta["short_name"],
            "architecture": meta["architecture"],
            "n_paired": int(len(shared)),
            "ber_with_system_prompt": ber_with,
            "ber_with_lo": lo_with,
            "ber_with_hi": hi_with,
            "ber_without_system_prompt": ber_without,
            "ber_without_lo": lo_without,
            "ber_without_hi": hi_without,
            "delta_without_minus_with": ber_without - ber_with,
            "mcnemar_b_novel_without_sys": res["b"],
            "mcnemar_c_recovered_with_sys": res["c"],
            "mcnemar_chi2_yates": res["chi2"],
            "mcnemar_p_value": res["p"],
            "mcnemar_odds_ratio": res["OR"],
        })
    return pd.DataFrame(rows)


def build_pattern_completion_ratio(
    df_ablation: pd.DataFrame,
    df_main: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per ablation model: BER(ngram_baseline) / BER(unanimous_confident).

    Both numerator and denominator are on the same 400 item_ids at T=0
    (verified via INTERSECT query during E0 audit). The ratio is the H2
    load-bearing number: what fraction of the confident-peer pattern
    completion survives when the entire social framing is stripped.
    """
    rows: List[Dict] = []
    for model_id in ABLATION_MODELS:
        ngram = df_ablation[
            (df_ablation["model_id"] == model_id)
            & (df_ablation["condition_name"] == _NGRAM)
        ]["judge_wrong_endorsed"].fillna(False).astype(int)
        uc = df_main[
            (df_main["model_id"] == model_id)
            & (df_main["temperature"] == 0.0)
            & (df_main["condition_name"] == _UNBIASED)
        ]["judge_wrong_endorsed"].fillna(False).astype(int)

        if len(ngram) == 0 or len(uc) == 0:
            continue

        ber_ngram = float(ngram.mean())
        ber_uc = float(uc.mean())
        ratio = ber_ngram / ber_uc if ber_uc > 0 else float("nan")

        _, lo_ngram, hi_ngram = wilson_ci(int(ngram.sum()), len(ngram))
        _, lo_uc, hi_uc = wilson_ci(int(uc.sum()), len(uc))

        meta = df_main[df_main["model_id"] == model_id].iloc[0]
        rows.append({
            "model_id": model_id,
            "short_name": meta["short_name"],
            "architecture": meta["architecture"],
            "ber_ngram_baseline": ber_ngram,
            "ber_ngram_lo": lo_ngram,
            "ber_ngram_hi": hi_ngram,
            "ber_unanimous_confident": ber_uc,
            "ber_uc_lo": lo_uc,
            "ber_uc_hi": hi_uc,
            "pattern_completion_ratio": ratio,
        })
    return pd.DataFrame(rows)


def build_combined_scorecard(
    ngram: pd.DataFrame,
    delta: pd.DataFrame,
    ratio: pd.DataFrame,
) -> pd.DataFrame:
    """One row per ablation model with all key ablation numbers side by side."""
    out = delta[[
        "model_id", "short_name", "architecture",
        "ber_with_system_prompt", "ber_without_system_prompt",
        "delta_without_minus_with", "mcnemar_p_value",
    ]].merge(
        ratio[["model_id", "ber_ngram_baseline", "pattern_completion_ratio"]],
        on="model_id", how="left",
    )
    return out


def main() -> int:
    args = build_cross_family_argparser(
        "Emit ablation-probe tables (ngram baseline + system-prompt removal) "
        "into April_analysis/tables/ablation_probes/"
    ).parse_args()
    df = load_cross_family_trials_from_args(args)
    print_summary("ablation_probes:loaded", df)

    df_main = df[df["experiment_group"] == "cross_family_main"].copy()
    df_ablation = df[df["experiment_group"] == "ablation_system_prompt_removed"].copy()
    if df_ablation.empty:
        raise SystemExit("[ablation_probes] No ablation rows found.")

    # Sanity re-check: shared item_ids between each ablation model's 400 items
    # and the matching cross_family_main unanimous_confident 400 items.
    for model_id in ABLATION_MODELS:
        a_items = set(df_ablation[
            (df_ablation["model_id"] == model_id)
            & (df_ablation["condition_name"] == _NAKED)
        ]["item_id"].unique())
        m_items = set(df_main[
            (df_main["model_id"] == model_id)
            & (df_main["temperature"] == 0.0)
            & (df_main["condition_name"] == _UNBIASED)
        ]["item_id"].unique())
        shared = a_items & m_items
        assert len(shared) == 400, (
            f"[ablation_probes] {model_id}: shared_items={len(shared)} != 400. "
            f"Ablation and main DB item_ids do not match."
        )

    out_dir = args.out_dir
    abl_dir = os.path.join(out_dir, "tables", "ablation_probes")
    ensure_dir(abl_dir)

    ngram = build_ngram_baseline_ber(df_ablation)
    ngram.to_csv(os.path.join(abl_dir, "ngram_baseline_ber.csv"), index=False)
    print_summary("ngram_baseline_ber", ngram)

    delta = build_system_prompt_delta(df_ablation, df_main)
    delta.to_csv(os.path.join(abl_dir, "system_prompt_ablation_delta.csv"), index=False)
    print_summary("system_prompt_ablation_delta", delta)

    ratio = build_pattern_completion_ratio(df_ablation, df_main)
    ratio.to_csv(os.path.join(abl_dir, "pattern_completion_ratio.csv"), index=False)
    print_summary("pattern_completion_ratio", ratio)

    scorecard = build_combined_scorecard(ngram, delta, ratio)
    scorecard.to_csv(os.path.join(abl_dir, "combined_ablation_scorecard.csv"), index=False)
    print_summary("combined_ablation_scorecard", scorecard)

    print(f"[ablation_probes] Wrote ablation tables to {abl_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
