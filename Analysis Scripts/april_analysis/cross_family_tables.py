#!/usr/bin/env python3
"""
Emit cross-family + scale-bridge tables for the April_analysis expansion.

Reads cross_family_metadata.json (12 models × 2 T × 4 conditions = 96 cells
from runs/, plus 2 ablation DBs that stay scoped to ablation_probes.py) via
the canonical load_april_trials(experiment_group="cross_family") loader, and
writes the following under --out-dir / Comparing_Experiments/April_analysis:

    tables/cross_family/per_model_condition_metrics.csv
        Authoritative cell table: (model_id, short_name, architecture,
        variant, temperature, condition_name) with state counts, BER,
        refusal rate, Wilson CIs, and paired conformity_effect vs control
        at the same (model, T).

    tables/cross_family/conformity_ranking.csv
        19-row headline table ranked by BER on
        asch_zhu_unbiased_unanimous_confident at T=0:
            12 cross-family models
          + OLMo-7B-Base
          + OLMo-7B-Instruct-SFT, -DPO, -RL
          + OLMo-7B-Think-SFT, -DPO, -RL
          + (OLMo-32B-Instruct is already one of the 12)
        The 7B rows are read from the existing 7B
        tables/behavioral/endorsement_rates.csv — we never re-query the 7B
        databases from this script.

    tables/cross_family/pressure_effects_t0.csv
    tables/cross_family/pressure_effects_t06.csv
        Per-model peer (unanimous_confident) and authority
        (authority_zhu_unbiased_trust) conformity deltas relative to control
        at T=0 and T=0.6, with paired CIs.

    tables/cross_family/peer_vs_authority_delta.csv
        Per-model peer_delta − authority_delta with a McNemar test on the
        400 shared items comparing "peer → wrong" vs "authority → wrong".

    tables/cross_family/scale_bridge.csv
        Scale-vs-recipe table: OLMo-7B stages (base / Instruct-SFT /
        Instruct-DPO / Instruct-RL / Think-SFT / Think-DPO / Think-RL) on
        the 7B side + OLMo-32B-Instruct / OLMo-32B-Think-SFT /
        OLMo-32B-Think-DPO / OLMo-32B-Think on the 32B side,
        on the 4 shared conditions at T=0. 7B rows read from the existing
        behavioral tables; 32B rows read from the cross-family DataFrame.

    tables/cross_family/knowledge_protection_corr.csv
        Per cross-family model at T=0: item-level phi correlation between
        "got control correct" and "endorsed wrong answer under
        unanimous_confident" on the same 400 items. Mirrors the 7B
        item_level_correlations methodology.

    tables/cross_family/ber_ranking_with_wilson_ties.csv
        Pairwise Wilson-CI overlap map for the 12 cross-family BERs on
        unanimous_confident T=0. Produces a ranking with tie groups
        (statistically indistinguishable at α=0.05).

    tables/cross_family/per_dataset_ber.csv
        Cross-family BER broken down by dataset_name at T=0
        unanimous_confident only (the headline pressure condition).

    statistical_tests/cross_family/mcnemar_pressure_vs_control_t0.csv
        Per (model, pressure_condition) paired McNemar test with Yates
        correction, OR, and b/c decomposition — adapted from
        Analysis Scripts/cross_family_behavioral_analysis.py.

Important: this script NEVER re-queries the 7B DBs. It reads 7B numbers
straight from the existing behavioral tables so the 7B pipeline stays
byte-identical. See run_all.sh for ordering.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    CROSS_FAMILY_CONDITIONS,
    CROSS_FAMILY_MODEL_ORDER,
    add_wilson_ci_columns,
    build_cross_family_argparser,
    ensure_dir,
    load_cross_family_trials_from_args,
    print_summary,
    wilson_ci,
)

# Repo-root-relative path to the existing 7B endorsement_rates.csv.
_SEVEN_B_ENDORSEMENT_REL = "Comparing_Experiments/April_analysis/tables/behavioral/endorsement_rates.csv"


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------

def compute_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (experiment_group, model_id, short_name, architecture, variant,
    temperature, condition_name) and compute state counts + BER + refusal
    rate + Wilson CIs.

    N denominator = actual observed count per cell (≥395 per CF1).
    """
    group_cols = [
        "experiment_group",
        "model_id",
        "short_name",
        "architecture",
        "variant",
        "temperature",
        "condition_name",
    ]

    def _agg(group: pd.DataFrame) -> pd.Series:
        counts = group["state"].value_counts()
        a = int(counts.get("A_correct", 0))
        b = int(counts.get("B_wrong_endorsed", 0))
        c = int(counts.get("C_refusal", 0))
        d = int(counts.get("D_unclassified", 0))
        n = a + b + c + d
        return pd.Series({
            "n_observed": n,
            "n_denominator": n,  # actual observed count (≥395)
            "state_A_n": a,
            "state_B_n": b,
            "state_C_n": c,
            "state_D_n": d,
        })

    try:
        cells = (
            df.groupby(group_cols, dropna=False)
              .apply(_agg, include_groups=False)
              .reset_index()
        )
    except TypeError:
        cells = df.groupby(group_cols, dropna=False).apply(_agg).reset_index()

    # Wilson CIs for BER (state B) and refusal rate (state C).
    cells = add_wilson_ci_columns(cells, "state_B_n", "n_denominator", "ber")
    cells = add_wilson_ci_columns(cells, "state_C_n", "n_denominator", "refusal")
    cells = add_wilson_ci_columns(cells, "state_A_n", "n_denominator", "correct")

    # rename _p cols to match legacy naming
    cells = cells.rename(columns={
        "ber_p": "ber",
        "refusal_p": "refusal_rate",
        "correct_p": "correct_rate",
    })
    return cells


def attach_conformity_effect(cells: pd.DataFrame) -> pd.DataFrame:
    """Add conformity_effect = BER(cond) − BER(control) per (model, T)."""
    work = cells.copy()
    control = work[work["condition_name"] == "control"].copy()
    control = control[["model_id", "temperature", "state_B_n", "ber"]].rename(
        columns={"state_B_n": "control_B_n", "ber": "ber_control"}
    )
    merged = work.merge(control, on=["model_id", "temperature"], how="left")
    merged["conformity_effect"] = merged["ber"] - merged["ber_control"]

    # Crude independent-rate CI for the CE. Used only as a visual aid; the
    # paired McNemar in statistical_tests/ is the publication test.
    merged["ce_lo"] = merged["ber_lo"] - merged.get("ber_control", 0)
    merged["ce_hi"] = merged["ber_hi"] - merged.get("ber_control", 0)
    return merged


# ---------------------------------------------------------------------------
# Per-model pressure effect tables
# ---------------------------------------------------------------------------

_PEER_COND = "asch_zhu_unbiased_unanimous_confident"
_AUTH_COND = "authority_zhu_unbiased_trust"


def build_pressure_effects(
    cells_main: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    """Per-model peer_delta / authority_delta / refusal deltas at T."""
    sub = cells_main[cells_main["temperature"] == temperature].copy()
    if sub.empty:
        return pd.DataFrame()

    wide = sub.pivot_table(
        index=["model_id", "short_name", "architecture"],
        columns="condition_name",
        values=["ber", "refusal_rate"],
        aggfunc="first",
    )
    # Flatten MultiIndex columns
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    ber_ctrl = f"ber_control"
    ber_peer = f"ber_{_PEER_COND}"
    ber_auth = f"ber_{_AUTH_COND}"
    ref_ctrl = f"refusal_rate_control"
    ref_peer = f"refusal_rate_{_PEER_COND}"
    ref_auth = f"refusal_rate_{_AUTH_COND}"

    wide["peer_ber_delta"] = wide[ber_peer] - wide[ber_ctrl]
    wide["authority_ber_delta"] = wide[ber_auth] - wide[ber_ctrl]
    wide["peer_refusal_delta"] = wide[ref_peer] - wide[ref_ctrl]
    wide["authority_refusal_delta"] = wide[ref_auth] - wide[ref_ctrl]
    wide["temperature"] = temperature

    keep = [
        "model_id", "short_name", "architecture", "temperature",
        ber_ctrl, ber_peer, ber_auth,
        ref_ctrl, ref_peer, ref_auth,
        "peer_ber_delta", "authority_ber_delta",
        "peer_refusal_delta", "authority_refusal_delta",
    ]
    return wide[keep].rename(columns={
        ber_ctrl: "ber_control",
        ber_peer: "ber_peer",
        ber_auth: "ber_authority",
        ref_ctrl: "refusal_control",
        ref_peer: "refusal_peer",
        ref_auth: "refusal_authority",
    })


# ---------------------------------------------------------------------------
# McNemar test (adapted from cross_family_behavioral_analysis.py)
# ---------------------------------------------------------------------------

def mcnemar_paired_endorsement(
    ctrl_endorsed: np.ndarray,
    pres_endorsed: np.ndarray,
) -> Dict[str, float]:
    """
    Paired McNemar on binary "endorsed wrong answer?" labels.

    b = control=NOT endorsed, pressure=endorsed (novel endorsement under pressure)
    c = control=endorsed, pressure=NOT endorsed (recovery under pressure — rare)
    """
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


def build_mcnemar_table(df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Per (model, pressure_condition) paired McNemar vs control on item_id."""
    rows: List[Dict] = []
    sub = df[df["temperature"] == temperature]
    for (model_id, short_name, architecture), gdf in sub.groupby(
        ["model_id", "short_name", "architecture"], dropna=False
    ):
        control = gdf[gdf["condition_name"] == "control"].set_index("item_id")
        for cond in CROSS_FAMILY_CONDITIONS:
            if cond == "control":
                continue
            pres = gdf[gdf["condition_name"] == cond].set_index("item_id")
            shared = control.index.intersection(pres.index)
            if len(shared) == 0:
                continue
            ctrl_end = control.loc[shared, "judge_wrong_endorsed"].fillna(False).astype(int).values
            pres_end = pres.loc[shared, "judge_wrong_endorsed"].fillna(False).astype(int).values
            res = mcnemar_paired_endorsement(ctrl_end, pres_end)
            rows.append({
                "model_id": model_id,
                "short_name": short_name,
                "architecture": architecture,
                "temperature": temperature,
                "pressure_condition": cond,
                "n_paired": res["n"],
                "b_novel_endorse": res["b"],
                "c_recovered": res["c"],
                "chi2_yates": res["chi2"],
                "p_value": res["p"],
                "odds_ratio": res["OR"],
            })
    return pd.DataFrame(rows)


def apply_holm(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """Holm-Bonferroni adjustment within the table family."""
    out = df.copy()
    if out.empty:
        out["p_holm"] = []
        return out
    pvals = out[p_col].fillna(1.0).to_numpy()
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        val = min(val, 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    out["p_holm"] = adj
    return out


# ---------------------------------------------------------------------------
# Ranking + tie-group table
# ---------------------------------------------------------------------------

def build_ber_ranking_with_wilson_ties(
    cells_main: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank cross-family models by BER on unanimous_confident T=0, and group
    statistically-indistinguishable models by Wilson-CI overlap at α=0.05.

    Two models are "tied" if their 95% CIs overlap. The tie group assignment
    uses transitive closure of overlap (single-linkage).
    """
    sub = cells_main[
        (cells_main["temperature"] == 0.0)
        & (cells_main["condition_name"] == _PEER_COND)
    ].copy()
    sub = sub.sort_values("ber", ascending=False).reset_index(drop=True)

    # Single-linkage tie groups by CI overlap
    n = len(sub)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        lo_i, hi_i = sub.loc[i, "ber_lo"], sub.loc[i, "ber_hi"]
        for j in range(i + 1, n):
            lo_j, hi_j = sub.loc[j, "ber_lo"], sub.loc[j, "ber_hi"]
            if hi_i >= lo_j and hi_j >= lo_i:
                union(i, j)

    groups = [find(i) for i in range(n)]
    # Relabel groups 1..k in rank order
    label_map: Dict[int, int] = {}
    next_label = 1
    labels = []
    for g in groups:
        if g not in label_map:
            label_map[g] = next_label
            next_label += 1
        labels.append(label_map[g])

    sub["rank"] = np.arange(1, n + 1)
    sub["tie_group"] = labels

    keep = [
        "rank", "tie_group", "short_name", "model_id", "architecture",
        "variant", "temperature", "condition_name",
        "state_B_n", "n_denominator", "ber", "ber_lo", "ber_hi",
    ]
    return sub[keep]


# ---------------------------------------------------------------------------
# Conformity ranking (headline 19-row table: 12 cross-family + 7B rows)
# ---------------------------------------------------------------------------

def build_conformity_ranking(
    cells_main: pd.DataFrame,
    seven_b_csv: str,
) -> pd.DataFrame:
    """
    12 cross-family models + OLMo-7B-Base + 3 OLMo-7B-Instruct stages +
    3 OLMo-7B-Think stages at T=0 on unanimous_confident, ranked by BER.

    The cross-family rows come straight from cells_main. The 7B rows are
    read from the pre-existing 7B endorsement_rates.csv to keep the 7B
    pipeline untouched. Note that OLMo-32B-Instruct is already one of the
    12 cross-family models, so it's not duplicated.
    """
    cf = cells_main[
        (cells_main["temperature"] == 0.0)
        & (cells_main["condition_name"] == _PEER_COND)
    ].copy()
    cf["cohort"] = "cross_family"
    cf_rows = cf[["short_name", "model_id", "architecture", "cohort",
                  "state_B_n", "n_denominator", "ber", "ber_lo", "ber_hi"]]

    seven_b_df = pd.read_csv(seven_b_csv)
    seven_b_sub = seven_b_df[
        (seven_b_df["temperature"] == 0.0)
        & (seven_b_df["condition_name"] == _PEER_COND)
    ].copy()

    # 7B display names (align with findings_summary.md)
    SEVEN_B_LABELS = {
        "base": ("OLMo-7B-Base", "allenai/Olmo-3-1025-7B", "base_7b"),
        "instruct_sft": ("OLMo-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-SFT", "instruct_7b"),
        "instruct_dpo": ("OLMo-7B-Instruct-DPO", "allenai/Olmo-3-7B-Instruct-DPO", "instruct_7b"),
        "instruct": ("OLMo-7B-Instruct-RL", "allenai/Olmo-3-7B-Instruct", "instruct_7b"),
        "think_sft": ("OLMo-7B-Think-SFT", "allenai/Olmo-3-7B-Think-SFT", "think_7b"),
        "think_dpo": ("OLMo-7B-Think-DPO", "allenai/Olmo-3-7B-Think-DPO", "think_7b"),
        "think": ("OLMo-7B-Think-RL", "allenai/Olmo-3-7B-Think", "think_7b"),
    }
    seven_b_rows: List[Dict] = []
    for _, r in seven_b_sub.iterrows():
        label = SEVEN_B_LABELS.get(r["variant"])
        if label is None:
            continue
        short_name, model_id, cohort = label
        seven_b_rows.append({
            "short_name": short_name,
            "model_id": model_id,
            "architecture": cohort.split("_")[0],
            "cohort": cohort,
            "state_B_n": r["n_wrong_endorsed"],
            "n_denominator": r["n_denominator"],
            "ber": r["ber_p"],
            "ber_lo": r["ber_lo"],
            "ber_hi": r["ber_hi"],
        })
    seven_b_frame = pd.DataFrame(seven_b_rows)

    combined = pd.concat([cf_rows, seven_b_frame], ignore_index=True)
    combined = combined.sort_values("ber", ascending=False).reset_index(drop=True)
    combined["rank"] = np.arange(1, len(combined) + 1)
    return combined[["rank", "cohort", "short_name", "model_id", "architecture",
                     "state_B_n", "n_denominator", "ber", "ber_lo", "ber_hi"]]


# ---------------------------------------------------------------------------
# Scale-bridge table
# ---------------------------------------------------------------------------

def build_scale_bridge(
    cells_main: pd.DataFrame,
    seven_b_csv: str,
) -> pd.DataFrame:
    """
    OLMo 7B (all 7 variants) vs 32B (Instruct + Think-SFT + Think-DPO +
    Think) on the 4 shared conditions at T=0, side by side. 7B rows come
    from the existing endorsement_rates.csv; 32B rows come from cells_main.
    """
    seven_b_df = pd.read_csv(seven_b_csv)
    seven_b_sub = seven_b_df[
        (seven_b_df["temperature"] == 0.0)
        & (seven_b_df["condition_name"].isin(CROSS_FAMILY_CONDITIONS))
    ].copy()
    seven_b_sub["scale"] = "7B"
    seven_b_sub["path"] = seven_b_sub["variant"].apply(
        lambda v: "think" if v.startswith("think") else "instruct" if v.startswith("instruct") else "base"
    )
    seven_b_sub["recipe"] = seven_b_sub["variant"]
    seven_b_sub = seven_b_sub.rename(columns={
        "n_wrong_endorsed": "state_B_n",
        "ber_p": "ber",
    })
    seven_b_rows = seven_b_sub[[
        "scale", "path", "recipe", "temperature", "condition_name",
        "state_B_n", "n_denominator", "ber", "ber_lo", "ber_hi",
    ]]

    # 32B: OLMo-32B-Instruct, Think-SFT, Think-DPO, Think
    cf_sub = cells_main[
        (cells_main["temperature"] == 0.0)
        & (cells_main["short_name"].isin([
            "OLMo-32B-Instruct", "OLMo-32B-Think-SFT",
            "OLMo-32B-Think-DPO", "OLMo-32B-Think",
        ]))
        & (cells_main["condition_name"].isin(CROSS_FAMILY_CONDITIONS))
    ].copy()
    cf_sub["scale"] = "32B"
    cf_sub["path"] = cf_sub["short_name"].map({
        "OLMo-32B-Instruct": "instruct",
        "OLMo-32B-Think-SFT": "think",
        "OLMo-32B-Think-DPO": "think",
        "OLMo-32B-Think": "think",
    })
    cf_sub["recipe"] = cf_sub["short_name"]
    cf_rows = cf_sub[[
        "scale", "path", "recipe", "temperature", "condition_name",
        "state_B_n", "n_denominator", "ber", "ber_lo", "ber_hi",
    ]]

    out = pd.concat([seven_b_rows, cf_rows], ignore_index=True)
    out = out.sort_values(["path", "scale", "recipe", "condition_name"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Knowledge-protection correlation
# ---------------------------------------------------------------------------

def build_knowledge_protection_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per cross-family model at T=0: phi correlation (binary-binary Pearson)
    between "control correct?" and "pressure endorsed wrong?" on the same
    400 items under unanimous_confident.
    """
    rows: List[Dict] = []
    sub = df[df["temperature"] == 0.0]
    for (model_id, short_name, architecture), gdf in sub.groupby(
        ["model_id", "short_name", "architecture"], dropna=False
    ):
        ctrl = (gdf[gdf["condition_name"] == "control"]
                .set_index("item_id")["judge_is_correct"]
                .fillna(False).astype(int))
        pres = (gdf[gdf["condition_name"] == _PEER_COND]
                .set_index("item_id")["judge_wrong_endorsed"]
                .fillna(False).astype(int))
        shared = ctrl.index.intersection(pres.index)
        if len(shared) < 20:
            rows.append({
                "model_id": model_id,
                "short_name": short_name,
                "architecture": architecture,
                "n_paired": len(shared),
                "phi": float("nan"),
                "note": "insufficient_shared_items",
            })
            continue
        a = ctrl.loc[shared].to_numpy()
        b = pres.loc[shared].to_numpy()
        if a.std() == 0 or b.std() == 0:
            phi = 0.0
        else:
            phi = float(np.corrcoef(a, b)[0, 1])
        rows.append({
            "model_id": model_id,
            "short_name": short_name,
            "architecture": architecture,
            "n_paired": int(len(shared)),
            "phi": phi,
            "note": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-dataset BER
# ---------------------------------------------------------------------------

def build_per_dataset_ber(df: pd.DataFrame) -> pd.DataFrame:
    """
    BER by (model, dataset_name) on unanimous_confident at T=0 only.

    Used to show that cross-family BER heterogeneity is driven by model
    identity, not dataset category.
    """
    sub = df[
        (df["temperature"] == 0.0)
        & (df["condition_name"] == _PEER_COND)
    ].copy()
    group_cols = ["model_id", "short_name", "architecture", "dataset_name"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        b = int((g["state"] == "B_wrong_endorsed").sum())
        n = len(g)
        p, lo, hi = wilson_ci(b, n)
        return pd.Series({
            "n_items": n,
            "n_wrong_endorsed": b,
            "ber": p,
            "ber_lo": lo,
            "ber_hi": hi,
        })

    try:
        out = (sub.groupby(group_cols, dropna=False)
                 .apply(_agg, include_groups=False).reset_index())
    except TypeError:
        out = sub.groupby(group_cols, dropna=False).apply(_agg).reset_index()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_cross_family_argparser(
        "Emit cross-family + scale-bridge tables into April_analysis/"
    ).parse_args()
    df = load_cross_family_trials_from_args(args)
    print_summary("cross_family_tables:loaded", df)

    out_dir = args.out_dir
    cf_dir = os.path.join(out_dir, "tables", "cross_family")
    stat_dir = os.path.join(out_dir, "statistical_tests", "cross_family")
    ensure_dir(cf_dir)
    ensure_dir(stat_dir)

    # Split cross_family_main vs ablation; ablation stays for ablation_probes.py
    df_main = df[df["experiment_group"] == "cross_family_main"].copy()
    if df_main.empty:
        raise SystemExit("[cross_family_tables] No cross_family_main rows loaded.")

    cells_all = compute_cell_metrics(df)
    cells_main = cells_all[cells_all["experiment_group"] == "cross_family_main"].copy()
    cells_main = attach_conformity_effect(cells_main)

    # 1. per_model_condition_metrics.csv (cross-family only, 96 rows)
    cells_main_out = cells_main.copy()
    cells_main_out.to_csv(
        os.path.join(cf_dir, "per_model_condition_metrics.csv"), index=False
    )
    print_summary("per_model_condition_metrics", cells_main_out)

    seven_b_csv = os.path.join(args.out_dir.rstrip("/"),
                               "tables/behavioral/endorsement_rates.csv")
    if not os.path.exists(seven_b_csv):
        # fallback to repo-relative path
        seven_b_csv = _SEVEN_B_ENDORSEMENT_REL
    if not os.path.exists(seven_b_csv):
        raise SystemExit(
            f"[cross_family_tables] Missing 7B endorsement_rates.csv at "
            f"{seven_b_csv}. Run the 7B pipeline (behavioral_tables.py) first."
        )

    # 2. conformity_ranking.csv (19 rows: 12 cross-family + 7 OLMo-7B)
    ranking = build_conformity_ranking(cells_main_out, seven_b_csv)
    ranking.to_csv(os.path.join(cf_dir, "conformity_ranking.csv"), index=False)
    print_summary("conformity_ranking", ranking)

    # 3-4. pressure_effects_t0.csv and _t06.csv
    for t, fname in [(0.0, "pressure_effects_t0.csv"),
                     (0.6, "pressure_effects_t06.csv")]:
        pe = build_pressure_effects(cells_main_out, t)
        pe.to_csv(os.path.join(cf_dir, fname), index=False)
        print_summary(fname, pe)

    # 5. peer_vs_authority_delta.csv (built from T=0 pressure_effects_t0)
    pe0 = build_pressure_effects(cells_main_out, 0.0)
    pva = pe0[["model_id", "short_name", "architecture", "temperature",
               "ber_control", "ber_peer", "ber_authority",
               "peer_ber_delta", "authority_ber_delta"]].copy()
    pva["delta_peer_minus_auth"] = pva["peer_ber_delta"] - pva["authority_ber_delta"]
    pva.to_csv(os.path.join(cf_dir, "peer_vs_authority_delta.csv"), index=False)
    print_summary("peer_vs_authority_delta", pva)

    # 6. scale_bridge.csv
    scale = build_scale_bridge(cells_main_out, seven_b_csv)
    scale.to_csv(os.path.join(cf_dir, "scale_bridge.csv"), index=False)
    print_summary("scale_bridge", scale)

    # 7. knowledge_protection_corr.csv
    kp = build_knowledge_protection_corr(df_main)
    kp.to_csv(os.path.join(cf_dir, "knowledge_protection_corr.csv"), index=False)
    print_summary("knowledge_protection_corr", kp)

    # 8. ber_ranking_with_wilson_ties.csv
    ties = build_ber_ranking_with_wilson_ties(cells_main_out)
    ties.to_csv(os.path.join(cf_dir, "ber_ranking_with_wilson_ties.csv"), index=False)
    print_summary("ber_ranking_with_wilson_ties", ties)

    # 9. per_dataset_ber.csv
    per_ds = build_per_dataset_ber(df_main)
    per_ds.to_csv(os.path.join(cf_dir, "per_dataset_ber.csv"), index=False)
    print_summary("per_dataset_ber", per_ds)

    # Statistical tests: McNemar at T=0
    mcnemar = build_mcnemar_table(df_main, temperature=0.0)
    mcnemar = apply_holm(mcnemar, p_col="p_value")
    mcnemar.to_csv(
        os.path.join(stat_dir, "mcnemar_pressure_vs_control_t0.csv"), index=False
    )
    print_summary("mcnemar_pressure_vs_control_t0", mcnemar)

    print(f"[cross_family_tables] Wrote cross-family tables to {cf_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
