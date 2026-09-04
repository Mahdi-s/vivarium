#!/usr/bin/env python3
"""
analysis_belief_probe.py — summarise belief_probe outputs (continuous belief + greedy readout)
across the structural factorial, per checkpoint.

Reads investigation/backstudy/data/belief_probe/*.jsonl (partial files are fine) and writes
investigation/backstudy/results/belief_probe_summary.csv, belief_probe_contrasts.csv, and belief_probe.md.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import argparse, os
REPO = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=os.environ.get("AAM_BELIEF_DIR") or str(REPO / "investigation/backstudy/data/belief_probe"))
_ap.add_argument("--out-dir", default=None, help="default: <data-dir>/bundle when --data-dir given, else repo results dir")
_args, _ = _ap.parse_known_args()
DATA = Path(_args.data_dir)
RES = Path(_args.out_dir) if _args.out_dir else (REPO / "investigation/backstudy/results" if str(DATA).startswith(str(REPO)) else DATA / "bundle")
RES.mkdir(parents=True, exist_ok=True)

ORDER = [
    "control", "pr_k0_plain", "pr_k5_filler", "pr_k5_correct",
    "pr_k1_plain", "pr_k2_plain", "pr_k3_plain", "pr_k5_plain", "pr_k8_plain",
    "pr_k5_confident", "pr_k5_uncertain", "pr_k5_diverse", "pr_k5_da", "pr_qd",
    "pr_k5_plain_warnsys", "pr_k5_plain_nosys",
    "pu_k1_history", "pu_k5_history", "pu_k5_history_ctrlsys",
    "auth_trust", "auth_bias", "auth_bias_ctrlsys",
    "ngram_orig", "ngram_matched", "ngram_matched_ctrlsys",
    "control_nolicense", "pr_k0_plain_nolicense", "pr_k5_plain_nolicense", "pr_k5_confident_nolicense", "auth_trust_nolicense",
    "user_reports_k1", "user_reports_k5",
]

CONTRASTS = [
    ("frame only (k=0) vs control", "pr_k0_plain", "control"),
    ("filler lines vs frame only", "pr_k5_filler", "pr_k0_plain"),
    ("correct peers vs frame only", "pr_k5_correct", "pr_k0_plain"),
    ("1 repeat vs frame only", "pr_k1_plain", "pr_k0_plain"),
    ("5 repeats vs 1 repeat", "pr_k5_plain", "pr_k1_plain"),
    ("8 repeats vs 5 repeats", "pr_k8_plain", "pr_k5_plain"),
    ("QD (stated once) vs 5 repeats", "pr_qd", "pr_k5_plain"),
    ("diverse (no consensus) vs 5 repeats", "pr_k5_diverse", "pr_k5_plain"),
    ("DA (4+1) vs 5 repeats", "pr_k5_da", "pr_k5_plain"),
    ("confident (varied) vs plain (identical)", "pr_k5_confident", "pr_k5_plain"),
    ("uncertain vs plain", "pr_k5_uncertain", "pr_k5_plain"),
    ("warning sys vs control sys (participant frame)", "pr_k5_plain_warnsys", "pr_k5_plain"),
    ("no sys vs control sys (participant frame)", "pr_k5_plain_nosys", "pr_k5_plain"),
    ("prior-users frame vs participant frame (both control sys, 5 identical)", "pu_k5_history_ctrlsys", "pr_k5_plain"),
    ("warning sys vs control sys (prior-users frame)", "pu_k5_history", "pu_k5_history_ctrlsys"),
    ("5 vs 1 repeats (prior-users frame, warning sys)", "pu_k5_history", "pu_k1_history"),
    ("authority trust (1 claim) vs 5 repeats participant", "auth_trust", "pr_k5_plain"),
    ("authority bias: warning vs control sys", "auth_bias", "auth_bias_ctrlsys"),
    ("ngram matched vs participant frame (both no sys)", "ngram_matched", "pr_k5_plain_nosys"),
    ("ngram orig vs ngram matched (instruction wording)", "ngram_orig", "ngram_matched"),
    ("ngram matched+ctrl sys vs participant frame+ctrl sys", "ngram_matched_ctrlsys", "pr_k5_plain"),
    ("LICENSE: control without 'say unsure' vs control", "control_nolicense", "control"),
    ("LICENSE: 5 repeats without clause vs with clause", "pr_k5_plain_nolicense", "pr_k5_plain"),
    ("LICENSE: frame-only without clause vs with clause", "pr_k0_plain_nolicense", "pr_k0_plain"),
    ("LICENSE: user claim without clause vs with clause", "auth_trust_nolicense", "auth_trust"),
    ("SOURCE: user reports 5 peers vs 5 peer lines", "user_reports_k5", "pr_k5_plain"),
    ("SOURCE: user reports 1 peer vs user's own claim", "user_reports_k1", "auth_trust"),
    ("SOURCE: user reports 5 vs 1", "user_reports_k5", "user_reports_k1"),
]


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """to_markdown without the optional 'tabulate' dependency."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(DATA / "*.jsonl"))):
        for l in open(f):
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no data yet")
    # Think models score after an empty think block; use that as the primary 'prefixed' margin when raw prefixed is absent
    for m in ("margin_first", "margin_mean", "lp_first_gt", "lp_first_wrong"):
        if f"{m}_empty_think_prefixed" in df.columns:
            if f"{m}_prefixed" not in df.columns:
                df[f"{m}_prefixed"] = np.nan
            df[f"{m}_prefixed"] = df[f"{m}_prefixed"].fillna(df[f"{m}_empty_think_prefixed"])
    for c in ("greedy_has_gt", "greedy_has_wrong", "greedy_refusal", "greedy_has_alt"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    df["belief_flip"] = df["margin_first_prefixed"] < 0          # forced-answer belief favours wrong
    df["belief_flip_mean"] = df["margin_mean_prefixed"] < 0
    df["greedy_wrong_only"] = df["greedy_has_wrong"] & ~df["greedy_has_gt"]
    df["greedy_gt_only"] = df["greedy_has_gt"] & ~df["greedy_has_wrong"]
    return df


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> None:
    df = load()
    metrics = ["margin_first_prefixed", "margin_mean_prefixed", "belief_flip", "belief_flip_mean",
               "greedy_has_gt", "greedy_has_wrong", "greedy_wrong_only", "greedy_refusal", "lp_first_wrong_prefixed", "lp_first_gt_prefixed"]
    for extra in ("margin_first_observed_think_prefixed", "think_closed_naturally", "think_tokens"):
        if extra in df.columns:
            metrics.append(extra)
    summ = df.groupby(["variant", "condition"])[metrics].mean().reset_index()
    summ["n"] = df.groupby(["variant", "condition"]).size().values
    summ["cond_order"] = summ.condition.map({c: i for i, c in enumerate(ORDER)})
    summ = summ.sort_values(["variant", "cond_order"]).drop(columns="cond_order")
    summ.to_csv(RES / "belief_probe_summary.csv", index=False)

    # paired contrasts per variant (items common to both conditions)
    out = []
    for v, g in df.groupby("variant"):
        piv = {m: g.pivot_table(index="item_id", columns="condition", values=m) for m in
               ["margin_first_prefixed", "belief_flip", "greedy_has_wrong", "greedy_has_gt", "greedy_refusal"]}
        for label, a, b in CONTRASTS:
            if a not in piv["margin_first_prefixed"].columns or b not in piv["margin_first_prefixed"].columns:
                continue
            rec = {"variant": v, "contrast": label, "a": a, "b": b}
            for m, P in piv.items():
                d = (P[a] - P[b]).dropna()
                mu, lo, hi = boot_ci(d.values.astype(float))
                rec[f"d_{m}"] = mu
                rec[f"d_{m}_lo"] = lo
                rec[f"d_{m}_hi"] = hi
                rec["n_items"] = len(d)
            out.append(rec)
    con = pd.DataFrame(out)
    con.to_csv(RES / "belief_probe_contrasts.csv", index=False)

    # suppression proxy: greedy says wrong-only, but forced-answer belief still favours gt
    sup = df[df.greedy_wrong_only].groupby(["variant", "condition"]).agg(
        n_greedy_wrong=("item_id", "size"),
        frac_belief_still_gt=("belief_flip", lambda s: float((~s).mean())),
    ).reset_index()
    sup.to_csv(RES / "belief_probe_suppression_proxy.csv", index=False)

    # dose-response
    dose = summ[summ.condition.isin(["pr_k0_plain", "pr_k1_plain", "pr_k2_plain", "pr_k3_plain", "pr_k5_plain", "pr_k8_plain"])].copy()
    dose["k"] = dose.condition.str.extract(r"k(\d+)").astype(int)

    with open(RES / "belief_probe.md", "w") as fh:
        fh.write("# Belief-probe (structural factorial)\n\n")
        fh.write(f"Rows: {len(df)}; variants: {sorted(df.variant.unique())}; items/variant: {df.groupby('variant').item_id.nunique().to_dict()}\n\n")
        fh.write("Metrics: `margin_first_prefixed` = logP(first token of GT) − logP(first token of wrong) after the assistant prefix 'The answer is ' "
                 "(think models: after an empty think block; `margin_first_observed_think_prefixed` = after the model's own budget-forced reasoning); "
                 "`belief_flip` = that margin < 0; `greedy_has_wrong/gt` = greedy continuation contains the wrong/GT string; `greedy_refusal` = hedging/refusal regex.\n\n")
        for v, g in summ.groupby("variant"):
            fh.write(f"## {v}\n\n")
            fh.write(_md(g.drop(columns=["variant"]).round(3), index=False) + "\n\n")
            dv = dose[dose.variant == v]
            if len(dv):
                fh.write("Dose–response (identical plain repeats, participant frame):\n\n" + _md(dv[["k", "margin_first_prefixed", "belief_flip", "greedy_has_wrong", "greedy_refusal", "n"]].round(3), index=False) + "\n\n")
            cv = con[con.variant == v]
            if len(cv):
                fh.write("Paired contrasts (a − b), bootstrap 95% CI over items:\n\n")
                fh.write(_md(cv[["contrast", "n_items", "d_margin_first_prefixed", "d_margin_first_prefixed_lo", "d_margin_first_prefixed_hi",
                                 "d_belief_flip", "d_greedy_has_wrong", "d_greedy_has_gt", "d_greedy_refusal"]].round(3), index=False) + "\n\n")
            sv = sup[sup.variant == v]
            if len(sv):
                fh.write("Suppression proxy — among greedy-wrong outputs, fraction whose forced-answer belief still favours GT:\n\n" + _md(sv.drop(columns="variant").round(3), index=False) + "\n\n")
    print(f"wrote {RES}/belief_probe.md ({len(df)} rows)")


if __name__ == "__main__":
    main()
