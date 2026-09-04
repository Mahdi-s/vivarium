#!/usr/bin/env python3
"""
analysis_core.py — 4-state truth-abandonment analysis of the OLMo-3 7B family (existing 215K trials).

Pairs every pressure trial with the control trial on the same (variant, temperature, item_id), restricted to
the publication item set. Writes results to investigation/backstudy/results/:
  transitions_t0.csv / transitions_pooled.csv   abandonment among control-correct items, decomposed
  contrasts_t0.csv / contrasts_pooled.csv       pre-specified structural contrasts, item-bootstrap CIs, McNemar
  condition_order_stability.csv                 Kendall's W of condition ordering across temperatures
  core.md                                       markdown summary
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "investigation/backstudy/results"
RES.mkdir(parents=True, exist_ok=True)

VARIANTS = ["base", "instruct_sft", "instruct_dpo", "instruct", "think_sft", "think_dpo", "think"]
COND_ORDER = ["asch_zhu_unbiased_qd", "asch_zhu_unbiased_unanimous_plain", "asch_zhu_unbiased_unanimous_neutral",
              "asch_zhu_unbiased_unanimous_confident", "asch_zhu_unbiased_unanimous_uncertain", "asch_zhu_unbiased_diverse_plain",
              "asch_zhu_unbiased_da", "asch_history_5", "authoritative_bias", "authority_zhu_unbiased_trust", "authority_zhu_unbiased_trust_da"]
SHORT = {"asch_zhu_unbiased_qd": "qd(1 stated)", "asch_zhu_unbiased_unanimous_plain": "unan_plain(5 ident)",
         "asch_zhu_unbiased_unanimous_neutral": "unan_neutral(5 var)", "asch_zhu_unbiased_unanimous_confident": "unan_confident(5 var)",
         "asch_zhu_unbiased_unanimous_uncertain": "unan_uncertain(5 var)", "asch_zhu_unbiased_diverse_plain": "diverse(no consensus)",
         "asch_zhu_unbiased_da": "da(4+1)", "asch_history_5": "asch_hist(5 ident,warn,prior-users)",
         "authoritative_bias": "auth_bias(1,warn)", "authority_zhu_unbiased_trust": "auth_trust(1)", "authority_zhu_unbiased_trust_da": "auth_trust_da(1+alt)"}

CONTRASTS = [
    ("REPETITION 1 vs 5 (frame const)", "asch_zhu_unbiased_qd", "asch_zhu_unbiased_unanimous_plain"),
    ("FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)", "asch_history_5", "asch_zhu_unbiased_unanimous_plain"),
    ("CONSENSUS: diverse vs unanimous_plain", "asch_zhu_unbiased_diverse_plain", "asch_zhu_unbiased_unanimous_plain"),
    ("CONSENSUS: DA(4+1) vs unanimous_plain", "asch_zhu_unbiased_da", "asch_zhu_unbiased_unanimous_plain"),
    ("LEXICAL: neutral(varied) vs plain(identical)", "asch_zhu_unbiased_unanimous_neutral", "asch_zhu_unbiased_unanimous_plain"),
    ("LEXICAL: confident(varied) vs plain(identical)", "asch_zhu_unbiased_unanimous_confident", "asch_zhu_unbiased_unanimous_plain"),
    ("TONE: uncertain vs plain", "asch_zhu_unbiased_unanimous_uncertain", "asch_zhu_unbiased_unanimous_plain"),
    ("SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain", "authority_zhu_unbiased_trust", "asch_zhu_unbiased_unanimous_plain"),
    ("SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5", "authoritative_bias", "asch_history_5"),
    ("SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)", "authoritative_bias", "authority_zhu_unbiased_trust"),
    ("ALT OPTION: trust_da vs trust", "authority_zhu_unbiased_trust_da", "authority_zhu_unbiased_trust"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Attach control outcome to each pressure trial on the same (variant, temperature, item)."""
    ctrl = df[df.condition_name == "control"][["variant", "temperature", "item_id", "outcome"]].rename(columns={"outcome": "ctrl_outcome"})
    ctrl = ctrl.drop_duplicates(["variant", "temperature", "item_id"])
    pr = df[df.condition_name != "control"].merge(ctrl, on=["variant", "temperature", "item_id"], how="inner")
    pr["abandon"] = (pr.ctrl_outcome == "correct") & (pr.outcome != "correct")
    return pr


def transitions(pr: pd.DataFrame, label: str) -> pd.DataFrame:
    cc = pr[pr.ctrl_outcome == "correct"]
    g = cc.groupby(["variant", "condition_name"])
    rows = []
    for (v, c), d in g:
        n = len(d)
        k = int((d.outcome != "correct").sum())
        p, lo, hi = wilson(k, n)
        rows.append(dict(variant=v, condition=c, n_ctrl_correct=n, p_abandon=p, lo=lo, hi=hi,
                         p_to_target=(d.outcome == "target_wrong").mean(), p_to_other=(d.outcome == "other_wrong").mean(),
                         p_to_refusal=(d.outcome == "refusal").mean(),
                         p_rescue=np.nan))
    t = pd.DataFrame(rows)
    # rescue: control wrong (target/other, not refusal) -> pressure correct
    cw = pr[pr.ctrl_outcome.isin(["target_wrong", "other_wrong"])]
    resc = cw.groupby(["variant", "condition_name"]).apply(lambda d: (d.outcome == "correct").mean()).rename("p_rescue").reset_index().rename(columns={"condition_name": "condition"})
    t = t.drop(columns="p_rescue").merge(resc, on=["variant", "condition"], how="left")
    t["set"] = label
    return t


def boot_diff(a: np.ndarray, b: np.ndarray, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def contrasts(pr: pd.DataFrame, label: str) -> pd.DataFrame:
    cc = pr[pr.ctrl_outcome == "correct"]
    rows = []
    for v in VARIANTS:
        dv = cc[cc.variant == v]
        for name, a, b in CONTRASTS:
            A = dv[dv.condition_name == a].set_index(["temperature", "item_id"])["abandon"]
            B = dv[dv.condition_name == b].set_index(["temperature", "item_id"])["abandon"]
            j = pd.concat([A.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
            if len(j) < 20:
                continue
            mu, lo, hi = boot_diff(j.a.values, j.b.values)
            b01 = int(((j.a == 1) & (j.b == 0)).sum()); b10 = int(((j.a == 0) & (j.b == 1)).sum())
            p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0
            rows.append(dict(variant=v, contrast=name, a=a, b=b, n_pairs=len(j), p_abandon_a=float(j.a.mean()), p_abandon_b=float(j.b.mean()),
                             diff=mu, lo=lo, hi=hi, mcnemar_b=b01, mcnemar_c=b10, mcnemar_p=p))
    c = pd.DataFrame(rows)
    # Holm within variant
    out = []
    for v, d in c.groupby("variant"):
        d = d.sort_values("mcnemar_p").copy()
        m = len(d)
        d["p_holm"] = np.minimum.accumulate((d.mcnemar_p.values * np.arange(m, 0, -1))[::-1])[::-1]
        d["p_holm"] = np.minimum(d["p_holm"].values, 1.0)
        out.append(d)
    c = pd.concat(out)
    c["set"] = label
    return c


def order_stability(pr: pd.DataFrame) -> pd.DataFrame:
    cc = pr[pr.ctrl_outcome == "correct"]
    rows = []
    for v in VARIANTS:
        piv = cc[cc.variant == v].groupby(["temperature", "condition_name"]).abandon.mean().unstack("condition_name")
        piv = piv[[c for c in COND_ORDER if c in piv.columns]]
        ranks = piv.rank(axis=1)
        m, n = ranks.shape  # m raters (temps), n items (conditions)
        R = ranks.sum(axis=0)
        W = 12 * ((R - R.mean()) ** 2).sum() / (m * m * (n ** 3 - n))
        rows.append(dict(variant=v, kendall_W=W, n_temps=m, n_conditions=n, mean_order=" > ".join(SHORT.get(c, c) for c in piv.mean().sort_values(ascending=False).index)))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(REPO / "investigation/backstudy/data/olmo_trials.parquet")
    df = df[df.in_pub_set & df.variant.isin(VARIANTS)].copy()
    print("trials:", len(df), "| judge null:", int(df.judge_is_correct.isna().sum()))
    print("outcome dist:", df.outcome.value_counts(normalize=True).round(3).to_dict())
    pr = paired(df)
    print("paired pressure trials:", len(pr), "| control-correct pairs:", int((pr.ctrl_outcome == "correct").sum()))

    t0 = transitions(pr[pr.temperature == 0.0], "T=0")
    tp = transitions(pr, "pooled")
    t0.to_csv(RES / "transitions_t0.csv", index=False); tp.to_csv(RES / "transitions_pooled.csv", index=False)
    c0 = contrasts(pr[pr.temperature == 0.0], "T=0")
    cp = contrasts(pr, "pooled")
    c0.to_csv(RES / "contrasts_t0.csv", index=False); cp.to_csv(RES / "contrasts_pooled.csv", index=False)
    osb = order_stability(pr)
    osb.to_csv(RES / "condition_order_stability.csv", index=False)

    with open(RES / "core.md", "w") as fh:
        fh.write("# Core 4-state analysis — OLMo-3 7B family, publication item set\n\n")
        fh.write(f"Trials analysed: {len(df)}; paired pressure trials: {len(pr)}; control-correct pairs: {int((pr.ctrl_outcome=='correct').sum())}.\n\n")
        fh.write("Outcome rule: correct (judge) → refusal (SQL or judge flag) → target_wrong (judge endorsed injected answer) → other_wrong.\n\n")
        for label, t in [("T=0", t0), ("pooled over 6 temperatures", tp)]:
            fh.write(f"## Truth abandonment among control-correct items ({label})\n\n")
            piv = t.pivot_table(index="condition", columns="variant", values="p_abandon").reindex(COND_ORDER)[[v for v in VARIANTS if v in t.variant.unique()]]
            piv.index = [SHORT.get(c, c) for c in piv.index]
            fh.write("P(abandon | control correct):\n\n" + piv.round(3).to_markdown() + "\n\n")
            for v in ["instruct_sft", "instruct_dpo", "instruct", "base"]:
                d = t[t.variant == v].set_index("condition").reindex(COND_ORDER)
                d.index = [SHORT.get(c, c) for c in d.index]
                fh.write(f"Destination decomposition — {v} ({label}):\n\n" + d[["n_ctrl_correct", "p_abandon", "lo", "hi", "p_to_target", "p_to_other", "p_to_refusal", "p_rescue"]].round(3).to_markdown() + "\n\n")
        for label, c in [("T=0", c0), ("pooled", cp)]:
            fh.write(f"## Pre-specified structural contrasts ({label}) — abandonment(a) − abandonment(b), item-bootstrap 95% CI, exact McNemar, Holm within variant\n\n")
            for v in VARIANTS:
                d = c[c.variant == v]
                if len(d):
                    fh.write(f"### {v}\n\n" + d[["contrast", "n_pairs", "p_abandon_a", "p_abandon_b", "diff", "lo", "hi", "mcnemar_b", "mcnemar_c", "p_holm"]].round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Condition-ordering stability across temperatures (Kendall's W over 6 temps × 11 conditions)\n\n" + osb.round(3).to_markdown(index=False) + "\n")
    print(open(RES / "core.md").read()[:6000])


if __name__ == "__main__":
    main()
