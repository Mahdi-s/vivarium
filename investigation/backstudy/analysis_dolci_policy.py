#!/usr/bin/env python3
"""
analysis_dolci_policy.py — what the Dolci-Instruct-DPO mixture rewards at the *policy* level.

Question: does DPO (chosen vs rejected) prefer hedging/abstention ("I'm not sure"), refusal, push-back against the
user's premise, or definite answering — i.e. the output policies we see change between SFT and DPO under pressure?
Also tests the confound in the paper's Pillar II: 'delta_learning' pairs contrast a strong chosen model with a weak
rejected model, so any Δ metric partly measures weak-vs-strong style, not a targeted preference.

Reads /Volumes/T7/conformity data analysis/dataset_analysis/data/raw/instruct-dpo/data/*.parquet (259,922 pairs).
Writes investigation/backstudy/results/dolci_dpo_policy.csv and dolci_dpo_policy.md
"""
from __future__ import annotations
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats

ROOT = Path("/Volumes/T7/conformity data analysis/dataset_analysis/data/raw/instruct-dpo/data")
RES = Path(__file__).resolve().parents[2] / "investigation/backstudy/results"

PAT = {
    "hedge": re.compile(r"\b(i(?:'m| am) (?:not (?:entirely |completely |fully )?sure|unsure|uncertain)|i don'?t know|i cannot (?:be certain|determine|verify)|i can'?t (?:be certain|determine|verify)|it(?:'s| is) (?:hard|difficult) to (?:say|determine|know)|not (?:entirely |completely )?certain|i (?:may|might) be (?:wrong|mistaken))\b", re.I),
    "refusal": re.compile(r"\b(i can'?t (?:help|assist|provide|comply)|i cannot (?:help|assist|provide|comply)|i(?:'m| am) (?:unable|not able) to|i won'?t be able to|as an ai(?: language model)?,? i)\b", re.I),
    "pushback": re.compile(r"\b(actually,|that(?:'s| is) (?:not|incorrect|a common misconception)|this is (?:not|incorrect)|i(?:'d| would) (?:respectfully )?disagree|i must (?:correct|clarify)|to clarify,|it(?:'s| is) important to note that|however, (?:this|that) is not)\b", re.I),
    "agree_user": re.compile(r"\b(you(?:'re| are) (?:absolutely |completely |totally )?(?:right|correct)|yes, (?:that(?:'s| is) (?:right|correct)|you(?:'re| are) right)|great question|good question|i agree)\b", re.I),
    "definite": re.compile(r"\b(the (?:correct |final )?answer is|therefore,? the answer|the answer:|final answer)\b", re.I),
}

def last_content(turns):
    if not turns:
        return ""
    for t in reversed(turns):
        if isinstance(t, dict) and t.get("role") == "assistant":
            return t.get("content") or ""
    return (turns[-1].get("content") or "") if isinstance(turns[-1], dict) else ""

def cliffs(x_rej, x_cho):
    """Cliff's delta P(rej > cho) - P(rej < cho) via Mann-Whitney U; positive = rejected stochastically greater = DPO penalises."""
    n1, n2 = len(x_rej), len(x_cho)
    u = stats.mannwhitneyu(x_rej, x_cho, alternative="two-sided").statistic
    return 2 * u / (n1 * n2) - 1

rows = []
for f in sorted(glob.glob(str(ROOT / "*.parquet"))):
    t = pq.read_table(f, columns=["chosen", "rejected", "preference_type", "chosen_model", "rejected_model"]).to_pylist()
    for r in t:
        c, j = last_content(r["chosen"]), last_content(r["rejected"])
        rec = {"preference_type": r["preference_type"], "chosen_model": r["chosen_model"], "rejected_model": r["rejected_model"],
               "c_len": len(c), "r_len": len(j)}
        for k, p in PAT.items():
            rec[f"c_{k}"] = int(bool(p.search(c))); rec[f"r_{k}"] = int(bool(p.search(j)))
        rows.append(rec)
    print(f"  {f.split('/')[-1]}: cumulative {len(rows):,}", flush=True)
df = pd.DataFrame(rows)
df["same_model"] = df.chosen_model == df.rejected_model
df.to_parquet(RES / "dolci_dpo_policy_pairs.parquet", index=False)

def summarise(d, label):
    out = []
    for k in list(PAT) + ["len"]:
        c, r = d[f"c_{k}"].values, d[f"r_{k}"].values
        diff = r - c
        b = int((diff > 0).sum()); a = int((diff < 0).sum())
        p = stats.binomtest(b, a + b, 0.5).pvalue if a + b else 1.0
        out.append(dict(subset=label, n=len(d), feature=k, chosen_rate=float(c.mean()), rejected_rate=float(r.mean()),
                        cliffs_delta_rej_minus_cho=cliffs(r, c), pairs_rej_only=b, pairs_cho_only=a,
                        chosen_over_rejected_ratio=(float(c.mean()) / float(r.mean()) if r.mean() else np.nan), sign_test_p=p))
    return out
summ = summarise(df, "all")
for pt, d in df.groupby("preference_type"):
    summ += summarise(d, f"pref={pt}")
summ += summarise(df[df.same_model], "same_model_pairs")
summ += summarise(df[~df.same_model], "different_model_pairs")
S = pd.DataFrame(summ); S.to_csv(RES / "dolci_dpo_policy.csv", index=False)

# model-pair confound: top chosen/rejected models and their hedge rates
mc = df.groupby("chosen_model").agg(n=("c_hedge", "size"), hedge=("c_hedge", "mean"), pushback=("c_pushback", "mean"), definite=("c_definite", "mean")).sort_values("n", ascending=False).head(8)
mr = df.groupby("rejected_model").agg(n=("r_hedge", "size"), hedge=("r_hedge", "mean"), pushback=("r_pushback", "mean"), definite=("r_definite", "mean")).sort_values("n", ascending=False).head(8)

with open(RES / "dolci_dpo_policy.md", "w") as fh:
    fh.write("# Dolci-Instruct-DPO: what the preference data reward at the policy level\n\n")
    fh.write(f"N pairs = {len(df):,}; same-model pairs = {int(df.same_model.sum()):,}; preference types = {df.preference_type.value_counts().to_dict()}\n\n")
    fh.write("Features are regex hits on the final assistant turn (chosen vs rejected). `cliffs_delta_rej_minus_cho` > 0 means the rejected response has the feature more often (DPO penalises it); < 0 means DPO *favours* it. Sign test on within-pair differences.\n\n")
    for sub in ["all", "pref=delta_learning", "pref=llm_judged", "pref=multiturn_self_talk", "pref=multiturn_synthetic_context", "same_model_pairs", "different_model_pairs"]:
        d = S[S.subset == sub]
        if len(d):
            fh.write(f"## {sub} (n={int(d.n.iloc[0]):,})\n\n" + d[["feature", "chosen_rate", "rejected_rate", "chosen_over_rejected_ratio", "cliffs_delta_rej_minus_cho", "pairs_rej_only", "pairs_cho_only", "sign_test_p"]].round(4).to_markdown(index=False) + "\n\n")
    fh.write("## Model-pair confound\n\nChosen models (top 8):\n\n" + mc.round(3).to_markdown() + "\n\nRejected models (top 8):\n\n" + mr.round(3).to_markdown() + "\n")
print(open(RES / "dolci_dpo_policy.md").read())
