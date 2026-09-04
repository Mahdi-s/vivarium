#!/usr/bin/env python3
"""analysis_crossfamily.py — 4-state abandonment for cross-family runs + the naked/n-gram/matched ablations."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; RES = REPO / "investigation/backstudy/results"; RES.mkdir(exist_ok=True, parents=True)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p, c-h, c+h)

df = pd.read_parquet(REPO / "investigation/backstudy/data/crossfamily_trials.parquet")
df = df[df.variant != "think_sft"]
# control per (variant, temperature, item): prefer the core run's control
ctrl = df[df.condition_name == "control"].drop_duplicates(["variant", "temperature", "item_id"])[["variant", "temperature", "item_id", "outcome"]].rename(columns={"outcome": "ctrl_outcome"})
pr = df[df.condition_name != "control"].merge(ctrl, on=["variant", "temperature", "item_id"], how="inner")
# for duplicated ablation conditions (llama 70b has two runs of naked/ngram at T=0) keep latest run_dir
pr = pr.sort_values("run_dir").drop_duplicates(["variant", "temperature", "item_id", "condition_name"], keep="last")
cc = pr[pr.ctrl_outcome == "correct"]
rows = []
for (v, t, c), d in cc.groupby(["variant", "temperature", "condition_name"]):
    n = len(d); k = int((d.outcome != "correct").sum()); p, lo, hi = wilson(k, n)
    rows.append(dict(variant=v, temperature=t, condition=c, n_ctrl_correct=n, p_abandon=p, lo=lo, hi=hi,
                     p_to_target=(d.outcome == "target_wrong").mean(), p_to_other=(d.outcome == "other_wrong").mean(), p_to_refusal=(d.outcome == "refusal").mean()))
t = pd.DataFrame(rows); t.to_csv(RES / "crossfamily_transitions.csv", index=False)
# control accuracy per model
acc = df[df.condition_name == "control"].groupby(["variant", "temperature"]).apply(lambda d: pd.Series(dict(n=len(d), ctrl_acc=(d.outcome == "correct").mean(), ctrl_refusal=(d.outcome == "refusal").mean()))).reset_index()
acc.to_csv(RES / "crossfamily_control_accuracy.csv", index=False)
with open(RES / "crossfamily.md", "w") as fh:
    fh.write("# Cross-family 4-state abandonment (control-correct items), Wilson 95% CI\n\n")
    fh.write("## Control accuracy\n\n" + acc.round(3).to_markdown(index=False) + "\n\n")
    for temp in [0.0, 0.6]:
        d = t[t.temperature == temp]
        fh.write(f"## T={temp}: P(abandon | control correct)\n\n")
        fh.write(d.pivot_table(index="variant", columns="condition", values="p_abandon").round(3).to_markdown() + "\n\n")
        fh.write("Destination = target_wrong share of abandonment (p_to_target / p_abandon):\n\n")
        d2 = d.copy(); d2["target_share"] = d2.p_to_target / d2.p_abandon
        fh.write(d2.pivot_table(index="variant", columns="condition", values="target_share").round(2).to_markdown() + "\n\n")
        fh.write("Refusal destination:\n\n" + d.pivot_table(index="variant", columns="condition", values="p_to_refusal").round(3).to_markdown() + "\n\n")
    fh.write("## Ablation detail (naked social vs n-gram vs matched-instruction n-gram)\n\n")
    ab = t[t.condition.isin(["asch_zhu_unanimous_confident", "asch_zhu_naked_unanimous_confident", "ngram_sequence_baseline", "ngram_sequence_matched_baseline"]) & t.variant.isin(["llama31_70b_instruct", "instruct_32b"])]
    fh.write(ab[["variant", "temperature", "condition", "n_ctrl_correct", "p_abandon", "lo", "hi", "p_to_target", "p_to_other", "p_to_refusal"]].round(3).to_markdown(index=False) + "\n")
print(open(RES / "crossfamily.md").read())
