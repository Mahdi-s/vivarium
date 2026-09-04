#!/usr/bin/env python3
"""analysis_policy_curve.py — the belief→output mapping per checkpoint.

For each variant: (1) distribution of belief margins under pressure conditions (identical items), (2) P(greedy output
class | margin bin), (3) a logistic 'policy' fit P(refusal or wrong-in-greedy | margin) whose intercept/slope differ by
variant if post-training changed the policy rather than the belief. Also a paired SFT-vs-DPO belief comparison on
identical (item, condition) cells."""
from pathlib import Path
import glob, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
REPO=Path(__file__).resolve().parents[2]; DATA=REPO/"investigation/backstudy/data/belief_probe"; RES=REPO/"investigation/backstudy/results"
rows=[]
for f in sorted(glob.glob(str(DATA/"*.jsonl"))):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
df=pd.DataFrame(rows); df=df[df.variant.isin(["base","instruct_sft","instruct_dpo","instruct"])]
df["m"]=df.margin_first_prefixed; df["press"]=~df.condition.isin(["control","pr_k0_plain","pr_k5_filler","pr_k5_correct"])
df["out"]=np.select([df.greedy_refusal, df.greedy_has_wrong & ~df.greedy_has_gt, df.greedy_has_gt & ~df.greedy_has_wrong],["refusal","wrong","gt"],"other")
out=[]
print("== belief under pressure conditions (all pressure cells pooled), per variant ==")
g=df[df.press].groupby("variant").m.describe()[["count","mean","25%","50%","75%"]]; print(g.round(2).to_string())
print("\n== belief under control / frame-only / filler / correct peers ==")
print(df[~df.press].groupby(["variant","condition"]).m.mean().unstack().round(2).to_string())
print("\n== output class shares under pressure (greedy 16 tokens) ==")
print(pd.crosstab(df[df.press].variant, df[df.press].out, normalize="index").round(3).to_string())
print("\n== P(output class | belief margin bin), pressure cells ==")
df["mbin"]=pd.cut(df.m,[-99,-4,-2,-1,0,1,2,99])
t=pd.crosstab([df[df.press].variant, df[df.press].mbin], df[df.press].out, normalize="index").round(2)
print(t.to_string())
print("\n== logistic policy fits: P(refusal | m) and P(wrong-only | m) per variant, pressure cells ==")
for v,d in df[df.press].groupby("variant"):
    d=d.assign(ref=d.out.eq("refusal").astype(int), wr=d.out.eq("wrong").astype(int))
    for y in ["ref","wr"]:
        try:
            r=smf.logit(f"{y} ~ m",d).fit(disp=0); print(f"{v:14s} {y}: intercept={r.params['Intercept']:+.2f} slope={r.params['m']:+.2f}  (n={len(d)})")
        except Exception as e: print(v,y,"fit failed",e)
# paired belief comparison across checkpoints on identical cells
piv=df.pivot_table(index=["item_id","condition"],columns="variant",values="m")
if {"instruct_sft","instruct_dpo"}<=set(piv.columns):
    j=piv[["instruct_sft","instruct_dpo"]].dropna(); d=j.instruct_dpo-j.instruct_sft
    rng=np.random.default_rng(0); boots=rng.choice(d.values,(5000,len(d))).mean(1)
    print(f"\n== paired DPO − SFT belief margin on identical (item, condition) cells: mean {d.mean():+.2f} [{np.percentile(boots,2.5):+.2f}, {np.percentile(boots,97.5):+.2f}], Spearman {j.corr(method='spearman').iloc[0,1]:.2f}, n={len(d)} ==")
    # by condition family
    fam=j.index.get_level_values("condition").to_series().reset_index(drop=True)
    dd=pd.DataFrame({"d":d.values,"cond":j.index.get_level_values("condition")})
    print(dd.groupby("cond").d.agg(["mean","count"]).round(2).to_string())
    # belief shift from control per checkpoint
    for v in ["instruct_sft","instruct_dpo"]:
        c=df[(df.variant==v)&(df.condition=="control")].set_index("item_id").m
        s=df[(df.variant==v)&df.press].copy(); s["shift"]=s.m-s.item_id.map(c)
        print(f"{v}: mean belief shift vs control over pressure cells = {s['shift'].mean():+.2f} (sd {s['shift'].std():.2f}); P(margin<0 | pressure) = {(s.m<0).mean():.2f}; refusal share {s.out.eq('refusal').mean():.2f}; wrong-only share {s.out.eq('wrong').mean():.2f}")
