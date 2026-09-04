#!/usr/bin/env python3
"""analysis_structure_items.py — (A) structural-feature logistic decomposition; (B) item susceptibility / Rasch; (C) temperature."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
REPO = Path(__file__).resolve().parents[2]; RES = REPO / "investigation/backstudy/results"
V=["base","instruct_sft","instruct_dpo","instruct","think_sft","think_dpo","think"]
df = pd.read_parquet(REPO/"investigation/backstudy/data/olmo_trials.parquet")
df = df[df.in_pub_set & df.variant.isin(V)].copy()
ctrl = df[df.condition_name=="control"][["variant","temperature","item_id","outcome"]].rename(columns={"outcome":"ctrl_outcome"}).drop_duplicates(["variant","temperature","item_id"])
pr = df[df.condition_name!="control"].merge(ctrl,on=["variant","temperature","item_id"])
cc = pr[pr.ctrl_outcome=="correct"].copy(); cc["abandon"]=(cc.outcome!="correct").astype(int)
cc["n_wrong"]=cc.n_wrong_mentions.astype(float); cc["has_lines"]=(cc.n_peer_lines>0).astype(int)
# ---------- A. feature decomposition per variant (all temps), item-clustered SEs, deviance drop per group
print("== A. Logistic abandon ~ features (all temps, control-correct pairs), item-clustered SE ==")
groups={"frame":"C(frame)","system":"C(system_prompt_type)","n_wrong":"n_wrong","lexical":"C(lexical_identity)","consensus":"C(consensus_type)","temperature":"temperature","domain":"C(dataset)"}
full="abandon ~ "+" + ".join(groups.values())
rows=[]
for v in V:
    d=cc[cc.variant==v]
    try:
        m=smf.logit(full,d).fit(disp=0,maxiter=200,cov_type="cluster",cov_kwds={"groups":d.item_id})
    except Exception as e:
        print(v,"full fit failed:",e); continue
    for g,term in groups.items():
        red=" + ".join(t for k,t in groups.items() if k!=g)
        try: m2=smf.logit("abandon ~ "+red,d).fit(disp=0,maxiter=200)
        except Exception: continue
        rows.append(dict(variant=v,feature=g,drop_in_deviance=2*(m.llf-m2.llf),df=int(m.df_model-m2.df_model),pseudoR2_full=m.prsquared))
imp=pd.DataFrame(rows); imp["share"]=imp.groupby("variant").drop_in_deviance.transform(lambda s:s/s.sum())
imp.to_csv(RES/"structural_feature_importance.csv",index=False)
print(imp.pivot_table(index="feature",columns="variant",values="share").round(3).to_string())
print("\nNOTE: features are only partially crossed across 11 conditions; shares are drop-one deviance and are order-independent but not free of aliasing (frame×system aliased: only asch_history_5/authoritative_bias use the warning system prompt).")
# ---------- B. item susceptibility
print("\n== B. Item susceptibility ==")
it = cc.groupby(["item_id","variant"]).abandon.mean().unstack("variant")
print("Spearman of per-item abandonment across variants (pooled temps & conditions):"); print(it.corr(method="spearman").round(2).to_string())
ic = cc.groupby(["item_id","condition_name"]).abandon.mean().unstack("condition_name")
print("\nSpearman across pressure conditions (pooled variants):"); print(ic.corr(method="spearman").round(2).loc[["asch_zhu_unbiased_unanimous_plain","asch_zhu_unbiased_diverse_plain","asch_zhu_unbiased_qd","authority_zhu_unbiased_trust","asch_history_5"],["asch_zhu_unbiased_unanimous_plain","asch_zhu_unbiased_diverse_plain","asch_zhu_unbiased_qd","authority_zhu_unbiased_trust","asch_history_5"]].to_string())
M = cc.groupby(["item_id","variant","condition_name"]).abandon.mean().unstack(["variant","condition_name"]).dropna(thresh=40)
Mf = M.fillna(M.mean())
U,S,Vt=np.linalg.svd(Mf.values-Mf.values.mean(),full_matrices=False); print(f"\nSVD of item × (variant,condition) abandonment matrix ({Mf.shape}): variance share of PC1={S[0]**2/(S**2).sum():.3f}, PC2={S[1]**2/(S**2).sum():.3f}")
# Rasch (1PL) via logistic with item and person(variant×condition×temp) fixed effects
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
d=cc.copy(); d["person"]=d.variant+"|"+d.condition_name+"|"+d.temperature.astype(str)
enc=OneHotEncoder(); Xs=enc.fit_transform(d[["item_id","person"]]); y=d.abandon.values
r=LogisticRegression(C=1e4,max_iter=2000,solver="lbfgs").fit(Xs,y)
ll=-log_loss(y,r.predict_proba(Xs)[:,1],normalize=False); p0=y.mean(); ll0=len(y)*(p0*np.log(p0)+(1-p0)*np.log(1-p0))
print(f"Rasch-style 1PL (item + person one-hot, near-unpenalised logistic): pseudo-R2={1-ll/ll0:.3f} (null llf {ll0:.0f} -> {ll:.0f}); n={len(y)}")
names=enc.get_feature_names_out(["item_id","person"]); coef=pd.Series(r.coef_[0],index=names)
b=coef[[n for n in names if n.startswith("item_id_")]]; th=coef[[n for n in names if n.startswith("person_")]]
print("item robustness (-b) quantiles:", (-b).quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
print("person (variant|condition|temp) pressure-strength theta quantiles:", th.quantile([.1,.5,.9]).round(2).to_dict())
# 2PL-ish check: does item ordering differ by pressure family? item difficulty fitted separately on participant-frame vs authority conditions
fam_mask=d.condition_name.str.startswith("asch_zhu")
bs={}
for name,mask in [("participant_frame",fam_mask),("other_frames",~fam_mask)]:
    dd=d[mask]; e=OneHotEncoder(); X2=e.fit_transform(dd[["item_id","person"]]); rr=LogisticRegression(C=1e4,max_iter=2000).fit(X2,dd.abandon.values)
    nn=e.get_feature_names_out(["item_id","person"]); bs[name]=pd.Series(rr.coef_[0],index=nn).filter(like="item_id_")
jj=pd.concat(bs,axis=1).dropna(); print(f"Spearman(item difficulty | participant-frame, item difficulty | other frames) = {stats.spearmanr(jj.iloc[:,0],jj.iloc[:,1]).statistic:.3f} (n={len(jj)})")
# prior-strength proxy: control correctness rate across all variants/temps
prior = df[df.condition_name=="control"].groupby("item_id").apply(lambda g:(g.outcome=="correct").mean()).rename("ctrl_acc_all")
sus = cc.groupby("item_id").abandon.mean().rename("abandon")
j=pd.concat([prior,sus],axis=1).dropna(); print(f"\nSpearman(item control-accuracy across variants/temps, item abandonment) = {stats.spearmanr(j.ctrl_acc_all,j.abandon).statistic:.3f} (n={len(j)})")
dom = cc.groupby(["dataset","variant"]).abandon.mean().unstack("variant"); print("\nAbandonment by dataset × variant (pooled):"); print(dom.round(2).to_string())
# ---------- C. temperature
print("\n== C. Temperature ==")
fam=np.where(cc.condition_name.str.startswith("asch_zhu"),"participant_frame",np.where(cc.condition_name.isin(["asch_history_5","authoritative_bias"]),"warning_frames","authority_ctrl"))
cc["fam"]=fam
print(cc.groupby(["variant","fam","temperature"]).abandon.mean().unstack("temperature").round(3).to_string())
# Tc: lowest temperature at which item abandons, per (variant, condition, item)
ab=cc[cc.abandon==1].groupby(["variant","condition_name","item_id"]).temperature.min().rename("Tc").reset_index()
print("\nT_c (lowest T at which the item is abandoned) — share with T_c=0 and mean, per variant:"); print(ab.groupby("variant").Tc.agg(share_Tc0=lambda s:(s==0).mean(), mean_Tc="mean", n="size").round(3).to_string())
