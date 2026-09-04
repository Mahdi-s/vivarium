#!/usr/bin/env python3
"""analysis_judge_null.py — robustness of abandonment to judge-null (undetermined) outputs; 5-state view."""
from pathlib import Path
import pandas as pd, numpy as np
REPO=Path(__file__).resolve().parents[2]; RES=REPO/"investigation/backstudy/results"
V=["base","instruct_sft","instruct_dpo","instruct"]
df=pd.read_parquet(REPO/"investigation/backstudy/data/olmo_trials.parquet"); df=df[df.in_pub_set&df.variant.isin(V)].copy()
df["state5"]=np.where(df.judge_is_correct.isna()&(df.sql_refusal_flag!=1),"undetermined",df.outcome)
ctrl=df[df.condition_name=="control"][["variant","temperature","item_id","state5"]].rename(columns={"state5":"ctrl"}).drop_duplicates(["variant","temperature","item_id"])
pr=df[df.condition_name!="control"].merge(ctrl,on=["variant","temperature","item_id"]); cc=pr[pr.ctrl=="correct"]
SHORT={"asch_zhu_unbiased_qd":"qd","asch_zhu_unbiased_unanimous_plain":"unan_plain","asch_zhu_unbiased_unanimous_neutral":"unan_neutral","asch_zhu_unbiased_unanimous_confident":"unan_confident","asch_zhu_unbiased_unanimous_uncertain":"unan_uncertain","asch_zhu_unbiased_diverse_plain":"diverse","asch_zhu_unbiased_da":"da","asch_history_5":"asch_hist","authoritative_bias":"auth_bias","authority_zhu_unbiased_trust":"auth_trust","authority_zhu_unbiased_trust_da":"auth_trust_da"}
for label,d in [("T=0",cc[cc.temperature==0]),("pooled",cc)]:
    g=d.groupby(["variant","condition_name"]).state5.value_counts(normalize=True).unstack().fillna(0)
    g["abandon_incl_null"]=1-g["correct"]; g["abandon_excl_null"]=1-g["correct"]/(1-g.get("undetermined",0))
    g=g.reset_index(); g["condition"]=g.condition_name.map(SHORT)
    print(f"\n== {label}: 5-state shares among control-correct pairs ==")
    print(g.set_index(["variant","condition"])[["correct","target_wrong","other_wrong","refusal","undetermined","abandon_incl_null","abandon_excl_null"]].round(3).to_string())
    g.to_csv(RES/f"five_state_{label.replace('=','')}.csv",index=False)
