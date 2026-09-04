#!/usr/bin/env python3
"""analysis_validity.py — measurement-validity checks: truncation/think-closure, output length by outcome, judge nulls."""
from pathlib import Path
import pandas as pd, numpy as np, re
REPO = Path(__file__).resolve().parents[2]; RES = REPO / "investigation/backstudy/results"
df = pd.read_parquet(REPO / "investigation/backstudy/data/olmo_trials.parquet")
df = df[df.in_pub_set & df.variant.isin(["base","instruct_sft","instruct_dpo","instruct","think_sft","think_dpo","think"])].copy()
df["has_think_open"] = df.raw_text.str.contains("<think>", regex=False)
df["has_think_close"] = df.raw_text.str.contains("</think>", regex=False)
df["truncated_think"] = df.has_think_open & ~df.has_think_close
df["is_empty"] = df.raw_len < 3
df["fam"] = np.where(df.condition_name=="control","control",np.where(df.condition_name.str.startswith("asch_zhu"),"participant_frame",np.where(df.condition_name.isin(["asch_history_5","authoritative_bias"]),"warning_sys_frames","authority_ctrl")))
out=[]
for (v,f),d in df[df.temperature==0.0].groupby(["variant","fam"]):
    out.append(dict(variant=v, family=f, n=len(d), judge_null=d.judge_is_correct.isna().mean(), think_open=d.has_think_open.mean(), think_unclosed=d.truncated_think.mean(), empty=d.is_empty.mean(), raw_len_med=d.raw_len.median(), raw_len_p90=d.raw_len.quantile(.9), p_correct=(d.outcome=="correct").mean(), p_refusal=(d.outcome=="refusal").mean(), p_other=(d.outcome=="other_wrong").mean()))
t=pd.DataFrame(out); t.to_csv(RES/"validity_truncation.csv",index=False)
print("== Truncation / think-closure / judge-null by variant × condition family (T=0) ==")
print(t.round(3).to_string(index=False))
# raw length by outcome
print("\n== median raw_len by variant × outcome (T=0) ==")
print(df[df.temperature==0.0].groupby(["variant","outcome"]).raw_len.median().unstack().round(0).to_string())
# what does 'other_wrong' look like for think models: unclosed think share
tw = df[(df.temperature==0.0)&(df.variant.str.startswith("think"))]
print("\n== think variants: outcome vs think_unclosed (T=0, all conditions) ==")
print(pd.crosstab(tw.outcome, tw.truncated_think, normalize="index").round(3).to_string())
print("\n== think variants: P(truncated_think) by condition (T=0) ==")
print(tw.groupby(["variant","condition_name"]).truncated_think.mean().unstack(0).round(3).to_string())
