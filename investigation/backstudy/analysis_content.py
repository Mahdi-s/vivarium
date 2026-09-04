#!/usr/bin/env python3
"""analysis_content.py — what the model says when it abandons the truth (T=0, OLMo family)."""
from pathlib import Path
import pandas as pd, numpy as np, re, json
REPO = Path(__file__).resolve().parents[2]; RES = REPO / "investigation/backstudy/results"
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
df = pd.read_parquet(REPO / "investigation/backstudy/data/olmo_trials.parquet")
df = df[df.in_pub_set & (df.temperature==0.0) & df.variant.isin(["base","instruct_sft","instruct_dpo","instruct","think"])].copy()
ctrl = df[df.condition_name=="control"][["variant","item_id","outcome"]].rename(columns={"outcome":"ctrl_outcome"})
pr = df[df.condition_name!="control"].merge(ctrl,on=["variant","item_id"])
cc = pr[pr.ctrl_outcome=="correct"]
# 1) other_wrong under diverse / da: does the answer match a shown peer answer?
def shown(row):
    try: L=json.loads(row.meta_confederate_lines)
    except Exception: return []
    outs=[]
    for x in L:
        if isinstance(x,dict): outs.append(x.get("base_answer") or x.get("utterance") or "")
        else: outs.append(str(x))
    return [norm(a) for a in outs if a]
rows=[]
for v,d in cc[cc.condition_name.isin(["asch_zhu_unbiased_diverse_plain","asch_zhu_unbiased_da"])&(cc.outcome=="other_wrong")].groupby(["variant","condition_name"]):
    m=0
    for _,r in d.iterrows():
        pa=norm(r.parsed_answer_text or r.raw_text[:200]); sh=[s for s in shown(r) if s!=norm(r.wrong_answer)]
        if any(s and (s in pa or pa in s) for s in sh): m+=1
    rows.append(dict(variant=v[0],condition=v[1],n_other_wrong=len(d),frac_matches_shown_peer=m/len(d) if len(d) else np.nan))
peer=pd.DataFrame(rows); print("== other_wrong under diverse/DA: fraction matching a *shown* non-target peer answer ==\n", peer.round(3).to_string(index=False))
# 2) rationalisation language among target_wrong under unanimous conditions
RAT=re.compile(r"(other participants|the participants|consensus|everyone|agree with|as (the )?others|the group|majority|prior users|the others)",re.I)
un=cc[cc.condition_name.str.contains("unanimous")&(cc.outcome=="target_wrong")]
print("\n== rationalisation phrases among target_wrong (unanimous conds) ==\n", un.groupby("variant").raw_text.apply(lambda s: s.str.contains(RAT).mean()).round(3).to_string())
# 3) refusal phrasing
REF=re.compile(r"(unsure|not sure|cannot determine|can't determine|don't know|insufficient|unable)",re.I)
rf=cc[cc.outcome=="refusal"]
print("\n== refusal outputs: share containing hedge phrase; median length ==\n", rf.groupby("variant").agg(n=("raw_text","size"),hedge=("raw_text",lambda s:s.str.contains(REF).mean()),len_med=("raw_len","median")).round(3).to_string())
# 4) examples
with open(RES/"content_examples.md","w") as fh:
    fh.write("# Verbatim examples (T=0, control-correct items)\n\n")
    for v in ["instruct_sft","instruct_dpo","instruct"]:
        for cond,oc in [("asch_zhu_unbiased_unanimous_plain","target_wrong"),("asch_zhu_unbiased_unanimous_plain","refusal"),("asch_zhu_unbiased_diverse_plain","other_wrong"),("asch_zhu_unbiased_qd","target_wrong"),("authority_zhu_unbiased_trust","target_wrong"),("asch_zhu_unbiased_unanimous_plain","other_wrong")]:
            d=cc[(cc.variant==v)&(cc.condition_name==cond)&(cc.outcome==oc)].head(4)
            fh.write(f"## {v} — {cond} — {oc} (n shown {len(d)})\n\n")
            for _,r in d.iterrows():
                fh.write(f"- **item** {r.item_id} | GT=`{r.ground_truth_text}` | injected wrong=`{r.wrong_answer}`\n  - output: {r.raw_text[:400].replace(chr(10),' ')}\n")
            fh.write("\n")
print("\nexamples -> results/content_examples.md")
