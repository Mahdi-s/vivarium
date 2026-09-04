#!/usr/bin/env python3
"""analysis_dolci_sft.py — hedging / 'say you are unsure' instructions / definiteness in the Dolci-Instruct-SFT mixture, by source (stratified sample)."""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path("/Volumes/T7/conformity data analysis/dataset_analysis/data/raw/instruct-sft/data")
RES = Path(__file__).resolve().parents[2] / "investigation/backstudy/results"
HEDGE = re.compile(r"\b(i(?:'m| am) (?:not (?:entirely |completely |fully )?sure|unsure|uncertain)|i don'?t know|i cannot (?:be certain|determine|verify)|it(?:'s| is) (?:hard|difficult) to (?:say|determine|know)|i (?:may|might) be (?:wrong|mistaken))\b", re.I)
UNSURE_INSTR = re.compile(r"(if you (?:are|'re) (?:unsure|not sure|uncertain)|say (?:that )?you (?:are|'re) unsure|say you don'?t know|admit (?:when|if) you)", re.I)
DEFINITE = re.compile(r"\b(the (?:correct |final )?answer is|therefore,? the answer|final answer)\b", re.I)
PUSHBACK = re.compile(r"\b(actually,|that(?:'s| is) (?:not|incorrect|a common misconception)|i(?:'d| would) (?:respectfully )?disagree|i must (?:correct|clarify))\b", re.I)
rows = []
STEP = 7  # ~1/7 sample of 2.15M ≈ 300K
for f in sorted(glob.glob(str(ROOT / "*.parquet"))):
    t = pq.read_table(f, columns=["messages", "source_dataset", "domain"])
    msgs = t.column("messages").to_pylist()[::STEP]; src = t.column("source_dataset").to_pylist()[::STEP]; dom = t.column("domain").to_pylist()[::STEP]
    for m, s, d in zip(msgs, src, dom):
        sysm = " ".join((x.get("content") or "") for x in m if x.get("role") == "system")
        user = " ".join((x.get("content") or "") for x in m if x.get("role") == "user")
        asst = " ".join((x.get("content") or "") for x in m if x.get("role") == "assistant")
        rows.append(dict(source=s, domain=d, n_turns=len(m), has_system=int(bool(sysm)), unsure_instr=int(bool(UNSURE_INSTR.search(sysm + " " + user))),
                         hedge=int(bool(HEDGE.search(asst))), definite=int(bool(DEFINITE.search(asst))), pushback=int(bool(PUSHBACK.search(asst))), asst_len=len(asst)))
    print(f"  {f.split('/')[-1]}: {len(rows):,}", flush=True)
df = pd.DataFrame(rows); df.to_parquet(RES / "dolci_sft_policy_sample.parquet", index=False)
g = df.groupby("source").agg(n=("hedge", "size"), hedge=("hedge", "mean"), unsure_instr=("unsure_instr", "mean"), definite=("definite", "mean"), pushback=("pushback", "mean"), has_system=("has_system", "mean"), len_med=("asst_len", "median")).sort_values("n", ascending=False)
g.to_csv(RES / "dolci_sft_policy_by_source.csv")
with open(RES / "dolci_sft_policy.md", "w") as fh:
    fh.write(f"# Dolci-Instruct-SFT (1/{STEP} stratified sample, n={len(df):,}): hedging, 'say you are unsure' instructions, definiteness, push-back — by source\n\n")
    fh.write(f"Overall: hedge {df.hedge.mean():.4f} · unsure-instruction in prompt {df.unsure_instr.mean():.4f} · definite-answer marker {df.definite.mean():.4f} · push-back {df.pushback.mean():.4f}\n\n")
    fh.write("P(hedge | unsure instruction present) = %.3f vs P(hedge | absent) = %.3f (n=%d vs %d)\n\n" % (df[df.unsure_instr==1].hedge.mean() if df.unsure_instr.sum() else float('nan'), df[df.unsure_instr==0].hedge.mean(), df.unsure_instr.sum(), (df.unsure_instr==0).sum()))
    fh.write(g.round(4).to_markdown() + "\n")
print(open(RES / "dolci_sft_policy.md").read())
