#!/usr/bin/env python3
"""steer_analysis.py — summarise the steering pass: Δ forced-answer margin and Δ greedy outcomes vs the unsteered run,
per (layer, direction, alpha, condition), item-paired, with bootstrap CIs; random-direction control alongside."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--out-dir", default=None)
a = ap.parse_args(); D = Path(a.data_dir); OUT = Path(a.out_dir) if a.out_dir else D / "bundle"; OUT.mkdir(parents=True, exist_ok=True)
st = pd.DataFrame([json.loads(l) for l in open(D / f"{a.variant}.steer.jsonl") if l.strip()])
base = pd.DataFrame([json.loads(l) for l in open(D / f"{a.variant}.jsonl") if l.strip()])
mcol = "margin_first_prefixed" if "margin_first_prefixed" in base.columns else "margin_first_empty_think_prefixed"
b = base.set_index(["item_id", "condition"])[[mcol, "greedy_refusal", "greedy_has_wrong", "greedy_has_gt"]].rename(columns={mcol: "m0", "greedy_refusal": "r0", "greedy_has_wrong": "w0", "greedy_has_gt": "g0"})
j = st.join(b, on=["item_id", "condition"], how="inner")
j["d_margin"] = j.margin_first_prefixed - j.m0; j["d_refusal"] = j.greedy_refusal.astype(int) - j.r0.astype(int)
j["d_wrong"] = j.greedy_has_wrong.astype(int) - j.w0.astype(int); j["d_gt"] = j.greedy_has_gt.astype(int) - j.g0.astype(int)
j["flip_rate_steered"] = (j.margin_first_prefixed < 0).astype(int); j["flip_rate_base"] = (j.m0 < 0).astype(int)
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3: return (np.nan, np.nan)
    bs = rng.choice(x, (2000, len(x))).mean(1); return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
rows = []
for (L, d, al, c), g in j.groupby(["layer", "dir", "alpha", "condition"]):
    lo, hi = ci(g.d_margin); rows.append(dict(variant=a.variant, layer=L, dir=d, alpha=al, condition=c, n=len(g), d_margin=g.d_margin.mean(), d_margin_lo=lo, d_margin_hi=hi,
        flip_base=g.flip_rate_base.mean(), flip_steered=g.flip_rate_steered.mean(), d_refusal=g.d_refusal.mean(), d_wrong=g.d_wrong.mean(), d_gt=g.d_gt.mean()))
S = pd.DataFrame(rows); S.to_csv(OUT / f"steer_{a.variant}.csv", index=False)
print(S.round(3).to_string(index=False))
