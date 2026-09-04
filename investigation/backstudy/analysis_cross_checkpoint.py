#!/usr/bin/env python3
"""analysis_cross_checkpoint.py — compare two checkpoints' captured activations on identical (item, condition, context) cells:
linear CKA per layer (Kornblith et al. 2019), cosine between centred mean vectors, and cosine between the two checkpoints'
pressure / truth directions (from directions_<variant>.npz). Same tokenizer/architecture assumed (OLMo-3 7B ladder).
Usage: python analysis_cross_checkpoint.py --data-dir <TAG dir> --a instruct_sft --b instruct_dpo [--bundle <dir>]"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--bundle", default=None)
ap.add_argument("--context", default="_prefixed"); args = ap.parse_args(); D = Path(args.data_dir); B = Path(args.bundle) if args.bundle else D / "bundle"
def load(v):
    X, keys = [], []
    for f in sorted(glob.glob(str(D / "activations" / f"{v}_*.npz"))):
        z = np.load(f); sel = z["context"].astype(str) == args.context
        X.append(z["X"][sel].astype(np.float32)); keys += list(zip(z["item_id"][sel].tolist(), z["condition"][sel].tolist())); layers = z["layers"]
    return np.concatenate(X), keys, [int(x) for x in np.atleast_1d(layers)]
Xa, ka, La = load(args.a); Xb, kb, Lb = load(args.b)
ib = {k: i for i, k in enumerate(kb)}; common = [k for k in ka if k in ib]
A = Xa[[ka.index(k) for k in common]]; Bm = Xb[[ib[k] for k in common]]
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((np.linalg.norm(X.T @ Y) ** 2) / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))
rows = []
da = np.load(B / f"directions_{args.a}.npz"); db = np.load(B / f"directions_{args.b}.npz")
for li, L in enumerate(La):
    if L not in Lb: continue
    lj = Lb.index(L); xa, xb = A[:, li], Bm[:, lj]
    r = dict(layer=L, n=len(common), cka=cka(xa, xb), cos_centered_mean=float(np.dot(xa.mean(0) - xa.mean(), xb.mean(0) - xb.mean()) / (np.linalg.norm(xa.mean(0) - xa.mean()) * np.linalg.norm(xb.mean(0) - xb.mean()) + 1e-8)))
    for key in ("pressure_dir", "truth_dir"):
        u, v = da[key][li], db[key][lj]
        if np.linalg.norm(u) > 0 and np.linalg.norm(v) > 0: r[f"cos_{key}"] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(B / f"cross_{args.a}_vs_{args.b}.csv", index=False); print(R.round(3).to_string(index=False))
