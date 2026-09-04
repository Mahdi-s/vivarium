#!/usr/bin/env python3
"""
probe_analysis.py — per-layer linear probes on the answer-slot activations captured by belief_probe.py (one variant).

For each captured layer:
  P1  belief probe: logistic regression predicting belief_flip (forced-answer margin < 0) — GroupKFold by item, AUC
  P2  frame probe : predicting the prompt frame (participant_role / assistant / prior_users / sequence) — accuracy
  P3  pressure direction: difference-in-means (pressure cells − control) unit vector; per-condition mean projection
Outputs <out-dir>/probe_<variant>.csv (+ _projections.csv). sklearn optional: skipped with a note if missing.

Usage: python probe_analysis.py --data-dir <TAG dir> --variant instruct_sft [--out-dir <TAG dir>/bundle]
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--out-dir", default=None)
ap.add_argument("--context", default=None, help="which captured context to use (default: _prefixed for non-think, _empty_think_prefixed for think)")
a = ap.parse_args()
D = Path(a.data_dir); OUT = Path(a.out_dir) if a.out_dir else D / "bundle"; OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(D / f"{a.variant}.jsonl") if l.strip()]
df = pd.DataFrame(rows)
mcol = "margin_first_prefixed" if "margin_first_prefixed" in df.columns else "margin_first_empty_think_prefixed"
ctx = a.context or ("_prefixed" if "margin_first_prefixed" in df.columns else "_empty_think_prefixed")
df["belief_flip"] = (df[mcol] < 0).astype(int)
meta = df.set_index(["item_id", "condition"])

X_parts, keys = [], []
for f in sorted(glob.glob(str(D / "activations" / f"{a.variant}_*.npz"))):
    z = np.load(f)
    sel = z["context"] == ctx
    X_parts.append(z["X"][sel].astype(np.float32)); keys += list(zip(z["item_id"][sel].tolist(), z["condition"][sel].tolist()))
    layers = z["layers"]
if not X_parts:
    print("no activations found"); raise SystemExit(0)
X = np.concatenate(X_parts)                       # n × L × d
layer_ids = list(range(X.shape[1])) if (np.ndim(layers) == 0 and int(layers) == -1) else [int(x) for x in np.atleast_1d(layers)]
idx = [k for k in keys if k in meta.index]
mask = np.array([k in meta.index for k in keys]); X = X[mask]
y_flip = meta.loc[idx, "belief_flip"].values.astype(int)
frame = meta.loc[idx, "feat_frame"].astype(str).values
cond = np.array([k[1] for k in idx], dtype=str)
items = np.array([k[0] for k in idx], dtype=str)
is_ctrl = np.isin(cond, ["control", "control_nolicense"])
print(f"{a.variant}: {len(idx)} captured rows, {X.shape[1]} layers × d={X.shape[2]}, context={ctx}, flip rate {y_flip.mean():.2f}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAVE_SK = True
except Exception:
    HAVE_SK = False; print("sklearn not available: probes P1/P2 skipped; P3 (diff-in-means) still computed")

res = []
for li, L in enumerate(layer_ids):
    Xl = X[:, li, :]
    rec = {"variant": a.variant, "layer": L, "context": ctx, "n": len(Xl)}
    n_groups = len(set(items))
    if HAVE_SK and len(set(y_flip)) == 2 and n_groups >= 3:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        aucs, accs = [], []
        for tr, te in gkf.split(Xl, y_flip, groups=items):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(Xl[tr], y_flip[tr]); p = clf.predict_proba(Xl[te])[:, 1]
            if len(set(y_flip[te])) == 2: aucs.append(roc_auc_score(y_flip[te], p))
            clf2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf2.fit(Xl[tr], frame[tr]); accs.append(accuracy_score(frame[te], clf2.predict(Xl[te])))
        rec.update(belief_probe_auc=float(np.mean(aucs)) if aucs else np.nan, frame_probe_acc=float(np.mean(accs)), frame_majority=float(pd.Series(frame).value_counts(normalize=True).iloc[0]))
    # P3: pressure direction = mean(pressure) - mean(control)
    if is_ctrl.sum() and (~is_ctrl).sum():
        v = Xl[~is_ctrl].mean(0) - Xl[is_ctrl].mean(0); v = v / (np.linalg.norm(v) + 1e-8)
        proj = Xl @ v
        rec["pressure_dir_sep_d"] = float((proj[~is_ctrl].mean() - proj[is_ctrl].mean()) / (proj.std() + 1e-8))
        pc = pd.Series(proj).groupby(cond).mean()
        for c, val in pc.items():
            rec[f"proj_{c}"] = float(val)
        # does the pressure direction also predict the belief flip?
        if len(set(y_flip)) == 2 and HAVE_SK:
            rec["pressure_dir_flip_auc"] = float(roc_auc_score(y_flip, proj))
    res.append(rec)
R = pd.DataFrame(res); R.to_csv(OUT / f"probe_{a.variant}.csv", index=False)
cols = [c for c in ["layer", "n", "belief_probe_auc", "frame_probe_acc", "frame_majority", "pressure_dir_sep_d", "pressure_dir_flip_auc"] if c in R.columns]
print(R[cols].round(3).to_string(index=False))
