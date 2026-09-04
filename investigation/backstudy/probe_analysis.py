#!/usr/bin/env python3
"""
probe_analysis.py — per-layer probes on the activations captured by belief_probe.py (one checkpoint).

Method (see investigation/backstudy/ACTIVATION_AUDIT.md for the literature basis):
  T  Truth probe on candidate STATEMENTS (last token of context+GT vs context+wrong), Marks & Tegmark-style mass-mean
     direction and an L2-regularised logistic probe. Trained on CONTROL-context statements only; evaluated on
     (a) held-out control items (GroupKFold by item), (b) every PRESSURE condition — the train-on-baseline / test-under-
     pressure erasure test (Joswin et al. 2026); chance = 0.5, below-chance = displacement; (c) leave-one-dataset-out
     (Marks & Tegmark generalisation test); (d) shuffled-label control (Hewitt & Liang control task).
  B  Belief-flip probe at the ANSWER SLOT (forced-answer margin < 0), GroupKFold by item, shuffled-label control, LODO.
  P  Pressure direction at the slot = mean(pressure) − mean(control): separability (Cohen's d), directional agreement
     (mean cosine of per-item difference vectors with the mean direction; Braun et al. 2025 — predicts steerability),
     per-condition projections with item-bootstrap CIs, and whether the direction predicts the belief flip (AUC).
  N  Residual norms per layer (activations are centred per layer before cosine/projection statistics).
Outputs: probe_<variant>.csv (per layer), probe_<variant>_truth_by_condition.csv, probe_<variant>_projections.csv,
         directions_<variant>.npz (layers, pressure_dir [raw mean difference], truth_dir [mass-mean], best_layer).
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--out-dir", default=None)
ap.add_argument("--context", default=None, help="_prefixed (default for non-think) or _empty_think_prefixed / _observed_think_prefixed")
ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
D = Path(a.data_dir); OUT = Path(a.out_dir) if a.out_dir else D / "bundle"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)

rows = [json.loads(l) for l in open(D / f"{a.variant}.jsonl") if l.strip()]
df = pd.DataFrame(rows)
ctx = a.context or ("_prefixed" if "margin_first_prefixed" in df.columns else "_empty_think_prefixed")
mcol = f"margin_first{ctx}"
df["belief_flip"] = (df[mcol] < 0).astype(int)
meta = df.set_index(["item_id", "condition"])
CONTROL = {"control", "control_nolicense"}

# ---------- load activations by capture kind
parts = {k: [[], []] for k in ("slot", "gt_end", "wrong_end")}
layers = None
for f in sorted(glob.glob(str(D / "activations" / f"{a.variant}_*.npz"))):
    z = np.load(f); c = z["context"].astype(str); layers = z["layers"]
    for kind, key in (("slot", ctx), ("gt_end", ctx + ":gt_end"), ("wrong_end", ctx + ":wrong_end")):
        sel = c == key
        if sel.any():
            parts[kind][0].append(z["X"][sel].astype(np.float32)); parts[kind][1] += list(zip(z["item_id"][sel].tolist(), z["condition"][sel].tolist()))
def _stack(kind):
    X, keys = parts[kind]
    if not X: return None, []
    X = np.concatenate(X); keep = [k in meta.index for k in keys]
    return X[np.array(keep)], [k for k, kp in zip(keys, keep) if kp]
Xs, ks = _stack("slot"); Xg, kg = _stack("gt_end"); Xw, kw = _stack("wrong_end")
if Xs is None:
    print("no slot activations found"); raise SystemExit(0)
layer_ids = list(range(Xs.shape[1])) if (np.ndim(layers) == 0 and int(layers) == -1) else [int(x) for x in np.atleast_1d(layers)]
print(f"{a.variant}: slot {Xs.shape}, gt_end {None if Xg is None else Xg.shape}, wrong_end {None if Xw is None else Xw.shape}, context={ctx}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False; print("sklearn unavailable: LR probes skipped, mass-mean probes still computed")

def mass_mean_fit(X, y):
    """Mass-mean probe (Marks & Tegmark 2023): direction = mean(pos) − mean(neg); threshold at the midpoint of projections."""
    mu = X.mean(0); Xc = X - mu
    d = Xc[y == 1].mean(0) - Xc[y == 0].mean(0); d = d / (np.linalg.norm(d) + 1e-8)
    p = Xc @ d; thr = 0.5 * (p[y == 1].mean() + p[y == 0].mean())
    return dict(mu=mu, d=d, thr=thr)
def mass_mean_score(m, X): return (X - m["mu"]) @ m["d"] - m["thr"]
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y)) < 2: return np.nan
    if HAVE_SK: return float(roc_auc_score(y, s))
    pos, neg = s[y == 1], s[y == 0]; return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
def acc(y, s): return float(((np.asarray(s) > 0).astype(int) == np.asarray(y)).mean())
def lr_fit(X, y, C=0.05):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    return dict(mu=mu, sd=sd, clf=clf)
def lr_score(m, X): return m["clf"].decision_function((X - m["mu"]) / m["sd"])
def boot_ci(vals, n=1000):
    vals = np.asarray(vals, dtype=float); vals = vals[~np.isnan(vals)]
    if len(vals) < 3: return (np.nan, np.nan)
    b = rng.choice(vals, (n, len(vals))).mean(1); return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))

results, truth_by_cond, proj_rows = [], [], []
directions = {"layers": np.array(layer_ids), "pressure_dir": [], "truth_dir": []}
items_s = np.array([k[0] for k in ks]); cond_s = np.array([k[1] for k in ks]); ds_s = meta.loc[ks, "dataset"].astype(str).values
is_ctrl_s = np.isin(cond_s, list(CONTROL)); yflip = meta.loc[ks, "belief_flip"].values.astype(int)

have_statements = Xg is not None and Xw is not None and len(kg) == len(kw)
if have_statements:
    idx_g = {k: i for i, k in enumerate(kg)}; idx_w = {k: i for i, k in enumerate(kw)}
    common = [k for k in kg if k in idx_w]
    items_t = np.array([k[0] for k in common]); cond_t = np.array([k[1] for k in common]); ds_t = meta.loc[common, "dataset"].astype(str).values
    is_ctrl_t = np.isin(cond_t, list(CONTROL))

for li, L in enumerate(layer_ids):
    rec = {"variant": a.variant, "layer": L, "context": ctx, "n_slot": len(Xs), "resid_norm_mean": float(np.linalg.norm(Xs[:, li], axis=1).mean())}
    # ---------- T: truth probe on statements (train on control only)
    if have_statements:
        G = Xg[[idx_g[k] for k in common], li]; W = Xw[[idx_w[k] for k in common], li]
        Xt = np.concatenate([G, W]); yt = np.concatenate([np.ones(len(G)), np.zeros(len(W))]).astype(int)
        it2 = np.concatenate([items_t, items_t]); c2 = np.concatenate([cond_t, cond_t]); d2 = np.concatenate([ds_t, ds_t]); ctrl2 = np.concatenate([is_ctrl_t, is_ctrl_t])
        ctrl_items = np.unique(it2[ctrl2]); n_groups = len(ctrl_items)
        if n_groups >= 3 and ctrl2.sum() >= 6:
            # (a) held-out control AUC via GroupKFold on control statements; (b) pressure AUC from a probe trained on ALL control statements
            folds = GroupKFold(n_splits=min(5, n_groups)) if HAVE_SK else None
            mm_auc, lr_auc, sh_auc, press_ho = [], [], [], {}
            Xc_, yc_, gc_ = Xt[ctrl2], yt[ctrl2], it2[ctrl2]
            splits = list(folds.split(Xc_, yc_, gc_)) if folds else [(np.arange(len(Xc_)), np.arange(len(Xc_)))]
            for tr, te in splits:
                m = mass_mean_fit(Xc_[tr], yc_[tr]); mm_auc.append(auc(yc_[te], mass_mean_score(m, Xc_[te])))
                ysh = rng.permutation(yc_[tr]); msh = mass_mean_fit(Xc_[tr], ysh); sh_auc.append(auc(yc_[te], mass_mean_score(msh, Xc_[te])))
                if HAVE_SK:
                    lm = lr_fit(Xc_[tr], yc_[tr]); lr_auc.append(auc(yc_[te], lr_score(lm, Xc_[te])))
                # held-out-ITEM erasure test: this fold's probe (trained on control statements of OTHER items) scored on the
                # pressure statements of the test-fold items — removes item-specific leakage from the same-item protocol
                ho_items = set(gc_[te]); sel_p = (~ctrl2) & np.isin(it2, list(ho_items))
                for cname in np.unique(c2[sel_p]):
                    sp = sel_p & (c2 == cname)
                    if len(set(yt[sp])) == 2:
                        press_ho.setdefault(cname, []).append(auc(yt[sp], mass_mean_score(m, Xt[sp])))
            rec.update(truth_mm_auc_control_heldout=float(np.nanmean(mm_auc)), truth_mm_auc_shuffled=float(np.nanmean(sh_auc)),
                       truth_lr_auc_control_heldout=float(np.nanmean(lr_auc)) if lr_auc else np.nan,
                       truth_mm_auc_pressure_heldout_items=float(np.nanmean([np.nanmean(v) for v in press_ho.values()])) if press_ho else np.nan)
            full = mass_mean_fit(Xc_, yc_); lrfull = lr_fit(Xc_, yc_) if HAVE_SK else None
            directions["truth_dir"].append(full["d"])
            # pressure conditions: accuracy/AUC of the CONTROL-trained probe (erasure test), per condition
            for cname in sorted(set(c2) - CONTROL):
                sel = c2 == cname
                s_mm = mass_mean_score(full, Xt[sel]); r = {"variant": a.variant, "layer": L, "condition": cname, "n_items": int(sel.sum() // 2),
                    "truth_mm_auc": auc(yt[sel], s_mm), "truth_mm_acc": acc(yt[sel], s_mm)}
                if lrfull is not None:
                    s_lr = lr_score(lrfull, Xt[sel]); r.update(truth_lr_auc=auc(yt[sel], s_lr), truth_lr_acc=acc(yt[sel], s_lr))
                # restricted to items whose belief flipped in this condition (the Joswin 'flipped questions' view)
                fl = meta.loc[list(zip(it2[sel], c2[sel])), "belief_flip"].values.astype(bool)
                if fl.sum() >= 4 and len(set(yt[sel][fl])) == 2:
                    r["truth_mm_auc_flipped_only"] = auc(yt[sel][fl], s_mm[fl]); r["truth_mm_acc_flipped_only"] = acc(yt[sel][fl], s_mm[fl])
                if cname in press_ho: r["truth_mm_auc_heldout_items"] = float(np.nanmean(press_ho[cname]))
                truth_by_cond.append(r)
            press = ~ctrl2
            if press.sum():
                rec["truth_mm_auc_pressure_all"] = auc(yt[press], mass_mean_score(full, Xt[press]))
                rec["truth_mm_acc_pressure_all"] = acc(yt[press], mass_mean_score(full, Xt[press]))
            # (c) leave-one-dataset-out on control statements
            lodo = []
            for dname in np.unique(d2[ctrl2]):
                tr = ctrl2 & (d2 != dname); te = ctrl2 & (d2 == dname)
                if tr.sum() >= 6 and te.sum() >= 4 and len(set(yt[te])) == 2:
                    lodo.append(auc(yt[te], mass_mean_score(mass_mean_fit(Xt[tr], yt[tr]), Xt[te])))
            rec["truth_mm_auc_lodo"] = float(np.nanmean(lodo)) if lodo else np.nan
        else:
            directions["truth_dir"].append(np.zeros(Xs.shape[2], dtype=np.float32))
    else:
        directions["truth_dir"].append(np.zeros(Xs.shape[2], dtype=np.float32))
    # ---------- B: belief-flip probe at the slot
    Xl = Xs[:, li]; n_groups = len(set(items_s))
    if HAVE_SK and len(set(yflip)) == 2 and n_groups >= 3:
        gkf = GroupKFold(n_splits=min(5, n_groups)); au, sh = [], []
        for tr, te in gkf.split(Xl, yflip, groups=items_s):
            m = lr_fit(Xl[tr], yflip[tr]); au.append(auc(yflip[te], lr_score(m, Xl[te])))
            ms = lr_fit(Xl[tr], rng.permutation(yflip[tr])); sh.append(auc(yflip[te], lr_score(ms, Xl[te])))
        rec.update(flip_lr_auc=float(np.nanmean(au)), flip_lr_auc_shuffled=float(np.nanmean(sh)))
        lodo = []
        for dname in np.unique(ds_s):
            tr, te = ds_s != dname, ds_s == dname
            if tr.sum() >= 6 and te.sum() >= 4 and len(set(yflip[te])) == 2:
                lodo.append(auc(yflip[te], lr_score(lr_fit(Xl[tr], yflip[tr]), Xl[te])))
        rec["flip_lr_auc_lodo"] = float(np.nanmean(lodo)) if lodo else np.nan
    # ---------- P: pressure direction (control vs pressure) at the slot
    if is_ctrl_s.sum() >= 2 and (~is_ctrl_s).sum() >= 2:
        mu = Xl.mean(0); Xc = Xl - mu
        v_raw = Xl[~is_ctrl_s].mean(0) - Xl[is_ctrl_s].mean(0); v = v_raw / (np.linalg.norm(v_raw) + 1e-8)
        proj = Xc @ v
        rec["pressure_sep_d"] = float((proj[~is_ctrl_s].mean() - proj[is_ctrl_s].mean()) / (np.sqrt(0.5 * (proj[~is_ctrl_s].var() + proj[is_ctrl_s].var())) + 1e-8))
        # directional agreement: cosine of per-item (pressure − control) difference vectors with the mean direction
        ctrl_by_item = {it: Xl[i] for i, it in enumerate(items_s) if is_ctrl_s[i] and cond_s[i] == "control"}
        cos = []
        for i in np.where(~is_ctrl_s)[0]:
            if items_s[i] in ctrl_by_item:
                dvec = Xl[i] - ctrl_by_item[items_s[i]]; cos.append(float(dvec @ v / (np.linalg.norm(dvec) + 1e-8)))
        rec["pressure_dir_agreement"] = float(np.mean(cos)) if cos else np.nan
        rec["pressure_dir_agreement_ci_lo"], rec["pressure_dir_agreement_ci_hi"] = boot_ci(cos, a.n_boot) if cos else (np.nan, np.nan)
        if len(set(yflip)) == 2: rec["pressure_dir_flip_auc"] = auc(yflip, proj)
        for cname in np.unique(cond_s):
            sel = cond_s == cname; lo, hi = boot_ci(proj[sel], a.n_boot)
            proj_rows.append({"variant": a.variant, "layer": L, "condition": cname, "n": int(sel.sum()), "proj_mean": float(proj[sel].mean()), "proj_ci_lo": lo, "proj_ci_hi": hi})
        directions["pressure_dir"].append(v_raw.astype(np.float32))
    else:
        directions["pressure_dir"].append(np.zeros(Xs.shape[2], dtype=np.float32))
    results.append(rec)

R = pd.DataFrame(results)
# best layer chosen on held-out CONTROL performance (never on the pressure test set)
crit = "truth_mm_auc_control_heldout" if "truth_mm_auc_control_heldout" in R.columns else ("flip_lr_auc" if "flip_lr_auc" in R.columns else "pressure_sep_d")
best = int(R.loc[R[crit].astype(float).idxmax(), "layer"]) if R[crit].notna().any() else layer_ids[len(layer_ids) // 2]
R["best_layer"] = best
R.to_csv(OUT / f"probe_{a.variant}.csv", index=False)
if truth_by_cond: pd.DataFrame(truth_by_cond).to_csv(OUT / f"probe_{a.variant}_truth_by_condition.csv", index=False)
if proj_rows: pd.DataFrame(proj_rows).to_csv(OUT / f"probe_{a.variant}_projections.csv", index=False)
np.savez(OUT / f"directions_{a.variant}.npz", layers=directions["layers"], pressure_dir=np.stack(directions["pressure_dir"]), truth_dir=np.stack(directions["truth_dir"]), best_layer=best, context=ctx)
show = [c for c in ["layer", "resid_norm_mean", "truth_mm_auc_control_heldout", "truth_mm_auc_shuffled", "truth_mm_auc_lodo", "truth_mm_auc_pressure_all", "truth_mm_auc_pressure_heldout_items", "truth_mm_acc_pressure_all",
                    "flip_lr_auc", "flip_lr_auc_shuffled", "flip_lr_auc_lodo", "pressure_sep_d", "pressure_dir_agreement", "pressure_dir_flip_auc"] if c in R.columns]
print(R[show].round(3).to_string(index=False)); print(f"best layer (held-out control criterion '{crit}'): {best}")
