#!/usr/bin/env python3
"""
Generate two sets of publication figures for v2 paper layout:

1.  Pillar II split panels:
    - fig_pillar_2_forest_plot.pdf  (main text, ~half-width wrapfigure)
    - fig_pillar_2_violins.pdf      (appendix)

2.  Updated 4-condition ablation figure:
    - fig3_ablation_ngram.pdf       (replaces existing; now shows matched-instruction condition)
    Data for Llama-3.1-70B comes from the HPC OpenRouter re-runs
    (20260429_184857 at T=0.0).  OLMo matched-instruction is still pending;
    the OLMo panel keeps three conditions and is marked accordingly.
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

DPO_CSV  = ROOT / "dataset_analysis/results/phase6_instruct-dpo_per_pair.csv"
DPO_JSON = ROOT / "dataset_analysis/results/phase6_instruct-dpo_summary.json"
ABLATION_CSV = ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/ablation_rates_t0.csv"

LLAMA_DB_T0 = ROOT / "runs/20260429_184857_5c9cbbbc-455c-4218-bd19-8439c44d6a60/simulation.db"

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


# ══════════════════════════════════════════════════════════════════════
# PILLAR II SPLIT
# ══════════════════════════════════════════════════════════════════════

def _load_pillar2_data():
    with open(DPO_JSON) as f:
        dpo_sum = json.load(f)

    VIOLIN_METRICS = [
        ("delta_struct_jaccard", r"$\Delta$Struct" + "\n" + "Jaccard"),
        ("delta_ngram_overlap",  r"$\Delta$N-gram" + "\n" + "Overlap"),
        ("delta_max_run",        r"$\Delta$Max" + "\n" + "Run"),
        ("delta_consensus_hits", r"$\Delta$Consensus" + "\n" + "Hits"),
    ]
    FOREST_METRICS = [
        ("delta_ngram_overlap",        r"$\Delta$ N-gram overlap"),
        ("delta_struct_jaccard",       r"$\Delta$ Struct Jaccard"),
        ("delta_has_run_5",            r"$\Delta$ has-run-$\geq$5"),
        ("delta_peer_frame_count",     r"$\Delta$ Peer-frame count"),
        ("delta_consensus_hits",       r"$\Delta$ Consensus hits"),
        ("delta_correction_per_1k",    r"$\Delta$ Correction / 1k"),
        ("delta_sycophancy",           r"$\Delta$ Sycophancy"),
        ("delta_correction",           r"$\Delta$ Correction"),
        ("delta_max_run",              r"$\Delta$ Max run"),
    ]

    # Cliff's delta and practical significance for violin annotations
    cliffs  = {k: dpo_sum.get(k, {}).get("cliffs_delta", float("nan")) for k, _ in VIOLIN_METRICS}
    prac_sig = {k: dpo_sum.get(k, {}).get("practical_significance", "n/a") for k, _ in VIOLIN_METRICS}

    # 50k subsample for violins
    dpo_df = pd.read_csv(DPO_CSV, usecols=[k for k, _ in VIOLIN_METRICS])
    if len(dpo_df) > 50000:
        dpo_df = dpo_df.sample(n=50000, random_state=0)

    # Forest rows
    forest_rows = []
    for key, label in FOREST_METRICS:
        entry = dpo_sum.get(key, {})
        ci = entry.get("boot_ci_cliffs_delta", [float("nan"), float("nan")])
        forest_rows.append({
            "key": key, "label": label,
            "cliffs":  entry.get("cliffs_delta", float("nan")),
            "ci_lo":   ci[0],  "ci_hi": ci[1],
            "p_holm":  entry.get("permutation_p_holm", float("nan")),
            "null_z":  entry.get("null_cliffs_delta_z", float("nan")),
        })

    return dpo_df, VIOLIN_METRICS, cliffs, prac_sig, forest_rows


def _draw_forest(ax, forest_rows):
    ax.axvspan(-0.147, 0.147, color="grey",    alpha=0.07, zorder=0,
               label=r"|$\delta$|<0.147 (Romano 'none')")
    ax.axvspan( 0.147,  0.33, color="#FFE4B5", alpha=0.20, zorder=0)
    ax.axvspan(-0.33, -0.147, color="#FFE4B5", alpha=0.20, zorder=0)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)

    y_pos = np.arange(len(forest_rows))[::-1]
    for y, row in zip(y_pos, forest_rows):
        cd, lo, hi = row["cliffs"], row["ci_lo"], row["ci_hi"]
        sig  = (not np.isnan(row["p_holm"])) and (row["p_holm"] < 0.05)
        face = "#1f77b4" if sig else "white"
        edge = "#1f77b4" if sig else "grey"
        ax.errorbar(cd, y, xerr=[[cd - lo], [hi - cd]],
                    fmt="none", ecolor=edge, elinewidth=1.2, capsize=3, zorder=4)
        ax.scatter(cd, y, s=55, marker="o",
                   facecolor=face, edgecolor=edge, linewidth=1.0, zorder=5)
        ax.text(hi + 0.008, y,
                f"  z$_{{\\mathrm{{null}}}}={row['null_z']:+.0f}$",
                fontsize=5.8, va="center", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in forest_rows], fontsize=6.5)
    ax.set_xlabel(r"Cliff's $\delta$ (rejected $-$ chosen)", fontsize=8)
    ax.set_xlim(-0.22, 0.22)
    ax.set_title(r"DPO Effect-Size Forest ($N{=}259{,}785$ pairs, Holm-corrected)",
                 fontsize=8, fontweight="bold")
    ax.grid(axis="x", alpha=0.12, zorder=0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#1f77b4", markeredgecolor="#1f77b4",
                   markersize=6, label=r"$p_{\mathrm{Holm}}<0.05$"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="white", markeredgecolor="grey",
                   markersize=6, label=r"$p_{\mathrm{Holm}}\geq 0.05$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.0,
              frameon=True, framealpha=0.85)


def _draw_violins(ax, dpo_df, VIOLIN_METRICS, cliffs, prac_sig):
    violin_data   = [dpo_df[k].dropna().values for k, _ in VIOLIN_METRICS]
    violin_labels = [lbl for _, lbl in VIOLIN_METRICS]

    parts = ax.violinplot(violin_data, positions=np.arange(len(VIOLIN_METRICS)),
                          showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#5B9BD5")
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.7)

    for i, (key, _) in enumerate(VIOLIN_METRICS):
        vals     = np.clip(dpo_df[key].dropna().values, -0.5, 0.5)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(plt.Rectangle(
            (i - 0.05, q25), 0.1, q75 - q25,
            facecolor="white", edgecolor="black", linewidth=0.8, zorder=4))
        ax.hlines(med, i - 0.08, i + 0.08, color="black", linewidth=1.2, zorder=5)
        cd   = cliffs.get(key, float("nan"))
        ps   = prac_sig.get(key, "n/a")
        cd_s = f"$\\delta$={cd:+.3f}\n({ps})" if not np.isnan(cd) else "n/a"
        ax.text(i, 0.52, cd_s, ha="center", va="bottom", fontsize=6.5, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="grey", linewidth=0.4, alpha=0.85))

    ax.set_xticks(np.arange(len(VIOLIN_METRICS)))
    ax.set_xticklabels(violin_labels, fontsize=7)
    ax.set_ylabel("Delta value (rejected $-$ chosen)", fontsize=8)
    ax.set_title("DPO Penalty: Structural-Prior Deltas\n(n=50k pairs, seed=0)",
                 fontsize=8, fontweight="bold")
    ax.set_ylim(-0.5, 0.65)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5, zorder=2)
    ax.grid(axis="y", alpha=0.12, zorder=0)


def make_pillar2_forest():
    """Standalone forest plot — fits as half-width wrapfigure in main text."""
    dpo_df, VIOLIN_METRICS, cliffs, prac_sig, forest_rows = _load_pillar2_data()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.8), constrained_layout=True)
    _draw_forest(ax, forest_rows)

    fig.savefig(OUT / "fig_pillar_2_forest_plot.pdf")
    fig.savefig(OUT / "fig_pillar_2_forest_plot.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_pillar_2_forest_plot.pdf  ({len(forest_rows)} metrics)")


def make_pillar2_violins():
    """Standalone violin panel — for appendix."""
    dpo_df, VIOLIN_METRICS, cliffs, prac_sig, forest_rows = _load_pillar2_data()

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), constrained_layout=True)
    _draw_violins(ax, dpo_df, VIOLIN_METRICS, cliffs, prac_sig)

    fig.savefig(OUT / "fig_pillar_2_violins.pdf")
    fig.savefig(OUT / "fig_pillar_2_violins.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_pillar_2_violins.pdf  (4 violin metrics)")


# ══════════════════════════════════════════════════════════════════════
# UPDATED 4-CONDITION ABLATION FIGURE  (Llama-3.1-70B, T=0.0)
# ══════════════════════════════════════════════════════════════════════

def _query_db_bers(db_path):
    """Return {condition_name: endorsed/400} from a simulation.db."""
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT cc.name, co.parsed_answer_json, co.refusal_flag
        FROM conformity_trials ct
        JOIN conformity_conditions cc ON ct.condition_id = cc.condition_id
        JOIN conformity_outputs co ON co.trial_id = ct.trial_id
    """).fetchall()
    con.close()

    cond = defaultdict(lambda: {"endorsed": 0, "refusals": 0})
    for cname, paj, rflag in rows:
        if rflag:
            cond[cname]["refusals"] += 1
        try:
            j = json.loads(paj)
            if j.get("wrong_answer_endorsed") == 1:
                cond[cname]["endorsed"] += 1
        except Exception:
            pass
    return {k: v["endorsed"] / 400 for k, v in cond.items()}


def make_ablation_figure():
    """
    4-condition ablation bar chart for Llama-3.1-70B (T=0.0 from HPC re-run).

    Conditions in narrative order:
      1. Social + system prompt  →  BER from existing ablation_rates_t0.csv
         (re-run only covers naked/ngram conditions; sys-prompt run is existing data)
      2. Naked social             →  BER from HPC re-run
      3. N-gram, matched instruction ("answer based on your knowledge")  →  re-run
      4. N-gram, original wording  ("answer based on provided sequence") →  re-run

    The three-level hierarchy is the key visual:
      sys-prompt (0.045) << social/matched-ngram (≈0.278) < original-ngram (0.365)
    """
    # ── Sys-prompt BER from existing CSV ──
    abl_df = pd.read_csv(ABLATION_CSV)
    llama_sys = abl_df[
        (abl_df["model_short"] == "Llama-3.1-70B") &
        (abl_df["condition"]   == "asch_zhu_unanimous_confident")
    ].iloc[0]
    ber_sys    = float(llama_sys["endorsement_rate"])
    ref_sys    = float(llama_sys["refusal_rate"])

    # ── Re-run BERs from HPC DB ──
    rerun = _query_db_bers(LLAMA_DB_T0)
    ber_naked   = rerun["asch_zhu_naked_unanimous_confident"]
    ber_matched = rerun["ngram_sequence_matched_baseline"]
    ber_orig    = rerun["ngram_sequence_baseline"]

    # ── Wilson 95% CI helper ──
    def wilson_ci(p, n=400):
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0, centre - half), min(1, centre + half)

    bars = [
        ("Social + sys prompt\n(89% refusal†)", ber_sys,    "#2980B9"),
        ("Naked social\n(no sys prompt)",        ber_naked,  "#E67E22"),
        ("N-gram, matched\ninstruction",         ber_matched,"#27AE60"),
        ("N-gram, original\nwording",            ber_orig,   "#C0392B"),
    ]

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    ci_los = [wilson_ci(v)[0] for v in values]
    ci_his = [wilson_ci(v)[1] for v in values]
    yerr_lo = [v - l for v, l in zip(values, ci_los)]
    yerr_hi = [h - v for v, h in zip(values, ci_his)]

    x = np.arange(len(bars))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.4), constrained_layout=True)

    rects = ax.bar(x, values, color=colors, alpha=0.85, width=0.55, zorder=3,
                   edgecolor="black", linewidth=0.5)
    ax.errorbar(x, values,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="black", elinewidth=1.0, capsize=4, zorder=4)

    # Annotate BER above each bar
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", zorder=5)

    # Horizontal reference lines
    ax.axhline(ber_naked, color="#E67E22", linewidth=0.7,
               linestyle="--", alpha=0.45, zorder=2)
    ax.axhline(ber_orig,  color="#C0392B", linewidth=0.7,
               linestyle="--", alpha=0.45, zorder=2)

    # Bracket annotation: instruction wording effect
    gap = ber_orig - ber_matched
    ax.annotate("", xy=(3, ber_orig + 0.01), xytext=(2, ber_orig + 0.01),
                arrowprops=dict(arrowstyle="<->", color="#C0392B",
                                lw=1.0, mutation_scale=8))
    ax.text(2.5, ber_orig + 0.025,
            f"+{gap:.3f}\n(wording\neffect)",
            ha="center", va="bottom", fontsize=6.2, color="#C0392B")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Baseline-Equalised Rate (BER,  $N{=}400$, $T{=}0$)", fontsize=8)
    ax.set_ylim(0, 0.52)
    ax.set_title("Llama-3.1-70B Ablation: Social vs.\ Structural Channel",
                 fontsize=9, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.14, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Footnote on refusal
    ax.text(0.01, -0.14,
            f"† BER with sys prompt coexists with a {ref_sys:.0%} refusal rate; "
            "the low endorsement reflects refusal-based abstention, not corrected answering.",
            transform=ax.transAxes, fontsize=5.8, va="top", color="grey",
            wrap=True)

    fig.savefig(OUT / "fig3_ablation_ngram.pdf")
    fig.savefig(OUT / "fig3_ablation_ngram.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig3_ablation_ngram.pdf")
    print(f"    sys={ber_sys:.3f}  naked={ber_naked:.3f}  "
          f"matched={ber_matched:.3f}  orig={ber_orig:.3f}")


# ── Runner ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import matplotlib.ticker as mticker

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="forest, violins, ablation  (comma-separated)")
    args = parser.parse_args()

    tasks = {
        "forest":   make_pillar2_forest,
        "violins":  make_pillar2_violins,
        "ablation": make_ablation_figure,
    }
    selected = [t.strip() for t in args.only.split(",")] if args.only else list(tasks.keys())
    for name in selected:
        print(f"\n→ {name}")
        tasks[name]()

    print("\nDone.")
