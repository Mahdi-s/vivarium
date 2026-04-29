#!/usr/bin/env python3
"""Generate publication-quality figures for the COLM 2026 paper.

Produces:
  figures/fig2_conformity_or.pdf            – McNemar OR across training pipeline
  figures/fig2_stage_trajectory_shared.pdf  – BER trajectory on 3 shared pressure
                                              conditions (Instruct vs Think, T=0, 7B)
  figures/fig3_tc_ursp.pdf                  – Combined Tc distribution + URSP bar
"""

import os
import csv
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VARIANT_ORDER = ["Base", "SFT", "DPO", "Instruct"]
COLORS = {
    "Base":      "#7f8c8d",  # grey
    "SFT":       "#e74c3c",  # red (amplifies)
    "DPO":       "#2ecc71",  # green (mitigates)
    "Instruct":  "#3498db",  # blue
}


def _holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values in original order."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = (n - rank) * p
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _sig_star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: McNemar OR across conditions — the headline result
# ═══════════════════════════════════════════════════════════════════════════

# ── Load McNemar data from CSV ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MCNEMAR_CSV = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "publication_V2_colm",
    "statistical_tests", "mcnemar_pressure_vs_control.csv",
)

# Map figure labels → CSV condition_name, and figure variant → CSV variant
COND_MAP = [
    ("Unan.\nConfident", "asch_zhu_unbiased_unanimous_confident"),
    ("Auth.\nBias",       "authoritative_bias"),
    ("Auth.\nTrust",      "authority_zhu_unbiased_trust"),
    ("Devil's\nAdvocate", "asch_zhu_unbiased_da"),
    ("Question\nDistill.","asch_zhu_unbiased_qd"),
]
VAR_MAP = {
    "Base":     "base",
    "SFT":      "instruct_sft",
    "DPO":      "instruct_dpo",
    "Instruct": "instruct",
}

conditions = [label for label, _ in COND_MAP]
cond_keys  = [key   for _, key in COND_MAP]

# Parse CSV into a lookup: (csv_variant, csv_condition) → row dict
mcnemar_rows = {}
with open(MCNEMAR_CSV, newline="") as f:
    for row in csv.DictReader(f):
        mcnemar_rows[(row["variant"], row["condition_name"])] = row

# Build arrays: OR, CI_lo, CI_hi, significance label
or_data   = {}
ci_lo     = {}
ci_hi     = {}
sig_label = {}

for var_fig, var_csv in VAR_MAP.items():
    or_data[var_fig]   = []
    ci_lo[var_fig]     = []
    ci_hi[var_fig]     = []
    sig_label[var_fig] = []
    for cond_csv in cond_keys:
        row = mcnemar_rows[(var_csv, cond_csv)]
        b = float(row["b_ctrl_correct_pres_wrong"])
        c = float(row["c_ctrl_wrong_pres_correct"])
        OR = b / c if c > 0 else float("inf")
        # Wald 95 % CI on log(OR)
        if b > 0 and c > 0:
            log_or = math.log(OR)
            se = math.sqrt(1.0 / b + 1.0 / c)
            lo = math.exp(log_or - 1.96 * se)
            hi = math.exp(log_or + 1.96 * se)
        else:
            lo, hi = OR, OR
        or_data[var_fig].append(OR)
        ci_lo[var_fig].append(lo)
        ci_hi[var_fig].append(hi)
        # Holm-corrected significance
        sig_label[var_fig].append(row["sig_adjusted"])


def _sig_star(label):
    """Convert sig_adjusted string to display text."""
    if label == "***":
        return "***"
    elif label == "**":
        return "**"
    elif label == "*":
        return "*"
    return "ns"


# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.3))

x = np.arange(len(conditions))
width = 0.19
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, var in enumerate(VARIANT_ORDER):
    ors = or_data[var]
    lo  = ci_lo[var]
    hi  = ci_hi[var]
    yerr_lo = [o - l for o, l in zip(ors, lo)]
    yerr_hi = [h - o for o, h in zip(ors, hi)]
    positions = x + offsets[i] * width
    bars = ax.bar(positions, ors, width,
                  yerr=[yerr_lo, yerr_hi],
                  capsize=1.5, error_kw=dict(lw=0.7, capthick=0.7, color="#444"),
                  label=var, color=COLORS[var], edgecolor="white", linewidth=0.5)
    # Significance stars above error bars
    for j, (bar, h_val, sig) in enumerate(zip(bars, hi, sig_label[var])):
        star_txt = _sig_star(sig)
        y_pos = h_val + 0.2
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                star_txt, ha="center", va="bottom",
                fontsize=7 if star_txt != "ns" else 6,
                fontweight="bold" if star_txt != "ns" else "normal",
                color="#000" if star_txt != "ns" else "#888")

ax.axhline(y=1.0, color="#bbb", linewidth=0.8, linestyle="--", zorder=0)
ax.set_ylabel("McNemar Odds Ratio")
ax.set_xticks(x)
ax.set_xticklabels(conditions)

# Dynamic y-limit based on data
max_hi = max(h for var in VARIANT_ORDER for h in ci_hi[var])
ax.set_ylim(0, max_hi + 2.5)

ax.legend(loc="upper right", ncol=2, frameon=True, fancybox=False,
          edgecolor="#ddd", facecolor="white", framealpha=0.85, fontsize=6,
          handlelength=0.7, handletextpad=0.3, borderpad=0.2, columnspacing=0.4,
          bbox_to_anchor=(1.0, 0.85))
ax.set_title("Conformity susceptibility across the post-training pipeline", fontsize=10, pad=8)

# SFT arrow — point to the tallest SFT bar (Unan. Confident)
sft_bar_x = x[0] + offsets[1] * width
sft_top = ci_hi["SFT"][0]
ax.annotate("SFT amplifies", xy=(sft_bar_x, sft_top + 0.6),
            xytext=(-0.15, max_hi + 1.8),
            fontsize=7.5, fontweight="bold", color=COLORS["SFT"],
            arrowprops=dict(arrowstyle="->", color=COLORS["SFT"], lw=1.0))

# DPO arrow — point to the DPO Unan. Confident bar
dpo_bar_x = x[0] + offsets[2] * width
dpo_top = ci_hi["DPO"][0]
ax.annotate("DPO mitigates", xy=(dpo_bar_x, dpo_top + 0.6),
            xytext=(1.2, max_hi + 1.8),
            fontsize=7.5, fontweight="bold", color=COLORS["DPO"],
            arrowprops=dict(arrowstyle="->", color=COLORS["DPO"], lw=1.0))

fig.savefig("paper/figures/fig2_conformity_or.pdf")
fig.savefig("paper/figures/fig2_conformity_or.png")
print("✓ fig2_conformity_or")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 (main paper): Shared-conditions stage trajectory
# Instruct path (Base → SFT → DPO → Instruct) and Think path (Base → Think-SFT
# → Think-DPO) compared on the 3 pressure conditions they share at T=0.
# Generalizes Figure 1 (one condition, two temperatures) across conditions.
# ═══════════════════════════════════════════════════════════════════════════

STAGE_CSV = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "April_analysis",
    "tables", "stage_decomposition", "instruct_vs_think_t0.csv",
)

# (csv_variant, x_position, stage_label) — x-labels are *training methods* applied at each
# step (SFT, DPO, RLVR), not checkpoint names. The corresponding released OLMo-3 checkpoint
# for the Instruct path at each x is: Base (Olmo-3-1025-7B) / Instruct-SFT / Instruct-DPO /
# Instruct (the final RLVR-trained release). Similarly for Think: Think-SFT / Think-DPO.
INSTRUCT_STAGES = [
    ("base",         0, "Base"),
    ("instruct_sft", 1, "SFT"),
    ("instruct_dpo", 2, "DPO"),
    ("instruct",     3, "RLVR"),
]
THINK_STAGES = [
    ("base",      0, "Base"),
    ("think_sft", 1, "SFT"),
    ("think_dpo", 2, "DPO"),
]

PRESSURE_CONDS = [
    ("asch_zhu_unbiased_unanimous_confident", "Unanimous confident"),
    ("authoritative_bias",                    "Authoritative bias"),
    ("authority_zhu_unbiased_trust",          "Authority trust"),
]

# Path colors for the connecting lines (stage markers still use the stage palette).
INSTRUCT_LINE_COLOR = COLORS["Instruct"]   # #3498db blue
THINK_LINE_COLOR    = "#e67e22"             # matched to fig1_test.tex warm orange family

# Stage-to-color mapping for scatter markers. Think-SFT/Think-DPO reuse SFT/DPO colors
# so the reader sees stage identity by color and path identity by linestyle/marker.
STAGE_MARKER_COLOR = {
    "base":         COLORS["Base"],
    "instruct_sft": COLORS["SFT"],
    "instruct_dpo": COLORS["DPO"],
    "instruct":     COLORS["Instruct"],
    "think_sft":    COLORS["SFT"],
    "think_dpo":    COLORS["DPO"],
    "think":        "#8e44ad",
}

# Parse stage-decomposition CSV into (variant, condition) → row dict.
stage_rows = {}
with open(STAGE_CSV, newline="") as f:
    for row in csv.DictReader(f):
        stage_rows[(row["variant"], row["condition_name"])] = row

CONTROL_BER = float(stage_rows[("base", "control")]["ber"])  # 0.0525

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6),
                         gridspec_kw={"wspace": 0.18})

DX = 0.06  # horizontal jitter to keep CI bars visually separable at shared x

for ax_i, (cond_csv, cond_title) in enumerate(PRESSURE_CONDS):
    ax = axes[ax_i]

    # Extract (ber, lo, hi) triples for each stage on each path.
    def _cell(variant):
        row = stage_rows[(variant, cond_csv)]
        return float(row["ber"]), float(row["ber_lo"]), float(row["ber_hi"])

    inst_ys, inst_lo, inst_hi = [], [], []
    for var, _, _ in INSTRUCT_STAGES:
        b, lo, hi = _cell(var)
        inst_ys.append(b); inst_lo.append(lo); inst_hi.append(hi)

    think_ys, think_lo, think_hi = [], [], []
    for var, _, _ in THINK_STAGES:
        b, lo, hi = _cell(var)
        think_ys.append(b); think_lo.append(lo); think_hi.append(hi)

    # Jittered x-positions (Base sits exactly at x=0 for both paths).
    x_inst  = [0.0, 1.0 - DX, 2.0 - DX, 3.0]
    x_think = [0.0, 1.0 + DX, 2.0 + DX]

    # Reference line: base control BER.
    ax.axhline(y=CONTROL_BER, color="#95a5a6", linestyle=":", lw=0.9, zorder=1)

    # Path lines (drawn first; stage-colored markers overlaid below).
    ax.plot(x_inst, inst_ys, color=INSTRUCT_LINE_COLOR, linestyle="-",
            lw=1.3, marker="None", zorder=2, label="_nolegend_")
    ax.plot(x_think, think_ys, color=THINK_LINE_COLOR, linestyle=(0, (4, 2)),
            lw=1.3, marker="None", zorder=2, label="_nolegend_")

    # Error bars (Wilson 95% precomputed upstream).
    inst_yerr = [[y - l for y, l in zip(inst_ys, inst_lo)],
                 [h - y for y, h in zip(inst_ys, inst_hi)]]
    think_yerr = [[y - l for y, l in zip(think_ys, think_lo)],
                  [h - y for y, h in zip(think_ys, think_hi)]]
    ax.errorbar(x_inst, inst_ys, yerr=inst_yerr, fmt="none",
                ecolor=INSTRUCT_LINE_COLOR, elinewidth=0.8, capsize=2, capthick=0.7,
                zorder=3)
    ax.errorbar(x_think, think_ys, yerr=think_yerr, fmt="none",
                ecolor=THINK_LINE_COLOR, elinewidth=0.8, capsize=2, capthick=0.7,
                zorder=3)

    # Stage-colored markers: circles for Instruct path, squares for Think.
    for xi, (var, _, _) in zip(x_inst, INSTRUCT_STAGES):
        if var == "base":
            continue  # shared Base point drawn once, below
        ax.scatter([xi], [_cell(var)[0]], marker="o", s=36,
                   color=STAGE_MARKER_COLOR[var],
                   edgecolors="white", linewidths=0.6, zorder=4)
    for xi, (var, _, _) in zip(x_think, THINK_STAGES):
        if var == "base":
            continue
        ax.scatter([xi], [_cell(var)[0]], marker="s", s=36,
                   color=STAGE_MARKER_COLOR[var],
                   edgecolors="white", linewidths=0.6, zorder=4)

    # Shared Base marker (single, slightly larger, drawn on top).
    ax.scatter([0.0], [_cell("base")[0]], marker="o", s=58,
               color=COLORS["Base"], edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Base", "SFT", "DPO", "RLVR"], fontsize=8)
    ax.set_ylim(0.0, 0.8)
    ax.set_title(cond_title, fontsize=9, pad=4)

    if ax_i == 0:
        ax.set_ylabel("Wrong-answer endorsement rate (BER)")
    else:
        ax.tick_params(labelleft=False)

    if ax_i == 1:
        ax.set_xlabel("Post-training stage")
        # Annotate the control reference once.
        ax.text(3.25, CONTROL_BER + 0.005,
                f"Base control BER {CONTROL_BER*100:.1f}%",
                fontsize=6.5, color="#7f8c8d", va="bottom", ha="right")

# Shared figure legend (paths + reference line). Stage-color encoding lives in caption.
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=INSTRUCT_LINE_COLOR, linestyle="-", lw=1.3,
           marker="o", markersize=5, markerfacecolor=COLORS["Instruct"],
           markeredgecolor="white", markeredgewidth=0.6,
           label="Instruct path: Base → Instruct-SFT → Instruct-DPO → Instruct (RLVR)"),
    Line2D([0], [0], color=THINK_LINE_COLOR, linestyle=(0, (4, 2)), lw=1.3,
           marker="s", markersize=5, markerfacecolor=COLORS["DPO"],
           markeredgecolor="white", markeredgewidth=0.6,
           label="Think path: Base → Think-SFT → Think-DPO"),
    Line2D([0], [0], color="#95a5a6", linestyle=":", lw=0.9,
           label=f"Base control BER ({CONTROL_BER*100:.1f}%)"),
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.5,
           handlelength=2.2, columnspacing=1.4)
fig.subplots_adjust(top=0.82)

fig.savefig("paper/figures/fig2_stage_trajectory_shared.pdf")
fig.savefig("paper/figures/fig2_stage_trajectory_shared.png")
print("✓ fig2_stage_trajectory_shared")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Shared-conditions stage trajectory at T={0.0, 0.6}
# Includes Think-RL at both temperatures.
# ═══════════════════════════════════════════════════════════════════════════

STAGE_T0_CSV = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "April_analysis",
    "tables", "stage_decomposition", "instruct_vs_think_t0.csv",
)
STAGE_T06_CSV = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "April_analysis",
    "tables", "stage_decomposition", "instruct_vs_think_t06.csv",
)

fig6_rows = {}
for temp_csv in (STAGE_T0_CSV, STAGE_T06_CSV):
    with open(temp_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["temperature"]), row["variant"], row["condition_name"])
            fig6_rows[key] = row

FIG6_TEMPS = [0.0, 0.6]
FIG6_CONDS = PRESSURE_CONDS
FIG6_INSTRUCT_STAGES = [
    ("base", 0, "Base"),
    ("instruct_sft", 1, "SFT"),
    ("instruct_dpo", 2, "DPO"),
    ("instruct", 3, "RLVR"),
]
FIG6_THINK_STAGES = [
    ("base", 0, "Base"),
    ("think_sft", 1, "SFT"),
    ("think_dpo", 2, "DPO"),
    ("think", 3, "RLVR"),
]

fig6, axes = plt.subplots(2, 3, figsize=(9.3, 4.9), gridspec_kw={"wspace": 0.18, "hspace": 0.30})
DX = 0.08

for ti, temp in enumerate(FIG6_TEMPS):
    for ci, (cond_csv, cond_title) in enumerate(FIG6_CONDS):
        ax = axes[ti, ci]

        def _cell(variant):
            row = fig6_rows[(temp, variant, cond_csv)]
            return float(row["ber"]), float(row["ber_lo"]), float(row["ber_hi"])

        inst_ys, inst_lo, inst_hi = [], [], []
        for var, _, _ in FIG6_INSTRUCT_STAGES:
            b, lo, hi = _cell(var)
            inst_ys.append(b)
            inst_lo.append(lo)
            inst_hi.append(hi)

        think_ys, think_lo, think_hi = [], [], []
        for var, _, _ in FIG6_THINK_STAGES:
            b, lo, hi = _cell(var)
            think_ys.append(b)
            think_lo.append(lo)
            think_hi.append(hi)

        x_inst = [0.0, 1.0 - DX, 2.0 - DX, 3.0 - DX]
        x_think = [0.0, 1.0 + DX, 2.0 + DX, 3.0 + DX]
        control_ber = _cell("base")[0] if cond_csv == "control" else float(fig6_rows[(temp, "base", "control")]["ber"])

        ax.axhline(y=control_ber, color="#95a5a6", linestyle=":", lw=0.9, zorder=1)
        ax.plot(x_inst, inst_ys, color=INSTRUCT_LINE_COLOR, linestyle="-", lw=1.3, zorder=2)
        ax.plot(x_think, think_ys, color=THINK_LINE_COLOR, linestyle=(0, (4, 2)), lw=1.3, zorder=2)

        inst_yerr = [[y - l for y, l in zip(inst_ys, inst_lo)], [h - y for y, h in zip(inst_ys, inst_hi)]]
        think_yerr = [[y - l for y, l in zip(think_ys, think_lo)], [h - y for y, h in zip(think_ys, think_hi)]]
        ax.errorbar(x_inst, inst_ys, yerr=inst_yerr, fmt="none", ecolor=INSTRUCT_LINE_COLOR, elinewidth=0.8, capsize=2, capthick=0.7, zorder=3)
        ax.errorbar(x_think, think_ys, yerr=think_yerr, fmt="none", ecolor=THINK_LINE_COLOR, elinewidth=0.8, capsize=2, capthick=0.7, zorder=3)

        for xi, (var, _, _) in zip(x_inst, FIG6_INSTRUCT_STAGES):
            if var != "base":
                ax.scatter([xi], [_cell(var)[0]], marker="o", s=28, color=STAGE_MARKER_COLOR[var], edgecolors="white", linewidths=0.6, zorder=4)
        for xi, (var, _, _) in zip(x_think, FIG6_THINK_STAGES):
            if var != "base":
                ax.scatter([xi], [_cell(var)[0]], marker="s", s=28, color=STAGE_MARKER_COLOR[var], edgecolors="white", linewidths=0.6, zorder=4)
        ax.scatter([0.0], [_cell("base")[0]], marker="o", s=48, color=COLORS["Base"], edgecolors="white", linewidths=0.8, zorder=5)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Base", "SFT", "DPO", "RLVR"], fontsize=7.5)
        ax.set_ylim(0.0, 0.85)
        if ti == 0:
            ax.set_title(cond_title, fontsize=8.5, pad=4)
        if ci == 0:
            ax.set_ylabel(f"T={temp:.1f}\nBER", fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
        if ti == 1 and ci == 1:
            ax.set_xlabel("Post-training stage", fontsize=8.5)

from matplotlib.lines import Line2D
fig6_handles = [
    Line2D([0], [0], color=INSTRUCT_LINE_COLOR, linestyle="-", lw=1.3, marker="o", markersize=4.5, markerfacecolor=COLORS["Instruct"], markeredgecolor="white", markeredgewidth=0.6, label="Instruct path"),
    Line2D([0], [0], color=THINK_LINE_COLOR, linestyle=(0, (4, 2)), lw=1.3, marker="s", markersize=4.5, markerfacecolor=COLORS["DPO"], markeredgecolor="white", markeredgewidth=0.6, label="Think path"),
    Line2D([0], [0], color="#95a5a6", linestyle=":", lw=0.9, label="Base control BER (per temperature)"),
]
fig6.legend(handles=fig6_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.2, handlelength=2.1, columnspacing=1.2)
fig6.suptitle("Figure 6: Shared-condition stage trajectories with Think-RL at T=0.0 and T=0.6", fontsize=10, y=1.06)
fig6.subplots_adjust(top=0.86)
fig6.savefig("paper/figures/fig6_stage_trajectory_shared.pdf")
fig6.savefig("paper/figures/fig6_stage_trajectory_shared.png")
print("✓ fig6_stage_trajectory_shared")
plt.close(fig6)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Conformity susceptibility (McNemar OR) across full pipeline
# Base + Instruct stages + Think stages, faceted by T={0.0, 0.6}.
# ═══════════════════════════════════════════════════════════════════════════

SRC_DIR = Path(REPO_ROOT) / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from vivarium.analytics.behavioral import load_april_trials  # noqa: E402

manifest_path = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "April_analysis", "metadata", "runs_metadata.json",
)
trials = load_april_trials(
    manifest_path=manifest_path,
    include_secondary=False,
    require_judge=True,
)
trials["judge_wrong_endorsed"] = trials["judge_wrong_endorsed"].fillna(False).astype(int)

FIG7_TEMPS = [0.0, 0.6]
FIG7_CONDS = [
    ("Unan.\nConfident", "asch_zhu_unbiased_unanimous_confident"),
    ("Auth.\nBias", "authoritative_bias"),
    ("Auth.\nTrust", "authority_zhu_unbiased_trust"),
]
FIG7_VARIANTS = [
    ("Base", "base"),
    ("Inst-SFT", "instruct_sft"),
    ("Inst-DPO", "instruct_dpo"),
    ("Instruct", "instruct"),
    ("Think-SFT", "think_sft"),
    ("Think-DPO", "think_dpo"),
    ("Think-RL", "think"),
]
FIG7_COLORS = {
    "Base": "#7f8c8d",
    "Inst-SFT": "#e74c3c",
    "Inst-DPO": "#2ecc71",
    "Instruct": "#3498db",
    "Think-SFT": "#f39c12",
    "Think-DPO": "#d35400",
    "Think-RL": "#8e44ad",
}

fig7, ax7 = plt.subplots(1, 2, figsize=(10.1, 3.6), sharey=True)
x = np.arange(len(FIG7_CONDS))
width = 0.11
offsets = np.linspace(-3, 3, len(FIG7_VARIANTS))
global_hi = []

for ti, temp in enumerate(FIG7_TEMPS):
    ax = ax7[ti]
    for vi, (label, variant) in enumerate(FIG7_VARIANTS):
        ors, ci_los, ci_his, pvals = [], [], [], []
        for _, cond in FIG7_CONDS:
            control = trials[
                (trials["variant"] == variant)
                & (trials["temperature"] == temp)
                & (trials["condition_name"] == "control")
            ][["item_id", "judge_wrong_endorsed"]].rename(columns={"judge_wrong_endorsed": "ctrl_wrong"})
            pressure = trials[
                (trials["variant"] == variant)
                & (trials["temperature"] == temp)
                & (trials["condition_name"] == cond)
            ][["item_id", "judge_wrong_endorsed"]].rename(columns={"judge_wrong_endorsed": "pres_wrong"})
            merged = control.merge(pressure, on="item_id", how="inner")
            b = int(((merged["ctrl_wrong"] == 0) & (merged["pres_wrong"] == 1)).sum())
            c = int(((merged["ctrl_wrong"] == 1) & (merged["pres_wrong"] == 0)).sum())
            # Haldane-Anscombe correction prevents infinite OR when c==0.
            b_cc = b + 0.5
            c_cc = c + 0.5
            OR = b_cc / c_cc
            log_or = math.log(OR)
            se = math.sqrt(1.0 / b_cc + 1.0 / c_cc)
            lo = math.exp(log_or - 1.96 * se)
            hi = math.exp(log_or + 1.96 * se)
            chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
            p_raw = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))

            ors.append(OR)
            ci_los.append(lo)
            ci_his.append(hi)
            pvals.append(p_raw)

        p_adj = _holm_adjust(pvals)
        yerr_lo = [o - l for o, l in zip(ors, ci_los)]
        yerr_hi = [h - o for o, h in zip(ors, ci_his)]
        positions = x + offsets[vi] * width
        bars = ax.bar(
            positions,
            ors,
            width,
            yerr=[yerr_lo, yerr_hi],
            capsize=1.4,
            error_kw=dict(lw=0.65, capthick=0.65, color="#444"),
            label=label,
            color=FIG7_COLORS[label],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, top, p in zip(bars, ci_his, p_adj):
            star = _sig_star_from_p(p)
            global_hi.append(top)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top * 1.12,
                star,
                ha="center",
                va="bottom",
                fontsize=6.2 if star != "ns" else 5.8,
                fontweight="bold" if star != "ns" else "normal",
                color="#000" if star != "ns" else "#888",
            )

    ax.axhline(y=1.0, color="#bbb", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in FIG7_CONDS], fontsize=8)
    ax.set_title(f"T = {temp:.1f}", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 300)
    if ti == 0:
        ax.set_ylabel("McNemar Odds Ratio")

ax7[1].tick_params(labelleft=False)
for ax in ax7:
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
fig7.legend(loc="upper center", ncol=4, frameon=True, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
fig7.suptitle("Figure 7: Conformity susceptibility across the post-training pipeline", fontsize=10, y=1.08)
fig7.subplots_adjust(top=0.82, wspace=0.08)
fig7.savefig("paper/figures/fig7_conformity_or.pdf")
fig7.savefig("paper/figures/fig7_conformity_or.png")
print("✓ fig7_conformity_or")
plt.close(fig7)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Combined — Tc distribution (left) + URSP & trace-length (right)
# ═══════════════════════════════════════════════════════════════════════════

# ── Load Tc and URSP data from CSVs ──────────────────────────────────────
TC_CSV = os.path.join(
    REPO_ROOT, "Comparing_Experiments", "publication_V2",
    "mechanistic", "tc_summary_by_variant.csv",
)
URSP_CSV = os.path.join(
    REPO_ROOT, "Comparing_Experiments", "publication_V2",
    "mechanistic", "ursp_by_variant.csv",
)

FIG3_VAR_MAP = {
    "Base":     "base",
    "SFT":      "instruct_sft",
    "DPO":      "instruct_dpo",
    "Instruct": "instruct",
}

# Parse Tc summary
tc_rows = {}
with open(TC_CSV, newline="") as f:
    for row in csv.DictReader(f):
        tc_rows[row["variant"]] = row

# Parse URSP
ursp_rows = {}
with open(URSP_CSV, newline="") as f:
    for row in csv.DictReader(f):
        ursp_rows[row["variant"]] = row

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.2),
                               gridspec_kw={"wspace": 0.4, "bottom": 0.22})

# ── Left panel: Tc distribution as stacked horizontal bars ──
# Build Tc distribution from CSV counts
tc_data = {}
for var_fig, var_csv in FIG3_VAR_MAP.items():
    row = tc_rows[var_csv]
    counts = [float(row[f"tc_at_{t}"]) for t in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]]
    total = sum(counts)
    tc_data[var_fig] = [100.0 * c / total for c in counts]

temps = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
temp_colors = ["#2c3e50", "#34495e", "#5d6d7e", "#85929e", "#aeb6bf", "#d5d8dc"]

bar_height = 0.45
y_pos = np.arange(len(VARIANT_ORDER)) * 1.3  # extra vertical spacing
for i, var in enumerate(VARIANT_ORDER):
    left = 0
    for j, (pct, tc_col) in enumerate(zip(tc_data[var], temp_colors)):
        bar = ax1.barh(y_pos[i], pct, left=left, height=bar_height,
                       color=tc_col, edgecolor="white", linewidth=0.3)
        if pct > 8:
            ax1.text(left + pct/2, y_pos[i], f"{pct:.0f}%",
                     ha="center", va="center", fontsize=6, color="white" if j < 3 else "#333")
        left += pct

# Mean Tc with 95% CI — placed below each bar to avoid overlap
# Scale: Tc ∈ [0,1] maps to x ∈ [0,100] (percentage axis)
tc_row_offset = 0.38  # vertical offset below bar center
for i, var in enumerate(VARIANT_ORDER):
    row = tc_rows[FIG3_VAR_MAP[var]]
    mean_tc = float(row["tc_mean"])
    ci_lo   = float(row["tc_mean_ci_lower"])
    ci_hi   = float(row["tc_mean_ci_upper"])
    x_mean = mean_tc * 100
    x_lo   = (mean_tc - ci_lo) * 100
    x_hi   = (ci_hi - mean_tc) * 100
    y_marker = y_pos[i] + tc_row_offset
    ax1.errorbar(x_mean, y_marker, xerr=[[x_lo], [x_hi]],
                 fmt="D", color=COLORS[var], markersize=3.5,
                 markeredgecolor="white", markeredgewidth=0.5,
                 ecolor=COLORS[var], elinewidth=1.0, capsize=2.5, capthick=0.7,
                 zorder=5)
    ax1.text(x_mean + x_hi + 1.0, y_marker,
             f"$\\bar{{T}}_c$={mean_tc:.2f}",
             va="center", fontsize=5.5, color=COLORS[var], fontweight="bold")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(VARIANT_ORDER, fontsize=8)
ax1.set_xlabel("% of conforming items")
ax1.set_title("$T_c$ distribution", fontsize=9, pad=6)
ax1.invert_yaxis()

# Legend below the left panel, anchored to the axes
from matplotlib.patches import Patch
tc_legend = [Patch(facecolor=c, label=f"$T_c$={t}") for c, t in zip(temp_colors, temps)]
ax1.legend(handles=tc_legend, loc="upper center", bbox_to_anchor=(0.5, -0.15),
           fontsize=6, frameon=False, ncol=6, handlelength=0.8, columnspacing=0.6)

# ── Right panel: URSP rate with 95% bootstrap CI + trace-length direction ──
# Order: Base, SFT, DPO, Instruct (chronological)
ursp_rates = []
ursp_ci_lo = []
ursp_ci_hi = []
for var in VARIANT_ORDER:
    row = ursp_rows[FIG3_VAR_MAP[var]]
    rate = float(row["ursp_given_conforming"]) * 100
    lo   = float(row["ursp_ci_lower"]) * 100
    hi   = float(row["ursp_ci_upper"]) * 100
    ursp_rates.append(rate)
    ursp_ci_lo.append(rate - lo)
    ursp_ci_hi.append(hi - rate)

trace_d = [1.25, -0.66, 0.03, -0.35]   # Cohen's d (positive = longer when conforming)

bars = ax2.bar(np.arange(4), ursp_rates,
               yerr=[ursp_ci_lo, ursp_ci_hi],
               capsize=3, error_kw=dict(lw=0.8, capthick=0.8, color="#444"),
               color=[COLORS[v] for v in VARIANT_ORDER],
               edgecolor="white", linewidth=0.5, width=0.6)

# Add trace-length annotation arrows + percentage labels
for i, (bar, d_val) in enumerate(zip(bars, trace_d)):
    ci_top = ursp_rates[i] + ursp_ci_hi[i]
    if abs(d_val) > 0.1:
        direction = "↑" if d_val > 0 else "↓"
        label = f"trace {direction}"
        color = "#c0392b" if d_val > 0 else "#27ae60"
        ax2.text(bar.get_x() + bar.get_width()/2, ci_top + 2.0,
                 label, ha="center", va="bottom", fontsize=6, color=color)
    ax2.text(bar.get_x() + bar.get_width()/2, ci_top + 0.3,
             f"{ursp_rates[i]:.1f}%", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(["Base", "SFT", "DPO", "Inst."], fontsize=8)
ax2.set_ylabel("URSP rate (%)")
ax2.set_ylim(0, 35)
ax2.set_title("Unfaithful reasoning", fontsize=9, pad=6)

fig.savefig("paper/figures/fig3_tc_ursp.pdf")
fig.savefig("paper/figures/fig3_tc_ursp.png")
print("✓ fig3_tc_ursp")
plt.close(fig)

print("\nAll figures generated in paper/figures/")
