#!/usr/bin/env python3
"""Generate publication-quality figures for the COLM 2026 paper.

Produces:
  figures/fig2_conformity_or.pdf   – McNemar OR across training pipeline
  figures/fig3_tc_ursp.pdf         – Combined Tc distribution + URSP bar
"""

import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: McNemar OR across conditions — the headline result
# ═══════════════════════════════════════════════════════════════════════════

# ── Load McNemar data from CSV ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MCNEMAR_CSV = os.path.join(
    REPO_ROOT,
    "Comparing_Experiments", "publication_V2_column",
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
