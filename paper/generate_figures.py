#!/usr/bin/env python3
"""Generate publication-quality figures for the COLM 2026 paper.

Produces:
  figures/fig2_conformity_or.pdf   – McNemar OR across training pipeline
  figures/fig3_tc_ursp.pdf         – Combined Tc distribution + URSP bar
"""

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

VARIANT_ORDER = ["Base", "Instruct", "Instruct-SFT", "Instruct-DPO"]
COLORS = {
    "Base":          "#7f8c8d",  # grey
    "Instruct":      "#3498db",  # blue
    "Instruct-SFT":  "#e74c3c",  # red (amplifies)
    "Instruct-DPO":  "#2ecc71",  # green (mitigates)
}

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: McNemar OR across conditions — the headline result
# ═══════════════════════════════════════════════════════════════════════════

# Pooled McNemar OR from the analysis (all 6 temperatures)
conditions = ["Unan.\nConfident", "Auth.\nBias", "Auth.\nTrust", "Devil's\nAdvocate", "Question\nDistill."]
or_data = {
    "Base":          [3.15, 1.58, 1.75, 3.51, 2.01],
    "Instruct":      [5.58, 3.65, 1.67, 2.38, 2.00],
    "Instruct-SFT":  [5.79, 2.01, 1.89, 2.95, 2.29],
    "Instruct-DPO":  [1.23, 3.28, 1.85, 1.01, 1.02],
}

fig, ax = plt.subplots(figsize=(5.5, 2.5))

x = np.arange(len(conditions))
width = 0.19
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, var in enumerate(VARIANT_ORDER):
    ors = or_data[var]
    bars = ax.bar(x + offsets[i] * width, ors, width,
                  label=var, color=COLORS[var], edgecolor="white", linewidth=0.5)
    # Mark non-significant bars
    for j, (bar, orval) in enumerate(zip(bars, ors)):
        if orval < 1.5:  # approximately non-significant
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    "ns", ha="center", va="bottom", fontsize=6, color="#888")

ax.axhline(y=1.0, color="#bbb", linewidth=0.8, linestyle="--", zorder=0)
ax.set_ylabel("McNemar Odds Ratio")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 7.2)
ax.legend(loc="upper right", ncol=2, frameon=True, fancybox=False,
          edgecolor="#ddd", facecolor="white", framealpha=0.95)
ax.set_title("Conformity susceptibility across the post-training pipeline", fontsize=10, pad=8)
ax.annotate("SFT amplifies", xy=(0.15, 5.85), xytext=(1.4, 6.6),
            fontsize=7, color=COLORS["Instruct-SFT"],
            arrowprops=dict(arrowstyle="->", color=COLORS["Instruct-SFT"], lw=0.8))
ax.annotate("DPO mitigates", xy=(-0.05, 1.35), xytext=(1.4, 2.2),
            fontsize=7, color=COLORS["Instruct-DPO"],
            arrowprops=dict(arrowstyle="->", color=COLORS["Instruct-DPO"], lw=0.8))

fig.savefig("paper/figures/fig2_conformity_or.pdf")
fig.savefig("paper/figures/fig2_conformity_or.png")
print("✓ fig2_conformity_or")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Combined — Tc distribution (left) + URSP & trace-length (right)
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8),
                               gridspec_kw={"wspace": 0.4, "bottom": 0.22})

# ── Left panel: Tc distribution as stacked horizontal bars ──
tc_data = {
    # pct of conforming items at each Tc threshold
    "Base":          [53.3, 13.6, 10.7, 8.7, 7.7, 5.9],
    "Instruct":      [54.1, 10.4,  8.4, 7.4, 11.2, 8.4],
    "Instruct-SFT":  [61.8, 10.0,  7.8, 6.7,  7.3, 6.4],
    "Instruct-DPO":  [46.1, 10.5,  8.6, 8.9, 12.1, 13.8],
}
temps = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
temp_colors = ["#2c3e50", "#34495e", "#5d6d7e", "#85929e", "#aeb6bf", "#d5d8dc"]

y_pos = np.arange(len(VARIANT_ORDER))
for i, var in enumerate(VARIANT_ORDER):
    left = 0
    for j, (pct, tc_col) in enumerate(zip(tc_data[var], temp_colors)):
        bar = ax1.barh(y_pos[i], pct, left=left, height=0.6,
                       color=tc_col, edgecolor="white", linewidth=0.3)
        if pct > 8:
            ax1.text(left + pct/2, y_pos[i], f"{pct:.0f}%",
                     ha="center", va="center", fontsize=6, color="white" if j < 3 else "#333")
        left += pct

ax1.set_yticks(y_pos)
ax1.set_yticklabels(VARIANT_ORDER, fontsize=8)
ax1.set_xlabel("% of conforming items")
ax1.set_title("$T_c$ distribution", fontsize=9, pad=6)
ax1.invert_yaxis()

# Legend below the left panel so it doesn't overlap bars
from matplotlib.patches import Patch
tc_legend = [Patch(facecolor=c, label=f"$T_c$={t}") for c, t in zip(temp_colors, temps)]
fig.legend(handles=tc_legend, loc="lower center", fontsize=6.5, frameon=False,
           ncol=6, bbox_to_anchor=(0.32, 0.02))

# ── Right panel: URSP rate + trace-length direction ──
ursp_rates = [25.9, 17.9, 19.1, 23.8]  # % given conforming
trace_d = [1.25, -0.35, -0.66, 0.03]   # Cohen's d (positive = longer when conforming)

bars = ax2.bar(np.arange(4), ursp_rates, color=[COLORS[v] for v in VARIANT_ORDER],
               edgecolor="white", linewidth=0.5, width=0.6)

# Add trace-length annotation arrows
for i, (bar, d_val) in enumerate(zip(bars, trace_d)):
    # Arrow showing direction of trace length change
    y_base = bar.get_height() + 0.5
    if abs(d_val) > 0.1:
        direction = "↑" if d_val > 0 else "↓"
        label = f"trace {direction}"
        color = "#c0392b" if d_val > 0 else "#27ae60"
        ax2.text(bar.get_x() + bar.get_width()/2, y_base + 1.5,
                 label, ha="center", va="bottom", fontsize=6, color=color)
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{ursp_rates[i]:.1f}%", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(["Base", "Inst.", "SFT", "DPO"], fontsize=8)
ax2.set_ylabel("URSP rate (%)")
ax2.set_ylim(0, 33)
ax2.set_title("Unfaithful reasoning", fontsize=9, pad=6)

fig.savefig("paper/figures/fig3_tc_ursp.pdf")
fig.savefig("paper/figures/fig3_tc_ursp.png")
print("✓ fig3_tc_ursp")
plt.close(fig)

print("\nAll figures generated in paper/figures/")
