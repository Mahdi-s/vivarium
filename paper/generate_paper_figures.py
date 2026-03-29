#!/usr/bin/env python3
"""
Generate publication-quality figures for the updated CoLM 2026 paper.

Produces:
  - fig3_cross_family_forest.pdf  (Forest plot: 14 models ranked by Δpeer)
  - fig4_refusal_endorsement.pdf  (Scatter: behavioral taxonomy)
  - fig5_peer_vs_authority.pdf    (Grouped bars: peer vs authority)

Data sources:
  - Comparing_Experiments/expanded_results/bridge/tables/calibrated_ranking.csv
  - Comparing_Experiments/expanded_results/cross_family/tables/conformity_ranking.csv
  - Comparing_Experiments/expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Global Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
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
    "lines.linewidth": 1.0,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ─── Color Scheme (colorblind-safe) ────────────────────────────────
C_DENSE = "#E74C3C"      # coral red
C_MOE = "#1ABC9C"        # teal
C_THINK = "#8E44AD"      # purple
C_OLMO = "#7F8C8D"       # grey
C_OLMO_STAGES = {
    "base": "#BDC3C7",
    "instruct_sft": "#E74C3C",
    "instruct_dpo": "#2ECC71",
    "instruct": "#3498DB",
}

ARCH_COLORS = {
    "Llama-3.1-70B": C_DENSE,
    "Llama-3-8B": C_DENSE,
    "OLMo-32B-Instruct": C_DENSE,
    "GPT-4o-Mini": C_DENSE,
    "Gemini-2.5-Flash-Lite": C_MOE,
    "Llama-4-Maverick": C_MOE,
    "GPT-OSS-20B": C_MOE,
    "OLMo-7B-Think": C_THINK,
    "OLMo-32B-Think": C_THINK,
    "Grok-4.1-Fast": C_THINK,
    # OLMo stages (from bridge)
    "OLMo-7B": C_OLMO,
    "Olmo-3-7B-Instruct": C_OLMO_STAGES["instruct"],
    "Olmo-3-7B-Instruct-SFT": C_OLMO_STAGES["instruct_sft"],
    "Olmo-3-7B-Instruct-DPO": C_OLMO_STAGES["instruct_dpo"],
}

ARCH_LABELS = {
    "Llama-3.1-70B": "Dense", "Llama-3-8B": "Dense",
    "OLMo-32B-Instruct": "Dense", "GPT-4o-Mini": "Dense",
    "Gemini-2.5-Flash-Lite": "MoE", "Llama-4-Maverick": "MoE", "GPT-OSS-20B": "MoE",
    "OLMo-7B-Think": "Think", "OLMo-32B-Think": "Think", "Grok-4.1-Fast": "Think",
}

def sig_stars(p):
    if p is None or np.isnan(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Cross-Family Forest Plot
# ═══════════════════════════════════════════════════════════════════
def make_forest_plot():
    bridge = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/bridge/tables/calibrated_ranking.csv")
    mcnemar = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control.csv")

    # Get peer McNemar ORs (average across temps per model)
    peer_mcn = mcnemar[mcnemar["condition"] == "asch_zhu_unanimous_confident"]
    mcn_avg = peer_mcn.groupby("short_name").agg(
        OR_mean=("odds_ratio", "mean"),
        p_min=("p_holm", "min"),
    ).reset_index()

    # Merge
    df = bridge.merge(mcn_avg, on="short_name", how="left")

    # For OLMo stages, we don't have cross-family McNemar — compute from bridge data
    # Fill missing with NaN
    df = df.sort_values("delta_mean", ascending=True).reset_index(drop=True)

    # Rename for display
    display_names = {
        "Olmo-3-7B-Instruct": "OLMo-7B Instruct",
        "Olmo-3-7B-Instruct-SFT": "OLMo-7B SFT",
        "Olmo-3-7B-Instruct-DPO": "OLMo-7B DPO",
        "OLMo-7B": "OLMo-7B Base",
    }
    df["display"] = df["short_name"].map(lambda x: display_names.get(x, x))

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

    y = np.arange(len(df))
    colors = [ARCH_COLORS.get(row["short_name"], C_OLMO) for _, row in df.iterrows()]

    # Horizontal bars
    bars = ax.barh(y, df["delta_mean"], height=0.55, color=colors,
                   edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)

    # Zero line
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4, zorder=2)

    # Separator between OLMo stages and cross-family
    # OLMo stages are the ones with "OLMo-7B" in display name
    olmo_idx = [i for i, row in df.iterrows() if "OLMo-7B" in row["display"]]
    if olmo_idx:
        sep_y = max(olmo_idx) + 0.5
        ax.axhline(sep_y, color="grey", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
        ax.text(0.52, sep_y + 0.15, "── OLMo-7B training stages ──",
                fontsize=6, color="grey", ha="center", va="bottom", style="italic",
                transform=ax.get_yaxis_transform())

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=7.5)

    # Right-side annotations: OR and significance
    for i, (_, row) in enumerate(df.iterrows()):
        delta = row["delta_mean"]
        or_val = row.get("OR_mean")
        p_val = row.get("p_min")

        # Delta value on the bar
        x_text = delta + 0.008 if delta >= 0 else delta - 0.008
        ha = "left" if delta >= 0 else "right"
        ax.text(x_text, i, f"{delta:+.3f}", va="center", ha=ha,
                fontsize=6.5, fontweight="bold", zorder=5)

        # OR + significance on right margin
        if or_val is not None and not np.isnan(or_val):
            stars = sig_stars(p_val)
            or_text = f"OR={or_val:.1f}" if or_val < 100 else f"OR={or_val:.0f}"
            ax.text(0.58, i, f"{or_text}  {stars}",
                    fontsize=6, va="center", ha="left", fontfamily="monospace",
                    transform=ax.get_yaxis_transform(),
                    color="black" if stars != "ns" else "#999999")

    ax.set_xlabel("Peer Pressure Effect (Δ error rate)", fontsize=9)
    ax.set_title("Cross-Family Conformity Under Structured Peer Consensus", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.05, 0.58)
    ax.grid(axis="x", alpha=0.2, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
        Patch(facecolor=C_OLMO, edgecolor="black", linewidth=0.4, label="OLMo-7B Stages"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              framealpha=0.9, edgecolor="grey")

    fig.savefig(OUT / "fig3_cross_family_forest.pdf")
    fig.savefig(OUT / "fig3_cross_family_forest.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig3_cross_family_forest.pdf")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Refusal vs Endorsement Scatter
# ═══════════════════════════════════════════════════════════════════
def make_scatter():
    ranking = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/conformity_ranking.csv")
    mcnemar = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control.csv")

    # Get avg McNemar OR per model for peer condition
    peer_mcn = mcnemar[mcnemar["condition"] == "asch_zhu_unanimous_confident"]
    mcn_avg = peer_mcn.groupby("short_name").agg(OR=("odds_ratio", "mean"), p=("p_holm", "min")).reset_index()
    df = ranking.merge(mcn_avg, on="short_name", how="left")

    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)

    # Quadrant lines
    ax.axhline(30, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(15, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)

    # Quadrant labels
    ax.text(35, 2, "Sycophantic\nCompliance", fontsize=6, color="#999", ha="center", va="bottom", style="italic")
    ax.text(4, 75, "Safety-Driven\nAvoidance", fontsize=6, color="#999", ha="center", va="center", style="italic")
    ax.text(4, 2, "Epistemic\nResistance", fontsize=6, color="#999", ha="center", va="bottom", style="italic")

    colors = [ARCH_COLORS.get(row["short_name"], C_OLMO) for _, row in df.iterrows()]
    sizes = np.abs(df["delta_mean"]) * 600 + 30

    scatter = ax.scatter(
        df["endorsed_mean"] * 100, df["refusal_mean"] * 100,
        s=sizes, c=colors, edgecolor="black", linewidth=0.5,
        alpha=0.85, zorder=5,
    )

    # Labels with manual offsets for readability
    offsets = {
        "Llama-3.1-70B": (5, -5), "Llama-3-8B": (5, -3),
        "GPT-4o-Mini": (-3, -8), "OLMo-32B-Instruct": (5, 3),
        "Gemini-2.5-Flash-Lite": (5, -5), "Llama-4-Maverick": (5, 5),
        "Grok-4.1-Fast": (5, 3), "GPT-OSS-20B": (-3, 5),
        "OLMo-7B-Think": (5, -5), "OLMo-32B-Think": (-5, 8),
    }

    for _, row in df.iterrows():
        name = row["short_name"]
        ox, oy = offsets.get(name, (5, 3))
        or_val = row.get("OR")
        p_val = row.get("p")
        stars = sig_stars(p_val) if p_val is not None else ""

        label = name
        if or_val is not None and not np.isnan(or_val):
            label += f"\nOR={or_val:.1f} {stars}"

        ax.annotate(label,
                    (row["endorsed_mean"] * 100, row["refusal_mean"] * 100),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=5.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.3, alpha=0.5) if abs(ox) > 3 else None,
                    zorder=6)

    ax.set_xlabel("Wrong Answer Endorsement Rate (%)", fontsize=9)
    ax.set_ylabel("Refusal Rate (%)", fontsize=9)
    ax.set_title("Behavioral Response Taxonomy\nUnder Peer Consensus Pressure", fontsize=10, fontweight="bold")
    ax.set_xlim(-2, 52)
    ax.set_ylim(-3, 95)
    ax.grid(alpha=0.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, framealpha=0.9)

    fig.savefig(OUT / "fig4_refusal_endorsement.pdf")
    fig.savefig(OUT / "fig4_refusal_endorsement.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig4_refusal_endorsement.pdf")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Peer vs Authority Grouped Bars
# ═══════════════════════════════════════════════════════════════════
def make_peer_vs_auth():
    ranking = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/conformity_ranking.csv")
    mcnemar = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control.csv")

    # Sort by peer delta
    df = ranking.sort_values("delta_mean", ascending=False).reset_index(drop=True)

    # Get peer significance per model (min p across temps)
    peer_mcn = mcnemar[mcnemar["condition"] == "asch_zhu_unanimous_confident"]
    peer_sig = peer_mcn.groupby("short_name").agg(p_peer=("p_holm", "min")).reset_index()

    # Get authority significance (authoritative_bias)
    auth_mcn = mcnemar[mcnemar["condition"] == "authoritative_bias"]
    auth_sig = auth_mcn.groupby("short_name").agg(p_auth=("p_holm", "min")).reset_index()

    df = df.merge(peer_sig, on="short_name", how="left")
    df = df.merge(auth_sig, on="short_name", how="left")

    fig, ax = plt.subplots(figsize=(7, 3.0), constrained_layout=True)

    x = np.arange(len(df))
    w = 0.35
    colors_peer = [ARCH_COLORS.get(row["short_name"], C_OLMO) for _, row in df.iterrows()]
    colors_auth = [plt.cm.Greys(0.35)] * len(df)

    # Peer bars
    peer_bars = ax.bar(x - w/2, df["delta_mean"], w, color=colors_peer,
                       edgecolor="black", linewidth=0.4, label="Peer Consensus", zorder=3)

    # Authority bars
    auth_vals = df["delta_auth_mean"].fillna(0)
    auth_bars = ax.bar(x + w/2, auth_vals, w, color=colors_auth,
                       edgecolor="black", linewidth=0.4, label="Authority Framing", zorder=3)

    # Significance annotations
    for i, (_, row) in enumerate(df.iterrows()):
        # Peer stars
        p_peer = row.get("p_peer")
        stars = sig_stars(p_peer)
        y_peer = row["delta_mean"] + 0.012
        ax.text(i - w/2, y_peer, stars, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
                color="black" if stars != "ns" else "#AAAAAA")

        # Authority: always ns
        y_auth = max(auth_vals.iloc[i], 0) + 0.012
        ax.text(i + w/2, y_auth, "ns", ha="center", va="bottom",
                fontsize=5, color="#AAAAAA", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_name"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Pressure Effect (Δ error rate)", fontsize=9)
    ax.set_title("Peer Consensus vs Authority Framing Across Model Families", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.12, 0.58)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(OUT / "fig5_peer_vs_authority.pdf")
    fig.savefig(OUT / "fig5_peer_vs_authority.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig5_peer_vs_authority.pdf")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    make_forest_plot()
    make_scatter()
    make_peer_vs_auth()
    print("Done!")
