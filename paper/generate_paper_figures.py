#!/usr/bin/env python3
"""
Generate publication-quality figures for the CoLM 2026 paper.

Produces:
  - fig3_cross_family_forest.pdf  (Forest plot: cross-family + OLMo bridge ranked by Δpeer)
  - fig4_refusal_endorsement.pdf  (Scatter: behavioral taxonomy)
  - fig5_peer_vs_authority.pdf    (Grouped bars: peer vs authority)

Data sources (corrected, includes Claude Sonnet 4):
  - cross_family/statistical_tests/mcnemar_pressure_vs_control_t0.csv  (fixed N=400, Holm-corrected)
  - cross_family/tables/pressure_effects_t0.csv                       (endorsement/refusal at T=0.0)
  - bridge/tables/calibrated_ranking.csv                              (OLMo-7B training stages)
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
C_CONST = "#2980B9"      # blue (Constitutional AI)
C_OLMO = "#7F8C8D"       # grey (OLMo-7B stages)
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
    "Claude-Sonnet-4": C_CONST,
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
    "Claude-Sonnet-4": "Const. AI",
}


def sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "ns"
    if isinstance(p, str):
        if p == "ns":
            return "ns"
        return p  # already "***", "**", "*"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_t0_data():
    """Load the authoritative T=0.0 data with Claude Sonnet 4."""
    mcn = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/statistical_tests/mcnemar_pressure_vs_control_t0.csv")
    pe = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/pressure_effects_t0.csv")
    bridge = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/bridge/tables/calibrated_ranking.csv")
    return mcn, pe, bridge


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Cross-Family Forest Plot (T=0.0, fixed N=400)
# ═══════════════════════════════════════════════════════════════════
def make_forest_plot():
    mcn, pe, bridge = load_t0_data()

    # --- Cross-family models from T=0.0 McNemar (peer condition) ---
    peer_mcn = mcn[mcn["condition"] == "asch_zhu_unanimous_confident"].copy()
    # Use the T=0.0 delta and OR directly (no averaging across temps)
    cross = peer_mcn[["model_short", "delta_error", "odds_ratio", "p_adjusted", "sig_adjusted"]].copy()
    cross.columns = ["short_name", "delta_mean", "OR_mean", "p_adj", "sig"]

    # --- OLMo-7B bridge stages (not in cross-family McNemar) ---
    olmo_stages = bridge[bridge["short_name"].str.contains("Olmo-3-7B|OLMo-7B")].copy()
    olmo_stages = olmo_stages[["short_name", "delta_mean"]].copy()
    olmo_stages["OR_mean"] = np.nan
    olmo_stages["p_adj"] = np.nan
    olmo_stages["sig"] = ""

    # Combine
    df = pd.concat([cross, olmo_stages], ignore_index=True)
    df = df.sort_values("delta_mean", ascending=True).reset_index(drop=True)

    # Display names
    display_names = {
        "Olmo-3-7B-Instruct": "OLMo-7B Instruct",
        "Olmo-3-7B-Instruct-SFT": "OLMo-7B SFT",
        "Olmo-3-7B-Instruct-DPO": "OLMo-7B DPO",
        "OLMo-7B": "OLMo-7B Base",
        "Claude-Sonnet-4": "Claude Sonnet 4",
    }
    df["display"] = df["short_name"].map(lambda x: display_names.get(x, x))

    fig, ax = plt.subplots(figsize=(7, 5.0), constrained_layout=True)

    y = np.arange(len(df))
    colors = [ARCH_COLORS.get(row["short_name"], C_OLMO) for _, row in df.iterrows()]

    # Horizontal bars
    bars = ax.barh(y, df["delta_mean"], height=0.55, color=colors,
                   edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)

    # Zero line
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4, zorder=2)

    # Separator between OLMo stages and cross-family
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

    # --- Two-column annotation layout to avoid overlap ---
    # Delta labels go at the end of each bar.
    # OR + significance go in a fixed right-margin column outside the plot area.
    for i, (_, row) in enumerate(df.iterrows()):
        delta = row["delta_mean"]
        or_val = row.get("OR_mean")
        sig = row.get("sig", "")

        # Delta value: placed just past the bar tip (or inside for very long bars)
        if delta >= 0.35:
            # Long bar — put delta INSIDE the bar to avoid collision with OR column
            ax.text(delta - 0.008, i, f"{delta:+.3f}", va="center", ha="right",
                    fontsize=6.5, fontweight="bold", color="white", zorder=5)
        elif delta >= 0:
            ax.text(delta + 0.008, i, f"{delta:+.3f}", va="center", ha="left",
                    fontsize=6.5, fontweight="bold", zorder=5)
        else:
            ax.text(delta - 0.008, i, f"{delta:+.3f}", va="center", ha="right",
                    fontsize=6.5, fontweight="bold", zorder=5)

        # OR + significance in a FIXED right column (axes fraction = 1.02, outside plot)
        if or_val is not None and not np.isnan(or_val):
            stars = sig_stars(sig) if sig else sig_stars(row.get("p_adj"))
            or_text = f"OR={or_val:.1f}" if or_val < 100 else f"OR={or_val:.0f}"
            ax.annotate(f"{or_text}  {stars}",
                        xy=(1.02, i), xycoords=("axes fraction", "data"),
                        fontsize=6, va="center", ha="left", fontfamily="monospace",
                        annotation_clip=False,
                        color="black" if stars != "ns" else "#999999")

    ax.set_xlabel(r"Peer Pressure Effect ($\Delta$ error rate, $T{=}0.0$, fixed $N{=}400$)", fontsize=9)
    ax.set_title("Cross-Family Conformity Under Structured Peer Consensus", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.08, 0.60)
    ax.grid(axis="x", alpha=0.2, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
        Patch(facecolor=C_CONST, edgecolor="black", linewidth=0.4, label="Constitutional AI"),
        Patch(facecolor=C_OLMO, edgecolor="black", linewidth=0.4, label="OLMo-7B Stages"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              framealpha=0.9, edgecolor="grey")

    fig.savefig(OUT / "fig3_cross_family_forest.pdf")
    fig.savefig(OUT / "fig3_cross_family_forest.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig3_cross_family_forest.pdf  ({len(df)} models)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Refusal vs Endorsement Scatter (T=0.0, fixed N=400)
# ═══════════════════════════════════════════════════════════════════
def make_scatter():
    mcn, pe, _ = load_t0_data()

    # Get peer-condition pressure effects at T=0.0 (endorsement/refusal rates)
    peer_pe = pe[pe["condition"] == "asch_zhu_unanimous_confident"].copy()
    peer_pe = peer_pe[["model_short", "pressure_endorsement_rate", "pressure_refusal_rate"]].copy()

    # Get peer McNemar OR, significance, and delta at T=0.0 (fixed-N=400 metric)
    peer_mcn = mcn[mcn["condition"] == "asch_zhu_unanimous_confident"].copy()
    peer_mcn = peer_mcn[["model_short", "odds_ratio", "delta_error", "p_adjusted", "sig_adjusted"]].copy()

    df = peer_pe.merge(peer_mcn, on="model_short", how="left")

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)

    # Quadrant lines
    ax.axhline(30, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(15, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)

    # Quadrant labels — positioned in far corners to avoid overlapping data points
    ax.text(0.95, 0.02, "Sycophantic\nCompliance", fontsize=7, color="#BBBBBB",
            ha="right", va="bottom", style="italic", fontweight="bold",
            transform=ax.transAxes, zorder=1)
    ax.text(0.03, 0.95, "Safety-Driven\nAvoidance", fontsize=7, color="#BBBBBB",
            ha="left", va="top", style="italic", fontweight="bold",
            transform=ax.transAxes, zorder=1)
    ax.text(0.03, 0.02, "Epistemic\nResistance", fontsize=7, color="#BBBBBB",
            ha="left", va="bottom", style="italic", fontweight="bold",
            transform=ax.transAxes, zorder=1)

    colors = [ARCH_COLORS.get(row["model_short"], C_OLMO) for _, row in df.iterrows()]
    sizes = np.abs(df["delta_error"]) * 600 + 40

    scatter = ax.scatter(
        df["pressure_endorsement_rate"] * 100, df["pressure_refusal_rate"] * 100,
        s=sizes, c=colors, edgecolor="black", linewidth=0.5,
        alpha=0.85, zorder=5,
    )

    # Labels with manual offsets for readability
    offsets = {
        "Llama-3.1-70B": (5, -5), "Llama-3-8B": (5, -5),
        "GPT-4o-Mini": (-3, -8), "OLMo-32B-Instruct": (5, 3),
        "Gemini-2.5-Flash-Lite": (5, -5), "Llama-4-Maverick": (5, 5),
        "Grok-4.1-Fast": (5, 3), "GPT-OSS-20B": (-3, 5),
        "OLMo-32B-Think": (-5, 8),
        "Claude-Sonnet-4": (-5, -8),
    }

    for _, row in df.iterrows():
        name = row["model_short"]
        ox, oy = offsets.get(name, (5, 3))
        or_val = row.get("odds_ratio")
        sig = row.get("sig_adjusted", "")

        stars = sig_stars(sig)
        label = name
        if or_val is not None and not np.isnan(or_val):
            or_str = f"{or_val:.1f}" if or_val < 100 else f"{or_val:.0f}"
            label += f"\nOR={or_str} {stars}"

        ax.annotate(label,
                    (row["pressure_endorsement_rate"] * 100, row["pressure_refusal_rate"] * 100),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=5.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.3, alpha=0.5) if abs(ox) > 3 else None,
                    zorder=6)

    ax.set_xlabel("Wrong Answer Endorsement Rate (%)", fontsize=9)
    ax.set_ylabel("Refusal Rate (%)", fontsize=9)
    ax.set_title("Behavioral Response Taxonomy Under Peer Consensus\n($T{=}0.0$, fixed $N{=}400$)", fontsize=10, fontweight="bold")
    ax.set_xlim(-2, 52)
    ax.set_ylim(-3, 95)
    ax.grid(alpha=0.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
        Patch(facecolor=C_CONST, edgecolor="black", linewidth=0.4, label="Constitutional AI"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, framealpha=0.9)

    fig.savefig(OUT / "fig4_refusal_endorsement.pdf")
    fig.savefig(OUT / "fig4_refusal_endorsement.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig4_refusal_endorsement.pdf  ({len(df)} models)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Peer vs Authority Grouped Bars (T=0.0, fixed N=400)
# ═══════════════════════════════════════════════════════════════════
def make_peer_vs_auth():
    mcn, _, _ = load_t0_data()

    # --- Use McNemar deltas exclusively (fixed-N=400, refusals as State C) ---
    # Peer delta and significance at T=0.0
    peer = mcn[mcn["condition"] == "asch_zhu_unanimous_confident"][
        ["model_short", "delta_error", "p_adjusted", "sig_adjusted"]].copy()
    peer.columns = ["model_short", "delta_peer", "p_peer", "sig_peer"]

    # Authority delta and significance at T=0.0 (authoritative_bias)
    auth = mcn[mcn["condition"] == "authoritative_bias"][
        ["model_short", "delta_error", "p_adjusted", "sig_adjusted"]].copy()
    auth.columns = ["model_short", "delta_auth", "p_auth", "sig_auth"]

    # Merge
    df = peer.merge(auth, on="model_short", how="left")

    # Sort by peer delta descending
    df = df.sort_values("delta_peer", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)

    x = np.arange(len(df))
    w = 0.35
    colors_peer = [ARCH_COLORS.get(row["model_short"], C_OLMO) for _, row in df.iterrows()]
    colors_auth = [plt.cm.Greys(0.35)] * len(df)

    # Peer bars
    ax.bar(x - w / 2, df["delta_peer"], w, color=colors_peer,
           edgecolor="black", linewidth=0.4, label="Peer Consensus", zorder=3)

    # Authority bars
    auth_vals = df["delta_auth"].fillna(0)
    ax.bar(x + w / 2, auth_vals, w, color=colors_auth,
           edgecolor="black", linewidth=0.4, label="Authority Framing", zorder=3)

    # Significance annotations — use ACTUAL p-values from data
    for i, (_, row) in enumerate(df.iterrows()):
        # Peer stars
        stars_peer = sig_stars(row.get("sig_peer", "ns"))
        y_peer = max(row["delta_peer"], 0) + 0.012
        ax.text(i - w / 2, y_peer, stars_peer, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
                color="black" if stars_peer != "ns" else "#AAAAAA")

        # Authority stars — use actual Holm-corrected significance
        stars_auth = sig_stars(row.get("sig_auth", "ns"))
        y_auth = max(auth_vals.iloc[i], 0) + 0.012
        ax.text(i + w / 2, y_auth, stars_auth, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold" if stars_auth != "ns" else "normal",
                color="black" if stars_auth != "ns" else "#AAAAAA",
                style="normal" if stars_auth != "ns" else "italic")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model_short"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(r"Pressure Effect ($\Delta$ error rate)", fontsize=9)
    ax.set_title("Peer Consensus vs Authority Framing ($T{=}0.0$, Holm corrected)", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.15, 0.65)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(OUT / "fig5_peer_vs_authority.pdf")
    fig.savefig(OUT / "fig5_peer_vs_authority.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig5_peer_vs_authority.pdf  ({len(df)} models)")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures from T=0.0 corrected data...")
    make_forest_plot()
    make_scatter()
    make_peer_vs_auth()
    print("Done!")
