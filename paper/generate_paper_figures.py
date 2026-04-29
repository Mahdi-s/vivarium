#!/usr/bin/env python3
"""
Generate publication-quality figures for the CoLM 2026 paper.

Produces:
  - fig1_stacked_decomposition.pdf  (100% stacked bar: 3-state decomposition)
  - fig3_cross_family_forest.pdf    (Forest plot: cross-family + OLMo bridge ranked by Δpeer)
  - fig4_refusal_endorsement.pdf    (Scatter: behavioral taxonomy + OLMo trajectory)
  - fig5_peer_vs_authority.pdf      (Grouped bars: peer vs authority)

Data sources (corrected, includes Claude Sonnet 4):
  - cross_family/statistical_tests/mcnemar_pressure_vs_control_t0.csv  (fixed N=400, Holm-corrected)
  - cross_family/tables/pressure_effects_t0.csv                       (endorsement/refusal at T=0.0)
  - cross_family/tables/ablation_rates_t0.csv                         (ablation 3-state decomposition)
  - olmo_family/tables/multinomial_rates_t0.csv                       (OLMo-7B training stages)
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
# FIGURE 3: Diverging Butterfly Chart — Endorsement vs Refusal Change
# ═══════════════════════════════════════════════════════════════════
def make_forest_plot():
    """
    Diverging butterfly chart: each model gets two opposing bars.
    RIGHT (warm red) = Δ endorsement rate (actual conformity / wrong-answer adoption)
    LEFT  (cool grey) = Δ refusal rate   (safety-driven avoidance)
    Models ranked by Δ endorsement (descending) — the real conformity signal.
    """
    mcn, pe, bridge = load_t0_data()

    # --- Cross-family: pressure effects with endorsement/refusal breakdown ---
    peer_pe = pe[pe["condition"] == "asch_zhu_unanimous_confident"].copy()

    # Control endorsement/refusal from per-model metrics
    ctrl_metrics = pd.read_csv(
        ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/per_model_condition_metrics.csv"
    )
    ctrl = ctrl_metrics[
        (ctrl_metrics["condition"] == "control") & (ctrl_metrics["temperature"] == 0.0)
    ][["short_name", "endorsed_rate", "refusal_rate"]].copy()
    ctrl.columns = ["model_short", "ctrl_endorse", "ctrl_refusal"]

    # Merge: pressure endorsement/refusal + control endorsement/refusal
    df = peer_pe[["model_short", "pressure_endorsement_rate", "pressure_refusal_rate"]].merge(
        ctrl, on="model_short", how="left"
    )
    df["delta_endorse"] = df["pressure_endorsement_rate"] - df["ctrl_endorse"]
    df["delta_refusal"] = df["pressure_refusal_rate"] - df["ctrl_refusal"]

    # Add OLMo-7B-Think from per-model metrics (not in pressure_effects_t0)
    peer_7bt = ctrl_metrics[
        (ctrl_metrics["short_name"] == "OLMo-7B-Think")
        & (ctrl_metrics["condition"] == "asch_zhu_unanimous_confident")
        & (ctrl_metrics["temperature"] == 0.0)
    ]
    ctrl_7bt = ctrl_metrics[
        (ctrl_metrics["short_name"] == "OLMo-7B-Think")
        & (ctrl_metrics["condition"] == "control")
        & (ctrl_metrics["temperature"] == 0.0)
    ]
    if len(peer_7bt) > 0 and len(ctrl_7bt) > 0:
        row_7bt = pd.DataFrame([{
            "model_short": "OLMo-7B-Think",
            "pressure_endorsement_rate": peer_7bt.iloc[0]["endorsed_rate"],
            "pressure_refusal_rate": peer_7bt.iloc[0]["refusal_rate"],
            "ctrl_endorse": ctrl_7bt.iloc[0]["endorsed_rate"],
            "ctrl_refusal": ctrl_7bt.iloc[0]["refusal_rate"],
            "delta_endorse": peer_7bt.iloc[0]["endorsed_rate"] - ctrl_7bt.iloc[0]["endorsed_rate"],
            "delta_refusal": peer_7bt.iloc[0]["refusal_rate"] - ctrl_7bt.iloc[0]["refusal_rate"],
        }])
        df = pd.concat([df, row_7bt], ignore_index=True)

    # McNemar significance for annotation
    peer_mcn = mcn[mcn["condition"] == "asch_zhu_unanimous_confident"].copy()
    sig_map = dict(zip(peer_mcn["model_short"], peer_mcn["sig_adjusted"]))
    or_map = dict(zip(peer_mcn["model_short"], peer_mcn["odds_ratio"]))

    # Sort by Δ endorsement descending (most conformist at top)
    df = df.sort_values("delta_endorse", ascending=True).reset_index(drop=True)

    # --- OLMo-7B bridge stages ---
    olmo_mn = pd.read_csv(
        ROOT / "Comparing_Experiments/expanded_results/olmo_family/tables/multinomial_rates_t0.csv"
    )
    ctrl_olmo = olmo_mn[olmo_mn["condition"] == "control"].copy()
    peer_olmo = olmo_mn[olmo_mn["condition"] == "asch_zhu_unbiased_unanimous_confident"].copy()

    stage_order = ["base", "instruct_sft", "instruct_dpo", "instruct"]
    stage_display = {
        "base": "OLMo-7B Base", "instruct_sft": "OLMo-7B SFT",
        "instruct_dpo": "OLMo-7B DPO", "instruct": "OLMo-7B Instruct",
    }
    olmo_rows = []
    for stage in stage_order:
        c = ctrl_olmo[ctrl_olmo["variant"] == stage]
        p = peer_olmo[peer_olmo["variant"] == stage]
        if len(c) == 0 or len(p) == 0:
            continue
        olmo_rows.append({
            "model_short": stage_display[stage],
            "delta_endorse": p.iloc[0]["endorsement_rate"] - c.iloc[0]["endorsement_rate"],
            "delta_refusal": p.iloc[0]["refusal_rate"] - c.iloc[0]["refusal_rate"],
        })
    olmo_df = pd.DataFrame(olmo_rows)
    # Sort OLMo by delta_endorse ascending (to match main ranking direction)
    olmo_df = olmo_df.sort_values("delta_endorse", ascending=True).reset_index(drop=True)

    # Display names for cross-family
    display_names = {"Claude-Sonnet-4": "Claude Sonnet 4"}
    df["display"] = df["model_short"].map(lambda x: display_names.get(x, x))

    # ── Build figure ──
    n_cross = len(df)
    n_olmo = len(olmo_df)
    n_total = n_cross + n_olmo + 1  # +1 for separator gap
    fig, ax = plt.subplots(figsize=(7, 0.42 * n_total + 1.2), constrained_layout=True)

    # Y positions: cross-family at bottom, gap, OLMo stages at top
    y_cross = np.arange(n_cross)
    y_olmo = np.arange(n_cross + 1, n_cross + 1 + n_olmo)  # +1 gap

    bar_h = 0.38

    # --- Endorsement bars (RIGHT, warm red) ---
    endorse_colors = [ARCH_COLORS.get(row["model_short"], C_OLMO) for _, row in df.iterrows()]
    ax.barh(y_cross, df["delta_endorse"] * 100, height=bar_h,
            color=endorse_colors, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)

    # --- Refusal bars (LEFT for increase, transparent for decrease) ---
    # Positive delta_refusal = more refusals under pressure → show LEFT (grey, solid)
    # Negative delta_refusal = fewer refusals under pressure → skip (not safety-driven)
    refusal_vals = []
    for _, row in df.iterrows():
        if row["delta_refusal"] > 0:
            refusal_vals.append(-row["delta_refusal"] * 100)  # negative = leftward
        else:
            refusal_vals.append(0)  # no bar for refusal decrease
    ax.barh(y_cross, refusal_vals, height=bar_h,
            color="#95A5A6", edgecolor="black", linewidth=0.3, alpha=0.6, zorder=3)

    # --- OLMo-7B stages ---
    for j, (_, row) in enumerate(olmo_df.iterrows()):
        stage_name = row["model_short"]
        stage_key = {v: k for k, v in stage_display.items()}.get(stage_name, "base")
        c = C_OLMO_STAGES.get(stage_key, C_OLMO)
        ax.barh(y_olmo[j], row["delta_endorse"] * 100, height=bar_h,
                color=c, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
        ax.barh(y_olmo[j], -abs(row["delta_refusal"]) * 100, height=bar_h,
                color="#95A5A6", edgecolor="black", linewidth=0.3, alpha=0.6, zorder=3)

    # --- Zero line ---
    ax.axvline(0, color="black", linewidth=0.8, zorder=2)

    # --- Separator ---
    sep_y = n_cross + 0.5 - 0.5  # gap position
    ax.axhline(sep_y, color="grey", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
    ax.text(0, sep_y + 0.2, "── OLMo-7B training stages ──",
            fontsize=6, color="grey", ha="center", va="bottom", style="italic")

    # --- Y-axis labels ---
    all_labels = list(df["display"]) + [""] + list(olmo_df["model_short"])
    all_y = list(y_cross) + [sep_y] + list(y_olmo)
    ax.set_yticks(list(y_cross) + list(y_olmo))
    ax.set_yticklabels(list(df["display"]) + list(olmo_df["model_short"]), fontsize=7.5)

    # --- Annotations: Δ values + significance ---
    for i, (_, row) in enumerate(df.iterrows()):
        de = row["delta_endorse"] * 100
        dr = row["delta_refusal"] * 100

        # Endorsement value (right side)
        if abs(de) > 1:
            x_pos = de + (1.5 if de >= 0 else -1.5)
            ha = "left" if de >= 0 else "right"
            ax.text(x_pos, y_cross[i], f"{de:+.1f}%", va="center", ha=ha,
                    fontsize=6, fontweight="bold", zorder=5)

        # Refusal value (left side)
        if abs(dr) > 2:
            x_pos = -abs(dr) - 1.5
            ax.text(x_pos, y_cross[i], f"+{abs(dr):.0f}%", va="center", ha="right",
                    fontsize=5.5, color="#666666", zorder=5)

        # Significance in right margin
        sig = sig_map.get(row["model_short"], "")
        stars = sig_stars(sig)
        color = "black" if stars != "ns" else "#999999"
        ax.annotate(stars, xy=(1.02, y_cross[i]),
                    xycoords=("axes fraction", "data"),
                    fontsize=6.5, va="center", ha="left", fontfamily="monospace",
                    annotation_clip=False, color=color)

    # OLMo stage labels
    for j, (_, row) in enumerate(olmo_df.iterrows()):
        de = row["delta_endorse"] * 100
        if abs(de) > 1:
            x_pos = de + (1.5 if de >= 0 else -1.5)
            ha = "left" if de >= 0 else "right"
            ax.text(x_pos, y_olmo[j], f"{de:+.1f}%", va="center", ha=ha,
                    fontsize=6, fontweight="bold", color="#555555", zorder=5)

    # --- Axis labels ---
    ax.set_xlabel(
        r"$\longleftarrow$ $\Delta$ Refusal Rate (%)          "
        r"$\Delta$ Endorsement Rate (%) $\longrightarrow$",
        fontsize=8.5,
    )
    ax.set_title(
        "Decomposed Pressure Response: Endorsement vs. Refusal\n"
        r"(Structured Peer Consensus, $T{=}0.0$, Fixed $N{=}400$)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.15, zorder=0)

    # --- Legend ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
        Patch(facecolor=C_CONST, edgecolor="black", linewidth=0.4, label="Constitutional AI"),
        Patch(facecolor="#95A5A6", edgecolor="black", linewidth=0.3, alpha=0.6,
              label=r"$\Delta$ Refusal (left)"),
        Patch(facecolor=C_OLMO, edgecolor="black", linewidth=0.4, label="OLMo-7B Stages"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6,
              framealpha=0.9, edgecolor="grey")

    fig.savefig(OUT / "fig3_cross_family_forest.pdf")
    fig.savefig(OUT / "fig3_cross_family_forest.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig3_cross_family_forest.pdf  ({n_cross} cross-family + {n_olmo} OLMo stages)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Refusal vs Endorsement Scatter (T=0.0, fixed N=400)
# ═══════════════════════════════════════════════════════════════════
def make_scatter():
    mcn, pe, _ = load_t0_data()

    # Get peer-condition pressure effects at T=0.0 (endorsement/refusal rates)
    peer_pe = pe[pe["condition"] == "asch_zhu_unanimous_confident"].copy()
    peer_pe = peer_pe[["model_short", "pressure_endorsement_rate", "pressure_refusal_rate",
                        "control_refusal_rate"]].copy()

    # Control endorsement rates from per-model metrics
    ctrl_metrics = pd.read_csv(
        ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/per_model_condition_metrics.csv"
    )
    ctrl = ctrl_metrics[
        (ctrl_metrics["condition"] == "control") & (ctrl_metrics["temperature"] == 0.0)
    ][["short_name", "endorsed_rate"]].copy()
    ctrl.columns = ["model_short", "ctrl_endorse"]
    peer_pe = peer_pe.merge(ctrl, on="model_short", how="left")
    peer_pe["delta_endorse"] = peer_pe["pressure_endorsement_rate"] - peer_pe["ctrl_endorse"]

    # Get peer McNemar significance at T=0.0
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
    # Bubble size proportional to Δ endorsement (actual conformity), NOT Δ error
    sizes = np.abs(df["delta_endorse"]) * 1200 + 40

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
        sig = row.get("sig_adjusted", "")
        stars = sig_stars(sig)
        de = row["delta_endorse"]

        # Annotate with Δ endorsement (the real conformity metric) instead of OR
        label = name
        if de is not None and not np.isnan(de):
            label += f"\n$\\Delta$end={de*100:+.1f}% {stars}"

        ax.annotate(label,
                    (row["pressure_endorsement_rate"] * 100, row["pressure_refusal_rate"] * 100),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=5.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.3, alpha=0.5) if abs(ox) > 3 else None,
                    zorder=6)

    # --- OLMo-7B Training Trajectory Overlay ---
    olmo_mn = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/olmo_family/tables/multinomial_rates_t0.csv")
    olmo_peer = olmo_mn[olmo_mn["condition"] == "asch_zhu_unbiased_unanimous_confident"].copy()
    stage_order = ["base", "instruct_sft", "instruct_dpo", "instruct"]
    stage_labels_short = {"base": "Base", "instruct_sft": "SFT", "instruct_dpo": "DPO", "instruct": "Instruct"}

    traj_x, traj_y = [], []
    for stage in stage_order:
        row = olmo_peer[olmo_peer["variant"] == stage]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        traj_x.append(r["endorsement_rate"] * 100)
        traj_y.append(r["refusal_rate"] * 100)

    # Plot trajectory path with arrows
    for i in range(len(traj_x) - 1):
        ax.annotate("", xy=(traj_x[i + 1], traj_y[i + 1]), xytext=(traj_x[i], traj_y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OLMO, linewidth=1.8, alpha=0.7),
                    zorder=4)

    # Plot trajectory points
    ax.scatter(traj_x, traj_y, s=60, c=C_OLMO, edgecolor="black", linewidth=0.6,
               marker="s", alpha=0.9, zorder=7)

    # Label trajectory points
    traj_offsets = {"Base": (-8, 8), "SFT": (5, 5), "DPO": (-10, 5), "Instruct": (5, -8)}
    for i, stage in enumerate(stage_order[:len(traj_x)]):
        lbl = stage_labels_short[stage]
        ox, oy = traj_offsets.get(lbl, (5, 3))
        ax.annotate(lbl, (traj_x[i], traj_y[i]), textcoords="offset points",
                    xytext=(ox, oy), fontsize=6, fontweight="bold", color=C_OLMO,
                    zorder=8)

    ax.set_xlabel("Wrong Answer Endorsement Rate (%)", fontsize=9)
    ax.set_ylabel("Refusal Rate (%)", fontsize=9)
    ax.set_title("Behavioral Phase Space Under Peer Consensus\n($T{=}0.0$, fixed $N{=}400$)", fontsize=10, fontweight="bold")
    ax.set_xlim(-2, 80)
    ax.set_ylim(-3, 95)
    ax.grid(alpha=0.15)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=C_DENSE, edgecolor="black", linewidth=0.4, label="Dense Instruct"),
        Patch(facecolor=C_MOE, edgecolor="black", linewidth=0.4, label="MoE"),
        Patch(facecolor=C_THINK, edgecolor="black", linewidth=0.4, label="Think/Reasoning"),
        Patch(facecolor=C_CONST, edgecolor="black", linewidth=0.4, label="Constitutional AI"),
        Line2D([0], [0], color=C_OLMO, linewidth=1.8, marker="s", markersize=5,
               markerfacecolor=C_OLMO, markeredgecolor="black", label="OLMo-7B Trajectory"),
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
    """
    Figure 5: Peer vs Authority, but decomposing peer bars into
    endorsement (solid) and refusal (hatched) components so readers
    can see which part of the pressure effect is actual conformity.
    """
    mcn, pe, _ = load_t0_data()

    # --- Peer: decompose into endorsement and refusal deltas ---
    peer_pe = pe[pe["condition"] == "asch_zhu_unanimous_confident"].copy()
    peer_pe = peer_pe[["model_short", "pressure_endorsement_rate",
                        "pressure_refusal_rate", "control_refusal_rate"]].copy()

    # Control endorsement rates
    ctrl_metrics = pd.read_csv(
        ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/per_model_condition_metrics.csv"
    )
    ctrl = ctrl_metrics[
        (ctrl_metrics["condition"] == "control") & (ctrl_metrics["temperature"] == 0.0)
    ][["short_name", "endorsed_rate", "refusal_rate"]].copy()
    ctrl.columns = ["model_short", "ctrl_endorse", "ctrl_refusal_metric"]

    peer_pe = peer_pe.merge(ctrl, on="model_short", how="left")
    peer_pe["delta_endorse"] = peer_pe["pressure_endorsement_rate"] - peer_pe["ctrl_endorse"]
    peer_pe["delta_refusal"] = peer_pe["pressure_refusal_rate"] - peer_pe["ctrl_refusal_metric"]

    # Peer significance from McNemar
    peer_mcn = mcn[mcn["condition"] == "asch_zhu_unanimous_confident"][
        ["model_short", "delta_error", "p_adjusted", "sig_adjusted"]].copy()
    peer_mcn.columns = ["model_short", "delta_peer_total", "p_peer", "sig_peer"]

    # Authority delta and significance
    auth = mcn[mcn["condition"] == "authoritative_bias"][
        ["model_short", "delta_error", "p_adjusted", "sig_adjusted"]].copy()
    auth.columns = ["model_short", "delta_auth", "p_auth", "sig_auth"]

    df = peer_pe.merge(peer_mcn, on="model_short", how="left")
    df = df.merge(auth, on="model_short", how="left")

    # Sort by peer endorsement delta descending (actual conformity)
    df = df.sort_values("delta_endorse", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)

    x = np.arange(len(df))
    w = 0.35
    colors_model = [ARCH_COLORS.get(row["model_short"], C_OLMO) for _, row in df.iterrows()]
    colors_auth = [plt.cm.Greys(0.35)] * len(df)

    # --- Peer bars: stacked endorsement (solid) + refusal (hatched) ---
    endorse_vals = df["delta_endorse"].fillna(0).values
    refusal_vals = df["delta_refusal"].clip(lower=0).fillna(0).values  # only positive refusal changes

    # Endorsement component (solid color)
    ax.bar(x - w / 2, endorse_vals, w, color=colors_model,
           edgecolor="black", linewidth=0.4, zorder=3, label="Peer: $\\Delta$ Endorsement")

    # Refusal component (hatched, stacked on top of endorsement if positive)
    ax.bar(x - w / 2, refusal_vals, w, bottom=np.maximum(endorse_vals, 0),
           color="#95A5A6", edgecolor="black", linewidth=0.3, alpha=0.6,
           hatch="///", zorder=3, label="Peer: $\\Delta$ Refusal")

    # Authority bars (single solid bar — authority framing has minimal refusal confound)
    auth_vals = df["delta_auth"].fillna(0)
    ax.bar(x + w / 2, auth_vals, w, color=colors_auth,
           edgecolor="black", linewidth=0.4, label="Authority Framing", zorder=3)

    # Significance annotations
    for i, (_, row) in enumerate(df.iterrows()):
        # Peer stars above the total stacked bar
        stars_peer = sig_stars(row.get("sig_peer", "ns"))
        y_peer = max(endorse_vals[i], 0) + max(refusal_vals[i], 0) + 0.015
        ax.text(i - w / 2, y_peer, stars_peer, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
                color="black" if stars_peer != "ns" else "#AAAAAA")

        # Authority stars
        stars_auth = sig_stars(row.get("sig_auth", "ns"))
        y_auth = max(auth_vals.iloc[i], 0) + 0.015
        ax.text(i + w / 2, y_auth, stars_auth, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold" if stars_auth != "ns" else "normal",
                color="black" if stars_auth != "ns" else "#AAAAAA",
                style="normal" if stars_auth != "ns" else "italic")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model_short"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(r"Pressure Effect ($\Delta$ rate)", fontsize=9)
    ax.set_title(
        "Peer Consensus vs Authority Framing ($T{=}0.0$, Holm corrected)\n"
        "Peer bars decomposed: solid = endorsement, hatched = refusal",
        fontsize=9, fontweight="bold",
    )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    fig.savefig(OUT / "fig5_peer_vs_authority.pdf")
    fig.savefig(OUT / "fig5_peer_vs_authority.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig5_peer_vs_authority.pdf  ({len(df)} models)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: 100% Stacked Bar — 3-State Decomposition
# ═══════════════════════════════════════════════════════════════════
def make_stacked_bar():
    _, pe, _ = load_t0_data()
    olmo_mn = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/olmo_family/tables/multinomial_rates_t0.csv")
    ablation = pd.read_csv(ROOT / "Comparing_Experiments/expanded_results/cross_family/tables/ablation_rates_t0.csv")

    # --- Group 1: Cross-family (peer condition, T=0.0) ---
    peer_pe = pe[pe["condition"] == "asch_zhu_unanimous_confident"].copy()
    peer_pe["correct"] = 1.0 - peer_pe["pressure_error_rate"] - peer_pe["pressure_refusal_rate"]
    peer_pe["error"] = peer_pe["pressure_error_rate"]
    peer_pe["refusal"] = peer_pe["pressure_refusal_rate"]
    # Sort by correct rate descending (most resistant at top)
    cross = peer_pe.sort_values("correct", ascending=True)[
        ["model_short", "correct", "error", "refusal"]].copy()

    # --- Group 2: OLMo-7B Training Trajectory (T=0.0, peer confident) ---
    olmo_peer = olmo_mn[olmo_mn["condition"] == "asch_zhu_unbiased_unanimous_confident"].copy()
    stage_order = {"base": 0, "instruct_sft": 1, "instruct_dpo": 2, "instruct": 3}
    olmo_peer = olmo_peer[olmo_peer["variant"].isin(stage_order)]
    olmo_peer["sort"] = olmo_peer["variant"].map(stage_order)
    olmo_peer = olmo_peer.sort_values("sort", ascending=False)
    stage_labels = {"base": "OLMo-7B Base", "instruct_sft": "OLMo-7B SFT",
                    "instruct_dpo": "OLMo-7B DPO", "instruct": "OLMo-7B Instruct"}
    olmo_rows = []
    for _, r in olmo_peer.iterrows():
        olmo_rows.append({
            "model_short": stage_labels[r["variant"]],
            "correct": r["correct_rate"],
            "error": r["error_rate"],
            "refusal": r["refusal_rate"],
        })
    olmo_df = pd.DataFrame(olmo_rows)

    # --- Group 3: Llama-3.1-70B Ablation ---
    abl_llama = ablation[ablation["model_short"] == "Llama-3.1-70B"].copy()
    cond_labels = {
        "asch_zhu_unanimous_confident": "70B: Peer + Sys Prompt",
        "asch_zhu_naked_unanimous_confident": "70B: Peer (No Sys Prompt)",
        "ngram_sequence_baseline": "70B: N-gram Baseline",
    }
    cond_order = list(cond_labels.keys())
    abl_rows = []
    for cond in reversed(cond_order):
        row = abl_llama[abl_llama["condition"] == cond]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        abl_rows.append({
            "model_short": cond_labels[cond],
            "correct": r["correct_rate"],
            "error": r["error_rate"],
            "refusal": r["refusal_rate"],
        })
    abl_df = pd.DataFrame(abl_rows)

    # --- Combine with group separators ---
    all_labels = []
    all_correct = []
    all_error = []
    all_refusal = []
    group_seps = []  # y-positions for group separators

    def add_group(df, label_col="model_short"):
        for _, r in df.iterrows():
            all_labels.append(r[label_col])
            all_correct.append(r["correct"] * 100)
            all_error.append(r["error"] * 100)
            all_refusal.append(r["refusal"] * 100)

    add_group(cross)
    group_seps.append(len(all_labels) - 0.5)
    add_group(olmo_df)
    group_seps.append(len(all_labels) - 0.5)
    add_group(abl_df)

    n = len(all_labels)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, max(5, n * 0.36)), constrained_layout=True)

    C_CORRECT = "#27AE60"  # green
    C_ERROR = "#C0392B"    # red
    C_REFUSAL = "#7F8C8D"  # grey

    # Stacked horizontal bars
    ax.barh(y, all_correct, color=C_CORRECT, edgecolor="white", linewidth=0.3,
            label="State A: Correct (Resisted)", zorder=3)
    ax.barh(y, all_error, left=all_correct, color=C_ERROR, edgecolor="white", linewidth=0.3,
            label="State B: Wrong Answer", zorder=3)
    lefts = [c + e for c, e in zip(all_correct, all_error)]
    ax.barh(y, all_refusal, left=lefts, color=C_REFUSAL, edgecolor="white", linewidth=0.3,
            label="State C: Refusal", zorder=3)

    # Percentage labels inside bars (if segment >= 8%)
    for i in range(n):
        # Correct
        if all_correct[i] >= 8:
            ax.text(all_correct[i] / 2, i, f"{all_correct[i]:.0f}%",
                    ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=4)
        # Error
        if all_error[i] >= 8:
            ax.text(all_correct[i] + all_error[i] / 2, i, f"{all_error[i]:.0f}%",
                    ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=4)
        # Refusal
        if all_refusal[i] >= 8:
            ax.text(lefts[i] + all_refusal[i] / 2, i, f"{all_refusal[i]:.0f}%",
                    ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=4)

    # Group separators
    for sep_y in group_seps:
        ax.axhline(sep_y, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=5)

    # Group labels
    n_cross = len(cross)
    n_olmo = len(olmo_df)
    ax.text(102, n_cross / 2 - 0.5, "Cross-Family\n(Study 2)", fontsize=7, va="center", ha="left",
            style="italic", color="grey", clip_on=False)
    ax.text(102, n_cross + n_olmo / 2 - 0.5, "OLMo-7B\n(Study 1)", fontsize=7, va="center", ha="left",
            style="italic", color="grey", clip_on=False)
    ax.text(102, n_cross + n_olmo + len(abl_df) / 2 - 0.5, "Ablation\n(Llama-70B)", fontsize=7, va="center", ha="left",
            style="italic", color="grey", clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=7.5)
    ax.set_xlabel("Percentage of $N{=}400$ Items", fontsize=9)
    ax.set_title("3-State Decomposition Under Peer Consensus ($T{=}0.0$, Fixed $N{=}400$)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", ncol=3, fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.45, -0.15))

    fig.savefig(OUT / "fig1_stacked_decomposition.pdf")
    fig.savefig(OUT / "fig1_stacked_decomposition.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig1_stacked_decomposition.pdf  ({n} bars)")


# ═══════════════════════════════════════════════════════════════════
# Domain color palette for data-audit figures (§3.3)
# ═══════════════════════════════════════════════════════════════════
DOMAIN_COLORS_AUDIT = {
    "math":       "#1f77b4",  # blue
    "science":    "#2ca02c",  # green
    "general":    "#ff7f0e",  # orange
    "history":    "#9467bd",  # purple
    "preference": "#e377c2",  # pink
    "unmapped":   "#7f7f7f",  # gray
}

# Source-to-domain mapping (mirrors audit_metrics.SOURCE_DATASET_TO_BER_DOMAIN)
SOURCE_DATASET_TO_DOMAIN = {
    "OpenMathInstruct 2": "math",
    "Tulu 3 Persona Algebra": "math",
    "Tulu 3 Persona GSM": "math",
    "Tulu 3 Persona MATH": "math",
    "Dolci Instruct OpenThoughts3+ Science": "science",
    "SciRiff": "science",
    "OpenAssistant": "general",
    "Wildchat": "general",
    "Aya": "unmapped",
    "CoCoNot": "unmapped",
    "Dolci Instruct Precise IF": "unmapped",
    "Dolci Instruct Python Algorithms": "unmapped",
    "Dolci Instruct Tool Use": "unmapped",
    "Evol CodeAlpaca": "unmapped",
    "FLAN": "unmapped",
    "Hardcoded Data": "unmapped",
    "Logic Puzzles": "unmapped",
    "TableGPT": "unmapped",
    "Tulu 3 Persona Python": "unmapped",
    "Verifiable Reasoning": "unmapped",
    "WildGuardMix": "unmapped",
    "WildJailbreak": "unmapped",
}


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Pillar I — Per-Source SFT Structural Priors
# ═══════════════════════════════════════════════════════════════════
def make_pillar1_fig():
    """
    2-panel figure for §3.3 Pillar I data audit.
    Left : horizontal bar chart of P_carryover (top 12 sources, sorted descending).
    Right: scatter affirm_rate x max_run_geq_5_rate, log-sized markers, Spearman rho.
    """
    import json
    from scipy import stats

    by_source_csv = ROOT / "dataset_analysis/results/phase5_instruct-sft_by_source.csv"
    summary_json = ROOT / "dataset_analysis/results/phase5_instruct-sft_summary.json"

    df = pd.read_csv(by_source_csv)
    with open(summary_json) as f:
        summary = json.load(f)

    # Corpus-wide P(resp_has_list | prompt_has_list)
    corpus_mean_carryover = summary.get("P(resp_has_list | prompt_has_list)", None)

    # Map domain and color
    df["domain"] = df["source_dataset"].map(lambda s: SOURCE_DATASET_TO_DOMAIN.get(s, "unmapped"))
    df["color"] = df["domain"].map(lambda d: DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"]))

    # --- Left panel: top 12 by P_carryover (drop NaN rows) ---
    bar_df = df.dropna(subset=["P_carryover"]).sort_values("P_carryover", ascending=False).head(12)

    # --- Right panel: scatter affirm_rate vs max_run_geq_5_rate ---
    scat_df = df.dropna(subset=["affirm_rate", "max_run_geq_5_rate", "n"]).copy()
    # Filter rows where n > 0
    scat_df = scat_df[scat_df["n"] > 0]
    rho, pval = stats.spearmanr(scat_df["affirm_rate"], scat_df["max_run_geq_5_rate"])

    fig, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    # ── Left: horizontal bar chart ──
    y_pos = np.arange(len(bar_df))
    bar_colors = [DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"])
                  for d in bar_df["domain"]]

    ax_bar.barh(y_pos, bar_df["P_carryover"].values, color=bar_colors,
                edgecolor="black", linewidth=0.4, height=0.65, zorder=3)

    # Corpus-wide mean vertical line
    if corpus_mean_carryover is not None:
        ax_bar.axvline(corpus_mean_carryover, color="black", linewidth=1.0,
                       linestyle="--", zorder=4,
                       label=f"Corpus mean ({corpus_mean_carryover:.2f})")

    # Short labels (truncate long names)
    short_labels = []
    for s in bar_df["source_dataset"]:
        lbl = s.replace("Dolci Instruct ", "Dolci:").replace("Tulu 3 Persona ", "Tulu:")
        if len(lbl) > 24:
            lbl = lbl[:23] + "…"
        short_labels.append(lbl)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(short_labels, fontsize=7)
    ax_bar.set_xlabel(r"$P$(\,resp has list\,|\,prompt has list\,)", fontsize=8)
    ax_bar.set_title("Formatting Carryover by Source\n(top 12)", fontsize=9, fontweight="bold")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.grid(axis="x", alpha=0.15, zorder=0)
    ax_bar.invert_yaxis()  # highest at top

    # Value labels on bars
    for i, val in enumerate(bar_df["P_carryover"].values):
        ax_bar.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=6.5, zorder=5)

    # Domain legend patches
    from matplotlib.patches import Patch
    seen_domains = list(dict.fromkeys(bar_df["domain"]))  # preserve order, deduplicate
    legend_patches = [
        Patch(facecolor=DOMAIN_COLORS_AUDIT[d], edgecolor="black", linewidth=0.4, label=d.capitalize())
        for d in seen_domains
    ]
    if corpus_mean_carryover is not None:
        from matplotlib.lines import Line2D
        legend_patches.append(Line2D([0], [0], color="black", linewidth=1.0,
                                     linestyle="--", label=f"Corpus mean ({corpus_mean_carryover:.2f})"))
    ax_bar.legend(handles=legend_patches, loc="lower right", fontsize=6,
                  framealpha=0.9, edgecolor="grey")

    # ── Right: scatter ──
    marker_sizes = np.log1p(scat_df["n"].values) * 8  # log scale, minimum ~8

    scatter_colors = [DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"])
                      for d in scat_df["domain"]]

    ax_scat.scatter(
        scat_df["affirm_rate"] * 100,
        scat_df["max_run_geq_5_rate"] * 100,
        s=marker_sizes,
        c=scatter_colors,
        edgecolor="black", linewidth=0.4,
        alpha=0.85, zorder=5,
    )

    # Label each point
    for _, row in scat_df.iterrows():
        lbl = row["source_dataset"].replace("Dolci Instruct ", "Dolci:").replace("Tulu 3 Persona ", "Tulu:")
        if len(lbl) > 20:
            lbl = lbl[:19] + "…"
        ax_scat.annotate(lbl,
                         (row["affirm_rate"] * 100, row["max_run_geq_5_rate"] * 100),
                         textcoords="offset points", xytext=(4, 2),
                         fontsize=5, zorder=6)

    # Spearman rho annotation
    pval_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
    ax_scat.text(0.97, 0.97,
                 rf"Spearman $\rho={rho:.2f}$" + f"\n$p={pval_str}$",
                 transform=ax_scat.transAxes,
                 fontsize=7, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="grey", linewidth=0.5, alpha=0.9))

    ax_scat.set_xlabel("Affirm-prefix rate (%)", fontsize=8)
    ax_scat.set_ylabel("Run-length $\geq 5$ rate (%)", fontsize=8)
    ax_scat.set_title("Affirmation vs. Run-Length\nper Source Dataset", fontsize=9, fontweight="bold")
    ax_scat.grid(alpha=0.15, zorder=0)

    fig.savefig(OUT / "fig6_pillar_1_sft_priors.pdf")
    fig.savefig(OUT / "fig6_pillar_1_sft_priors.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig6_pillar_1_sft_priors.pdf  ({len(bar_df)} bars, {len(scat_df)} scatter pts, rho={rho:.3f})")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: Pillar II — DPO Penalty Effect + Pillar III Scatter
# ═══════════════════════════════════════════════════════════════════
def make_pillar2_fig():
    """
    2-panel figure for §3.3 Pillar II.
    Left : violin plots for delta_struct_jaccard, delta_ngram_overlap,
           delta_max_run, delta_consensus_hits (50k subsample, seed=0).
           Annotated with pre-computed Cliff's delta from phase6 summary.
    Right: effect-size forest plot — Cliff's delta + 95% bootstrap CI for
           all 9 DPO delta metrics, with Holm-Bonferroni significance
           markers and the Romano (2006) practical-significance thresholds.

    Note (post-review revision): the original right panel was an n=3
    Spearman scatter (Pillar III). With only 3 macro-domains, ρ takes
    one of 6 discrete values and the bootstrap CI spans the full support
    [-1, +1]; the panel was uninformative. We replace it with the
    effect-size forest plot, which surfaces the directional sign
    structure of DPO's penalty across all 9 metrics — the substantive
    Pillar II finding.
    """
    import json

    dpo_csv = ROOT / "dataset_analysis/results/phase6_instruct-dpo_per_pair.csv"
    dpo_summary_json = ROOT / "dataset_analysis/results/phase6_instruct-dpo_summary.json"

    # ── Load DPO summary (Cliff's delta + practical_significance) ──
    with open(dpo_summary_json) as f:
        dpo_sum = json.load(f)

    VIOLIN_METRICS = [
        ("delta_struct_jaccard", r"$\Delta$Struct\nJaccard"),
        ("delta_ngram_overlap",  r"$\Delta$N-gram\nOverlap"),
        ("delta_max_run",        r"$\Delta$Max\nRun"),
        ("delta_consensus_hits", r"$\Delta$Consensus\nHits"),
    ]

    # Extract Cliff's delta and practical_significance from summary
    cliffs = {}
    prac_sig = {}
    for key, _ in VIOLIN_METRICS:
        entry = dpo_sum.get(key, {})
        cliffs[key] = entry.get("cliffs_delta", float("nan"))
        prac_sig[key] = entry.get("practical_significance", "n/a")

    # ── Subsample DPO pairs (50k, seed=0) ──
    dpo_df = pd.read_csv(dpo_csv, usecols=[k for k, _ in VIOLIN_METRICS])
    if len(dpo_df) > 50000:
        dpo_df = dpo_df.sample(n=50000, random_state=0)

    # ── Pillar II forest plot data: all 9 delta metrics ──
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
    forest_rows = []
    for key, label in FOREST_METRICS:
        entry = dpo_sum.get(key, {})
        cd = entry.get("cliffs_delta", float("nan"))
        ci = entry.get("boot_ci_cliffs_delta", [float("nan"), float("nan")])
        p_holm = entry.get("permutation_p_holm", float("nan"))
        null_z = entry.get("null_cliffs_delta_z", float("nan"))
        forest_rows.append({
            "key": key, "label": label,
            "cliffs": cd, "ci_lo": ci[0], "ci_hi": ci[1],
            "p_holm": p_holm, "null_z": null_z,
        })

    # ── Build figure ──
    fig, (ax_vio, ax_p3) = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    # ── Left: violin plots ──
    violin_data = [dpo_df[key].dropna().values for key, _ in VIOLIN_METRICS]
    violin_labels = [lbl for _, lbl in VIOLIN_METRICS]

    parts = ax_vio.violinplot(violin_data, positions=np.arange(len(VIOLIN_METRICS)),
                               showmedians=False, showextrema=False)

    # Style each violin body
    vio_color = "#5B9BD5"
    for body in parts["bodies"]:
        body.set_facecolor(vio_color)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.7)

    # Overlay IQR box and median per violin
    for i, (key, _) in enumerate(VIOLIN_METRICS):
        vals = dpo_df[key].dropna().values
        # Clip to [-0.5, 0.5] for display
        vals_clipped = np.clip(vals, -0.5, 0.5)
        q25, median, q75 = np.percentile(vals_clipped, [25, 50, 75])

        # IQR box
        ax_vio.add_patch(plt.Rectangle(
            (i - 0.05, q25), 0.1, q75 - q25,
            facecolor="white", edgecolor="black", linewidth=0.8, zorder=4,
        ))
        # Median line
        ax_vio.hlines(median, i - 0.08, i + 0.08,
                      color="black", linewidth=1.2, zorder=5)

        # Annotate Cliff's delta above violin
        cd = cliffs.get(key, float("nan"))
        ps = prac_sig.get(key, "n/a")
        cd_str = f"$\\delta$={cd:+.3f}\n({ps})" if not np.isnan(cd) else "n/a"
        ax_vio.text(i, 0.52, cd_str, ha="center", va="bottom", fontsize=6.5,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="grey", linewidth=0.4, alpha=0.85))

    ax_vio.set_xticks(np.arange(len(VIOLIN_METRICS)))
    ax_vio.set_xticklabels(violin_labels, fontsize=7)
    ax_vio.set_ylabel("Delta value (rejected − chosen)", fontsize=8)
    ax_vio.set_title("DPO Penalty: Structural-Prior Deltas\n(n=50k pairs, seed=0)", fontsize=9, fontweight="bold")
    ax_vio.set_ylim(-0.5, 0.65)
    ax_vio.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5, zorder=2)
    ax_vio.grid(axis="y", alpha=0.12, zorder=0)

    # ── Right: effect-size forest plot for all 9 delta metrics ──
    # Pre-registered Romano (2006) thresholds: |δ|<0.147 negligible, <0.33 small, <0.474 medium.
    ax_p3.axvspan(-0.147, 0.147, color="grey", alpha=0.07, zorder=0,
                  label=r"|$\delta$|<0.147 (Romano 'none')")
    ax_p3.axvspan(0.147, 0.33,  color="#FFE4B5", alpha=0.20, zorder=0)
    ax_p3.axvspan(-0.33, -0.147, color="#FFE4B5", alpha=0.20, zorder=0)
    ax_p3.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)

    y_positions = np.arange(len(forest_rows))[::-1]  # top-to-bottom
    for y, row in zip(y_positions, forest_rows):
        cd, lo, hi = row["cliffs"], row["ci_lo"], row["ci_hi"]
        # Holm-significant marker style: filled circle if p_holm<0.05, else open
        sig = (not np.isnan(row["p_holm"])) and (row["p_holm"] < 0.05)
        face = "#1f77b4" if sig else "white"
        edge = "#1f77b4" if sig else "grey"
        # CI bar
        ax_p3.errorbar(cd, y, xerr=[[cd - lo], [hi - cd]],
                       fmt="none", ecolor=edge, elinewidth=1.2,
                       capsize=3, zorder=4)
        # Point
        ax_p3.scatter(cd, y, s=55, marker="o",
                      facecolor=face, edgecolor=edge, linewidth=1.0, zorder=5)
        # Annotation: null-z value
        z_str = f"  z$_{{\\mathrm{{null}}}}={row['null_z']:+.0f}$"
        ax_p3.text(hi + 0.01, y, z_str, fontsize=6.0, va="center", zorder=6)

    ax_p3.set_yticks(y_positions)
    ax_p3.set_yticklabels([r["label"] for r in forest_rows], fontsize=6.5)
    ax_p3.set_xlabel(r"Cliff's $\delta$ (rejected − chosen)", fontsize=8)
    ax_p3.set_xlim(-0.20, 0.20)
    ax_p3.set_title("DPO Effect-Size Forest\n($N{=}259{,}785$ pairs, Holm-corrected)",
                    fontsize=9, fontweight="bold")
    ax_p3.grid(axis="x", alpha=0.12, zorder=0)
    # Legend: filled = Holm-significant, open = not
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4",
                   markeredgecolor="#1f77b4", markersize=6, label=r"$p_{\mathrm{Holm}}<0.05$"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="grey", markersize=6, label=r"$p_{\mathrm{Holm}}\geq 0.05$"),
    ]
    ax_p3.legend(handles=handles, loc="lower right", fontsize=6.0,
                 frameon=True, framealpha=0.85)

    fig.savefig(OUT / "fig7_pillar_2_dpo_effect.pdf")
    fig.savefig(OUT / "fig7_pillar_2_dpo_effect.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig7_pillar_2_dpo_effect.pdf  ({len(forest_rows)} forest metrics, {len(dpo_df)} violin pairs)")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of figures to generate: "
                             "stacked,forest,scatter,peer_auth,pillar1,pillar2")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None

    dispatch = {
        "stacked":   make_stacked_bar,
        "forest":    make_forest_plot,
        "scatter":   make_scatter,
        "peer_auth": make_peer_vs_auth,
        "pillar1":   make_pillar1_fig,
        "pillar2":   make_pillar2_fig,
    }

    if only:
        for key in only:
            key = key.strip()
            if key in dispatch:
                print(f"Generating {key}...")
                dispatch[key]()
            else:
                print(f"Unknown figure key: {key!r}. Available: {list(dispatch.keys())}")
    else:
        print("Generating publication figures from T=0.0 corrected data...")
        for name, fn in dispatch.items():
            print(f"  -> {name}")
            fn()
    print("Done!")
