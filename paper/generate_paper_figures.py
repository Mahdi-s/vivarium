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
if __name__ == "__main__":
    print("Generating publication figures from T=0.0 corrected data...")
    make_stacked_bar()
    make_forest_plot()
    make_scatter()
    make_peer_vs_auth()
    print("Done!")
