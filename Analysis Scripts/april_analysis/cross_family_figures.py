#!/usr/bin/env python3
"""
Generate cross-family + ablation figures for the April_analysis expansion.

Reads the cross-family tables produced by cross_family_tables.py and the
ablation tables produced by ablation_probes.py (both under
Comparing_Experiments/April_analysis/tables/). Writes PDFs under
Comparing_Experiments/April_analysis/figures/cross_family/.

Figures produced:

  Replacement for paper/generate_paper_figures.py outputs
  (reads from April_analysis tables, not expanded_results):
    fig1_stacked_decomposition.pdf    3-state stacked bars across 12 models
    fig3_cross_family_forest.pdf      forest plot ranked by Δ endorsement
    fig4_refusal_endorsement.pdf      scatter: refusal Δ vs endorsement Δ
    fig5_peer_vs_authority.pdf        grouped bars: peer vs authority Δ

  New figures for the expansion narrative:
    fig_cross_family_headline_ber.pdf       HEADLINE for H1 — 19-bar BER
                                             ranking (12 cross-family + 7B
                                             base + 3 Instruct + 3 Think)
    fig_cross_family_t0_vs_t06.pdf           BER(T=0) vs BER(T=0.6) scatter
    fig_scale_bridge.pdf                     paired bars: 7B stages vs 32B
                                             on 4 shared conditions
    fig_ablation_ngram_vs_pressure.pdf       HEADLINE for H2 — BER on ngram
                                             vs unanimous_confident, Llama-70B
                                             + OLMo-32B
    fig_system_prompt_ablation.pdf           paired bars: system_style=none
                                             vs system_style=default BER

Color scheme and ARCH_COLORS are adapted from paper/generate_paper_figures.py
(see attribution header in each plot function).
"""
# adapted from paper/generate_paper_figures.py (color scheme + sig_stars helper)
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from _common import build_cross_family_argparser, ensure_dir


def _save(fig, path_stem: Path) -> None:
    """Save figure as both PDF and PNG (for PowerPoint/preview)."""
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Global style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
})

# ── Color scheme (adapted from paper/generate_paper_figures.py) ────────
C_DENSE = "#E74C3C"          # coral red
C_MOE = "#1ABC9C"            # teal
C_THINK = "#8E44AD"          # purple
C_CONST = "#2980B9"          # blue (Constitutional AI)
C_BASE_7B = "#7F8C8D"        # grey
C_INSTRUCT_7B = "#F39C12"    # orange (Instruct-path 7B stages)
C_THINK_7B = "#9B59B6"       # light purple (Think-path 7B stages)

ARCH_COLOR = {
    "dense": C_DENSE,
    "moe": C_MOE,
    "think": C_THINK,
    "constitutional": C_CONST,
    "base": C_BASE_7B,
    "instruct": C_INSTRUCT_7B,
}

# Cohort colors for the 19-bar headline
COHORT_COLOR = {
    "cross_family": None,  # will use architecture color
    "base_7b": C_BASE_7B,
    "instruct_7b": C_INSTRUCT_7B,
    "think_7b": C_THINK_7B,
}


def _out(out_dir: str) -> Path:
    p = Path(out_dir) / "figures" / "cross_family"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _color_for_row(row: pd.Series) -> str:
    cohort = row.get("cohort", "cross_family")
    if cohort == "cross_family":
        return ARCH_COLOR.get(row["architecture"], "#95A5A6")
    return COHORT_COLOR.get(cohort, "#95A5A6")


# ═══════════════════════════════════════════════════════════════════════
# HEADLINE: fig_cross_family_headline_ber.pdf  (H1)
# ═══════════════════════════════════════════════════════════════════════

def make_headline_ber(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    df = tables["conformity_ranking"].copy().reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.35 * n + 1.2))

    y = np.arange(n)
    colors = [_color_for_row(r) for _, r in df.iterrows()]
    ber = df["ber"].to_numpy()
    lo = df["ber_lo"].to_numpy()
    hi = df["ber_hi"].to_numpy()
    xerr_lo = ber - lo
    xerr_hi = hi - ber

    ax.barh(y, ber, color=colors, edgecolor="black", linewidth=0.4, alpha=0.92)
    ax.errorbar(
        ber, y, xerr=[xerr_lo, xerr_hi],
        fmt="none", ecolor="black", elinewidth=0.7, capsize=2,
    )
    for i, v in enumerate(ber):
        label_x = max(hi[i], v) + 0.012
        ax.text(label_x, i, f"{v*100:.1f}%", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["short_name"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.85, hi.max() + 0.08))
    ax.set_xlabel("Wrong-answer endorsement rate (BER), T=0.0, asch_zhu_unbiased_unanimous_confident")
    ax.set_title("OLMo family and other families")
    ax.axvline(0.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_DENSE, label="Dense"),
        mpatches.Patch(color=C_MOE, label="MoE"),
        mpatches.Patch(color=C_THINK, label="Think"),
        mpatches.Patch(color=C_CONST, label="Constitutional"),
        mpatches.Patch(color=C_BASE_7B, label="OLMo-7B-Base"),
        mpatches.Patch(color=C_INSTRUCT_7B, label="OLMo-7B Instruct stages"),
        mpatches.Patch(color=C_THINK_7B, label="OLMo-7B Think stages"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    _save(fig, out_dir / "fig_cross_family_headline_ber")


# ═══════════════════════════════════════════════════════════════════════
# fig_cross_family_t0_vs_t06.pdf  (BER scatter)
# ═══════════════════════════════════════════════════════════════════════

def make_t0_vs_t06_scatter(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    cells = tables["per_model_condition_metrics"]
    sub = cells[cells["condition_name"] == "asch_zhu_unbiased_unanimous_confident"].copy()

    pivot = sub.pivot_table(
        index=["short_name", "architecture"],
        columns="temperature",
        values="ber",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    pivot = pivot.rename(columns={"0.0": "ber_t0", "0.6": "ber_t06"})

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for _, r in pivot.iterrows():
        color = ARCH_COLOR.get(r["architecture"], "#95A5A6")
        ax.scatter(r["ber_t0"], r["ber_t06"], s=110, color=color,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(
            r["short_name"], (r["ber_t0"], r["ber_t06"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7,
        )

    lim = max(pivot[["ber_t0", "ber_t06"]].max()) + 0.05
    ax.plot([0, lim], [0, lim], ":", color="gray", linewidth=0.7, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("BER at T=0.0")
    ax.set_ylabel("BER at T=0.6")
    ax.set_title("Cross-family BER concentration: cold vs warm")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=C_DENSE, label="Dense"),
        mpatches.Patch(color=C_MOE, label="MoE"),
        mpatches.Patch(color=C_THINK, label="Think"),
        mpatches.Patch(color=C_CONST, label="Constitutional"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", frameon=True)
    fig.tight_layout()
    _save(fig, out_dir / "fig_cross_family_t0_vs_t06")


# ═══════════════════════════════════════════════════════════════════════
# fig_scale_bridge.pdf  (7B stages vs 32B on 4 conditions)
# ═══════════════════════════════════════════════════════════════════════

def make_scale_bridge(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    sb = tables["scale_bridge"].copy()

    # Panel layout: 2 rows (instruct / think) × 1 column, conditions on x-axis.
    cond_order = [
        "control",
        "asch_zhu_unbiased_unanimous_confident",
        "authoritative_bias",
        "authority_zhu_unbiased_trust",
    ]
    cond_labels = ["control", "peer (unanimous\nconfident)",
                   "authoritative\nbias", "authority\ntrust"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    recipe_order = {
        "instruct": ["base", "instruct_sft", "instruct_dpo", "instruct", "OLMo-32B-Instruct"],
        "think": ["think_sft", "think_dpo", "think",
                  "OLMo-32B-Think-SFT", "OLMo-32B-Think-DPO", "OLMo-32B-Think"],
    }
    recipe_label = {
        "base": "7B Base",
        "instruct_sft": "7B Instruct-SFT",
        "instruct_dpo": "7B Instruct-DPO",
        "instruct": "7B Instruct-RL",
        "OLMo-32B-Instruct": "32B Instruct",
        "think_sft": "7B Think-SFT",
        "think_dpo": "7B Think-DPO",
        "think": "7B Think-RL",
        "OLMo-32B-Think-SFT": "32B Think-SFT",
        "OLMo-32B-Think-DPO": "32B Think-DPO",
        "OLMo-32B-Think": "32B Think",
    }
    recipe_color = {
        "base": C_BASE_7B,
        "instruct_sft": "#F4A261",
        "instruct_dpo": "#E76F51",
        "instruct": C_INSTRUCT_7B,
        "OLMo-32B-Instruct": "#B35900",
        "think_sft": "#C39BD3",
        "think_dpo": "#A569BD",
        "think": "#7D3C98",
        "OLMo-32B-Think-SFT": "#6C3483",
        "OLMo-32B-Think-DPO": "#5B2C6F",
        "OLMo-32B-Think": "#4A235A",
    }

    for ax, path in zip(axes, ["instruct", "think"]):
        recipes = recipe_order[path]
        bw = 0.8 / len(recipes)
        x = np.arange(len(cond_order))
        for i, r in enumerate(recipes):
            sub = sb[sb["recipe"] == r]
            heights = [
                float(sub[sub["condition_name"] == c]["ber"].iloc[0])
                if not sub[sub["condition_name"] == c].empty else 0.0
                for c in cond_order
            ]
            offset = (i - (len(recipes) - 1) / 2) * bw
            ax.bar(x + offset, heights, width=bw,
                   color=recipe_color[r], edgecolor="black",
                   linewidth=0.3, label=recipe_label[r])
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=0)
        ax.set_ylim(0, 0.85)
        ax.set_title(f"{path.capitalize()} path")
        ax.set_ylabel("BER" if path == "instruct" else "")
        ax.legend(loc="upper left", frameon=True, fontsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    fig.suptitle("Scale vs recipe: OLMo-7B stages against OLMo-32B at T=0", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "fig_scale_bridge")


# ═══════════════════════════════════════════════════════════════════════
# HEADLINE: fig_ablation_ngram_vs_pressure.pdf  (H2)
# ═══════════════════════════════════════════════════════════════════════

def make_ablation_ngram_vs_pressure(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    sc = tables["combined_ablation_scorecard"].copy()
    models = sc["short_name"].tolist()
    ber_with = sc["ber_with_system_prompt"].to_numpy()
    ber_without = sc["ber_without_system_prompt"].to_numpy()
    ber_ngram = sc["ber_ngram_baseline"].to_numpy()
    ratio = sc["pattern_completion_ratio"].to_numpy()
    refusal_with = sc.get("refusal_unanimous_confident", pd.Series([np.nan] * len(sc))).to_numpy()
    refusal_without = sc.get("refusal_naked_unanimous", pd.Series([np.nan] * len(sc))).to_numpy()
    refusal_ngram = sc.get("refusal_ngram_baseline", pd.Series([np.nan] * len(sc))).to_numpy()

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(models))
    bw = 0.25

    bars1 = ax.bar(x - bw, ber_with, bw,
                   color="#5DADE2", edgecolor="black", linewidth=0.4,
                   label="Unanimous confident (with system prompt)")
    bars2 = ax.bar(x, ber_without, bw,
                   color="#E59866", edgecolor="black", linewidth=0.4,
                   label="Naked unanimous confident (no system prompt)")
    bars3 = ax.bar(x + bw, ber_ngram, bw,
                   color="#CD6155", edgecolor="black", linewidth=0.4,
                   label="N-gram sequence baseline (no social framing)")

    def _annot(bars, values):
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{v*100:.1f}%", ha="center", fontsize=7)
    _annot(bars1, ber_with)
    _annot(bars2, ber_without)
    _annot(bars3, ber_ngram)

    # Pattern-completion ratio annotation
    for i, (r, bn) in enumerate(zip(ratio, ber_ngram)):
        ax.annotate(
            f"ratio = {r:.2f}×",
            xy=(i + bw, bn), xytext=(i + bw, bn + 0.08),
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
        )

    # Explicit refusal context avoids conflating refusal behavior with BER.
    for i, (rw, rn, rg) in enumerate(zip(refusal_with, refusal_without, refusal_ngram)):
        if np.isnan(rw) and np.isnan(rn) and np.isnan(rg):
            continue
        y_text = min(0.62, max(ber_with[i], ber_without[i], ber_ngram[i]) + 0.15)
        ax.text(
            i,
            y_text,
            f"Refusal: with {rw*100:.1f}% | naked {rn*100:.1f}% | ngram {rg*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Wrong-answer endorsement rate (BER)")
    ax.set_ylim(0, 0.65)
    ax.set_title(
        "Pattern completion without social framing (H2 headline)\n"
        "BER on the abstract N-gram probe vs social-pressure baselines, T=0\n"
        "(refusal rates shown separately above each model)"
    )
    ax.legend(loc="upper left", frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_dir / "fig_ablation_ngram_vs_pressure")


# ═══════════════════════════════════════════════════════════════════════
# fig_system_prompt_ablation.pdf  (paired bars for sys prompt removal)
# ═══════════════════════════════════════════════════════════════════════

def make_system_prompt_ablation(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    d = tables["system_prompt_ablation_delta"].copy()
    models = d["short_name"].tolist()
    with_sys = d["ber_with_system_prompt"].to_numpy()
    without_sys = d["ber_without_system_prompt"].to_numpy()
    p_vals = d["mcnemar_p_value"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    x = np.arange(len(models))
    bw = 0.34
    bars1 = ax.bar(x - bw/2, with_sys, bw,
                   color="#5DADE2", edgecolor="black", linewidth=0.4,
                   label='With "be truthful" system prompt')
    bars2 = ax.bar(x + bw/2, without_sys, bw,
                   color="#E59866", edgecolor="black", linewidth=0.4,
                   label="System prompt stripped")

    for b, v in list(zip(bars1, with_sys)) + list(zip(bars2, without_sys)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{v*100:.1f}%", ha="center", fontsize=7)

    for i, p in enumerate(p_vals):
        if p < 0.001:
            mark = "***"
        elif p < 0.01:
            mark = "**"
        elif p < 0.05:
            mark = "*"
        else:
            mark = "ns"
        top = max(with_sys[i], without_sys[i]) + 0.04
        ax.plot([i - bw/2, i + bw/2], [top, top], color="black", linewidth=0.6)
        ax.text(i, top + 0.003, f"McNemar: {mark}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("BER (asch_zhu_unanimous_confident peers)")
    ax.set_ylim(0, 0.55)
    ax.set_title('System-prompt ablation: "be truthful" on vs off, T=0')
    ax.legend(loc="upper right", frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_dir / "fig_system_prompt_ablation")


# ═══════════════════════════════════════════════════════════════════════
# fig1_stacked_decomposition.pdf  (replacement for paper fig1)
# ═══════════════════════════════════════════════════════════════════════

def make_stacked_decomposition(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    """
    3-state stacked bar (correct / wrong endorsed / refusal) across
    cross-family models at T=0 on asch_zhu_unbiased_unanimous_confident,
    ranked by BER. Uses per_model_condition_metrics.csv.
    """
    cells = tables["per_model_condition_metrics"]
    sub = cells[
        (cells["temperature"] == 0.0)
        & (cells["condition_name"] == "asch_zhu_unbiased_unanimous_confident")
    ].copy()
    sub = sub.sort_values("ber", ascending=False).reset_index(drop=True)

    correct = sub["state_A_n"].to_numpy() / 400.0
    wrong = sub["state_B_n"].to_numpy() / 400.0
    refuse = sub["state_C_n"].to_numpy() / 400.0
    other = 1 - correct - wrong - refuse
    labels = sub["short_name"].tolist()

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(labels))
    bw = 0.7

    ax.bar(x, correct, bw, color="#27AE60", edgecolor="black", linewidth=0.3,
           label="Correct (A)")
    ax.bar(x, wrong, bw, bottom=correct, color="#C0392B",
           edgecolor="black", linewidth=0.3, label="Wrong-answer endorsed (B)")
    ax.bar(x, refuse, bw, bottom=correct + wrong, color="#7F8C8D",
           edgecolor="black", linewidth=0.3, label="Refused (C)")
    ax.bar(x, other, bw, bottom=correct + wrong + refuse, color="#F1C40F",
           edgecolor="black", linewidth=0.3, label="Unclassified (D)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Fraction of trials (N=400)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-family 3-state decomposition at T=0 (ranked by BER)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig1_stacked_decomposition")


# ═══════════════════════════════════════════════════════════════════════
# fig3_cross_family_forest.pdf  (replacement for paper fig3)
# ═══════════════════════════════════════════════════════════════════════

def make_forest_plot(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    pe = tables["pressure_effects_t0"].copy()
    pe = pe.sort_values("peer_ber_delta", ascending=True).reset_index(drop=True)
    n = len(pe)

    fig, ax = plt.subplots(figsize=(7.5, 0.38 * n + 1))
    y = np.arange(n)
    colors = [ARCH_COLOR.get(a, "#95A5A6") for a in pe["architecture"]]
    peer = pe["peer_ber_delta"].to_numpy()
    auth = pe["authority_ber_delta"].to_numpy()

    ax.barh(y - 0.18, peer, height=0.34, color=colors,
            edgecolor="black", linewidth=0.3, label="Peer Δ")
    ax.barh(y + 0.18, auth, height=0.34, color=colors, alpha=0.45,
            edgecolor="black", linewidth=0.3, label="Authority Δ")
    ax.axvline(0, color="black", linewidth=0.5)

    for i, (p, a) in enumerate(zip(peer, auth)):
        ax.text(p + (0.006 if p >= 0 else -0.006), i - 0.18,
                f"{p*100:+.1f}", va="center",
                ha="left" if p >= 0 else "right", fontsize=6.5)
        ax.text(a + (0.006 if a >= 0 else -0.006), i + 0.18,
                f"{a*100:+.1f}", va="center",
                ha="left" if a >= 0 else "right", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(pe["short_name"])
    ax.set_xlabel("Δ BER vs control (percentage points)")
    ax.set_title("Cross-family Δ endorsement under peer vs authority pressure, T=0")
    ax.legend(loc="lower right", frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "fig3_cross_family_forest")


# ═══════════════════════════════════════════════════════════════════════
# fig4_refusal_endorsement.pdf  (replacement for paper fig4)
# ═══════════════════════════════════════════════════════════════════════

def make_refusal_endorsement_scatter(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    pe = tables["pressure_effects_t0"].copy()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for _, r in pe.iterrows():
        color = ARCH_COLOR.get(r["architecture"], "#95A5A6")
        ax.scatter(
            r["peer_ber_delta"], r["peer_refusal_delta"],
            s=130, color=color, edgecolor="black", linewidth=0.5, zorder=3,
        )
        ax.annotate(r["short_name"],
                    (r["peer_ber_delta"], r["peer_refusal_delta"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Peer Δ endorsement (BER change)")
    ax.set_ylabel("Peer Δ refusal rate")
    ax.set_title("Behavioral taxonomy under peer pressure, T=0")
    x_pad = 0.035
    y_pad = 0.035
    ax.set_xlim(pe["peer_ber_delta"].min() - x_pad, pe["peer_ber_delta"].max() + x_pad)
    ax.set_ylim(pe["peer_refusal_delta"].min() - y_pad, pe["peer_refusal_delta"].max() + y_pad)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=C_DENSE, label="Dense"),
        mpatches.Patch(color=C_MOE, label="MoE"),
        mpatches.Patch(color=C_THINK, label="Think"),
        mpatches.Patch(color=C_CONST, label="Constitutional"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    _save(fig, out_dir / "fig4_refusal_endorsement")


# ═══════════════════════════════════════════════════════════════════════
# fig5_peer_vs_authority.pdf  (replacement for paper fig5)
# ═══════════════════════════════════════════════════════════════════════

def make_peer_vs_authority(
    tables: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    pe = tables["pressure_effects_t0"].copy()
    pe = pe.sort_values("peer_ber_delta", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    x = np.arange(len(pe))
    bw = 0.38
    ax.bar(x - bw/2, pe["peer_ber_delta"], bw, color="#E67E22",
           edgecolor="black", linewidth=0.3, label="Peer Δ")
    ax.bar(x + bw/2, pe["authority_ber_delta"], bw, color="#3498DB",
           edgecolor="black", linewidth=0.3, label="Authority Δ")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pe["short_name"], rotation=35, ha="right")
    ax.set_ylabel("Δ BER vs control")
    ax.set_title("Peer vs authority conformity delta per model, T=0")
    ax.legend(loc="upper right", frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "fig5_peer_vs_authority")


# ═══════════════════════════════════════════════════════════════════════
# Loader helper
# ═══════════════════════════════════════════════════════════════════════

def load_tables(out_dir: str) -> Dict[str, pd.DataFrame]:
    cf_dir = Path(out_dir) / "tables" / "cross_family"
    abl_dir = Path(out_dir) / "tables" / "ablation_probes"
    tables = {
        "per_model_condition_metrics": pd.read_csv(
            cf_dir / "per_model_condition_metrics.csv"),
        "conformity_ranking": pd.read_csv(cf_dir / "conformity_ranking.csv"),
        "pressure_effects_t0": pd.read_csv(cf_dir / "pressure_effects_t0.csv"),
        "pressure_effects_t06": pd.read_csv(cf_dir / "pressure_effects_t06.csv"),
        "peer_vs_authority_delta": pd.read_csv(cf_dir / "peer_vs_authority_delta.csv"),
        "scale_bridge": pd.read_csv(cf_dir / "scale_bridge.csv"),
        "knowledge_protection_corr": pd.read_csv(cf_dir / "knowledge_protection_corr.csv"),
        "combined_ablation_scorecard": pd.read_csv(
            abl_dir / "combined_ablation_scorecard.csv"),
        "system_prompt_ablation_delta": pd.read_csv(
            abl_dir / "system_prompt_ablation_delta.csv"),
        "pattern_completion_ratio": pd.read_csv(
            abl_dir / "pattern_completion_ratio.csv"),
    }
    return tables


def main() -> int:
    args = build_cross_family_argparser(
        "Emit cross-family + ablation figures into April_analysis/figures/cross_family/"
    ).parse_args()
    tables = load_tables(args.out_dir)
    out_path = _out(args.out_dir)

    # Expansion figures
    make_headline_ber(tables, out_path)
    make_t0_vs_t06_scatter(tables, out_path)
    make_scale_bridge(tables, out_path)
    make_ablation_ngram_vs_pressure(tables, out_path)
    make_system_prompt_ablation(tables, out_path)
    # Replacements for paper/generate_paper_figures.py
    make_stacked_decomposition(tables, out_path)
    make_forest_plot(tables, out_path)
    make_refusal_endorsement_scatter(tables, out_path)
    make_peer_vs_authority(tables, out_path)

    print(f"[cross_family_figures] Wrote 9 figures to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
