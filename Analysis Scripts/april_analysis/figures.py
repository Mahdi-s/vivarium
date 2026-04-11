#!/usr/bin/env python3
"""
Emit the canonical April_analysis figures.

Remediated 2026-04-08 for asymmetric Think-family coverage. Think
variants only exist at T ∈ {0, 0.6} (SFT/DPO) or T = 0 (Think-RL),
and only on the 4 shared conditions (control, unanimous_confident,
authoritative_bias, authority_zhu_unbiased_trust). Every figure below
handles that asymmetry explicitly; the old six-temperature Think curves
were a data-provenance bug (runs_latest Think variants were truncated
at ~1,400 chars).

Figures produced (all PDF + PNG):

    fig_stage_trajectory                Two-panel (T=0, T=0.6) BER curves
                                         for Instruct path vs Think path on
                                         unanimous_confident. Think-RL has
                                         no T=0.6 → broken segment.
    fig_2axis_heatmap_instruct          Panel A: base + Instruct family ×
                                         12 conditions at T=0 (single panel).
    fig_2axis_heatmap_think             Panel B: 3 Think variants × 4
                                         shared conditions at T=0.
    fig_2axis_heatmap_combined          Combined two-panel figure with a
                                         shared colorbar, column-aligned on
                                         the 4 shared conditions.
    fig_pattern_match_gradient          BER vs target-answer repetition
                                         count. Instruct family curves across
                                         12 conditions; Think family markers
                                         at reps ∈ {0, 1, 5} only.
    fig_instruct_temperature_sweep      (Q1a) Instruct + base × 6 T points
                                         on unanimous_confident and
                                         unanimous_plain (contrast panel).
    fig_temperature_t0_vs_t06_scatter   (Q1b) BER at T=0 vs BER at T=0.6
                                         across 7 variants × 4 shared
                                         conditions; y=x diagonal.
    fig_temperature_slope_bars          (Q1c) Instruct-family polyfit slopes
                                         per condition; positive bars flag
                                         the tone-cue × temperature finding.
    fig_mitigation_delta                Instruct + base only. Think path
                                         DA/QD/Diverse cells are N/A
                                         (conditions not collected for
                                         Think path — see taxonomy table).
    fig_think_prefix_proxy              Cross-path proxy: ber(think) vs
                                         ber(instruct) on unanimous_confident
                                         at T=0 per stage.

The old `fig_temperature_surface` is no longer produced.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    CONDITION_GRADIENT_ORDER,
    PATTERN_MATCH_REPS,
    SHARED_4_CONDITIONS,
    VARIANT_ORDER,
    april_cell_metrics,
    build_argparser,
    ensure_dir,
    load_trials_from_args,
)

# Colors: blues for Instruct path, oranges for Think path, grey for base.
VARIANT_COLORS = {
    "base": "#666666",
    "instruct_sft": "#1f77b4",
    "instruct_dpo": "#4e9ec8",
    "instruct": "#0b3d5c",
    "think_sft": "#ff7f0e",
    "think_dpo": "#ffa852",
    "think": "#8a4506",
}

VARIANT_DISPLAY = {
    "base": "Base",
    "instruct_sft": "Instruct-SFT",
    "instruct_dpo": "Instruct-DPO",
    "instruct": "Instruct (RLVR)",
    "think_sft": "Think-SFT",
    "think_dpo": "Think-DPO",
    "think": "Think (RL)",
}

INSTRUCT_FAMILY = ("base", "instruct_sft", "instruct_dpo", "instruct")
THINK_FAMILY = ("think_sft", "think_dpo", "think")
STAGE_ORDER = ["base", "sft", "dpo", "rl"]

# Short display labels for the 12 conditions so x-axis doesn't wrap.
CONDITION_SHORT = {
    "control": "control",
    "asch_history_5": "asch_history_5",
    "asch_zhu_unbiased_unanimous_plain": "unan_plain",
    "asch_zhu_unbiased_unanimous_confident": "unan_confident",
    "asch_zhu_unbiased_unanimous_uncertain": "unan_uncertain",
    "asch_zhu_unbiased_unanimous_neutral": "unan_neutral",
    "asch_zhu_unbiased_da": "devil's_advocate",
    "asch_zhu_unbiased_qd": "question_dist",
    "asch_zhu_unbiased_diverse_plain": "diverse_peers",
    "authoritative_bias": "auth_bias",
    "authority_zhu_unbiased_trust": "auth_trust",
    "authority_zhu_unbiased_trust_da": "auth_trust_da",
}


def _save(fig, path_noext: str) -> None:
    fig.savefig(path_noext + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(path_noext + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_stage_trajectory
# ---------------------------------------------------------------------------


def plot_stage_trajectory(cells: pd.DataFrame, out_dir: str) -> None:
    """Two side-by-side plots: T=0 and T=0.6, base -> SFT -> DPO -> RL."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, temp in zip(axes, (0.0, 0.6)):
        sub = cells[
            (cells["temperature"] == temp)
            & (cells["condition_name"] == "asch_zhu_unbiased_unanimous_confident")
        ].copy()
        sub = sub.set_index("variant")

        def _series(variants):
            return [
                sub.loc[v, "ber"] if v in sub.index else np.nan
                for v in variants
            ]

        instr = _series(["base", "instruct_sft", "instruct_dpo", "instruct"])
        think = _series(["base", "think_sft", "think_dpo", "think"])

        ax.plot(STAGE_ORDER, instr, "o-", color="#1f77b4", lw=2.2, ms=7,
                label="Instruct path")
        # Think at T=0.6 has no Think-RL point. Draw the SFT→DPO segment
        # solid and leave the DPO→RL segment unplotted.
        valid_pairs = [(s, v) for s, v in zip(STAGE_ORDER, think) if not np.isnan(v)]
        if len(valid_pairs) >= 2:
            xs = [p[0] for p in valid_pairs]
            ys = [p[1] for p in valid_pairs]
            ax.plot(xs, ys, "s--", color="#ff7f0e", lw=2.2, ms=7,
                    label="Think path")
        # Individual markers to anchor visible points even when a stage is missing.
        for s, v in zip(STAGE_ORDER, think):
            if not np.isnan(v):
                ax.plot([s], [v], "s", color="#ff7f0e", ms=7)
        if temp == 0.6 and any(np.isnan(v) for v in think):
            ax.annotate(
                "Think-RL not collected at T=0.6",
                xy=("rl", 0.05), xytext=(2.2, 0.12),
                fontsize=8, color="#8a4506",
                arrowprops=dict(arrowstyle="->", color="#8a4506", lw=0.8),
            )
        ax.set_xlabel("Post-training stage")
        if temp == 0.0:
            ax.set_ylabel("BER (wrong-answer endorsement)")
        ax.set_title(f"T = {temp:.1f}")
        ax.set_ylim(0, 0.9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Post-training stage effects on unanimous-confident pressure (7B OLMo)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_stage_trajectory"))


# ---------------------------------------------------------------------------
# fig_2axis_heatmap (split + combined)
# ---------------------------------------------------------------------------


def _heatmap_single(ax, pivot, title, vmin=0, vmax=0.8):
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{CONDITION_SHORT.get(c, c)}\n(reps={PATTERN_MATCH_REPS[c]})"
         for c in pivot.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([VARIANT_DISPLAY[v] for v in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    ax.set_title(title, fontsize=10)
    return im


def plot_2axis_heatmap(cells: pd.DataFrame, out_dir: str) -> None:
    t0 = cells[cells["temperature"] == 0.0].copy()
    t0 = t0[t0["variant"].isin(VARIANT_ORDER)]
    full_pivot = t0.pivot(index="variant", columns="condition_name", values="ber")

    # Panel A: Instruct family (base + 3 Instruct stages) × 12 conditions
    instruct_idx = [v for v in INSTRUCT_FAMILY if v in full_pivot.index]
    panel_a = full_pivot.reindex(
        index=instruct_idx,
        columns=list(CONDITION_GRADIENT_ORDER),
    )
    # Panel B: Think family × 4 shared conditions
    think_idx = [v for v in THINK_FAMILY if v in full_pivot.index]
    shared_ordered = [c for c in CONDITION_GRADIENT_ORDER if c in SHARED_4_CONDITIONS]
    panel_b = full_pivot.reindex(
        index=think_idx,
        columns=shared_ordered,
    )

    # --- Panel A standalone -----------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(11, 3.6))
    im_a = _heatmap_single(
        ax_a, panel_a,
        "BER per (Instruct variant × condition) at T=0 — conditions ordered by target-answer repetition count",
    )
    cbar = fig_a.colorbar(im_a, ax=ax_a, shrink=0.85)
    cbar.set_label("BER")
    fig_a.tight_layout()
    _save(fig_a, os.path.join(out_dir, "figures", "fig_2axis_heatmap_instruct"))

    # --- Panel B standalone -----------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(5.5, 2.8))
    im_b = _heatmap_single(
        ax_b, panel_b,
        "BER per (Think variant × shared condition) at T=0",
    )
    cbar = fig_b.colorbar(im_b, ax=ax_b, shrink=0.85)
    cbar.set_label("BER")
    fig_b.tight_layout()
    _save(fig_b, os.path.join(out_dir, "figures", "fig_2axis_heatmap_think"))

    # --- Combined two-panel figure with a single shared colorbar ---------
    # GridSpec: top panel wide (12 cond), bottom panel narrow (4 cond),
    # bottom x-aligned under the 4 matching columns of the top panel.
    n_top = len(panel_a.columns)
    n_bot = len(panel_b.columns)
    fig_c, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12.5, 6.6),
        gridspec_kw={"height_ratios": [len(panel_a.index), len(panel_b.index)]},
    )
    _heatmap_single(
        ax_top, panel_a,
        "Panel A — Instruct family × 12 conditions (T=0)",
    )
    _heatmap_single(
        ax_bot, panel_b,
        "Panel B — Think family × 4 shared conditions (T=0)",
    )
    # Align Panel B's x-axis under the matching 4 columns of Panel A
    # (both panels use the same column ordering because CONDITION_GRADIENT_ORDER
    # controls it). We adjust subplot widths so the 4-column Panel B is
    # horizontally aligned with the 4-column subset of Panel A.
    box_top = ax_top.get_position()
    box_bot = ax_bot.get_position()
    # Find the x-pixels of the 4 shared columns in Panel A
    col_positions = {c: i for i, c in enumerate(panel_a.columns)}
    shared_indices = [col_positions[c] for c in panel_b.columns]
    left_frac = (shared_indices[0] + 0.0) / n_top
    right_frac = (shared_indices[-1] + 1.0) / n_top
    new_left = box_top.x0 + (box_top.x1 - box_top.x0) * left_frac
    new_width = (box_top.x1 - box_top.x0) * (right_frac - left_frac)
    ax_bot.set_position([new_left, box_bot.y0, new_width, box_bot.height])
    fig_c.suptitle(
        "BER heatmap at T=0 (panels column-aligned on the 4 shared conditions)",
        fontsize=11,
    )
    # Colorbar sized against the top panel
    cax = fig_c.add_axes([0.93, 0.25, 0.015, 0.5])
    fig_c.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=0.8), cmap="RdYlBu_r"),
        cax=cax,
    ).set_label("BER")
    _save(fig_c, os.path.join(out_dir, "figures", "fig_2axis_heatmap_combined"))


# ---------------------------------------------------------------------------
# fig_pattern_match_gradient  (asymmetric Think coverage)
# ---------------------------------------------------------------------------


def plot_pattern_match_gradient(cells: pd.DataFrame, out_dir: str) -> None:
    t0 = cells[cells["temperature"] == 0.0].copy()
    t0["reps"] = t0["condition_name"].map(PATTERN_MATCH_REPS)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    # Instruct family: solid lines across full 12-condition range
    for variant in INSTRUCT_FAMILY:
        sub = t0[t0["variant"] == variant].sort_values("reps")
        ax.plot(
            sub["reps"].to_numpy(),
            sub["ber"].to_numpy(),
            "o-",
            color=VARIANT_COLORS[variant],
            lw=1.8, ms=5,
            label=VARIANT_DISPLAY[variant],
            alpha=0.9,
        )
    # Think family: markers only at the 3 distinct rep counts available
    # from the 4 shared conditions ({0, 1, 5}); dashed segments.
    for variant in THINK_FAMILY:
        sub = t0[t0["variant"] == variant].copy()
        if sub.empty:
            continue
        # Average BER per rep count for this variant (handles rep=1 having
        # 2 conditions in the shared set: authoritative_bias + authority_trust).
        grouped = sub.groupby("reps")["ber"].mean().sort_index()
        ax.plot(
            grouped.index.to_numpy(),
            grouped.values,
            "s--",
            color=VARIANT_COLORS[variant],
            lw=1.6, ms=6,
            label=VARIANT_DISPLAY[variant],
            alpha=0.9,
        )
    ax.set_xlabel("Target-answer repetitions in peer/authority block")
    ax.set_ylabel("BER")
    ax.set_title(
        "Pattern-match gradient at T=0\n"
        "(Instruct: 12 conditions; Think: 3 distinct rep counts from 4 shared conditions)",
        fontsize=10,
    )
    ax.set_xticks([0, 1, 4, 5])
    ax.set_ylim(-0.02, 0.82)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_pattern_match_gradient"))


# ---------------------------------------------------------------------------
# fig_instruct_temperature_sweep  (Q1a, replaces one of the 3 old surface plots)
# ---------------------------------------------------------------------------


def plot_instruct_temperature_sweep(cells: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    conds = [
        ("asch_zhu_unbiased_unanimous_confident", "Unanimous confident tone"),
        ("asch_zhu_unbiased_unanimous_plain", "Unanimous plain (no tone cue)"),
    ]
    for ax, (cond, title) in zip(axes, conds):
        sub = cells[cells["condition_name"] == cond].copy()
        for variant in INSTRUCT_FAMILY:
            s = sub[sub["variant"] == variant].sort_values("temperature")
            if s.empty:
                continue
            ax.plot(
                s["temperature"].to_numpy(),
                s["ber"].to_numpy(),
                "o-",
                color=VARIANT_COLORS[variant],
                lw=1.8, ms=5,
                label=VARIANT_DISPLAY[variant],
                alpha=0.9,
            )
        ax.set_xlabel("Sampling temperature T")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.9)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3)
    # Annotate the Finding-4 peak on the left panel
    sft = cells[
        (cells["variant"] == "instruct_sft")
        & (cells["condition_name"] == "asch_zhu_unbiased_unanimous_confident")
    ].sort_values("temperature")
    if not sft.empty:
        peak_row = sft.loc[sft["ber"].idxmax()]
        axes[0].annotate(
            f"Peak T={peak_row['temperature']:.1f}\nBER={peak_row['ber']:.2f}",
            xy=(peak_row["temperature"], peak_row["ber"]),
            xytext=(0.25, 0.88),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
        )
    axes[0].set_ylabel("BER (wrong-answer endorsement)")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.suptitle(
        "Instruct family temperature sweep (7B OLMo; Think family cannot be plotted at mid-T — coverage gap)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_instruct_temperature_sweep"))


# ---------------------------------------------------------------------------
# fig_temperature_t0_vs_t06_scatter  (Q1b)
# ---------------------------------------------------------------------------


def plot_temperature_t0_vs_t06_scatter(cells: pd.DataFrame, out_dir: str) -> None:
    # Build 28 points: 7 variants × 4 shared conditions
    sub = cells[cells["condition_name"].isin(SHARED_4_CONDITIONS)].copy()
    sub = sub[sub["variant"].isin(VARIANT_ORDER)]
    t0 = sub[sub["temperature"] == 0.0][["variant", "condition_name", "ber"]]
    t0 = t0.rename(columns={"ber": "ber_t0"})
    t06 = sub[sub["temperature"] == 0.6][["variant", "condition_name", "ber"]]
    t06 = t06.rename(columns={"ber": "ber_t06"})
    merged = t0.merge(t06, on=["variant", "condition_name"], how="inner")
    # Think-RL (variant=="think") only has T=0, so it drops out of the merge.
    # Document that explicitly in the caption.

    marker_map = {
        "control": "s",
        "asch_zhu_unbiased_unanimous_confident": "o",
        "authoritative_bias": "^",
        "authority_zhu_unbiased_trust": "D",
    }
    cond_label_map = {
        "control": "control",
        "asch_zhu_unbiased_unanimous_confident": "unan_confident",
        "authoritative_bias": "auth_bias",
        "authority_zhu_unbiased_trust": "auth_trust",
    }

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    # Diagonal y=x
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", zorder=1)
    # Scatter points
    for _, row in merged.iterrows():
        variant = row["variant"]
        cond = row["condition_name"]
        ax.scatter(
            row["ber_t0"], row["ber_t06"],
            s=85,
            marker=marker_map.get(cond, "o"),
            color=VARIANT_COLORS[variant],
            edgecolor="black", lw=0.6,
            zorder=3,
        )
    # Label the instruct_sft unanimous_confident anchor point
    anchor = merged[
        (merged["variant"] == "instruct_sft")
        & (merged["condition_name"] == "asch_zhu_unbiased_unanimous_confident")
    ]
    if not anchor.empty:
        x = float(anchor["ber_t0"].iloc[0])
        y = float(anchor["ber_t06"].iloc[0])
        ax.annotate(
            "Instruct-SFT\nunan_confident\n(Finding 4)",
            xy=(x, y), xytext=(x - 0.22, y + 0.12),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        )
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("BER at T = 0.0")
    ax.set_ylabel("BER at T = 0.6")
    ax.set_title(
        "Temperature concentration map — 4 shared conditions × 6 variants\n"
        "(Points above diagonal = BER rises with sampling noise; below = argmax concentrates endorsement.)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.25)

    # Variant legend
    var_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=VARIANT_COLORS[v], markerfacecolor=VARIANT_COLORS[v],
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label=VARIANT_DISPLAY[v])
        for v in INSTRUCT_FAMILY + THINK_FAMILY[:2]  # Think-RL drops out
    ]
    # Condition legend
    cond_handles = [
        plt.Line2D([0], [0], marker=m, linestyle="",
                   color="gray", markerfacecolor="gray",
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label=cond_label_map[c])
        for c, m in marker_map.items()
    ]
    leg1 = ax.legend(handles=var_handles, title="Variant", loc="upper left",
                     fontsize=7, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cond_handles, title="Condition", loc="lower right",
              fontsize=7, framealpha=0.9)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_temperature_t0_vs_t06_scatter"))


# ---------------------------------------------------------------------------
# fig_temperature_slope_bars  (Q1c — Instruct family only)
# ---------------------------------------------------------------------------


def plot_temperature_slope_bars(cells: pd.DataFrame, out_dir: str) -> None:
    # Slopes: polyfit of BER vs T per (variant, condition) across all 6 T points.
    rows = []
    for (variant, cond), g in cells[
        cells["variant"].isin(INSTRUCT_FAMILY)
    ].groupby(["variant", "condition_name"]):
        g = g.sort_values("temperature")
        ts = g["temperature"].to_numpy(dtype=float)
        bers = g["ber"].to_numpy(dtype=float)
        if len(ts) < 3:
            slope = np.nan
        else:
            slope, _ = np.polyfit(ts, bers, deg=1)
        rows.append({
            "variant": variant,
            "condition_name": cond,
            "slope": float(slope),
        })
    slopes = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    conds = list(CONDITION_GRADIENT_ORDER)
    x = np.arange(len(conds))
    bar_w = 0.19
    for i, variant in enumerate(INSTRUCT_FAMILY):
        sub = slopes[slopes["variant"] == variant].set_index("condition_name")
        vals = [sub.loc[c, "slope"] if c in sub.index else 0.0 for c in conds]
        ax.bar(
            x + (i - 1.5) * bar_w,
            vals,
            width=bar_w,
            color=VARIANT_COLORS[variant],
            edgecolor="black", lw=0.4,
            label=VARIANT_DISPLAY[variant],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CONDITION_SHORT.get(c, c)}\n(reps={PATTERN_MATCH_REPS[c]})" for c in conds],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylabel("Polyfit slope of BER vs T")
    ax.set_title(
        "Temperature sensitivity per condition (Instruct family; slopes fit across 6 T points)\n"
        "Negative = pattern-completion signature (cooler concentrates BER). Positive = SFT tone-cue amplification at mid-T.",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_temperature_slope_bars"))


# ---------------------------------------------------------------------------
# fig_mitigation_delta  (Instruct + base only)
# ---------------------------------------------------------------------------


def plot_mitigation_delta(cells: pd.DataFrame, out_dir: str) -> None:
    t0 = cells[cells["temperature"] == 0.0].copy()
    anchor = (
        t0[t0["condition_name"] == "asch_zhu_unbiased_unanimous_plain"]
        .set_index("variant")["ber"]
    )

    mitigation_conds = {
        "Devil's Advocate": "asch_zhu_unbiased_da",
        "Question Distillation": "asch_zhu_unbiased_qd",
        "Diverse peers": "asch_zhu_unbiased_diverse_plain",
    }

    records = []
    for label, cond in mitigation_conds.items():
        ber_by_variant = t0[t0["condition_name"] == cond].set_index("variant")["ber"]
        for variant in INSTRUCT_FAMILY:
            if variant in ber_by_variant.index and variant in anchor.index:
                delta = float(ber_by_variant.loc[variant] - anchor.loc[variant])
                records.append({
                    "mitigation": label,
                    "variant": variant,
                    "delta": delta,
                })
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mitigations = list(mitigation_conds.keys())
    x = np.arange(len(mitigations))
    bar_w = 0.19
    for i, variant in enumerate(INSTRUCT_FAMILY):
        sub = df[df["variant"] == variant].set_index("mitigation")
        vals = [sub.loc[m, "delta"] if m in sub.index else 0.0 for m in mitigations]
        ax.bar(
            x + (i - 1.5) * bar_w,
            vals,
            width=bar_w,
            color=VARIANT_COLORS[variant],
            edgecolor="black", lw=0.4,
            label=VARIANT_DISPLAY[variant],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mitigations)
    ax.set_ylabel("Delta BER vs unanimous_plain anchor")
    ax.set_title(
        "Pattern-break mitigation effects on the Instruct family (T=0)\n"
        "Think-path mitigation cells intentionally absent — DA/QD/Diverse conditions not collected on HPC Think runs.",
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_mitigation_delta"))


# ---------------------------------------------------------------------------
# fig_think_prefix_proxy  (Q3 weak-proxy claim for Think-path mitigation)
# ---------------------------------------------------------------------------


def plot_think_prefix_proxy(cells: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: BER(instruct_*) vs BER(think_*) per stage on unanimous_confident at T=0."""
    conf = "asch_zhu_unbiased_unanimous_confident"
    t0 = cells[
        (cells["temperature"] == 0.0) & (cells["condition_name"] == conf)
    ].set_index("variant")["ber"]

    stages = [
        ("base", "base", "base"),
        ("sft", "instruct_sft", "think_sft"),
        ("dpo", "instruct_dpo", "think_dpo"),
        ("rl", "instruct", "think"),
    ]
    labels = []
    instr_vals = []
    think_vals = []
    for stage, ivar, tvar in stages:
        if ivar not in t0.index or tvar not in t0.index:
            continue
        labels.append(stage)
        instr_vals.append(float(t0.loc[ivar]))
        think_vals.append(float(t0.loc[tvar]))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(labels))
    bar_w = 0.35
    ax.bar(x - bar_w/2, instr_vals, width=bar_w,
           color="#1f77b4", edgecolor="black", lw=0.5, label="Instruct path")
    ax.bar(x + bar_w/2, think_vals, width=bar_w,
           color="#ff7f0e", edgecolor="black", lw=0.5, label="Think path")
    for xi, iv, tv in zip(x, instr_vals, think_vals):
        ax.text(xi - bar_w/2, iv + 0.01, f"{iv:.2f}", ha="center", fontsize=8)
        ax.text(xi + bar_w/2, tv + 0.01, f"{tv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("BER on unanimous_confident (T=0)")
    ax.set_title(
        "<think> prefix as a weak pattern-break proxy\n"
        "(same items, same condition, matched post-training stage)",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figures", "fig_think_prefix_proxy"))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    args = build_argparser("Emit April_analysis figures").parse_args()
    df = load_trials_from_args(args)
    cells = april_cell_metrics(df)

    fig_dir = ensure_dir(os.path.join(args.out_dir, "figures"))

    # Sweep out any stale PDFs from the previous run (fig_temperature_surface
    # and any abandoned variants) to avoid a confusing mix of old+new figures.
    for stale in (
        "fig_temperature_surface.pdf",
        "fig_temperature_surface.png",
        "fig_2axis_heatmap.pdf",
        "fig_2axis_heatmap.png",
    ):
        stale_path = Path(fig_dir) / stale
        if stale_path.exists():
            stale_path.unlink()

    plot_stage_trajectory(cells, args.out_dir)
    plot_2axis_heatmap(cells, args.out_dir)
    plot_pattern_match_gradient(cells, args.out_dir)
    plot_instruct_temperature_sweep(cells, args.out_dir)
    plot_temperature_t0_vs_t06_scatter(cells, args.out_dir)
    plot_temperature_slope_bars(cells, args.out_dir)
    plot_mitigation_delta(cells, args.out_dir)
    plot_think_prefix_proxy(cells, args.out_dir)

    written = sorted(Path(fig_dir).glob("*.pdf"))
    print(f"[figures] wrote {len(written)} PDFs to {fig_dir}")
    for p in written:
        print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
