"""Split the Pillar I figure into two standalone, slides-ready figures.

The original Pillar I figure (paper/figures/fig6_pillar_1_sft_priors.pdf) packs
two distinct stories into one cramped 9.5x4 inch frame:

  (a) Bar chart: P(response has list | prompt has list) per source, top 12.
  (b) Scatter:   per-source affirm-prefix rate vs. max-run-≥5 rate.

For talks the bar chart is the headline ('this is sharply concentrated in
math/science'), and the scatter is a sanity check ('the two surface-mirroring
features are partially distinct'). Each deserves its own slide. This script
produces two separate output files:

  paper/figures/fig_pillar_1_carryover_bars.{pdf,png}
  paper/figures/fig_pillar_1_affirm_vs_runlength.{pdf,png}

The scatter version uses a wider canvas with explicit axis padding and
two-tier label placement so the dense cluster of sources between 0 and 10%
on the affirm-prefix axis is readable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Same maps as paper/generate_paper_figures.py — keep in sync.
DOMAIN_COLORS_AUDIT = {
    "math": "#3F88C5",
    "science": "#2EA853",
    "general": "#E69138",
    "unmapped": "#888888",
}

# Curated mapping from source dataset to broad domain label. Pulled directly
# from paper/generate_paper_figures.py so the colors stay consistent.
SOURCE_DATASET_TO_DOMAIN = {
    "Tulu 3 Persona GSM": "math",
    "Tulu 3 Persona MATH": "math",
    "Tulu 3 Persona Algebra": "math",
    "OpenMathInstruct 2": "math",
    "Dolci Instruct OpenThoughts3+ Science": "science",
    "SciRiff": "science",
    "WildGuardMix": "general",
    "Wildchat": "general",
    "OpenAssistant": "general",
    "Aya": "general",
    "FLAN": "general",
    "CoCoNot": "general",
}

BY_SOURCE_CSV = ROOT / "dataset_analysis/results/phase5_instruct-sft_by_source.csv"
SUMMARY_JSON = ROOT / "dataset_analysis/results/phase5_instruct-sft_summary.json"

OUT_DIR = ROOT / "paper" / "figures"
OUT_BARS_PDF = OUT_DIR / "fig_pillar_1_carryover_bars.pdf"
OUT_BARS_PNG = OUT_DIR / "fig_pillar_1_carryover_bars.png"
OUT_SCAT_PDF = OUT_DIR / "fig_pillar_1_affirm_vs_runlength.pdf"
OUT_SCAT_PNG = OUT_DIR / "fig_pillar_1_affirm_vs_runlength.png"


def short_label(name: str) -> str:
    return (
        name.replace("Dolci Instruct ", "Dolci:")
            .replace("Tulu 3 Persona ", "Tulu:")
    )


def load_data() -> tuple[pd.DataFrame, float | None]:
    df = pd.read_csv(BY_SOURCE_CSV)
    summary = json.load(open(SUMMARY_JSON))
    corpus_mean = summary.get("P(resp_has_list | prompt_has_list)", None)
    df["domain"] = df["source_dataset"].map(lambda s: SOURCE_DATASET_TO_DOMAIN.get(s, "unmapped"))
    df["color"] = df["domain"].map(lambda d: DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"]))
    return df, corpus_mean


# ─────────────────────────────────────────────────────────────────────────────
# Figure A: Formatting Carryover by Source (top 12)
# ─────────────────────────────────────────────────────────────────────────────

def render_carryover_bars(df: pd.DataFrame, corpus_mean: float | None) -> None:
    bar_df = (
        df.dropna(subset=["P_carryover"])
          .sort_values("P_carryover", ascending=False)
          .head(12)
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    y_pos = np.arange(len(bar_df))
    bar_colors = [
        DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"])
        for d in bar_df["domain"]
    ]

    ax.barh(
        y_pos, bar_df["P_carryover"].values,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        height=0.65, zorder=3,
    )

    if corpus_mean is not None:
        ax.axvline(
            corpus_mean, color="black", linewidth=1.1, linestyle="--", zorder=4,
            label=f"Corpus mean ({corpus_mean:.2f})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_label(s) for s in bar_df["source_dataset"]], fontsize=10)
    ax.set_xlabel(r"$P(\,\mathrm{response\ has\ list}\,\mid\,\mathrm{prompt\ has\ list}\,)$",
                  fontsize=11)
    ax.set_title("Dolci-Instruct-SFT: list-formatting carryover by source\n"
                 "(top 12 of 22; math and science demonstrations dominate)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.invert_yaxis()

    for i, val in enumerate(bar_df["P_carryover"].values):
        ax.text(val + 0.012, i, f"{val:.2f}", va="center", fontsize=9, zorder=5)

    seen_domains = list(dict.fromkeys(bar_df["domain"]))
    legend_patches = [
        Patch(facecolor=DOMAIN_COLORS_AUDIT[d], edgecolor="black", linewidth=0.4,
              label=d.capitalize())
        for d in seen_domains
    ]
    if corpus_mean is not None:
        legend_patches.append(
            Line2D([0], [0], color="black", linewidth=1.1, linestyle="--",
                   label=f"Corpus mean ({corpus_mean:.2f})")
        )
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
              framealpha=0.9, edgecolor="grey")

    fig.savefig(OUT_BARS_PDF, bbox_inches="tight")
    fig.savefig(OUT_BARS_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_BARS_PDF.relative_to(ROOT)}")
    print(f"Wrote: {OUT_BARS_PNG.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B: Affirm-prefix rate vs Run-length≥5 rate per source (de-overlapped)
# ─────────────────────────────────────────────────────────────────────────────

# Leader-line callouts for the densely-clustered sources in the left panel.
# Each entry maps source_dataset -> (label_x_data, label_y_data) in the same
# axis units as the scatter (% on both axes). The marker stays where the data
# says, and a thin grey line connects the marker to the label position.
# This keeps the left panel readable without needing log axes or jittering.
LABEL_CALLOUTS: dict[str, tuple[float, float]] = {
    # Cluster near (0, 0). Stagger labels in two columns up the right side
    # of the cluster so leader lines don't tangle and labels don't overlap.
    # Column at x=2.0 (lower row), column at x=3.5 (upper row).
    "Dolci Instruct Python Algorithms":      (2.0, 6.6),
    "OpenMathInstruct 2":                    (2.0, 5.7),
    "Tulu 3 Persona Algebra":                (2.0, 4.8),
    "Aya":                                   (2.0, 3.9),
    "Tulu 3 Persona GSM":                    (2.0, 3.0),
    "SciRiff":                               (2.0, 2.1),
    "FLAN":                                  (4.6, 6.6),
    "Logic Puzzles":                         (4.6, 5.7),
    "TableGPT":                              (4.6, 4.8),  # actual y=1.38, push up to lane
    # Mid-range cluster (1.5-9% affirm). These are spread enough that
    # short offsets work; we DON'T use callouts so the panel stays clean.
    "Dolci Instruct OpenThoughts3+ Science": (4.6, 3.9),
    "Dolci Instruct Precise IF":             (4.6, 3.0),
    "Tulu 3 Persona MATH":                   (4.6, 2.1),
    "WildGuardMix":                          (8.5, 0.4),
    "OpenAssistant":                         (8.5, 1.2),
    "Dolci Instruct Tool Use":               (8.5, 2.0),
    "CoCoNot":                               (8.5, 2.8),
    "WildJailbreak":                         (8.5, 3.6),
    "Verifiable Reasoning":                  (10.5, 6.3),
}

# Right-panel labels in data coords (mirroring the left-panel callout style).
# Without callouts the three labels at x ∈ [22, 44], y < 0.5 collide; placing
# them in three distinct vertical lanes resolves it.
LABEL_CALLOUTS_RIGHT: dict[str, tuple[float, float]] = {
    "Hardcoded Data":        (16.5, 4.0),
    "Tulu 3 Persona Python": (22.5, 1.5),
    "Evol CodeAlpaca":       (33.0, 3.2),
    "Wildchat":              (43.0, 1.5),
}


def _draw_scatter(ax, scat_df: pd.DataFrame) -> None:
    """Common scatter rendering for both axes; caller sets xlim, ylim, ticks."""
    marker_sizes = np.log1p(scat_df["n"].values) * 32
    scatter_colors = [
        DOMAIN_COLORS_AUDIT.get(d, DOMAIN_COLORS_AUDIT["unmapped"])
        for d in scat_df["domain"]
    ]
    ax.scatter(
        scat_df["affirm_pct"], scat_df["runlen_pct"],
        s=marker_sizes, c=scatter_colors,
        edgecolor="black", linewidth=0.6,
        alpha=0.85, zorder=5,
    )
    ax.grid(alpha=0.2, zorder=0)


def render_scatter(df: pd.DataFrame) -> None:
    """Two-panel broken-x scatter: zoom into the dense 0-12% region on the
    left, show the right-tail outliers (12-50%) on the right. The split keeps
    every label readable without resorting to a log axis (which would distort
    the visual interpretation of effect sizes).
    """
    scat_df = df.dropna(subset=["affirm_rate", "max_run_geq_5_rate", "n"]).copy()
    scat_df = scat_df[scat_df["n"] > 0]
    scat_df["affirm_pct"] = scat_df["affirm_rate"] * 100
    scat_df["runlen_pct"] = scat_df["max_run_geq_5_rate"] * 100

    rho, pval = stats.spearmanr(scat_df["affirm_rate"], scat_df["max_run_geq_5_rate"])

    SPLIT = 12.0  # boundary between zoom panel and right-tail panel

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16.0, 7.5), sharey=True,
        gridspec_kw={"width_ratios": [3.0, 1.4], "wspace": 0.05},
    )

    # Render the same scatter on both axes, then constrain x-limits.
    _draw_scatter(ax_left, scat_df)
    _draw_scatter(ax_right, scat_df)

    ax_left.set_xlim(-1.0, SPLIT)
    ax_right.set_xlim(SPLIT, 50.0)
    ax_left.set_ylim(-0.4, 7.4)
    ax_right.set_ylim(-0.4, 7.4)

    ax_left.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax_right.set_xticks([15, 20, 25, 30, 35, 40, 45, 50])
    ax_left.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])

    # "Broken axis" decoration — hide the inner spines, draw small diagonal
    # marks where the axis is broken so the reader sees the discontinuity.
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(axis="y", which="both", length=0)
    d = 0.012
    kw = dict(transform=ax_left.transAxes, color="grey", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kw)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw = dict(transform=ax_right.transAxes, color="grey", clip_on=False, linewidth=1.0)
    ax_right.plot((-d, +d), (-d, +d), **kw)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kw)

    # Left panel: leader-line callouts. Each label is positioned at a
    # hand-tuned (x, y) in data coords and connected back to its marker by a
    # thin grey line. This decouples label placement from marker position so
    # the dense 0-1% cluster becomes legible.
    for _, row in scat_df.iterrows():
        name = row["source_dataset"]
        x_marker, y_marker = row["affirm_pct"], row["runlen_pct"]

        if x_marker <= SPLIT:
            label_xy = LABEL_CALLOUTS.get(name)
            if label_xy is None:
                # Fallback: simple offset.
                ax_left.annotate(
                    short_label(name),
                    xy=(x_marker, y_marker),
                    xytext=(8, 6),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=9.5, zorder=6,
                )
                continue
            x_lbl, y_lbl = label_xy
            # Decide horizontal alignment so the label hangs off the leader-line nicely.
            ha = "left" if x_lbl >= x_marker else "right"
            # Draw the leader line (thin, grey, behind the marker).
            ax_left.annotate(
                "",
                xy=(x_marker, y_marker),
                xytext=(x_lbl, y_lbl),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    linewidth=0.7,
                    shrinkA=2, shrinkB=4,
                ),
                zorder=4,
            )
            ax_left.text(
                x_lbl, y_lbl, short_label(name),
                fontsize=9.5, ha=ha, va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none", alpha=0.9),
            )
        else:
            label_xy = LABEL_CALLOUTS_RIGHT.get(name)
            if label_xy is None:
                ax_right.annotate(
                    short_label(name),
                    xy=(x_marker, y_marker),
                    xytext=(8, 6),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=9.5, zorder=6,
                )
                continue
            x_lbl, y_lbl = label_xy
            ha = "left" if x_lbl >= x_marker else "right"
            ax_right.annotate(
                "",
                xy=(x_marker, y_marker),
                xytext=(x_lbl, y_lbl),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    linewidth=0.7,
                    shrinkA=2, shrinkB=4,
                ),
                zorder=4,
            )
            ax_right.text(
                x_lbl, y_lbl, short_label(name),
                fontsize=9.5, ha=ha, va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none", alpha=0.9),
            )

    # Stats annotation in the right panel (lots of empty space there).
    pval_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
    ax_right.text(
        0.97, 0.97,
        rf"Spearman $\rho = {rho:.2f}$" + f"\n$p = {pval_str}$" +
        f"\n$n = {len(scat_df)}$ sources",
        transform=ax_right.transAxes,
        fontsize=10.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="grey", linewidth=0.6, alpha=0.95),
    )

    # Single shared x-label, centered across both panels.
    fig.supxlabel("Affirm-prefix rate (% of responses)", fontsize=12, y=0.02)
    ax_left.set_ylabel(r"Max literal run-length $\geq$ 5 rate (% of responses)", fontsize=12)

    fig.suptitle(
        "Two surface-mirroring features in Dolci-Instruct-SFT, per source dataset\n"
        "(left: zoomed dense 0–12% cluster • right: right-tail outliers 12–50%)",
        fontsize=12.5, fontweight="bold", y=0.99,
    )

    # Domain legend (left panel, upper-left).
    seen_domains = list(dict.fromkeys(scat_df["domain"]))
    legend_patches = [
        Patch(facecolor=DOMAIN_COLORS_AUDIT[d], edgecolor="black", linewidth=0.4,
              label=d.capitalize())
        for d in seen_domains
    ]
    # Domain legend lives on the right panel near the top; the right panel
    # has plenty of empty space and the left panel needs every inch for labels.
    leg = ax_right.legend(handles=legend_patches, loc="upper left", fontsize=10,
                          framealpha=0.95, edgecolor="grey",
                          title="Domain", title_fontsize=10.5)
    leg.set_zorder(20)
    ax_right.add_artist(leg)

    # Corpus-size marker legend on the right panel.
    n_min, n_max = scat_df["n"].min(), scat_df["n"].max()
    n_med = int(np.exp((np.log1p(n_min) + np.log1p(n_max)) / 2)) - 1
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=np.sqrt(np.log1p(n) * 32),
               label=f"n ≈ {n:,}")
        for n in [int(n_min), n_med, int(n_max)]
    ]
    ax_right.legend(handles=size_legend, loc="lower right", fontsize=9.5,
                    framealpha=0.95, edgecolor="grey",
                    title="Corpus size", title_fontsize=10,
                    labelspacing=1.4, handletextpad=1.0)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.10)
    fig.savefig(OUT_SCAT_PDF, bbox_inches="tight")
    fig.savefig(OUT_SCAT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_SCAT_PDF.relative_to(ROOT)}")
    print(f"Wrote: {OUT_SCAT_PNG.relative_to(ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, corpus_mean = load_data()
    render_carryover_bars(df, corpus_mean)
    render_scatter(df)


if __name__ == "__main__":
    main()
