"""Render a stage trajectory across post-training stages with paired
McNemar significance annotations placed inline next to each point.

Convention:
  - Instruct path: significance label sits ABOVE each point.
  - Think path:    significance label sits BELOW each point.

Two panels: T=0.0 (left), T=0.6 (right). Same condition as the paper figure:
asch_zhu_unbiased_unanimous_confident vs control.

McNemar p-values are computed FROM paired trial-level data using the canonical
April loader (vivarium.analytics.behavioral.load_april_trials), pairing on
item_id within each (variant, temperature) cell. Refusals/unclassified are
excluded from the contingency table (consistent with paper Table 3 methodology).

Usage from repo root:
    python scripts/render_slide_fig1_stage_trajectory.py
Outputs: paper/figures/fig_stage_trajectory_with_mcnemar.{pdf,png}
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vivarium.analytics.behavioral import (  # noqa: E402
    april_classify_state,
    load_april_trials,
)

MANIFEST = ROOT / "Comparing_Experiments" / "April_analysis" / "metadata" / "runs_metadata.json"

OUT_DIR = ROOT / "paper" / "figures"
OUT_PDF = OUT_DIR / "fig_stage_trajectory_with_mcnemar.pdf"
OUT_PNG = OUT_DIR / "fig_stage_trajectory_with_mcnemar.png"

CONTROL = "control"
PRESSURE = "asch_zhu_unbiased_unanimous_confident"

# Stage ordering (shared by both paths).
STAGES = ["base", "sft", "dpo", "rl"]
STAGE_LABEL = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rl": "RLVR"}

INSTRUCT_VARIANT = {"base": "base", "sft": "instruct_sft", "dpo": "instruct_dpo", "rl": "instruct"}
THINK_VARIANT = {"base": "base", "sft": "think_sft", "dpo": "think_dpo", "rl": "think"}

INSTRUCT_COLOR = "#2E5BFF"
THINK_COLOR = "#FF9500"


# -- McNemar helpers ---------------------------------------------------------

def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial CDF on min(b, c) of n=b+c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_half_n = -n * math.log(2)
    cdf = 0.0
    for i in range(k + 1):
        cdf += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
    return min(1.0, 2 * cdf)


def sig_marker(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "p<0.0001"
    if p < 1e-3:
        return f"p={p:.4f}"
    return f"p={p:.3f}"


# -- Paired McNemar from trial data -----------------------------------------

def paired_mcnemar_for_cell(
    df: pd.DataFrame, *, variant: str, temperature: float
) -> tuple[float, int, int, int]:
    """Compute paired (BER on pressure, b, c, n_pairs) for one (variant, T) cell.

    Pairing is on item_id. We include only items where BOTH the control AND the
    pressure trial produced a definite answer (state ∈ {A_correct, B_wrong_endorsed}).
    Refusals (C) and unclassified (D) trials are dropped from the pairing,
    matching the paper Table 3 methodology.
    """
    cell = df[(df["variant"] == variant) & (df["temperature"] == temperature)]
    if cell.empty:
        return float("nan"), 0, 0, 0

    ctl = cell[cell["condition_name"] == CONTROL].set_index("item_id")["state"]
    prs = cell[cell["condition_name"] == PRESSURE].set_index("item_id")["state"]
    common = ctl.index.intersection(prs.index)

    paired = pd.DataFrame({
        "control_state": ctl.loc[common],
        "pressure_state": prs.loc[common],
    })
    keep = paired[
        paired["control_state"].isin(["A_correct", "B_wrong_endorsed"])
        & paired["pressure_state"].isin(["A_correct", "B_wrong_endorsed"])
    ]
    if keep.empty:
        return float("nan"), 0, 0, 0

    # Pressure-condition BER on the full cell (B / 400 over all items in the
    # pressure condition for this variant/temp). This matches the paper's BER
    # definition (fixed-N denominator), distinct from the McNemar contingency
    # which uses only paired definite-answer items.
    pressure_full = cell[cell["condition_name"] == PRESSURE]
    n_pressure = len(pressure_full)
    n_wrong_pressure = (pressure_full["state"] == "B_wrong_endorsed").sum()
    ber = n_wrong_pressure / n_pressure if n_pressure else float("nan")

    # b = correct under control AND wrong under pressure (truth_override)
    # c = wrong under control AND correct under pressure (truth_rescue)
    b = int(((keep["control_state"] == "A_correct") & (keep["pressure_state"] == "B_wrong_endorsed")).sum())
    c = int(((keep["control_state"] == "B_wrong_endorsed") & (keep["pressure_state"] == "A_correct")).sum())
    return ber, b, c, len(keep)


def annotate_point(ax, x: float, y: float, label: str, *, above: bool, color: str) -> None:
    offset = 18 if above else -22
    va = "bottom" if above else "top"
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )


def render_panel(ax, df: pd.DataFrame, *, temperature: float) -> None:
    xs = list(range(len(STAGES)))

    # Instruct path
    inst_ys, inst_lbls = [], []
    for stage in STAGES:
        v = INSTRUCT_VARIANT[stage]
        ber, b, c, n = paired_mcnemar_for_cell(df, variant=v, temperature=temperature)
        inst_ys.append(ber)
        if math.isnan(ber):
            inst_lbls.append("missing")
        else:
            p = mcnemar_exact_two_sided(b, c)
            inst_lbls.append(f"{sig_marker(p)}\n{fmt_p(p)}\nb={b}, c={c}, n={n}")
    ax.plot(xs, inst_ys, marker="o", markersize=10, linewidth=2.4,
            color=INSTRUCT_COLOR, label="Instruct path", zorder=3)
    for x, y, lbl in zip(xs, inst_ys, inst_lbls):
        if not math.isnan(y):
            annotate_point(ax, x, y, lbl, above=True, color=INSTRUCT_COLOR)

    # Think path
    th_ys, th_lbls = [], []
    for stage in STAGES:
        v = THINK_VARIANT[stage]
        ber, b, c, n = paired_mcnemar_for_cell(df, variant=v, temperature=temperature)
        th_ys.append(ber)
        if math.isnan(ber):
            th_lbls.append("missing")
        else:
            p = mcnemar_exact_two_sided(b, c)
            th_lbls.append(f"{sig_marker(p)}\n{fmt_p(p)}\nb={b}, c={c}, n={n}")
    valid = [(x, y) for x, y in zip(xs, th_ys) if not math.isnan(y)]
    if valid:
        ax.plot(
            [p[0] for p in valid],
            [p[1] for p in valid],
            marker="s", markersize=10, linewidth=2.4,
            color=THINK_COLOR, linestyle="--",
            label="Think path", zorder=3,
        )
    for x, y, lbl in zip(xs, th_ys, th_lbls):
        if not math.isnan(y):
            annotate_point(ax, x, y, lbl, above=False, color=THINK_COLOR)

    ax.set_xticks(xs)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
    ax.set_xlabel("Post-training stage")
    ax.set_ylabel("BER (wrong-answer endorsement)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(-0.4, len(STAGES) - 0.6)
    ax.set_title(f"T = {temperature}")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading paired trial-level data via load_april_trials() ...")
    df = load_april_trials(manifest_path=str(MANIFEST), include_secondary=True, require_judge=True)
    df["state"] = april_classify_state(df)

    print(f"Loaded {len(df):,} trials. Variants present: {sorted(df['variant'].unique())}")
    print(f"Temperatures present: {sorted(df['temperature'].unique())}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    render_panel(ax0, df, temperature=0.0)
    render_panel(ax1, df, temperature=0.6)

    fig.suptitle(
        "Post-training stage effects on unanimous-confident peer pressure (Olmo-3 7B)\n"
        "Paired McNemar (exact, two-sided) — Instruct path: above markers • Think path: below markers",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"Wrote: {OUT_PDF.relative_to(ROOT)}")
    print(f"Wrote: {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
