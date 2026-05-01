"""Reproduce the BER heatmap (Figure 2-style) with two configurable variants.

Outputs (both written to paper/figures/):
  1. Per-temperature panel: one heatmap per temperature ∈ {0.0, 0.2, 0.4, 0.6,
     0.8, 1.0}. Each subplot is variants × 12 conditions, BER as the cell value.
     File: fig_ber_heatmap_per_temperature{_instruct,_all}.{pdf,png}

  2. Temperature × condition (averaged): a single heatmap whose rows are
     variants and whose columns are temperatures, with each cell holding the
     mean BER across the 11 pressure conditions (control excluded) for that
     (variant, temperature) cell. Shows how each variant's average
     pressure-response BER moves with temperature.
     File: fig_ber_heatmap_temp_averaged{_instruct,_all}.{pdf,png}

CLI flags:
  --instruct-only   Restrict to {base, instruct_sft, instruct_dpo, instruct}.
                    Default: True (per user direction; ignore Think for now).
  --include-think   Include Think variants too (overrides --instruct-only).

Run from the parent worktree (the .claude worktree's DBs are LFS stubs).
    cd /Users/mahdi/repos/abstractAgentMachine
    python scripts/render_fig2_heatmap_variants.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

INSTRUCT_FAMILY = ("base", "instruct_sft", "instruct_dpo", "instruct")
THINK_FAMILY = ("think_sft", "think_dpo", "think")

VARIANT_DISPLAY = {
    "base": "Base",
    "instruct_sft": "Instruct-SFT",
    "instruct_dpo": "Instruct-DPO",
    "instruct": "Instruct (RLVR)",
    "think_sft": "Think-SFT",
    "think_dpo": "Think-DPO",
    "think": "Think (RL)",
}

# Same 12-condition catalog and ordering used by the original Figure 2.
PATTERN_MATCH_REPS = {
    "control": 0,
    "asch_zhu_unbiased_diverse_plain": 1,
    "asch_zhu_unbiased_qd": 1,
    "authoritative_bias": 1,
    "authority_zhu_unbiased_trust": 1,
    "authority_zhu_unbiased_trust_da": 1,
    "asch_zhu_unbiased_da": 4,
    "asch_history_5": 5,
    "asch_zhu_unbiased_unanimous_confident": 5,
    "asch_zhu_unbiased_unanimous_neutral": 5,
    "asch_zhu_unbiased_unanimous_plain": 5,
    "asch_zhu_unbiased_unanimous_uncertain": 5,
}
CONDITION_GRADIENT_ORDER = sorted(
    PATTERN_MATCH_REPS.keys(),
    key=lambda c: (PATTERN_MATCH_REPS[c], c),
)
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

PRESSURE_CONDITIONS = [c for c in CONDITION_GRADIENT_ORDER if c != "control"]


def compute_ber_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BER (= P(state == B_wrong_endorsed) over a fixed N=400 cell)
    for every (variant, temperature, condition_name)."""
    df = df.copy()
    df["state"] = april_classify_state(df)
    df["is_b"] = (df["state"] == "B_wrong_endorsed").astype(int)
    grp = df.groupby(["variant", "temperature", "condition_name"])
    out = grp["is_b"].mean().reset_index().rename(columns={"is_b": "ber"})
    out["n"] = grp.size().values
    return out


def heatmap_panel(ax, pivot: pd.DataFrame, *, title: str, vmin: float = 0.0,
                  vmax: float = 0.8, show_xlabels: bool = True,
                  cmap: str = "RdYlBu_r") -> "plt.cm.ScalarMappable":
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    if show_xlabels:
        ax.set_xticklabels(
            [
                f"{CONDITION_SHORT.get(c, c)}\n(reps={PATTERN_MATCH_REPS.get(c, '?')})"
                if c in PATTERN_MATCH_REPS else str(c)
                for c in pivot.columns
            ],
            rotation=40, ha="right", fontsize=8,
        )
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [VARIANT_DISPLAY.get(v, v) for v in pivot.index], fontsize=9,
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=7, color="grey")
                continue
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title(title, fontsize=10)
    return im


# ---------------------------------------------------------------------------
# Mode 1: one heatmap per temperature
# ---------------------------------------------------------------------------

def render_per_temperature(
    ber_table: pd.DataFrame, variants: list[str], suffix: str,
) -> None:
    temps = sorted(ber_table["temperature"].unique())
    n_temps = len(temps)

    n_cols = 2
    n_rows = (n_temps + n_cols - 1) // n_cols
    height_per_row = max(2.6, 0.55 * len(variants) + 1.2)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(13.5, height_per_row * n_rows),
        squeeze=False,
    )

    for i, temp in enumerate(temps):
        ax = axes[i // n_cols][i % n_cols]
        cell = ber_table[ber_table["temperature"] == temp]
        pivot = (
            cell.pivot(index="variant", columns="condition_name", values="ber")
                .reindex(index=variants, columns=CONDITION_GRADIENT_ORDER)
        )
        last_row = (i // n_cols) == (n_rows - 1) or i + n_cols >= n_temps
        heatmap_panel(
            ax, pivot,
            title=f"T = {temp:.1f}",
            show_xlabels=last_row,
        )

    # Hide unused subplots if grid > n_temps.
    for j in range(n_temps, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    family_label = "Instruct family + Base" if "think" not in suffix else "all OLMo-3 7B variants"
    fig.suptitle(
        f"BER heatmap per temperature ({family_label})\n"
        "rows = variant • columns = 12 conditions ordered by target-answer repetition count",
        fontsize=12, y=1.005,
    )

    cbar_ax = fig.add_axes([0.94, 0.20, 0.012, 0.6])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=0.8), cmap="RdYlBu_r")
    fig.colorbar(sm, cax=cbar_ax).set_label("BER")

    fig.subplots_adjust(left=0.10, right=0.92, top=0.94, bottom=0.10, hspace=0.55, wspace=0.20)
    pdf = OUT_DIR / f"fig_ber_heatmap_per_temperature{suffix}.pdf"
    png = OUT_DIR / f"fig_ber_heatmap_per_temperature{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pdf.relative_to(ROOT)}")
    print(f"Wrote: {png.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Mode 2: variant × temperature heatmap, averaged over pressure conditions
# ---------------------------------------------------------------------------

def render_temp_averaged(
    ber_table: pd.DataFrame, variants: list[str], suffix: str,
) -> None:
    pressure = ber_table[ber_table["condition_name"].isin(PRESSURE_CONDITIONS)]

    avg = (
        pressure.groupby(["variant", "temperature"])["ber"]
                .mean()
                .reset_index()
    )
    pivot = avg.pivot(index="variant", columns="temperature", values="ber")
    pivot = pivot.reindex(index=variants)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.55 * len(variants) + 1.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=0.8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"T = {t:.1f}" for t in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([VARIANT_DISPLAY.get(v, v) for v in pivot.index], fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="grey")
                continue
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("Mean BER (across 11 pressure conditions)")

    family_label = "Instruct family + Base" if "think" not in suffix else "all OLMo-3 7B variants"
    ax.set_title(
        f"BER averaged across the 11 pressure conditions ({family_label})\n"
        "rows = variant • columns = decoding temperature • control excluded from the mean",
        fontsize=11,
    )
    fig.tight_layout()
    pdf = OUT_DIR / f"fig_ber_heatmap_temp_averaged{suffix}.pdf"
    png = OUT_DIR / f"fig_ber_heatmap_temp_averaged{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pdf.relative_to(ROOT)}")
    print(f"Wrote: {png.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-think",
        action="store_true",
        help="Include Think variants. Default: only Base + Instruct family.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trials via load_april_trials() ...")
    df = load_april_trials(
        manifest_path=str(MANIFEST),
        include_secondary=True,
        require_judge=True,
    )
    ber_table = compute_ber_table(df)

    if args.include_think:
        variants = list(INSTRUCT_FAMILY) + list(THINK_FAMILY)
        suffix = "_all"
    else:
        variants = list(INSTRUCT_FAMILY)
        suffix = "_instruct"

    # Filter to the requested variants.
    ber_table = ber_table[ber_table["variant"].isin(variants)]
    print(f"Variants in figure: {variants}")
    print(f"Temperatures present: {sorted(ber_table['temperature'].unique())}")

    render_per_temperature(ber_table, variants, suffix)
    render_temp_averaged(ber_table, variants, suffix)


if __name__ == "__main__":
    main()
