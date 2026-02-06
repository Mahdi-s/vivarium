"""Render the Asch social-pressure effect plot from aggregate tables.

This figure is used in the paper (paper/figures/social_pressure_effect.png).

It is intentionally lightweight and reproducible: it reads the strict table
produced by `expanded_suite_behavioral_breakdown.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VARIANT_ORDER = ["base", "instruct", "instruct_sft", "think", "think_sft", "rl_zero"]


def compute_asch_effects(factual_rates_path: Path) -> pd.DataFrame:
    """Compute Asch-Control error-rate deltas pooled across factual categories."""
    df = pd.read_csv(factual_rates_path)

    # Pool across topics using counts to keep weighting correct.
    pooled = (
        df.groupby(["temperature", "variant", "condition_name"], as_index=False)
        .agg(n_trials=("n_trials", "sum"), n_correct=("n_correct", "sum"))
        .copy()
    )
    pooled["error_rate"] = 1.0 - (pooled["n_correct"] / pooled["n_trials"])

    piv = pooled.pivot_table(
        index=["temperature", "variant"], columns="condition_name", values="error_rate", aggfunc="first"
    )

    # Defensive: if a condition is missing, keep it as NA.
    for col in ["control", "asch_history_5"]:
        if col not in piv.columns:
            piv[col] = np.nan

    out = piv.reset_index()
    out["delta_asch"] = out["asch_history_5"] - out["control"]
    return out[["temperature", "variant", "delta_asch"]]


def render_social_pressure_effect(
    effects: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path | None = None,
    title: str = "Social Pressure Effect by Temperature\n(Positive = Pressure Increases Errors)",
) -> None:
    sns.set_theme(style="whitegrid")

    # Keep a stable, paper-friendly ordering.
    variants = [v for v in VARIANT_ORDER if v in set(effects["variant"].unique())]
    temps = sorted(float(t) for t in effects["temperature"].unique())

    x = np.arange(len(variants))
    n_temps = len(temps)
    width = min(0.85 / max(n_temps, 1), 0.14)

    # Wide + taller than the previous version so the legend doesn't dominate.
    fig, ax = plt.subplots(figsize=(18, 7))

    # Temperature palette (cool -> warm).
    palette = sns.color_palette("viridis", n_colors=n_temps)

    for i, (temp, color) in enumerate(zip(temps, palette, strict=True)):
        sub = effects[effects["temperature"] == temp].set_index("variant")
        y = [float(sub.loc[v, "delta_asch"]) if v in sub.index else np.nan for v in variants]
        offset = (i - (n_temps - 1) / 2) * width
        ax.bar(
            x + offset,
            y,
            width,
            label=f"T={temp:.1f}",
            color=color,
            alpha=0.90,
            edgecolor="none",
        )

    ax.axhline(y=0.0, color="black", linestyle="-", linewidth=0.9)
    ax.set_ylabel("Social Pressure Effect\n(Asch Error Rate - Control Error Rate)")
    ax.set_xlabel("Model Variant")
    ax.set_title(title, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")

    # Tight but not clipped y-lims.
    finite = effects["delta_asch"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) > 0:
        lo, hi = float(finite.min()), float(finite.max())
        pad = max(0.02, 0.15 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)

    # Vertical legend at the top-right (inside the axes, so it doesn't steal width).
    ax.legend(
        title="Temperature",
        loc="upper right",
        ncol=1,
        frameon=True,
        fontsize=10,
        title_fontsize=10,
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--factual-rates-csv",
        type=str,
        default="Comparing_Experiments/expanded_suite_analysis_db_endorse/tables/factual_rates_by_temp_variant_condition_category.csv",
        help="Path to factual_rates_by_temp_variant_condition_category.csv",
    )
    ap.add_argument(
        "--also-write-paper-figures",
        action="store_true",
        help="Write paper/figures/social_pressure_effect.(png|pdf) as well as Comparing_Experiments/figures.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    table = root / args.factual_rates_csv
    if not table.exists():
        raise SystemExit(f"Missing table: {table}")

    effects = compute_asch_effects(table)

    # Always update the Comparing_Experiments snapshot.
    render_social_pressure_effect(
        effects,
        out_png=root / "Comparing_Experiments" / "figures" / "social_pressure_effect.png",
        out_pdf=root / "Comparing_Experiments" / "figures" / "social_pressure_effect.pdf",
    )

    if args.also_write_paper_figures:
        render_social_pressure_effect(
            effects,
            out_png=root / "paper" / "figures" / "social_pressure_effect.png",
            out_pdf=root / "paper" / "figures" / "social_pressure_effect.pdf",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
