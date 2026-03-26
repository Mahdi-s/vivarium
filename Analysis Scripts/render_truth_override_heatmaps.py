#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = ["base", "instruct", "instruct_sft", "think", "think_sft", "rl_zero"]
TOPIC_ORDER = ["general", "knowledge", "math", "reasoning", "science", "truthfulness"]


def _draw_heatmap(ax: plt.Axes, data: np.ndarray, *, xticklabels: list[str], yticklabels: list[str]) -> None:
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="Reds")
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=10)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if np.isnan(v):
                continue
            color = "white" if v >= 0.6 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xlabel("Temperature", fontsize=11)
    return im


def render(
    df: pd.DataFrame,
    *,
    pressure_condition: str,
    out_pdf: Path,
    out_png: Path,
    title: str,
) -> None:
    sub = df[df["pressure_condition"] == pressure_condition].copy()
    if sub.empty:
        raise SystemExit(f"No rows for pressure_condition={pressure_condition}")

    temps = sorted({float(t) for t in sub["temperature"].unique()})
    temp_labels = [f"{t:.1f}" if t % 1 else f"{int(t)}.0" for t in temps]

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 8.5), constrained_layout=True)
    axes = axes.flatten()

    ims = []
    for i, variant in enumerate(VARIANT_ORDER):
        ax = axes[i]
        vsub = sub[sub["variant"] == variant]
        pivot = (
            vsub.pivot_table(index="dataset_category", columns="temperature", values="truth_override_rate", aggfunc="mean")
            .reindex(index=TOPIC_ORDER, columns=temps)
        )
        data = pivot.to_numpy(dtype=float)
        im = _draw_heatmap(ax, data, xticklabels=temp_labels, yticklabels=TOPIC_ORDER if i % ncols == 0 else [""] * len(TOPIC_ORDER))
        ax.set_title(variant, fontsize=12)
        ims.append(im)

    for j in range(len(VARIANT_ORDER), len(axes)):
        axes[j].axis("off")

    cbar = fig.colorbar(ims[0], ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("truth_override_rate", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--truth-override-csv",
        type=str,
        default="Comparing_Experiments/expanded_suite_analysis/tables/factual_truth_override_rates.csv",
    )
    ap.add_argument("--out-dir", type=str, default="Comparing_Experiments/expanded_suite_analysis/figures")
    ap.add_argument("--also-write-paper-figures", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.truth_override_csv)
    out_dir = Path(args.out_dir)

    render(
        df,
        pressure_condition="asch_history_5",
        out_pdf=out_dir / "factual_truth_override_asch_heatmaps.pdf",
        out_png=out_dir / "factual_truth_override_asch_heatmaps.png",
        title="Factual Tasks: Truth-Override Rate Under asch_history_5 (P(pressure incorrect | control correct))",
    )
    render(
        df,
        pressure_condition="authoritative_bias",
        out_pdf=out_dir / "factual_truth_override_authority_heatmaps.pdf",
        out_png=out_dir / "factual_truth_override_authority_heatmaps.png",
        title="Factual Tasks: Truth-Override Rate Under authoritative_bias (P(pressure incorrect | control correct))",
    )

    if args.also_write_paper_figures:
        paper_dir = Path("paper/figures")
        for name in [
            "factual_truth_override_asch_heatmaps.pdf",
            "factual_truth_override_asch_heatmaps.png",
            "factual_truth_override_authority_heatmaps.pdf",
            "factual_truth_override_authority_heatmaps.png",
        ]:
            (paper_dir / name).write_bytes((out_dir / name).read_bytes())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

