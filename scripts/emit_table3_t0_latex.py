"""Emit the LaTeX rows for paper Table 3 (within-OLMo McNemar) at T=0 only.

Reads Comparing_Experiments/expanded_results/olmo_family/tables/mcnemar_pressure_vs_control_t0.csv
and prints LaTeX rows in the same column order paper Table 3 uses.

Usage: python scripts/emit_table3_t0_latex.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    ROOT
    / "Comparing_Experiments"
    / "expanded_results"
    / "olmo_family"
    / "tables"
    / "mcnemar_pressure_vs_control_t0.csv"
)

# Display order of variants and conditions, mirroring the paper.
VARIANT_ORDER = ["base", "instruct_sft", "instruct_dpo", "instruct"]
VARIANT_DISPLAY = {
    "base": "Base",
    "instruct_sft": "Instruct-SFT",
    "instruct_dpo": "Instruct-DPO",
    "instruct": "Instruct",
}

CONDITION_ORDER = [
    ("asch_history_5", "Asch-5 free-text"),
    ("asch_zhu_unbiased_unanimous_plain", "Unan.\\ plain"),
    ("asch_zhu_unbiased_unanimous_neutral", "Unan.\\ neutral"),
    ("asch_zhu_unbiased_unanimous_confident", "Unan.\\ confident"),
    ("asch_zhu_unbiased_unanimous_uncertain", "Unan.\\ uncertain"),
    ("asch_zhu_unbiased_diverse_plain", "Diverse peers"),
    ("asch_zhu_unbiased_da", "Devil's advocate"),
    ("asch_zhu_unbiased_qd", "Question distill."),
    ("authoritative_bias", "Auth.\\ bias"),
    ("authority_zhu_unbiased_trust", "Auth.\\ trust"),
    ("authority_zhu_unbiased_trust_da", "Trust + DA"),
]


def haldane_log_or_ci(b: int, c: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Haldane–Anscombe log-OR with Wald 95 % CI on the log scale.
    Returns (OR, OR_lower, OR_upper)."""
    log_or = math.log((b + 0.5) / (c + 0.5))
    se = math.sqrt(1.0 / (b + 0.5) + 1.0 / (c + 0.5))
    z = 1.959963984540054  # two-sided 95 %
    lo = math.exp(log_or - z * se)
    hi = math.exp(log_or + z * se)
    return math.exp(log_or), lo, hi


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "$<10^{-4}$"
    if p < 1e-3:
        return "$<10^{-3}$"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def sig_marker(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def main() -> None:
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_key = {(r["variant"], r["condition"]): r for r in rows}

    for variant in VARIANT_ORDER:
        first = True
        for cond_key, cond_label in CONDITION_ORDER:
            r = by_key.get((variant, cond_key))
            if r is None:
                continue
            n_pairs = int(r["n_pairs"])
            b = int(r["b_truth_override"])
            c = int(r["c_truth_rescue"])
            p_adj = float(r["p_adjusted"])
            or_, lo, hi = haldane_log_or_ci(b, c)
            sig = sig_marker(p_adj)

            variant_cell = VARIANT_DISPLAY[variant] if first else ""
            first = False
            print(
                rf" {variant_cell} & {cond_label} & {n_pairs} & {b} & {c} & "
                rf"{or_:.2f} [{lo:.2f}, {hi:.2f}]\, {{\tiny {sig}}} & {fmt_p(p_adj)} \\"
            )
        print(r"\midrule" if variant != VARIANT_ORDER[-1] else "")


if __name__ == "__main__":
    main()
