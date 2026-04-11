"""
Shared helpers for Analysis Scripts/april_analysis/* drivers.

All April_analysis scripts load trial rows through the canonical
vivarium.analytics.behavioral.load_april_trials() function and write their
output under Comparing_Experiments/April_analysis/. This module centralizes
CLI parsing, Wilson CIs, and a small set of classification helpers so that
every driver script behaves consistently.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Repo root is the parent of "Analysis Scripts/"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vivarium.analytics.behavioral import (  # noqa: E402
    APRIL_PATH_OF,
    APRIL_STAGE_OF,
    APRIL_VALID_VARIANTS,
    april_cell_denominator,
    april_cell_metrics,
    april_classify_state,
    load_april_trials,
)


DEFAULT_MANIFEST = str(
    _REPO_ROOT / "Comparing_Experiments" / "April_analysis" / "metadata" / "runs_metadata.json"
)
DEFAULT_CROSS_FAMILY_MANIFEST = str(
    _REPO_ROOT / "Comparing_Experiments" / "April_analysis" / "metadata" / "cross_family_metadata.json"
)
DEFAULT_OUT_DIR = str(_REPO_ROOT / "Comparing_Experiments" / "April_analysis")

# The four conditions that are shared between the Instruct path's 12-condition
# suite and the Think-RL 4-condition suite. Cross-path comparisons live on
# this subset.
SHARED_4_CONDITIONS = (
    "control",
    "asch_zhu_unbiased_unanimous_confident",
    "authoritative_bias",
    "authority_zhu_unbiased_trust",
)

# Ordered variant list used for every row-wise table / plot so that tables can
# be visually compared without hand-sorting.
VARIANT_ORDER = (
    "base",
    "instruct_sft",
    "instruct_dpo",
    "instruct",
    "think_sft",
    "think_dpo",
    "think",
)


def build_argparser(description: str) -> argparse.ArgumentParser:
    """Shared CLI for every April_analysis driver."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to runs_metadata.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output root under which tables/, figures/, etc. are written "
             f"(default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also load sources_secondary from the manifest (runs/think/ Think-RL 4-cond DB)",
    )
    return parser


def load_trials_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Shared trial loader: applies the canonical label policy."""
    df = load_april_trials(
        manifest_path=args.manifest,
        include_secondary=bool(args.include_secondary),
        require_judge=True,
    )
    if df.empty:
        raise SystemExit(f"[april_analysis] load_april_trials returned 0 rows from {args.manifest}")
    df["state"] = april_classify_state(df)
    return df


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Wilson score interval for a binomial proportion.
# Scoped here (rather than importing scipy) so the Phase B tables can be
# generated in low-dependency environments.
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """
    Wilson score interval (two-sided) for a binomial proportion.

    Returns (point_estimate, ci_low, ci_high).
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (p, lo, hi)


def add_wilson_ci_columns(
    df: pd.DataFrame,
    k_col: str,
    n_col: str,
    out_prefix: str,
) -> pd.DataFrame:
    """Append {prefix}_p, {prefix}_lo, {prefix}_hi to `df`."""
    out = df.copy()
    rows = out.apply(lambda r: wilson_ci(int(r[k_col]), int(r[n_col])), axis=1)
    out[f"{out_prefix}_p"] = [r[0] for r in rows]
    out[f"{out_prefix}_lo"] = [r[1] for r in rows]
    out[f"{out_prefix}_hi"] = [r[2] for r in rows]
    return out


# ---------------------------------------------------------------------------
# Target-answer repetition count: the pattern-match score used for the
# pre-registered 12-condition gradient test (see April_analysis plan C2).
# ---------------------------------------------------------------------------
# The rendered prompt for each condition contains a fixed number of
# peer/authority "votes" for the wrong answer (target). This count is the
# simplest principled pattern-completion score: how many times does the
# target token-sequence appear in the context before the model answers.
PATTERN_MATCH_REPS = {
    # Pressure conditions with a single anchor target repetition count.
    "control": 0,
    "asch_history_5": 5,
    "asch_zhu_unbiased_unanimous_plain": 5,
    "asch_zhu_unbiased_unanimous_confident": 5,
    "asch_zhu_unbiased_unanimous_uncertain": 5,
    "asch_zhu_unbiased_unanimous_neutral": 5,
    "asch_zhu_unbiased_da": 4,          # devil's advocate = 4 of 5 repeat the target
    "asch_zhu_unbiased_qd": 1,          # question distillation rewrites once
    "asch_zhu_unbiased_diverse_plain": 1,  # diverse peers = only 1 target mention
    "authoritative_bias": 1,            # one authority statement
    "authority_zhu_unbiased_trust": 1,
    "authority_zhu_unbiased_trust_da": 1,
}

# Condition ordering by pattern-match repetition count (ascending). This is the
# x-axis of fig_pattern_match_gradient.pdf.
CONDITION_GRADIENT_ORDER = sorted(
    PATTERN_MATCH_REPS.keys(), key=lambda c: (PATTERN_MATCH_REPS[c], c)
)


def print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"[{label}] rows={len(df):,}", file=sys.stderr)


# ===========================================================================
# Cross-family + ablation helpers (April_analysis expansion, 2026-04-09)
# ===========================================================================
#
# These are parallel to the 7B helpers above but scoped entirely to the
# cross_family_metadata.json manifest + experiment_group="cross_family"
# assertion bundle in load_april_trials(). They MUST NOT be called from the
# existing 7B driver scripts — the 7B pipeline stays byte-identical.

# Ordered cross-family model list (used for consistent bar-chart / table
# ordering). Rank is by descending BER on unanimous_confident T=0 from the
# pre-expansion cross-family analysis, but because that ordering is
# data-dependent we fall back to "alphabetical within architecture cluster"
# here. The actual headline figure re-sorts on the fly.
CROSS_FAMILY_MODEL_ORDER = (
    # dense first
    "OLMo-32B-Instruct",
    "Llama-3-8B",
    "Llama-3.1-70B",
    "GPT-4o-Mini",
    # MoE
    "Llama-4-Maverick (MoE)",
    "GPT-OSS-20B",
    "Gemini-2.5-Flash-Lite",
    # Think
    "OLMo-32B-Think",
    "OLMo-32B-Think-SFT",
    "OLMo-32B-Think-DPO",
    "Grok-4.1-Fast",
    # Constitutional
    "Claude-Sonnet-4",
)

# The 4 shared conditions that every cross-family model was evaluated on.
# Same set as SHARED_4_CONDITIONS; re-exported here so cross_family_* scripts
# import one single name.
CROSS_FAMILY_CONDITIONS = SHARED_4_CONDITIONS

# The 2 ablation conditions (not canonicalized).
ABLATION_CONDITIONS = (
    "asch_zhu_naked_unanimous_confident",
    "ngram_sequence_baseline",
)

# The 2 models that had system_style:none ablation runs.
ABLATION_MODELS = (
    "meta-llama/llama-3.1-70b-instruct",
    "allenai/olmo-3.1-32b-instruct",
)


def build_cross_family_argparser(description: str) -> argparse.ArgumentParser:
    """Shared CLI for every cross-family / ablation driver.

    Uses a distinct default manifest (DEFAULT_CROSS_FAMILY_MANIFEST) so no
    driver can accidentally point at the 7B runs_metadata.json. A hard guard
    in load_cross_family_trials_from_args() asserts the resolved manifest is
    NOT the 7B default.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_CROSS_FAMILY_MANIFEST,
        help=f"Path to cross_family_metadata.json "
             f"(default: {DEFAULT_CROSS_FAMILY_MANIFEST})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output root under which tables/, figures/, etc. are written "
             f"(default: {DEFAULT_OUT_DIR})",
    )
    return parser


def load_cross_family_trials_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Cross-family trial loader.

    Routes through the canonical load_april_trials() with
    experiment_group="cross_family" so the CF1-CF5 assertion bundle fires
    instead of the Think R1-R4 bundle. Attaches the state column and reads
    the manifest dict once more for short_name / architecture annotation.
    """
    manifest_path = os.path.abspath(args.manifest)
    if manifest_path == os.path.abspath(DEFAULT_MANIFEST):
        raise SystemExit(
            "[cross_family] Refusing to run cross-family loader against the "
            "7B runs_metadata.json. Pass --manifest cross_family_metadata.json."
        )

    df = load_april_trials(
        manifest_path=manifest_path,
        include_secondary=False,
        require_judge=True,
        experiment_group="cross_family",
    )
    if df.empty:
        raise SystemExit(
            f"[cross_family] load_april_trials returned 0 rows from {manifest_path}"
        )

    # Attach 3-state label (A/B/C/D), same classifier as the 7B pipeline.
    df["state"] = april_classify_state(df)

    # Attach short_name + architecture columns from the manifest so downstream
    # scripts don't re-parse JSON.
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    short_names = manifest.get("model_short_names") or {}
    arch_tags_raw = manifest.get("architecture_tags") or {}
    # Strip helper "comment" keys
    arch_tags = {k: v for k, v in arch_tags_raw.items() if k != "comment"}

    df["short_name"] = df["model_id"].map(short_names).fillna(df["model_id"])
    df["architecture"] = df["short_name"].map(arch_tags).fillna("unknown")
    return df


def load_cross_family_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the cross_family_metadata.json dict (utility)."""
    path = manifest_path or DEFAULT_CROSS_FAMILY_MANIFEST
    with open(path, "r") as f:
        return json.load(f)
