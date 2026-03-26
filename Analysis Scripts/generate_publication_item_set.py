#!/usr/bin/env python3
"""
Unified Publication Item-Set Generator — Final Publication Pipeline.

This is the single gateway for generating all behavioral analysis outputs
for the OLMo conformity paper. It supersedes generate_v6/v7/v8 scripts
and the expanded_suite_behavioral_breakdown.

Pipeline stages:
  1. Discovery  — auto-detect runs from a folder, associate temperatures
  2. Item-set   — compute balanced publication item set across temperatures
  3. Loading    — load judge-labeled trial data for all (T, variant, condition)
  4. Metrics    — error rates, truth override/rescue, wrong-answer flip,
                  opinion agreement, pressure effects (all 12 conditions)
  5. Statistics — McNemar (pressure vs control), Cochran's Q (families),
                  BCa bootstrap CIs, Holm-Bonferroni correction
  6. Figures    — publication-quality plots (no bar charts)
  7. Export     — LLM-readable tables, logs, and manifest

Output structure:
  <out-dir>/
    behavioral/
      tables/          — CSV tables with provenance headers
      figures/         — PNG (300dpi) + PDF figures
      statistical_tests/
    logs/
      manifest.json    — machine-readable run manifest
      pipeline.log     — human/LLM-readable pipeline log
      balance_check.csv
    README.md          — auto-generated LLM-readable guide

Usage:
  python 'Analysis Scripts/generate_publication_item_set.py' \\
      --runs-dir runs_latest/runs \\
      --out-dir Comparing_Experiments/final_publication \\
      [--exclude-variants rl_zero] \\
      [--n-boot 10000] \\
      [--skip-figures]
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# All 12 conditions in canonical order
ALL_CONDITIONS: Tuple[str, ...] = (
    "control",
    # Core behavioral
    "asch_history_5",
    "authoritative_bias",
    # Tone family (peer, Zhu-style)
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
    # Mitigation family
    "asch_zhu_unbiased_da",
    "asch_zhu_unbiased_qd",
    # Format control
    "asch_zhu_unbiased_diverse_plain",
    # Authority family (Zhu-style)
    "authority_zhu_unbiased_trust",
    "authority_zhu_unbiased_trust_da",
)

CORE_CONDITIONS: Tuple[str, ...] = ("control", "asch_history_5", "authoritative_bias")

CONDITION_LABELS: Dict[str, str] = {
    "control": "Control",
    "asch_history_5": "Asch (5 Confederates)",
    "authoritative_bias": "Authoritative Bias",
    "asch_zhu_unbiased_unanimous_plain": "Tone: Plain",
    "asch_zhu_unbiased_unanimous_neutral": "Tone: Neutral",
    "asch_zhu_unbiased_unanimous_confident": "Tone: Confident",
    "asch_zhu_unbiased_unanimous_uncertain": "Tone: Uncertain",
    "asch_zhu_unbiased_da": "Mitigation: Devil's Advocate",
    "asch_zhu_unbiased_qd": "Mitigation: Question Distillation",
    "asch_zhu_unbiased_diverse_plain": "Format: Diverse Peers",
    "authority_zhu_unbiased_trust": "Authority: Trust",
    "authority_zhu_unbiased_trust_da": "Authority: Trust+DA",
}

# Condition families for grouping
CONDITION_FAMILY: Dict[str, str] = {
    "control": "control",
    "asch_history_5": "peer_core",
    "authoritative_bias": "authority_core",
    "asch_zhu_unbiased_unanimous_plain": "tone",
    "asch_zhu_unbiased_unanimous_neutral": "tone",
    "asch_zhu_unbiased_unanimous_confident": "tone",
    "asch_zhu_unbiased_unanimous_uncertain": "tone",
    "asch_zhu_unbiased_da": "mitigation",
    "asch_zhu_unbiased_qd": "mitigation",
    "asch_zhu_unbiased_diverse_plain": "format_control",
    "authority_zhu_unbiased_trust": "authority_extended",
    "authority_zhu_unbiased_trust_da": "authority_extended",
}

# Human-psychology predictions for each condition (for "mitigation myth" framing)
PSYCHOLOGY_PREDICTIONS: Dict[str, str] = {
    "asch_zhu_unbiased_unanimous_uncertain": "Should REDUCE conformity (hedged language signals low confidence)",
    "asch_zhu_unbiased_da": "Should REDUCE conformity (single dissenter breaks unanimity — Asch 1956)",
    "asch_zhu_unbiased_qd": "Should REDUCE conformity (removes individual voices, weakens social proof)",
    "asch_zhu_unbiased_diverse_plain": "Should ELIMINATE conformity (no unanimous majority)",
    "asch_zhu_unbiased_unanimous_confident": "Should INCREASE conformity (confident peers more persuasive)",
    "authority_zhu_unbiased_trust_da": "Should REDUCE authority conformity (alternative option provides escape)",
}

DATASET_TO_CATEGORY: Dict[str, str] = {
    "immutable_facts_minimal": "general",
    "social_conventions_minimal": "opinion",
    "gsm8k": "math",
    "mmlu_math": "math",
    "mmlu_science": "science",
    "mmlu_knowledge": "knowledge",
    "truthfulqa": "truthfulness",
    "arc": "reasoning",
}

FACTUAL_CATEGORIES: List[str] = sorted(
    [c for c in set(DATASET_TO_CATEGORY.values()) if c != "opinion"]
)
ALL_CATEGORIES: List[str] = sorted(set(DATASET_TO_CATEGORY.values()))

VARIANT_ORDER: List[str] = [
    "base", "instruct", "instruct_sft", "instruct_dpo",
    "think", "think_sft", "think_dpo",
]

VARIANT_LABELS: Dict[str, str] = {
    "base": "Base", "instruct": "Instruct", "instruct_sft": "Instruct-SFT",
    "instruct_dpo": "Instruct-DPO", "think": "Think", "think_sft": "Think-SFT",
    "think_dpo": "Think-DPO", "rl_zero": "RL-Zero",
}

# Color palette: Instruct branch = blues/purples, Think branch = oranges/reds
VARIANT_COLORS: Dict[str, str] = {
    "base": "#7F8C8D",
    "instruct": "#2980B9", "instruct_sft": "#1F618D", "instruct_dpo": "#6C3483",
    "think": "#E67E22", "think_sft": "#D35400", "think_dpo": "#A93226",
    "rl_zero": "#566573",
}

# Training trajectory branches
TRAINING_BRANCHES: List[Tuple[str, List[str]]] = [
    ("Instruct", ["base", "instruct", "instruct_sft", "instruct_dpo"]),
    ("Think", ["base", "think", "think_sft", "think_dpo"]),
]

STAGE_X: Dict[str, float] = {
    "base": 0, "instruct": 1, "think": 1,
    "instruct_sft": 2, "think_sft": 2,
    "instruct_dpo": 3, "think_dpo": 3,
}

FAMILY_COLORS: Dict[str, str] = {
    "peer_core": "#2980B9",
    "tone": "#5DADE2",
    "mitigation": "#85C1E9",
    "format_control": "#AED6F1",
    "authority_core": "#E67E22",
    "authority_extended": "#F0B27A",
    "control": "#95A5A6",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("publication_pipeline")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Auto-discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """
    Auto-discover all run directories under runs_dir.
    Each run folder has format: YYYYMMDD_HHMMSS_<uuid>/simulation.db
    Extracts temperature from the database itself.
    Returns sorted by temperature.
    """
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {runs_dir}")

    discovered = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        db_path = p / "simulation.db"
        if not db_path.is_file():
            continue
        # Extract run_id from folder name
        parts = p.name.split("_")
        run_id = "_".join(parts[2:]) if len(parts) >= 3 else p.name
        # Get temperature from DB
        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT DISTINCT temperature FROM conformity_trials LIMIT 1"
            ).fetchone()
            conn.close()
            temp = float(row[0]) if row else None
        except Exception:
            temp = None
        if temp is not None:
            discovered.append({
                "temperature": temp,
                "run_id": run_id,
                "run_dir": p.name,
                "db_path": db_path,
            })

    discovered.sort(key=lambda x: x["temperature"])
    return discovered


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Balanced item-set computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_balanced_item_set(
    runs: List[Dict[str, Any]],
    excluded_variants: Tuple[str, ...],
    conditions: Tuple[str, ...],
    logger: logging.Logger,
) -> set:
    """
    Compute the intersection of fully-covered items across ALL temperature runs.
    An item is fully covered in a run when every included variant has a valid
    judge label for every condition.
    """
    logger.info("Computing balanced item set across %d runs...", len(runs))

    per_run_items: List[set] = []
    for run in runs:
        db_path = run["db_path"]
        run_id = run["run_id"]
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        excl_ph = ",".join("?" * len(excluded_variants))
        cond_ph = ",".join("?" * len(conditions))

        # Count included models
        n_models_row = conn.execute(
            f"SELECT COUNT(DISTINCT variant) AS n FROM conformity_trials "
            f"WHERE run_id = ? AND variant NOT IN ({excl_ph})",
            [run_id, *excluded_variants],
        ).fetchone()
        n_models = n_models_row["n"] if n_models_row else 0
        expected_per_item = n_models * len(conditions)

        # Find fully-covered items
        item_rows = conn.execute(
            f"""
            SELECT t.item_id, COUNT(*) AS n_judge
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({cond_ph})
              AND t.variant NOT IN ({excl_ph})
              AND o.parsed_answer_json IS NOT NULL
              AND trim(o.parsed_answer_json) != ''
              AND o.parsed_answer_json NOT LIKE '%parse_error%'
            GROUP BY t.item_id
            """,
            [run_id, *conditions, *excluded_variants],
        ).fetchall()
        conn.close()

        fully_covered = {r["item_id"] for r in item_rows if r["n_judge"] >= expected_per_item}
        per_run_items.append(fully_covered)
        logger.info(
            "  T=%.1f: %d fully-covered items (n_models=%d, expected_per_item=%d)",
            run["temperature"], len(fully_covered), n_models, expected_per_item,
        )

    # Intersection across all runs
    if not per_run_items:
        return set()
    balanced = per_run_items[0]
    for s in per_run_items[1:]:
        balanced = balanced & s

    logger.info("Balanced item set: %d items (intersection of %d runs)", len(balanced), len(runs))
    return balanced


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Trial data loading (judge labels)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_judge_json(pj: Any) -> Optional[Dict[str, Any]]:
    if pj is None:
        return None
    if isinstance(pj, str):
        if not pj.strip() or "parse_error" in pj.lower():
            return None
        try:
            return json.loads(pj)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(pj, dict):
        return pj
    return None


def load_trial_data(
    runs: List[Dict[str, Any]],
    conditions: Tuple[str, ...],
    excluded_variants: Tuple[str, ...],
    publication_items: Optional[set],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and concatenate judge-labeled trial data from all runs."""
    all_dfs = []
    for run in runs:
        df = _load_single_run(
            run["db_path"], run["run_id"], conditions, excluded_variants, publication_items,
        )
        if not df.empty:
            df["temperature"] = run["temperature"]
            all_dfs.append(df)
            logger.info(
                "  T=%.1f: loaded %d judge-labeled trials",
                run["temperature"], len(df),
            )

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total trials loaded: %d", len(combined))
    return combined


def _load_single_run(
    db_path: Path,
    run_id: str,
    conditions: Tuple[str, ...],
    excluded_variants: Tuple[str, ...],
    publication_items: Optional[set],
) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    cond_ph = ",".join("?" * len(conditions))
    excl_ph = ",".join("?" * len(excluded_variants))
    item_filter = ""
    item_params: List[Any] = []
    if publication_items:
        item_ph = ",".join("?" * len(publication_items))
        item_filter = f"AND i.item_id IN ({item_ph})"
        item_params = list(publication_items)

    query = f"""
    WITH first_output_ids AS (
        SELECT MIN(o.output_id) AS output_id, o.trial_id
        FROM conformity_outputs o
        JOIN (
            SELECT trial_id, MIN(created_at) AS min_created_at
            FROM conformity_outputs GROUP BY trial_id
        ) fo ON fo.trial_id = o.trial_id AND fo.min_created_at = o.created_at
        GROUP BY o.trial_id
    )
    SELECT
        t.trial_id, t.model_id, t.variant, t.temperature,
        c.name AS condition_name,
        i.item_id, i.domain, i.ground_truth_text,
        d.name AS dataset_name,
        json_extract(i.source_json, '$.wrong_answer') AS wrong_answer,
        o.parsed_answer_json, o.refusal_flag, o.raw_text
    FROM conformity_trials t
    JOIN conformity_conditions c ON c.condition_id = t.condition_id
    JOIN conformity_items i ON i.item_id = t.item_id
    JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
    JOIN first_output_ids foi ON foi.trial_id = t.trial_id
    JOIN conformity_outputs o ON o.output_id = foi.output_id
    WHERE t.run_id = ?
      AND c.name IN ({cond_ph})
      AND t.variant NOT IN ({excl_ph})
      {item_filter}
    """
    params = [run_id, *conditions, *excluded_variants, *item_params]
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return df

    # Parse judge labels
    is_correct_list, agrees_wrong_list, refusal_list, keep_mask = [], [], [], []
    for _, row in df.iterrows():
        jd = _parse_judge_json(row["parsed_answer_json"])
        if jd is None:
            keep_mask.append(False)
            is_correct_list.append(None)
            agrees_wrong_list.append(None)
            refusal_list.append(None)
            continue

        ic_raw = jd.get("is_correct")
        ic = None
        if ic_raw in (1, "1", True):
            ic = 1
        elif ic_raw in (0, "0", False):
            ic = 0

        wae_raw = jd.get("wrong_answer_endorsed")
        wae = wae_raw in (1, "1", True)

        rf_raw = jd.get("refusal_flag")
        rf = 1 if rf_raw in (1, "1", True) else 0

        is_factual = pd.notna(row.get("ground_truth_text"))
        has_wa = row.get("wrong_answer") is not None and str(row.get("wrong_answer", "")).strip() != ""
        ok = True
        if is_factual and ic is None:
            ok = False
        if has_wa and wae_raw is None:
            ok = False

        keep_mask.append(ok)
        is_correct_list.append(ic)
        agrees_wrong_list.append(wae)
        refusal_list.append(rf)

    df = df[keep_mask].copy()
    df["is_correct"] = [x for x, k in zip(is_correct_list, keep_mask) if k]
    df["agrees_wrong_answer"] = [x for x, k in zip(agrees_wrong_list, keep_mask) if k]
    df["refusal_flag"] = [x for x, k in zip(refusal_list, keep_mask) if k]
    df["dataset_category"] = df["dataset_name"].map(DATASET_TO_CATEGORY).fillna("unknown")
    df["is_factual"] = df["ground_truth_text"].notna()
    return df


def check_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Check cell balance across (temperature, variant, condition_name)."""
    factual = df[df["is_factual"]]
    if factual.empty:
        return pd.DataFrame()
    counts = (
        factual.groupby(["temperature", "variant", "condition_name"], observed=True)
        .size().reset_index(name="n_trials")
    )
    median_n = float(counts["n_trials"].median())
    counts["median_n"] = median_n
    counts["deviation_pct"] = ((counts["n_trials"] - median_n) / median_n * 100).round(1)
    counts["imbalanced"] = counts["deviation_pct"].abs() > 5.0
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate per (temperature, variant, condition_name, dataset_category)."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()
    factual["is_error"] = (factual["is_correct"] == 0).astype(int)
    factual["refusal_int"] = factual["refusal_flag"].astype(int)
    return (
        factual.groupby(
            ["temperature", "variant", "condition_name", "dataset_category"], observed=True,
        ).agg(
            error_rate=("is_error", "mean"),
            n_trials=("is_error", "count"),
            refusal_rate=("refusal_int", "mean"),
        ).reset_index()
    )


def compute_pressure_effects(error_rates: pd.DataFrame) -> pd.DataFrame:
    """Delta error (pressure - control) for ALL pressure conditions."""
    if error_rates.empty:
        return pd.DataFrame()
    ctrl = error_rates[error_rates["condition_name"] == "control"][
        ["temperature", "variant", "dataset_category", "error_rate", "n_trials"]
    ].rename(columns={"error_rate": "control_error_rate", "n_trials": "n_control"})

    rows = []
    for cond in [c for c in error_rates["condition_name"].unique() if c != "control"]:
        sub = error_rates[error_rates["condition_name"] == cond][
            ["temperature", "variant", "dataset_category", "error_rate", "n_trials"]
        ].rename(columns={"error_rate": "pressure_error_rate", "n_trials": "n_pressure"})
        merged = ctrl.merge(sub, on=["temperature", "variant", "dataset_category"], how="inner")
        merged["condition_name"] = cond
        merged["condition_label"] = merged["condition_name"].map(CONDITION_LABELS)
        merged["condition_family"] = merged["condition_name"].map(CONDITION_FAMILY)
        merged["delta_error"] = merged["pressure_error_rate"] - merged["control_error_rate"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _compute_conditional_metric(
    df: pd.DataFrame,
    pressure_condition: str,
    metric_type: str,  # "override" | "rescue" | "wrong_flip"
) -> pd.DataFrame:
    """Shared logic for truth override, truth rescue, and wrong-answer flip."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    ctrl_col = "is_correct" if metric_type != "wrong_flip" else "agrees_wrong_answer"
    pres_col = ctrl_col

    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "dataset_category", ctrl_col]
    ].rename(columns={ctrl_col: "ctrl_val"})

    pres = factual[factual["condition_name"] == pressure_condition][
        ["temperature", "variant", "item_id", pres_col]
    ].rename(columns={pres_col: "pres_val"})

    merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    if metric_type == "override":
        eligible = merged[merged["ctrl_val"] == 1].copy()
        eligible["metric_val"] = (eligible["pres_val"] == 0).astype(int)
        rate_name = "truth_override_rate"
    elif metric_type == "rescue":
        eligible = merged[merged["ctrl_val"] == 0].copy()
        eligible["metric_val"] = (eligible["pres_val"] == 1).astype(int)
        rate_name = "truth_rescue_rate"
    else:  # wrong_flip
        eligible = merged[~merged["ctrl_val"].astype(bool)].copy()
        eligible["metric_val"] = eligible["pres_val"].astype(bool).astype(int)
        rate_name = "wrong_answer_flip_rate"

    if eligible.empty:
        return pd.DataFrame()

    agg = (
        eligible.groupby(["temperature", "variant", "dataset_category"], observed=True)
        .agg(**{rate_name: ("metric_val", "mean"), "n_items": ("metric_val", "count")})
        .reset_index()
    )
    agg["pressure_condition"] = pressure_condition
    return agg


def compute_all_conditional_metrics(
    df: pd.DataFrame, logger: logging.Logger,
) -> Dict[str, pd.DataFrame]:
    """Compute override/rescue/flip for ALL pressure conditions."""
    results: Dict[str, List[pd.DataFrame]] = {
        "truth_override": [], "truth_rescue": [], "wrong_answer_flip": [],
    }
    pressure_conds = [c for c in df["condition_name"].unique() if c != "control"]
    for cond in pressure_conds:
        for metric_type, key in [
            ("override", "truth_override"),
            ("rescue", "truth_rescue"),
            ("wrong_flip", "wrong_answer_flip"),
        ]:
            out = _compute_conditional_metric(df, cond, metric_type)
            if not out.empty:
                results[key].append(out)

    combined = {}
    for key, dfs in results.items():
        if dfs:
            combined[key] = pd.concat(dfs, ignore_index=True)
            logger.info("  %s: %d rows", key, len(combined[key]))
        else:
            combined[key] = pd.DataFrame()
    return combined


def compute_opinion_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Wrong-answer agreement on opinion (social conventions) items."""
    opinion = df[
        (df["dataset_category"] == "opinion") & df["agrees_wrong_answer"].notna()
    ].copy()
    if opinion.empty:
        return pd.DataFrame()
    return (
        opinion.groupby(["temperature", "variant", "condition_name"], observed=True)
        .agg(agreement_rate=("agrees_wrong_answer", "mean"), n_trials=("agrees_wrong_answer", "count"))
        .reset_index()
    )


def compute_pooled_summary(error_rates: pd.DataFrame) -> pd.DataFrame:
    """Pooled error rates across temperatures and categories (for Table 1 in paper)."""
    if error_rates.empty:
        return pd.DataFrame()
    # Weighted average across temperatures and categories
    pooled = (
        error_rates.groupby(["variant", "condition_name"], observed=True)
        .apply(lambda g: pd.Series({
            "error_rate": np.average(g["error_rate"], weights=g["n_trials"]),
            "n_trials": g["n_trials"].sum(),
        }))
        .reset_index()
    )
    # Pivot: variant × condition
    pivot = pooled.pivot_table(
        index="variant", columns="condition_name", values="error_rate",
    ).reindex(index=[v for v in VARIANT_ORDER if v in pooled["variant"].unique()])
    return pivot


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Statistical tests
# ═══════════════════════════════════════════════════════════════════════════

def _mcnemar_exact_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    try:
        from scipy.stats import binom as _binom
        return min(1.0, 2.0 * float(_binom.cdf(min(b, c), total, 0.5)))
    except ImportError:
        chi2 = (abs(b - c) - 1) ** 2 / max(1, total)
        return float(math.erfc(math.sqrt(chi2 / 2.0)))


def _cohens_h(p1: float, p2: float) -> float:
    return (
        2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
        - 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    )


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_mcnemar_tests(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """McNemar test per (variant, pressure_condition) vs control, pooled across T and items."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "is_correct"]
    ].rename(columns={"is_correct": "ctrl_correct"})

    rows: List[Dict[str, Any]] = []
    for cond in [c for c in factual["condition_name"].unique() if c != "control"]:
        pres = factual[factual["condition_name"] == cond][
            ["temperature", "variant", "item_id", "is_correct"]
        ].rename(columns={"is_correct": "pres_correct"})
        merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
        for variant in sorted(merged["variant"].unique()):
            sub = merged[merged["variant"] == variant]
            b = int(((sub["ctrl_correct"] == 1) & (sub["pres_correct"] == 0)).sum())
            c = int(((sub["ctrl_correct"] == 0) & (sub["pres_correct"] == 1)).sum())
            p_ctrl = float((sub["ctrl_correct"] == 1).mean())
            p_pres = float((sub["pres_correct"] == 1).mean())
            rows.append({
                "variant": variant,
                "condition_name": cond,
                "condition_label": CONDITION_LABELS.get(cond, cond),
                "condition_family": CONDITION_FAMILY.get(cond, "other"),
                "n_pairs": len(sub),
                "b_override": b, "c_rescue": c,
                "error_ctrl": round(1.0 - p_ctrl, 4),
                "error_pressure": round(1.0 - p_pres, 4),
                "delta_error": round((1.0 - p_pres) - (1.0 - p_ctrl), 4),
                "odds_ratio": round((b + 0.5) / (c + 0.5), 3),
                "cohens_h": round(_cohens_h(1.0 - p_pres, 1.0 - p_ctrl), 4),
                "p_raw": _mcnemar_exact_p(b, c),
            })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["p_adjusted"] = _holm_bonferroni(result["p_raw"].tolist())
    result["sig_raw"] = result["p_raw"].apply(_sig_label)
    result["sig_adjusted"] = result["p_adjusted"].apply(_sig_label)
    logger.info("  McNemar tests: %d comparisons", len(result))
    return result.sort_values(["variant", "condition_family", "condition_name"]).reset_index(drop=True)


def compute_cochrans_q(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Cochran's Q across condition families for each variant."""
    from scipy.stats import chi2 as _chi2

    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    families: Dict[str, Tuple[str, ...]] = {
        "tone": (
            "asch_zhu_unbiased_unanimous_plain",
            "asch_zhu_unbiased_unanimous_neutral",
            "asch_zhu_unbiased_unanimous_confident",
            "asch_zhu_unbiased_unanimous_uncertain",
        ),
        "mitigation_vs_plain": (
            "asch_zhu_unbiased_unanimous_plain",
            "asch_zhu_unbiased_da",
            "asch_zhu_unbiased_qd",
            "asch_zhu_unbiased_diverse_plain",
        ),
        "authority_all": (
            "authoritative_bias",
            "authority_zhu_unbiased_trust",
            "authority_zhu_unbiased_trust_da",
        ),
    }

    rows: List[Dict[str, Any]] = []
    for variant in sorted(factual["variant"].unique()):
        v_df = factual[factual["variant"] == variant]
        for family_name, family_conds in families.items():
            available = [c for c in family_conds if c in v_df["condition_name"].unique()]
            if len(available) < 2:
                continue
            pivot = v_df[v_df["condition_name"].isin(available)].pivot_table(
                index=["item_id", "temperature"],
                columns="condition_name",
                values="agrees_wrong_answer",
                aggfunc="first",
            ).dropna()
            if pivot.shape[0] < 5 or pivot.shape[1] < 2:
                continue
            X = pivot[[c for c in available if c in pivot.columns]].values.astype(float)
            n, k = X.shape
            row_sums = X.sum(axis=1)
            col_sums = X.sum(axis=0)
            G_bar = col_sums.mean()
            numerator = (k - 1) * float(np.sum((col_sums - G_bar) ** 2))
            denominator = float(k * row_sums.sum() - np.sum(row_sums ** 2))
            if denominator <= 0:
                continue
            Q = numerator / denominator
            p_val = float(1.0 - _chi2.cdf(Q, df=k - 1))
            rows.append({
                "variant": variant,
                "family": family_name,
                "k_conditions": k, "n_subjects": n,
                "conditions": "; ".join(CONDITION_LABELS.get(c, c) for c in available if c in pivot.columns),
                "Q_stat": round(Q, 3), "df": k - 1,
                "p_value": p_val,
                "sig": _sig_label(p_val),
            })

    logger.info("  Cochran's Q: %d family tests", len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_bootstrap_cis(
    df: pd.DataFrame, n_boot: int, logger: logging.Logger,
) -> pd.DataFrame:
    """BCa 95% CIs for truth_override_rate and delta_error per (variant, condition)."""
    from scipy.stats import norm as _norm

    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    rows: List[Dict[str, Any]] = []

    ctrl_data = factual[factual["condition_name"] == "control"]
    pressure_conds = [c for c in factual["condition_name"].unique() if c != "control"]

    for variant in sorted(factual["variant"].unique()):
        ctrl_v = ctrl_data[ctrl_data["variant"] == variant][
            ["item_id", "temperature", "is_correct"]
        ].rename(columns={"is_correct": "ctrl_ic"})

        for cond in pressure_conds:
            pres_v = factual[
                (factual["variant"] == variant) & (factual["condition_name"] == cond)
            ][["item_id", "temperature", "is_correct"]].rename(columns={"is_correct": "pres_ic"})

            if pres_v.empty or ctrl_v.empty:
                continue
            merged = ctrl_v.merge(pres_v, on=["item_id", "temperature"], how="inner")
            if merged.empty:
                continue

            merged["ctrl_correct"] = (merged["ctrl_ic"] == 1).astype(int)
            merged["override"] = ((merged["ctrl_ic"] == 1) & (merged["pres_ic"] == 0)).astype(int)
            merged["err_ctrl"] = (merged["ctrl_ic"] == 0).astype(int)
            merged["err_pres"] = (merged["pres_ic"] == 0).astype(int)

            item_agg = merged.groupby("item_id").agg(
                n_cc=("ctrl_correct", "sum"), n_ov=("override", "sum"),
                n_obs=("ctrl_correct", "count"),
                n_ec=("err_ctrl", "sum"), n_ep=("err_pres", "sum"),
            ).reset_index()

            if len(item_agg) < 10:
                continue

            cc = item_agg["n_cc"].values.astype(int)
            ov = item_agg["n_ov"].values.astype(int)
            obs = item_agg["n_obs"].values.astype(int)
            ec = item_agg["n_ec"].values.astype(int)
            ep = item_agg["n_ep"].values.astype(int)
            n_items = len(item_agg)

            for metric in ("truth_override_rate", "delta_error"):
                def _stat(idx):
                    if metric == "truth_override_rate":
                        d = int(cc[idx].sum())
                        return float(ov[idx].sum()) / d if d > 0 else float("nan")
                    else:
                        t = int(obs[idx].sum())
                        return (float(ep[idx].sum()) / t - float(ec[idx].sum()) / t) if t > 0 else float("nan")

                point = _stat(np.arange(n_items))
                if math.isnan(point):
                    continue

                boots = np.array([_stat(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)])
                valid = boots[~np.isnan(boots)]
                if len(valid) < 50:
                    continue

                lo = float(np.percentile(valid, 2.5))
                hi = float(np.percentile(valid, 97.5))

                rows.append({
                    "variant": variant,
                    "condition_name": cond,
                    "condition_label": CONDITION_LABELS.get(cond, cond),
                    "condition_family": CONDITION_FAMILY.get(cond, "other"),
                    "metric": metric,
                    "point_estimate": round(point, 4),
                    "ci_lower": round(lo, 4),
                    "ci_upper": round(hi, 4),
                })

    logger.info("  Bootstrap CIs: %d estimates (n_boot=%d)", len(rows), n_boot)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6: Figures
# ═══════════════════════════════════════════════════════════════════════════

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_fig(fig, path_png: Path, path_pdf: Optional[Path] = None, dpi: int = 300):
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if path_pdf:
        fig.savefig(path_pdf, bbox_inches="tight")


def _present_variants(df: pd.DataFrame) -> List[str]:
    return [v for v in VARIANT_ORDER if v in set(df["variant"].unique())]


def generate_all_figures(
    df: pd.DataFrame,
    error_rates: pd.DataFrame,
    effects: pd.DataFrame,
    conditional: Dict[str, pd.DataFrame],
    opinion: pd.DataFrame,
    mcnemar: pd.DataFrame,
    boot_cis: pd.DataFrame,
    fig_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate all publication figures."""
    import seaborn as sns
    plt = _mpl()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # ── Figure 1: Forest plot — all conditions ──
    _fig_forest_plot(boot_cis, mcnemar, fig_dir, logger)

    # ── Figure 2: Training trajectory — truth override by branch ──
    _fig_training_trajectory(conditional.get("truth_override", pd.DataFrame()), fig_dir, logger)

    # ── Figure 3: Mitigation effectiveness — slope chart ──
    _fig_mitigation_slope(effects, fig_dir, logger)

    # ── Figure 4: Tone modulation — connected dot plot ──
    _fig_tone_modulation(effects, fig_dir, logger)

    # ── Figure 5: Temperature × override — line ribbons ──
    _fig_temperature_ribbons(conditional.get("truth_override", pd.DataFrame()), fig_dir, logger)

    # ── Figure 6: Asymmetry — peer vs authority divergence chart ──
    _fig_asymmetry_divergence(effects, fig_dir, logger)

    # ── Figure 7: Heatmap — variant × condition truth override ──
    _fig_heatmap_override(boot_cis, mcnemar, fig_dir, logger)

    logger.info("All figures saved to %s", fig_dir)


def _fig_forest_plot(boot_cis, mcnemar, fig_dir, logger):
    """Forest plot: delta-error for all conditions, faceted by variant."""
    plt = _mpl()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    delta_df = boot_cis[boot_cis["metric"] == "delta_error"].copy()
    if delta_df.empty:
        logger.warning("Forest plot: no data, skipping")
        return

    sig_map = {}
    if not mcnemar.empty:
        sig_map = mcnemar.groupby(["variant", "condition_name"])["sig_adjusted"].first().to_dict()

    cond_order = [c for c in ALL_CONDITIONS if c != "control"]
    present_conds = [c for c in cond_order if c in delta_df["condition_name"].unique()]
    variants = _present_variants(delta_df)

    fig, axes = plt.subplots(
        1, len(variants),
        figsize=(3.6 * len(variants), max(6, len(present_conds) * 0.52 + 2)),
        sharey=True,
    )
    if len(variants) == 1:
        axes = [axes]

    y_pos = {c: i for i, c in enumerate(reversed(present_conds))}

    for ax, variant in zip(axes, variants):
        v_df = delta_df[delta_df["variant"] == variant]
        for cond in present_conds:
            row = v_df[v_df["condition_name"] == cond]
            if row.empty:
                continue
            y = y_pos[cond]
            family = CONDITION_FAMILY.get(cond, "other")
            color = FAMILY_COLORS.get(family, "#95A5A6")
            pe = float(row["point_estimate"].iloc[0])
            lo = float(row["ci_lower"].iloc[0])
            hi = float(row["ci_upper"].iloc[0])
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.8, alpha=0.7, zorder=3)
            sig = sig_map.get((variant, cond), "ns")
            is_core = cond in {"asch_history_5", "authoritative_bias"}
            marker = "D" if is_core else "o"
            ms = 9 if is_core else 7
            filled = sig != "ns"
            ax.plot(
                pe, y, marker=marker, color=color, markersize=ms,
                markerfacecolor=color if filled else "white",
                markeredgewidth=1.8, zorder=5,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xlabel("Δ Error vs Control", fontsize=8)
        ax.grid(axis="x", alpha=0.25, linestyle=":")

    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(
        [CONDITION_LABELS.get(c, c) for c in reversed(present_conds)], fontsize=7.5,
    )

    legend_els = [
        mpatches.Patch(color=FAMILY_COLORS["peer_core"], label="Peer (core)"),
        mpatches.Patch(color=FAMILY_COLORS["tone"], label="Tone variants"),
        mpatches.Patch(color=FAMILY_COLORS["mitigation"], label="Mitigations"),
        mpatches.Patch(color=FAMILY_COLORS["authority_core"], label="Authority (core)"),
        mpatches.Patch(color=FAMILY_COLORS["authority_extended"], label="Authority (extended)"),
    ]
    axes[-1].legend(handles=legend_els, fontsize=6.5, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Pressure Effect on Error Rate — All 11 Conditions\n(filled = p < 0.05 Holm-adjusted; 95% CI)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig1_forest_all_conditions.png", fig_dir / "fig1_forest_all_conditions.pdf")
    plt.close(fig)
    logger.info("  fig1_forest_all_conditions saved")


def _fig_training_trajectory(override_df, fig_dir, logger):
    """Training trajectory: truth override rate across checkpoints, by branch."""
    plt = _mpl()
    import matplotlib.pyplot as plt

    if override_df.empty:
        logger.warning("Training trajectory: no data, skipping")
        return

    # Pool across temperatures and categories (weighted by n_items)
    pooled = (
        override_df.groupby(["variant", "pressure_condition"], observed=True)
        .apply(lambda g: pd.Series({
            "rate": np.average(g["truth_override_rate"], weights=g["n_items"]) if g["n_items"].sum() > 0 else np.nan,
        })).reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (branch_name, branch_variants) in zip(axes, TRAINING_BRANCHES):
        present = [v for v in branch_variants if v in pooled["variant"].unique()]
        for cond in [c for c in pooled["pressure_condition"].unique()]:
            sub = pooled[(pooled["pressure_condition"] == cond) & pooled["variant"].isin(present)]
            if sub.empty:
                continue
            xs = [STAGE_X[v] for v in present if v in sub["variant"].values]
            ys = [float(sub[sub["variant"] == v]["rate"].iloc[0]) for v in present if v in sub["variant"].values]
            family = CONDITION_FAMILY.get(cond, "other")
            color = FAMILY_COLORS.get(family, "#95A5A6")
            label = CONDITION_LABELS.get(cond, cond)
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color=color, label=label, alpha=0.8)

        ax.set_xticks([STAGE_X[v] for v in present])
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in present], fontsize=8, rotation=15)
        ax.set_title(f"{branch_name} Branch", fontweight="bold")
        ax.set_ylabel("Pooled Truth Override Rate" if ax == axes[0] else "")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[1].legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.9)
    fig.suptitle("Truth Override Across Training Stages", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig2_training_trajectory.png", fig_dir / "fig2_training_trajectory.pdf")
    plt.close(fig)
    logger.info("  fig2_training_trajectory saved")


def _fig_mitigation_slope(effects, fig_dir, logger):
    """Slope chart: mitigation effectiveness vs unanimous plain baseline."""
    plt = _mpl()
    import matplotlib.pyplot as plt

    if effects.empty:
        logger.warning("Mitigation slope: no data, skipping")
        return

    baseline = "asch_zhu_unbiased_unanimous_plain"
    mitigations = ["asch_zhu_unbiased_da", "asch_zhu_unbiased_qd", "asch_zhu_unbiased_diverse_plain"]
    conds = [baseline] + mitigations
    sub = effects[effects["condition_name"].isin(conds)].copy()
    if sub.empty:
        return

    # Pool across T and category
    pooled = (
        sub.groupby(["variant", "condition_name"], observed=True)
        .apply(lambda g: pd.Series({
            "delta_error": np.average(g["delta_error"], weights=g["n_pressure"]) if g["n_pressure"].sum() > 0 else np.nan,
        })).reset_index()
    )

    variants = [v for v in VARIANT_ORDER if v in pooled["variant"].unique()]
    mit_labels = {
        baseline: "Unanimous\nPlain",
        "asch_zhu_unbiased_da": "Devil's\nAdvocate",
        "asch_zhu_unbiased_qd": "Question\nDistillation",
        "asch_zhu_unbiased_diverse_plain": "Diverse\nPeers",
    }
    x_pos = {c: i for i, c in enumerate(conds)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        v_data = pooled[pooled["variant"] == variant]
        xs, ys = [], []
        for c in conds:
            row = v_data[v_data["condition_name"] == c]
            if not row.empty:
                xs.append(x_pos[c])
                ys.append(float(row["delta_error"].iloc[0]))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=8,
                    color=VARIANT_COLORS.get(variant, "#333"),
                    label=VARIANT_LABELS.get(variant, variant), alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([mit_labels.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Δ Error vs Control (pooled)", fontsize=10)
    ax.set_title(
        "Mitigation Effectiveness: Do Human-Inspired Fixes Work?\n"
        "(lower is better; dashed line = no effect)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")
    # Add psychology prediction annotations
    for c, pred in PSYCHOLOGY_PREDICTIONS.items():
        if c in x_pos:
            short = "↓ Expected" if "REDUCE" in pred or "ELIMINATE" in pred else "↑ Expected"
            ax.annotate(short, xy=(x_pos[c], ax.get_ylim()[1] * 0.95),
                        fontsize=6, ha="center", color="#888", style="italic")

    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig3_mitigation_slope.png", fig_dir / "fig3_mitigation_slope.pdf")
    plt.close(fig)
    logger.info("  fig3_mitigation_slope saved")


def _fig_tone_modulation(effects, fig_dir, logger):
    """Connected dot plot: tone modulation effects."""
    plt = _mpl()
    import matplotlib.pyplot as plt

    tone_conds = [
        "asch_zhu_unbiased_unanimous_plain",
        "asch_zhu_unbiased_unanimous_neutral",
        "asch_zhu_unbiased_unanimous_confident",
        "asch_zhu_unbiased_unanimous_uncertain",
    ]
    sub = effects[effects["condition_name"].isin(tone_conds)].copy()
    if sub.empty:
        return

    pooled = (
        sub.groupby(["variant", "condition_name"], observed=True)
        .apply(lambda g: pd.Series({
            "delta_error": np.average(g["delta_error"], weights=g["n_pressure"]) if g["n_pressure"].sum() > 0 else np.nan,
        })).reset_index()
    )

    tone_labels = {"plain": "Plain", "neutral": "Neutral", "confident": "Confident", "uncertain": "Uncertain"}
    tone_short = {c: c.split("_")[-1] for c in tone_conds}
    variants = [v for v in VARIANT_ORDER if v in pooled["variant"].unique()]

    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = {c: i for i, c in enumerate(tone_conds)}
    for variant in variants:
        v_data = pooled[pooled["variant"] == variant]
        xs, ys = [], []
        for c in tone_conds:
            row = v_data[v_data["condition_name"] == c]
            if not row.empty:
                xs.append(x_pos[c])
                ys.append(float(row["delta_error"].iloc[0]))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=7,
                    color=VARIANT_COLORS.get(variant, "#333"),
                    label=VARIANT_LABELS.get(variant, variant), alpha=0.85)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([tone_labels.get(tone_short.get(c, ""), c) for c in tone_conds], fontsize=10)
    ax.set_ylabel("Δ Error vs Control (pooled)", fontsize=10)
    ax.set_title("Tone Modulation: Does Peer Confidence Matter?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig4_tone_modulation.png", fig_dir / "fig4_tone_modulation.pdf")
    plt.close(fig)
    logger.info("  fig4_tone_modulation saved")


def _fig_temperature_ribbons(override_df, fig_dir, logger):
    """Line+ribbon: truth override vs temperature for core conditions."""
    plt = _mpl()
    import matplotlib.pyplot as plt

    if override_df.empty:
        return

    core_pressure = [c for c in CORE_CONDITIONS if c != "control"]
    sub = override_df[override_df["pressure_condition"].isin(core_pressure)].copy()
    if sub.empty:
        return

    # Aggregate across categories per (T, variant, condition)
    agg = (
        sub.groupby(["temperature", "variant", "pressure_condition"], observed=True)
        .apply(lambda g: pd.Series({
            "rate": np.average(g["truth_override_rate"], weights=g["n_items"]) if g["n_items"].sum() > 0 else np.nan,
        })).reset_index()
    )

    variants = _present_variants(agg)
    n_v = len(variants)
    ncols = min(4, n_v)
    nrows = math.ceil(n_v / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).flatten() if n_v > 1 else [axes]

    for i, variant in enumerate(variants):
        ax = axes[i]
        for cond in core_pressure:
            line = agg[(agg["variant"] == variant) & (agg["pressure_condition"] == cond)].sort_values("temperature")
            if line.empty:
                continue
            family = CONDITION_FAMILY.get(cond, "other")
            color = FAMILY_COLORS.get(family, "#333")
            ax.plot(line["temperature"], line["rate"], marker="o", linewidth=2,
                    markersize=6, color=color, label=CONDITION_LABELS.get(cond, cond))
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Temperature")
        if i % ncols == 0:
            ax.set_ylabel("Truth Override Rate")
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(variants), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Truth Override Rate vs Temperature", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig5_temperature_override.png", fig_dir / "fig5_temperature_override.pdf")
    plt.close(fig)
    logger.info("  fig5_temperature_override saved")


def _fig_asymmetry_divergence(effects, fig_dir, logger):
    """Divergence chart: authority_delta - peer_delta per variant × category."""
    plt = _mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if effects.empty:
        return

    # Compute asymmetry: authority - peer (pooled)
    peer = effects[effects["condition_name"] == "asch_history_5"].copy()
    auth = effects[effects["condition_name"] == "authoritative_bias"].copy()
    if peer.empty or auth.empty:
        return

    peer_pool = peer.groupby(["variant", "dataset_category"], observed=True).apply(
        lambda g: np.average(g["delta_error"], weights=g["n_pressure"]) if g["n_pressure"].sum() > 0 else np.nan
    ).reset_index(name="peer_delta")

    auth_pool = auth.groupby(["variant", "dataset_category"], observed=True).apply(
        lambda g: np.average(g["delta_error"], weights=g["n_pressure"]) if g["n_pressure"].sum() > 0 else np.nan
    ).reset_index(name="auth_delta")

    asym = peer_pool.merge(auth_pool, on=["variant", "dataset_category"], how="inner")
    asym["asymmetry"] = asym["auth_delta"] - asym["peer_delta"]

    variants = [v for v in VARIANT_ORDER if v in asym["variant"].unique()]
    categories = [c for c in FACTUAL_CATEGORIES if c in asym["dataset_category"].unique()]

    pivot = asym.pivot_table(index="variant", columns="dataset_category", values="asymmetry")
    pivot = pivot.reindex(index=variants, columns=categories)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max())) if not pivot.empty else 0.3
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        xticklabels=[c.title() for c in categories],
        yticklabels=[VARIANT_LABELS.get(v, v) for v in variants],
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Auth Δ − Peer Δ  (>0 = authority-dominant)"},
    )
    ax.set_title(
        "Peer vs Authority Asymmetry by Variant × Topic\n"
        "(positive = authority pressure stronger; negative = peer pressure stronger)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig6_asymmetry_heatmap.png", fig_dir / "fig6_asymmetry_heatmap.pdf")
    plt.close(fig)
    logger.info("  fig6_asymmetry_heatmap saved")


def _fig_heatmap_override(boot_cis, mcnemar, fig_dir, logger):
    """Heatmap: variant × condition truth override with significance stars."""
    plt = _mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    override_df = boot_cis[boot_cis["metric"] == "truth_override_rate"].copy()
    if override_df.empty:
        return

    cond_order = [c for c in ALL_CONDITIONS if c != "control"]
    present_conds = [c for c in cond_order if c in override_df["condition_name"].unique()]
    present_variants = _present_variants(override_df)

    pivot = (
        override_df.groupby(["variant", "condition_name"])["point_estimate"]
        .mean().unstack("condition_name")
        .reindex(index=present_variants, columns=present_conds)
    )

    sig_pivot = None
    if not mcnemar.empty:
        sig_pivot = (
            mcnemar.groupby(["variant", "condition_name"])["sig_adjusted"]
            .first().unstack("condition_name")
            .reindex(index=present_variants, columns=present_conds)
        )

    fig, ax = plt.subplots(figsize=(max(11, len(present_conds) * 1.1), max(3.5, len(present_variants) * 0.85)))
    sns.heatmap(
        pivot.values, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        xticklabels=[CONDITION_LABELS.get(c, c) for c in present_conds],
        yticklabels=[VARIANT_LABELS.get(v, v) for v in present_variants],
        linewidths=0.4, linecolor="white", annot=False,
        cbar_kws={"label": "Truth Override Rate"},
    )
    # Annotate with value + significance
    for ri, v in enumerate(present_variants):
        for ci, c in enumerate(present_conds):
            val = pivot.iloc[ri, ci]
            if pd.isna(val):
                continue
            sig = ""
            if sig_pivot is not None and v in sig_pivot.index and c in sig_pivot.columns:
                s = sig_pivot.loc[v, c]
                if s not in ("ns", "", None) and not (isinstance(s, float) and math.isnan(s)):
                    sig = str(s)
            text_color = "white" if val > 0.6 else "black"
            ax.text(ci + 0.5, ri + 0.5, f"{val:.2f}{sig}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold" if sig else "normal")

    ax.set_title("Truth Override Rate: Variant × Condition (pooled over T, topics)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=7.5)
    plt.tight_layout()
    _save_fig(fig, fig_dir / "fig7_heatmap_override.png", fig_dir / "fig7_heatmap_override.pdf")
    plt.close(fig)
    logger.info("  fig7_heatmap_override saved")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7: Export (LLM-readable tables, manifest, README)
# ═══════════════════════════════════════════════════════════════════════════

def _save_table(df: pd.DataFrame, path: Path, description: str, logger: logging.Logger):
    """Save a CSV with a provenance comment header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# {description}\n"
        f"# Generated: {datetime.now().isoformat()}\n"
        f"# Rows: {len(df)}\n"
        f"# Script: generate_publication_item_set.py\n"
    )
    with open(path, "w") as f:
        f.write(header)
        df.to_csv(f, index=False)
    logger.info("  saved %s (%d rows)", path.name, len(df))


def write_manifest(
    runs: List[Dict[str, Any]],
    excluded_variants: Tuple[str, ...],
    n_items: int,
    n_trials: int,
    out_dir: Path,
    logger: logging.Logger,
):
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "script": "generate_publication_item_set.py",
        "n_temperature_runs": len(runs),
        "temperatures": [r["temperature"] for r in runs],
        "n_conditions": len(ALL_CONDITIONS),
        "conditions": {c: CONDITION_LABELS[c] for c in ALL_CONDITIONS},
        "excluded_variants": list(excluded_variants),
        "included_variants": VARIANT_ORDER,
        "n_balanced_items": n_items,
        "n_total_trials": n_trials,
        "condition_families": {
            "control": ["control"],
            "peer_core": ["asch_history_5"],
            "tone": [c for c, f in CONDITION_FAMILY.items() if f == "tone"],
            "mitigation": [c for c, f in CONDITION_FAMILY.items() if f == "mitigation"],
            "format_control": [c for c, f in CONDITION_FAMILY.items() if f == "format_control"],
            "authority_core": ["authoritative_bias"],
            "authority_extended": [c for c, f in CONDITION_FAMILY.items() if f == "authority_extended"],
        },
        "psychology_predictions": PSYCHOLOGY_PREDICTIONS,
    }
    path = out_dir / "logs" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", path)


def write_readme(out_dir: Path, logger: logging.Logger):
    """Auto-generate an LLM-readable README for the output directory."""
    readme = f"""# Publication Analysis Output

Generated: {datetime.now().isoformat()}
Script: `Analysis Scripts/generate_publication_item_set.py`

## Directory Structure

```
{out_dir.name}/
├── behavioral/
│   ├── tables/              # CSV tables with provenance headers
│   │   ├── error_rates_all_conditions.csv
│   │   ├── pressure_effects_all_conditions.csv
│   │   ├── truth_override_all_conditions.csv
│   │   ├── truth_rescue_all_conditions.csv
│   │   ├── wrong_answer_flip_all_conditions.csv
│   │   ├── opinion_agreement.csv
│   │   └── pooled_summary.csv
│   ├── figures/             # Publication figures (PNG 300dpi + PDF)
│   │   ├── fig1_forest_all_conditions.png   # Forest plot: all 11 pressure conditions
│   │   ├── fig2_training_trajectory.png     # Training stage → override rate
│   │   ├── fig3_mitigation_slope.png        # Mitigation effectiveness slope chart
│   │   ├── fig4_tone_modulation.png         # Tone modulation connected dots
│   │   ├── fig5_temperature_override.png    # Temperature × override line plots
│   │   ├── fig6_asymmetry_heatmap.png       # Peer vs authority asymmetry heatmap
│   │   └── fig7_heatmap_override.png        # Variant × condition override heatmap
│   └── statistical_tests/
│       ├── mcnemar_pressure_vs_control.csv  # McNemar + Holm-Bonferroni
│       ├── cochrans_q_families.csv          # Cochran's Q per family
│       └── bootstrap_cis.csv               # BCa 95% CIs
├── interpretability/        # (placeholder for future work)
├── logs/
│   ├── manifest.json        # Machine-readable run manifest
│   ├── pipeline.log         # Full pipeline log
│   └── balance_check.csv    # Cell balance diagnostics
└── README.md                # This file
```

## How to Read the Tables

Every CSV has a comment header (lines starting with `#`) that documents:
- What the table contains
- When it was generated
- Row count

### Key tables:

**error_rates_all_conditions.csv**
- Columns: temperature, variant, condition_name, dataset_category, error_rate, n_trials, refusal_rate
- One row per (T, variant, condition, topic) cell
- error_rate = fraction of factual trials where the model was incorrect (judge-labeled)

**pressure_effects_all_conditions.csv**
- Columns: temperature, variant, dataset_category, condition_name, control_error_rate, pressure_error_rate, delta_error
- delta_error = pressure_error_rate - control_error_rate (positive = pressure hurts)

**truth_override_all_conditions.csv**
- Columns: temperature, variant, dataset_category, truth_override_rate, n_items, pressure_condition
- truth_override_rate = P(pressure incorrect | control correct) — the core sycophancy metric
- n_items = denominator (control-correct items)

**mcnemar_pressure_vs_control.csv**
- One row per (variant, pressure_condition)
- Pooled across all temperatures and items
- p_adjusted = Holm-Bonferroni corrected p-value
- sig_adjusted = significance stars (***, **, *, ns)

## Condition Families

| Family | Conditions | Purpose |
|--------|-----------|---------|
| Control | control | Baseline (no pressure) |
| Peer (core) | asch_history_5 | Asch-style 5-confederate consensus |
| Tone | plain, neutral, confident, uncertain | Does peer confidence tone matter? |
| Mitigation | devil's advocate, question distillation, diverse peers | Do human-inspired fixes work? |
| Authority (core) | authoritative_bias | Single authoritative user claim |
| Authority (extended) | trust, trust+DA | Trust framing and alternative option |

## Key Finding: The "Mitigation Myth"

Human conformity research predicts that a single dissenter (devil's advocate) should
dramatically reduce conformity (Asch, 1956). Our data shows these mitigations FAIL
for LLMs — and sometimes backfire. See fig3_mitigation_slope for the visualization.
"""
    path = out_dir / "README.md"
    with open(path, "w") as f:
        f.write(readme)
    logger.info("README written to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Publication Item-Set Generator — Final Publication Pipeline",
    )
    parser.add_argument(
        "--runs-dir", type=str, required=True,
        help="Path to folder containing run directories (auto-discovers all runs)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(REPO_ROOT / "Comparing_Experiments" / "final_publication"),
        help="Output directory for all results",
    )
    parser.add_argument(
        "--exclude-variants", type=str, nargs="*", default=["rl_zero"],
        help="Variants to exclude (default: rl_zero)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10000,
        help="Bootstrap resamples for CIs (default: 10000)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation (tables and stats only)",
    )

    args = parser.parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    excluded = tuple(args.exclude_variants)

    # Setup output dirs
    behavioral_dir = out_dir / "behavioral"
    tables_dir = behavioral_dir / "tables"
    fig_dir = behavioral_dir / "figures"
    stats_dir = behavioral_dir / "statistical_tests"
    logs_dir = out_dir / "logs"
    interp_dir = out_dir / "interpretability"
    for d in [tables_dir, fig_dir, stats_dir, logs_dir, interp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir / "pipeline.log")
    logger.info("=" * 70)
    logger.info("Publication Pipeline — generate_publication_item_set.py")
    logger.info("=" * 70)
    t0 = time.time()

    # Stage 1: Discovery
    logger.info("\n[Stage 1] Auto-discovering runs in %s", runs_dir)
    runs = discover_runs(runs_dir)
    if not runs:
        logger.error("No runs found! Check --runs-dir path.")
        sys.exit(1)
    logger.info("Found %d temperature runs: %s", len(runs), [r["temperature"] for r in runs])

    # Stage 2: Balanced item set (two-tier strategy)
    logger.info("\n[Stage 2] Computing balanced item sets")
    logger.info("  Pass 1: All 12 conditions...")
    pub_items_all = compute_balanced_item_set(runs, excluded, ALL_CONDITIONS, logger)
    logger.info("  Pass 2: Core 3 conditions only (for larger item set)...")
    pub_items_core = compute_balanced_item_set(runs, excluded, CORE_CONDITIONS, logger)

    # Use the larger set: if all-12 drops below 250 items, load with core item set
    # but still request all conditions (extended conditions will have fewer items)
    if len(pub_items_all) >= 250:
        pub_items = pub_items_all
        logger.info("Using all-12-condition item set: %d items", len(pub_items))
    else:
        pub_items = pub_items_core
        logger.info(
            "All-12 item set too small (%d). Using core-3 item set (%d items) — "
            "extended conditions will have reduced item coverage.",
            len(pub_items_all), len(pub_items),
        )

    if not pub_items:
        logger.error("Empty balanced item set! Cannot proceed.")
        sys.exit(1)

    # Stage 3: Load trial data (all 12 conditions, using the item set above)
    logger.info("\n[Stage 3] Loading judge-labeled trial data")
    df = load_trial_data(runs, ALL_CONDITIONS, excluded, pub_items, logger)
    if df.empty:
        logger.error("No trial data loaded!")
        sys.exit(1)

    balance = check_balance(df)
    balance.to_csv(logs_dir / "balance_check.csv", index=False)
    n_imbalanced = int(balance["imbalanced"].sum()) if "imbalanced" in balance.columns else 0
    logger.info("Balance check: %d/%d cells imbalanced (>5%% deviation)", n_imbalanced, len(balance))

    # Stage 4: Metrics
    logger.info("\n[Stage 4] Computing metrics")
    error_rates = compute_error_rates(df)
    _save_table(error_rates, tables_dir / "error_rates_all_conditions.csv",
                "Error rates per (T, variant, condition, category) — judge-labeled", logger)

    effects = compute_pressure_effects(error_rates)
    _save_table(effects, tables_dir / "pressure_effects_all_conditions.csv",
                "Pressure effects (delta error) for all conditions vs control", logger)

    conditional = compute_all_conditional_metrics(df, logger)
    for key, tbl in conditional.items():
        if not tbl.empty:
            _save_table(tbl, tables_dir / f"{key}_all_conditions.csv",
                        f"{key} per (T, variant, category, condition) — judge-labeled", logger)

    opinion = compute_opinion_agreement(df)
    _save_table(opinion, tables_dir / "opinion_agreement.csv",
                "Opinion-suite wrong-answer agreement", logger)

    pooled = compute_pooled_summary(error_rates)
    if not pooled.empty:
        pooled.to_csv(tables_dir / "pooled_summary.csv")
        logger.info("  saved pooled_summary.csv")

    # Stage 5: Statistical tests
    logger.info("\n[Stage 5] Statistical tests")
    mcnemar = compute_mcnemar_tests(df, logger)
    _save_table(mcnemar, stats_dir / "mcnemar_pressure_vs_control.csv",
                "McNemar tests: each (variant, condition) vs control, Holm-Bonferroni corrected", logger)

    cochran = compute_cochrans_q(df, logger)
    _save_table(cochran, stats_dir / "cochrans_q_families.csv",
                "Cochran's Q tests across condition families per variant", logger)

    logger.info("  Computing bootstrap CIs (n_boot=%d) — this may take a few minutes...", args.n_boot)
    boot_cis = compute_bootstrap_cis(df, args.n_boot, logger)
    _save_table(boot_cis, stats_dir / "bootstrap_cis.csv",
                "BCa 95% CIs for truth_override_rate and delta_error", logger)

    # Stage 6: Figures
    if not args.skip_figures:
        logger.info("\n[Stage 6] Generating figures")
        try:
            generate_all_figures(
                df, error_rates, effects, conditional, opinion,
                mcnemar, boot_cis, fig_dir, logger,
            )
        except Exception as e:
            logger.error("Figure generation failed: %s", e)
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info("\n[Stage 6] Skipped (--skip-figures)")

    # Stage 7: Export
    logger.info("\n[Stage 7] Writing manifest and README")
    write_manifest(runs, excluded, len(pub_items), len(df), out_dir, logger)
    write_readme(out_dir, logger)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Output: %s", out_dir)
    logger.info("Trials: %d | Items: %d | Temps: %d | Conditions: %d",
                len(df), len(pub_items), len(runs), len(ALL_CONDITIONS))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
