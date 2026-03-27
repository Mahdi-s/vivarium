#!/usr/bin/env python3
"""
Publication V2 Column – Non-Think Variant Analysis with Domain Heatmaps.

Focused on the "column" models (no extended-reasoning variants):
  base / instruct / instruct_sft / instruct_dpo

Excluded: think, think_sft, think_dpo, rl_zero

Extends V8 pipeline with additional per-domain/topic heatmaps designed
for slide decks and detailed post-talk analysis:
  G. Domain × Variant error-rate heatmap under control
  H. Domain × Variant truth-override heatmap (peer pressure)
  I. Domain × Variant truth-override heatmap (authority pressure)
  J. Domain-level pressure-delta grouped bar chart
  K. Temperature × Domain truth-override heatmap (pooled across variants)

Usage:
  python 'Analysis Scripts/generate_publication_V2_column.py' \\
      --runs-dir runs_latest/runs \\
      --metadata Comparing_Experiments/runs_metadata_v6.json \\
      --out-dir Comparing_Experiments/publication_V2_column \\
      --run-analysis

Add --n-boot N to change bootstrap resamples (default 10000).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Condition catalogue
# ---------------------------------------------------------------------------

CORE_CONDITIONS: Tuple[str, ...] = (
    "control",
    "asch_history_5",
    "authoritative_bias",
)

PEER_SUBCONDITIONS: Tuple[str, ...] = (
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
    "asch_zhu_unbiased_diverse_plain",
    "asch_zhu_unbiased_qd",
    "asch_zhu_unbiased_da",
)

AUTHORITY_SUBCONDITIONS: Tuple[str, ...] = (
    "authority_zhu_unbiased_trust",
    "authority_zhu_unbiased_trust_da",
)

TONE_CONDITIONS: Tuple[str, ...] = (
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
)

MITIGATION_CONDITIONS: Tuple[str, ...] = (
    "asch_zhu_unbiased_da",
    "asch_zhu_unbiased_qd",
    "asch_zhu_unbiased_diverse_plain",
)

ALL_CONDITIONS: Tuple[str, ...] = (
    "control",
    "asch_history_5",
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
    "asch_zhu_unbiased_diverse_plain",
    "asch_zhu_unbiased_qd",
    "asch_zhu_unbiased_da",
    "authoritative_bias",
    "authority_zhu_unbiased_trust",
    "authority_zhu_unbiased_trust_da",
)

CONDITION_LABELS: Dict[str, str] = {
    "control": "Control",
    "asch_history_5": "Asch (5 Confederates)",
    "asch_zhu_unbiased_unanimous_plain": "Unanimous Plain",
    "asch_zhu_unbiased_unanimous_neutral": "Unanimous Neutral",
    "asch_zhu_unbiased_unanimous_confident": "Unanimous Confident",
    "asch_zhu_unbiased_unanimous_uncertain": "Unanimous Uncertain",
    "asch_zhu_unbiased_diverse_plain": "Diverse Plain",
    "asch_zhu_unbiased_qd": "Asch: Question Distillation",
    "asch_zhu_unbiased_da": "Asch: Devil's Advocate",
    "authoritative_bias": "Authoritative Bias",
    "authority_zhu_unbiased_trust": "Authority: Trust",
    "authority_zhu_unbiased_trust_da": "Authority: Trust+DA",
}

CONDITION_FAMILY: Dict[str, str] = {
    "control": "control",
    "asch_history_5": "peer",
    "asch_zhu_unbiased_unanimous_plain": "peer_tone",
    "asch_zhu_unbiased_unanimous_neutral": "peer_tone",
    "asch_zhu_unbiased_unanimous_confident": "peer_tone",
    "asch_zhu_unbiased_unanimous_uncertain": "peer_tone",
    "asch_zhu_unbiased_diverse_plain": "peer_mitigation",
    "asch_zhu_unbiased_qd": "peer_mitigation",
    "asch_zhu_unbiased_da": "peer_mitigation",
    "authoritative_bias": "authority",
    "authority_zhu_unbiased_trust": "authority",
    "authority_zhu_unbiased_trust_da": "authority",
}

# ── Column-only: exclude ALL think variants and rl_zero ──
EXCLUDED_VARIANTS: Tuple[str, ...] = (
    "think", "think_sft", "think_dpo", "rl_zero",
)

VARIANT_ORDER: List[str] = [
    "base", "instruct", "instruct_sft", "instruct_dpo",
]

VARIANT_LABELS: Dict[str, str] = {
    "base": "Base",
    "instruct": "Instruct",
    "instruct_sft": "Instruct-SFT",
    "instruct_dpo": "Instruct-DPO",
}

VARIANT_COLORS: Dict[str, str] = {
    "base": "#7F8C8D",
    "instruct": "#2980B9",
    "instruct_sft": "#1F618D",
    "instruct_dpo": "#6C3483",
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

FACTUAL_CATEGORIES: List[str] = [
    "general", "math", "science", "knowledge", "truthfulness", "reasoning",
]

DOMAIN_LABELS: Dict[str, str] = {
    "general": "General Facts",
    "math": "Math",
    "science": "Science",
    "knowledge": "Knowledge",
    "truthfulness": "Truthfulness",
    "reasoning": "Reasoning",
    "opinion": "Opinion/Social",
}

INTENDED_MAX_ITEMS = 50


# ---------------------------------------------------------------------------
# Phase 1: Item-set generation
# ---------------------------------------------------------------------------

def discover_runs(
    runs_dir: Path, metadata_path: Path
) -> List[Tuple[str, str, Path]]:
    """Return [(label, run_id, db_path)] sorted by temperature (ascending)."""
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {runs_dir}")
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    out: List[Tuple[str, str, Path]] = []
    for label, info in meta.get("experiments", {}).items():
        run_id = str(info.get("run_id", ""))
        run_dir_name = str(info.get("run_dir", ""))
        if not run_id or not run_dir_name:
            continue
        db_path = runs_dir / run_dir_name / "simulation.db"
        if not db_path.is_file():
            print(f"  [warn] DB not found for temp={label}: {db_path}", file=sys.stderr)
            continue
        out.append((label, run_id, db_path))
    return sorted(out, key=lambda x: float(x[0]))


def query_run_judge_coverage(
    db_path: Path,
    run_id: str,
    excluded_variants: Tuple[str, ...],
    judge_conditions: Tuple[str, ...],
) -> Dict[str, Any]:
    """Return coverage statistics and the set of fully-covered items for a single run."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    excl_ph = ",".join("?" * len(excluded_variants))
    cond_ph = ",".join("?" * len(judge_conditions))
    try:
        n_models_row = conn.execute(
            f"SELECT COUNT(DISTINCT variant) AS n FROM conformity_trials "
            f"WHERE run_id = ? AND variant NOT IN ({excl_ph})",
            [run_id, *excluded_variants],
        ).fetchone()
        n_included_models: int = n_models_row["n"] if n_models_row else 0
        expected_labels_per_item = n_included_models * len(judge_conditions)

        item_rows = conn.execute(
            f"""
            SELECT
                t.item_id,
                d.name AS dataset_name,
                COUNT(*) AS n_judge
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_items i ON i.item_id = t.item_id
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({cond_ph})
              AND t.variant NOT IN ({excl_ph})
              AND o.parsed_answer_json IS NOT NULL
              AND trim(o.parsed_answer_json) != ''
              AND o.parsed_answer_json NOT LIKE '%parse_error%'
            GROUP BY t.item_id, d.name
            """,
            [run_id, *judge_conditions, *excluded_variants],
        ).fetchall()

        fully_covered: set = set()
        per_dataset: Dict[str, set] = {}
        for r in item_rows:
            if r["n_judge"] >= expected_labels_per_item:
                fully_covered.add(r["item_id"])
                per_dataset.setdefault(r["dataset_name"], set()).add(r["item_id"])

        model_rows = conn.execute(
            f"""
            SELECT
                t.variant,
                t.model_id,
                COUNT(*) AS n_trials,
                SUM(
                    CASE WHEN o.parsed_answer_json IS NOT NULL
                          AND trim(o.parsed_answer_json) != ''
                          AND o.parsed_answer_json NOT LIKE '%parse_error%'
                    THEN 1 ELSE 0 END
                ) AS n_judge
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?
              AND c.name IN ({cond_ph})
              AND t.variant NOT IN ({excl_ph})
            GROUP BY t.variant, t.model_id
            ORDER BY t.variant
            """,
            [run_id, *judge_conditions, *excluded_variants],
        ).fetchall()

        per_model: List[Dict[str, Any]] = []
        for r in model_rows:
            n_exp = r["n_trials"]
            n_j = r["n_judge"]
            per_model.append({
                "variant": r["variant"],
                "model_id": r["model_id"],
                "n_expected": n_exp,
                "n_judge_labeled": n_j,
                "coverage_pct": round(100.0 * n_j / n_exp, 1) if n_exp else 0.0,
                "complete": "Y" if n_j == n_exp else "N",
            })

        return {
            "fully_covered_items": fully_covered,
            "n_included_models": n_included_models,
            "expected_labels_per_item": expected_labels_per_item,
            "per_model_coverage": per_model,
            "per_dataset": per_dataset,
        }
    finally:
        conn.close()


def query_item_metadata(
    db_path: Path, item_ids: set
) -> List[Dict[str, str]]:
    """Return [{item_id, dataset_name, domain, ground_truth_text}] for item_ids."""
    if not item_ids:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ph = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT DISTINCT
                i.item_id,
                d.name AS dataset_name,
                i.domain,
                i.ground_truth_text
            FROM conformity_items i
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            WHERE i.item_id IN ({ph})
            """,
            list(item_ids),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 2: Trial data loading
# ---------------------------------------------------------------------------

def _parse_judge_json(pj: Any) -> Optional[Dict[str, Any]]:
    """Parse and validate judge JSON from parsed_answer_json."""
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


def load_trial_data_for_run(
    db_path: Path,
    run_id: str,
    conditions: Tuple[str, ...],
    excluded_variants: Tuple[str, ...],
    publication_item_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Load all trial-level rows for a single run using judge labels."""
    conn = sqlite3.connect(str(db_path))
    cond_ph = ",".join("?" * len(conditions))
    excl_ph = ",".join("?" * len(excluded_variants))
    item_filter = ""
    item_params: List[Any] = []
    if publication_item_ids:
        item_ph = ",".join("?" * len(publication_item_ids))
        item_filter = f"AND i.item_id IN ({item_ph})"
        item_params = list(publication_item_ids)

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
        t.trial_id,
        t.model_id,
        t.variant,
        t.temperature,
        c.name  AS condition_name,
        i.item_id,
        i.domain,
        i.ground_truth_text,
        d.name  AS dataset_name,
        json_extract(i.source_json, '$.wrong_answer') AS wrong_answer,
        o.parsed_answer_json,
        o.refusal_flag
    FROM conformity_trials t
    JOIN conformity_conditions c  ON c.condition_id = t.condition_id
    JOIN conformity_items i       ON i.item_id      = t.item_id
    JOIN conformity_datasets d    ON d.dataset_id   = i.dataset_id
    JOIN first_output_ids foi     ON foi.trial_id   = t.trial_id
    JOIN conformity_outputs o     ON o.output_id    = foi.output_id
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

    is_correct_list: List[Any] = []
    agrees_wrong_list: List[Any] = []
    refusal_list: List[Any] = []
    keep_mask: List[bool] = []

    for _, row in df.iterrows():
        jd = _parse_judge_json(row["parsed_answer_json"])
        if jd is None:
            keep_mask.append(False)
            is_correct_list.append(None)
            agrees_wrong_list.append(None)
            refusal_list.append(None)
            continue

        ic_raw = jd.get("is_correct")
        ic: Optional[int] = None
        if ic_raw in (1, "1", True):
            ic = 1
        elif ic_raw in (0, "0", False):
            ic = 0

        wae_raw = jd.get("wrong_answer_endorsed")
        wae_bool = wae_raw in (1, "1", True)

        rf_raw = jd.get("refusal_flag")
        rf_int = 1 if rf_raw in (1, "1", True) else 0

        is_factual = pd.notna(row.get("ground_truth_text"))
        has_pressure = (
            row.get("wrong_answer") is not None
            and str(row.get("wrong_answer", "")).strip() != ""
        )
        ok = True
        if is_factual and ic is None:
            ok = False
        if has_pressure and wae_raw is None:
            ok = False

        keep_mask.append(ok)
        is_correct_list.append(ic)
        agrees_wrong_list.append(wae_bool)
        refusal_list.append(rf_int)

    df = df[keep_mask].copy()
    df["is_correct"] = [x for x, k in zip(is_correct_list, keep_mask) if k]
    df["agrees_wrong_answer"] = [x for x, k in zip(agrees_wrong_list, keep_mask) if k]
    df["refusal_flag"] = [x for x, k in zip(refusal_list, keep_mask) if k]
    df["dataset_category"] = df["dataset_name"].map(DATASET_TO_CATEGORY).fillna("unknown")
    df["is_factual"] = df["ground_truth_text"].notna()
    return df


def load_all_trial_data(
    runs: List[Tuple[str, str, Path]],
    conditions: Tuple[str, ...],
    excluded_variants: Tuple[str, ...],
    publication_item_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Load and concatenate judge-labeled trial data from all temperature runs."""
    dfs = []
    for label, run_id, db_path in runs:
        df = load_trial_data_for_run(
            db_path, run_id, conditions, excluded_variants, publication_item_ids
        )
        if not df.empty:
            df["temperature"] = float(label)
        dfs.append(df)
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def check_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Check cell balance across (temperature, variant, condition_name)."""
    factual = df[df["is_factual"]]
    if factual.empty:
        return pd.DataFrame()
    counts = (
        factual.groupby(["temperature", "variant", "condition_name"], observed=True)
        .size()
        .reset_index(name="n_trials")
    )
    if counts.empty:
        return counts
    median_n = float(counts["n_trials"].median())
    counts["median_n"] = median_n
    counts["deviation_pct"] = ((counts["n_trials"] - median_n) / median_n * 100).round(1)
    counts["imbalanced"] = counts["deviation_pct"].abs() > 5.0
    return counts


# ---------------------------------------------------------------------------
# Phase 3: Metric computation
# ---------------------------------------------------------------------------

def compute_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Error rates per (temperature, variant, condition_name, dataset_category)."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()
    factual["is_error"] = (factual["is_correct"] == 0).astype(int)
    return (
        factual.groupby(
            ["temperature", "variant", "condition_name", "dataset_category"],
            observed=True,
        )
        .agg(error_rate=("is_error", "mean"), n_trials=("is_error", "count"))
        .reset_index()
    )


def compute_pressure_effects(error_rates: pd.DataFrame) -> pd.DataFrame:
    """Delta error (pressure - control) per (temperature, variant, condition, category)."""
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
        merged["delta_error"] = merged["pressure_error_rate"] - merged["control_error_rate"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_truth_override(
    df: pd.DataFrame, pressure_condition: str
) -> pd.DataFrame:
    """P(pressure incorrect | control correct) per (temperature, variant, dataset_category)."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()
    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "dataset_category", "is_correct"]
    ].rename(columns={"is_correct": "ctrl_correct"})
    pres = factual[factual["condition_name"] == pressure_condition][
        ["temperature", "variant", "item_id", "is_correct"]
    ].rename(columns={"is_correct": "pres_correct"})
    merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
    cc = merged[merged["ctrl_correct"] == 1].copy()
    if cc.empty:
        return pd.DataFrame()
    cc["override"] = (cc["pres_correct"] == 0).astype(int)
    agg = (
        cc.groupby(["temperature", "variant", "dataset_category"], observed=True)
        .agg(truth_override_rate=("override", "mean"), n_items=("override", "count"))
        .reset_index()
    )
    agg["pressure_condition"] = pressure_condition
    return agg


def compute_truth_rescue(
    df: pd.DataFrame, pressure_condition: str
) -> pd.DataFrame:
    """P(pressure correct | control incorrect) per (temperature, variant, dataset_category)."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()
    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "dataset_category", "is_correct"]
    ].rename(columns={"is_correct": "ctrl_correct"})
    pres = factual[factual["condition_name"] == pressure_condition][
        ["temperature", "variant", "item_id", "is_correct"]
    ].rename(columns={"is_correct": "pres_correct"})
    merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
    cw = merged[merged["ctrl_correct"] == 0].copy()
    if cw.empty:
        return pd.DataFrame()
    cw["rescue"] = (cw["pres_correct"] == 1).astype(int)
    agg = (
        cw.groupby(["temperature", "variant", "dataset_category"], observed=True)
        .agg(truth_rescue_rate=("rescue", "mean"), n_items=("rescue", "count"))
        .reset_index()
    )
    agg["pressure_condition"] = pressure_condition
    return agg


def compute_wrong_answer_flip(
    df: pd.DataFrame, pressure_condition: str
) -> pd.DataFrame:
    """P(pressure endorses wrong | control does not) per (temperature, variant, category)."""
    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()
    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "dataset_category", "agrees_wrong_answer"]
    ].rename(columns={"agrees_wrong_answer": "ctrl_agrees"})
    pres = factual[factual["condition_name"] == pressure_condition][
        ["temperature", "variant", "item_id", "agrees_wrong_answer"]
    ].rename(columns={"agrees_wrong_answer": "pres_agrees"})
    merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
    not_wrong = merged[~merged["ctrl_agrees"].astype(bool)].copy()
    if not_wrong.empty:
        return pd.DataFrame()
    not_wrong["flip"] = not_wrong["pres_agrees"].astype(bool).astype(int)
    agg = (
        not_wrong.groupby(["temperature", "variant", "dataset_category"], observed=True)
        .agg(wrong_answer_flip_rate=("flip", "mean"), n_items=("flip", "count"))
        .reset_index()
    )
    agg["pressure_condition"] = pressure_condition
    return agg


def compute_opinion_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Wrong-answer agreement on opinion items per (temperature, variant, condition_name)."""
    opinion = df[~df["is_factual"] & df["agrees_wrong_answer"].notna()].copy()
    if opinion.empty:
        return pd.DataFrame()
    return (
        opinion.groupby(["temperature", "variant", "condition_name"], observed=True)
        .agg(agreement_rate=("agrees_wrong_answer", "mean"), n_trials=("agrees_wrong_answer", "count"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Phase 4: Statistical tests
# ---------------------------------------------------------------------------

def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-tailed exact McNemar p-value via binomial CDF."""
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


def compute_mcnemar_tests(df: pd.DataFrame) -> pd.DataFrame:
    """McNemar test per (variant, pressure_condition) vs control."""
    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "is_correct"]
    ].rename(columns={"is_correct": "ctrl_correct"})

    pressure_conds = [c for c in factual["condition_name"].unique() if c != "control"]
    rows: List[Dict[str, Any]] = []

    for cond in pressure_conds:
        pres = factual[factual["condition_name"] == cond][
            ["temperature", "variant", "item_id", "is_correct"]
        ].rename(columns={"is_correct": "pres_correct"})
        merged = ctrl.merge(pres, on=["temperature", "variant", "item_id"], how="inner")
        if merged.empty:
            continue
        for variant in sorted(merged["variant"].unique()):
            sub = merged[merged["variant"] == variant]
            b = int(((sub["ctrl_correct"] == 1) & (sub["pres_correct"] == 0)).sum())
            c = int(((sub["ctrl_correct"] == 0) & (sub["pres_correct"] == 1)).sum())
            n_pairs = len(sub)
            p_ctrl = float((sub["ctrl_correct"] == 1).mean())
            p_pres = float((sub["pres_correct"] == 1).mean())
            b_c, c_c = b + 0.5, c + 0.5
            odds_ratio = b_c / c_c
            rows.append({
                "variant": variant,
                "condition_name": cond,
                "condition_label": CONDITION_LABELS.get(cond, cond),
                "condition_family": CONDITION_FAMILY.get(cond, "other"),
                "n_pairs": n_pairs,
                "b_ctrl_correct_pres_wrong": b,
                "c_ctrl_wrong_pres_correct": c,
                "error_rate_control": round(1.0 - p_ctrl, 4),
                "error_rate_pressure": round(1.0 - p_pres, 4),
                "delta_error": round((1.0 - p_pres) - (1.0 - p_ctrl), 4),
                "odds_ratio": round(odds_ratio, 3),
                "cohens_h": round(_cohens_h(1.0 - p_pres, 1.0 - p_ctrl), 4),
                "p_raw": _mcnemar_exact_p(b, c),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["p_adjusted"] = _holm_bonferroni(result["p_raw"].tolist())

    def _sig(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return "ns"

    result["sig_raw"] = result["p_raw"].apply(_sig)
    result["sig_adjusted"] = result["p_adjusted"].apply(_sig)
    return result.sort_values(["variant", "condition_family", "condition_name"]).reset_index(drop=True)


def _cochrans_q_stat(X: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import chi2 as _chi2
    n, k = X.shape
    if k < 2 or n < 2:
        return float("nan"), float("nan")
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)
    G_bar = col_sums.mean()
    numerator = (k - 1) * float(np.sum((col_sums - G_bar) ** 2))
    denominator = float(k * row_sums.sum() - np.sum(row_sums ** 2))
    if denominator <= 0:
        return float("nan"), float("nan")
    Q = numerator / denominator
    p_value = float(1.0 - _chi2.cdf(Q, df=k - 1))
    return float(Q), p_value


def compute_cochrans_q_families(df: pd.DataFrame) -> pd.DataFrame:
    """Cochran's Q across condition families for each variant."""
    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()].copy()
    if factual.empty:
        return pd.DataFrame()

    families: Dict[str, Tuple[str, ...]] = {
        "tone": TONE_CONDITIONS,
        "mitigation": MITIGATION_CONDITIONS,
        "authority": (
            "authoritative_bias",
            "authority_zhu_unbiased_trust",
            "authority_zhu_unbiased_trust_da",
        ),
        "peer_full": (
            "asch_history_5",
            "asch_zhu_unbiased_unanimous_plain",
            "asch_zhu_unbiased_unanimous_neutral",
            "asch_zhu_unbiased_unanimous_confident",
            "asch_zhu_unbiased_unanimous_uncertain",
            "asch_zhu_unbiased_diverse_plain",
            "asch_zhu_unbiased_qd",
            "asch_zhu_unbiased_da",
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
            Q, p = _cochrans_q_stat(X)
            rows.append({
                "variant": variant,
                "family": family_name,
                "k_conditions": X.shape[1],
                "n_subjects": X.shape[0],
                "conditions": "; ".join(CONDITION_LABELS.get(c, c) for c in available if c in pivot.columns),
                "Q_stat": round(Q, 3) if not math.isnan(Q) else float("nan"),
                "df": X.shape[1] - 1,
                "p_value": round(p, 4) if not math.isnan(p) else float("nan"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _bca_ci_from_item_arrays(
    n_ctrl_correct: np.ndarray,
    n_override: np.ndarray,
    n_obs: np.ndarray,
    n_error_ctrl: np.ndarray,
    n_error_pres: np.ndarray,
    metric: str,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: Optional[Any] = None,
) -> Tuple[float, float]:
    from scipy.stats import norm as _norm

    if rng is None:
        rng = np.random.default_rng(42)

    n_items = len(n_obs)
    if n_items < 10:
        return float("nan"), float("nan")

    def _compute(idx: np.ndarray) -> float:
        if metric == "truth_override_rate":
            total_cc = int(n_ctrl_correct[idx].sum())
            if total_cc == 0:
                return float("nan")
            return float(n_override[idx].sum()) / total_cc
        else:
            total_obs = int(n_obs[idx].sum())
            if total_obs == 0:
                return float("nan")
            ctrl_err = float(n_error_ctrl[idx].sum()) / total_obs
            pres_err = float(n_error_pres[idx].sum()) / total_obs
            return pres_err - ctrl_err

    theta_hat = _compute(np.arange(n_items))
    if math.isnan(theta_hat):
        return float("nan"), float("nan")

    boot = np.array([_compute(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)])
    valid = boot[~np.isnan(boot)]
    if len(valid) < 50:
        return float("nan"), float("nan")

    z0 = float(_norm.ppf(np.clip(np.mean(valid < theta_hat), 1e-6, 1 - 1e-6)))

    if metric == "truth_override_rate":
        total_cc = n_ctrl_correct.sum()
        total_ov = n_override.sum()
        jack_cc = total_cc - n_ctrl_correct
        jack_ov = total_ov - n_override
        with np.errstate(invalid="ignore", divide="ignore"):
            jack_stats = np.where(jack_cc > 0, jack_ov.astype(float) / jack_cc, float("nan"))
    else:
        total_obs = n_obs.sum()
        jack_obs = total_obs - n_obs
        jack_ec = n_error_ctrl.sum() - n_error_ctrl
        jack_ep = n_error_pres.sum() - n_error_pres
        with np.errstate(invalid="ignore", divide="ignore"):
            jack_stats = np.where(
                jack_obs > 0,
                jack_ep.astype(float) / jack_obs - jack_ec.astype(float) / jack_obs,
                float("nan"),
            )

    jack_valid = jack_stats[~np.isnan(jack_stats)]
    if len(jack_valid) < 3:
        lo = float(np.quantile(valid, alpha / 2))
        hi = float(np.quantile(valid, 1.0 - alpha / 2))
        return lo, hi

    jm = float(np.mean(jack_valid))
    diff = jm - jack_valid
    num = float(np.sum(diff ** 3))
    denom = 6.0 * float(np.sum(diff ** 2) ** 1.5)
    a = num / denom if abs(denom) > 1e-12 else 0.0

    def _adj_p(z: float) -> float:
        d = 1.0 - a * (z0 + z)
        if abs(d) < 1e-12:
            return alpha / 2 if z < 0 else 1.0 - alpha / 2
        return float(_norm.cdf(z0 + (z0 + z) / d))

    z_lo = _norm.ppf(alpha / 2)
    z_hi = _norm.ppf(1.0 - alpha / 2)
    p_lo = _adj_p(z_lo)
    p_hi = _adj_p(z_hi)

    lo = float(np.quantile(valid, max(0.001, min(0.999, p_lo))))
    hi = float(np.quantile(valid, max(0.001, min(0.999, p_hi))))
    return lo, hi


def compute_bootstrap_cis(
    df: pd.DataFrame, n_boot: int = 10000
) -> pd.DataFrame:
    """BCa 95% CIs for truth_override_rate and delta_error per (variant, condition)."""
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
                n_ctrl_correct=("ctrl_correct", "sum"),
                n_override=("override", "sum"),
                n_obs=("ctrl_correct", "count"),
                n_err_ctrl=("err_ctrl", "sum"),
                n_err_pres=("err_pres", "sum"),
            ).reset_index()

            n_items = len(item_agg)
            if n_items < 10:
                continue

            cc_arr = item_agg["n_ctrl_correct"].values.astype(int)
            ov_arr = item_agg["n_override"].values.astype(int)
            obs_arr = item_agg["n_obs"].values.astype(int)
            ec_arr = item_agg["n_err_ctrl"].values.astype(int)
            ep_arr = item_agg["n_err_pres"].values.astype(int)

            for metric in ("truth_override_rate", "delta_error"):
                if metric == "truth_override_rate":
                    denom = int(cc_arr.sum())
                    point = float(ov_arr.sum()) / denom if denom > 0 else float("nan")
                else:
                    total = int(obs_arr.sum())
                    point = (
                        float(ep_arr.sum()) / total - float(ec_arr.sum()) / total
                        if total > 0 else float("nan")
                    )
                if math.isnan(point):
                    continue

                try:
                    lo, hi = _bca_ci_from_item_arrays(
                        cc_arr, ov_arr, obs_arr, ec_arr, ep_arr,
                        metric=metric, n_boot=n_boot, rng=rng,
                    )
                except Exception as e:
                    print(f"    [warn] BCa failed {variant}/{cond}/{metric}: {e}")
                    lo, hi = float("nan"), float("nan")

                rows.append({
                    "variant": variant,
                    "condition_name": cond,
                    "condition_label": CONDITION_LABELS.get(cond, cond),
                    "condition_family": CONDITION_FAMILY.get(cond, "other"),
                    "metric": metric,
                    "point_estimate": round(point, 4),
                    "ci_lower": round(lo, 4) if not math.isnan(lo) else float("nan"),
                    "ci_upper": round(hi, 4) if not math.isnan(hi) else float("nan"),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 5: Publication figures (A–F from V8 + G–K domain heatmaps)
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_fig(fig: Any, path_png: Path, path_pdf: Optional[Path] = None, dpi: int = 300) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if path_pdf is not None:
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, bbox_inches="tight")


def _wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    lo = (
        p + z ** 2 / (2 * n) - z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    ) / (1 + z ** 2 / n)
    hi = (
        p + z ** 2 / (2 * n) + z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    ) / (1 + z ** 2 / n)
    return max(0.0, lo), min(1.0, hi)


# ── Figure A: Forest plot ──────────────────────────────────────────────────

def plot_forest_pressure_effects(
    boot_cis: pd.DataFrame,
    mcnemar: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    mpl = _mpl()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    delta_df = boot_cis[boot_cis["metric"] == "delta_error"].copy()
    if delta_df.empty or mcnemar.empty:
        print("[fig A] Forest plot: insufficient data, skipping.")
        return

    sig_map = (
        mcnemar.groupby(["variant", "condition_name"])["sig_adjusted"]
        .first()
        .to_dict()
    )

    cond_order = [
        "asch_history_5",
        "asch_zhu_unbiased_unanimous_plain",
        "asch_zhu_unbiased_unanimous_neutral",
        "asch_zhu_unbiased_unanimous_confident",
        "asch_zhu_unbiased_unanimous_uncertain",
        "asch_zhu_unbiased_diverse_plain",
        "asch_zhu_unbiased_da",
        "asch_zhu_unbiased_qd",
        "authoritative_bias",
        "authority_zhu_unbiased_trust",
        "authority_zhu_unbiased_trust_da",
    ]
    present_conds = [c for c in cond_order if c in delta_df["condition_name"].unique()]
    variants_present = [v for v in VARIANT_ORDER if v in delta_df["variant"].unique()]
    n_v = len(variants_present)

    fig, axes = plt.subplots(
        1, n_v,
        figsize=(3.8 * n_v, max(6, len(present_conds) * 0.52 + 2)),
        sharey=True,
    )
    if n_v == 1:
        axes = [axes]

    family_colors = {
        "peer": "#2980B9",
        "peer_tone": "#5DADE2",
        "peer_mitigation": "#85C1E9",
        "authority": "#E67E22",
    }
    core_set = {"asch_history_5", "authoritative_bias"}
    y_pos = {c: i for i, c in enumerate(reversed(present_conds))}

    peer_last = next(
        (c for c in reversed(cond_order) if CONDITION_FAMILY.get(c) in family_colors and "peer" in CONDITION_FAMILY.get(c, "")),
        None,
    )
    auth_first = next(
        (c for c in cond_order if CONDITION_FAMILY.get(c) == "authority"),
        None,
    )
    sep_y = None
    if peer_last in y_pos and auth_first in y_pos:
        sep_y = (y_pos[peer_last] + y_pos[auth_first]) / 2.0

    for ax, variant in zip(axes, variants_present):
        v_df = delta_df[delta_df["variant"] == variant]
        for cond in present_conds:
            row = v_df[v_df["condition_name"] == cond]
            y = y_pos[cond]
            family = CONDITION_FAMILY.get(cond, "peer")
            color = family_colors.get(family, "#95A5A6")
            if row.empty:
                continue
            pe = float(row["point_estimate"].iloc[0])
            lo = float(row["ci_lower"].iloc[0])
            hi = float(row["ci_upper"].iloc[0])
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.8, alpha=0.7, zorder=3)
            sig = sig_map.get((variant, cond), "ns")
            is_core = cond in core_set
            marker = "D" if is_core else "o"
            ms = 9 if is_core else 7
            lw = 2.0 if is_core else 1.5
            filled = sig != "ns"
            ax.plot(
                pe, y, marker=marker, color=color, markersize=ms,
                markerfacecolor=color if filled else "white",
                markeredgewidth=lw, zorder=5,
            )
            if sig not in ("ns", ""):
                ax.text(
                    hi + 0.008, y, sig, va="center", ha="left",
                    fontsize=7, color=color, fontweight="bold",
                )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        if sep_y is not None:
            ax.axhline(sep_y, color="gray", linewidth=0.6, linestyle="-", alpha=0.35)
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xlabel("Delta Error Rate vs Control", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25, linestyle=":")

    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(
        [CONDITION_LABELS.get(c, c) for c in reversed(present_conds)], fontsize=7.5
    )

    legend_elements = [
        mpatches.Patch(color="#2980B9", label="Peer family"),
        mpatches.Patch(color="#E67E22", label="Authority family"),
        plt.Line2D([0], [0], marker="D", color="gray", markersize=8, linestyle="None", label="Core condition"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=7,
                   markerfacecolor="white", markeredgewidth=1.5, linestyle="None",
                   label="Sub-condition (open = ns)"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Pressure Effect on Error Rate Across All Conditions\n"
        "(filled = McNemar p < 0.05, Holm-Bonferroni corrected; 95% BCa CI)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure B: Heatmap with significance stars ──────────────────────────────

def plot_heatmap_with_stars(
    boot_cis: pd.DataFrame,
    mcnemar: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    import seaborn as sns
    mpl = _mpl()
    import matplotlib.pyplot as plt

    override_df = boot_cis[boot_cis["metric"] == "truth_override_rate"].copy()
    if override_df.empty or mcnemar.empty:
        print("[fig B] Heatmap: insufficient data, skipping.")
        return

    cond_order = [
        "asch_history_5",
        "asch_zhu_unbiased_unanimous_plain",
        "asch_zhu_unbiased_unanimous_neutral",
        "asch_zhu_unbiased_unanimous_confident",
        "asch_zhu_unbiased_unanimous_uncertain",
        "asch_zhu_unbiased_diverse_plain",
        "asch_zhu_unbiased_da",
        "asch_zhu_unbiased_qd",
        "authoritative_bias",
        "authority_zhu_unbiased_trust",
        "authority_zhu_unbiased_trust_da",
    ]
    present_conds = [c for c in cond_order if c in override_df["condition_name"].unique()]
    present_variants = [v for v in VARIANT_ORDER if v in override_df["variant"].unique()]

    pivot = (
        override_df.groupby(["variant", "condition_name"])["point_estimate"]
        .mean()
        .unstack("condition_name")
        .reindex(index=present_variants, columns=present_conds)
    )
    sig_pivot = (
        mcnemar.groupby(["variant", "condition_name"])["sig_adjusted"]
        .first()
        .unstack("condition_name")
        .reindex(index=present_variants, columns=present_conds)
    )

    col_labels = [CONDITION_LABELS.get(c, c) for c in present_conds]
    row_labels = [VARIANT_LABELS.get(v, v) for v in present_variants]

    fig, ax = plt.subplots(
        figsize=(max(11, len(present_conds) * 1.15), max(3.5, len(present_variants) * 0.85))
    )
    sns.heatmap(
        pivot.values,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0.0, vmax=1.0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.4,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Truth Override Rate", "shrink": 0.75},
    )

    for r_i, variant in enumerate(present_variants):
        for c_i, cond in enumerate(present_conds):
            val = pivot.iloc[r_i, c_i]
            if pd.isna(val):
                continue
            sig = (
                sig_pivot.iloc[r_i, c_i]
                if variant in sig_pivot.index and cond in sig_pivot.columns
                else ""
            )
            sig_str = "" if sig in ("ns", "", None) or (isinstance(sig, float) and math.isnan(sig)) else str(sig)
            text = f"{val:.2f}{sig_str}"
            text_color = "white" if val > 0.6 or val < 0.2 else "black"
            ax.text(
                c_i + 0.5, r_i + 0.5, text,
                ha="center", va="center", fontsize=8,
                color=text_color,
                fontweight="bold" if sig_str else "normal",
            )

    n_peer = sum(
        1 for c in present_conds if CONDITION_FAMILY.get(c, "") in ("peer", "peer_tone", "peer_mitigation")
    )
    if 0 < n_peer < len(present_conds):
        ax.axvline(n_peer, color="black", linewidth=2.2)

    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)

    peer_mid = n_peer / 2.0
    auth_mid = n_peer + (len(present_conds) - n_peer) / 2.0
    trans = ax.get_xaxis_transform()
    ax.text(peer_mid, -0.05, "Peer Pressure Family", ha="center", va="top",
            fontsize=8.5, style="italic", color="#2980B9", transform=trans)
    ax.text(auth_mid, -0.05, "Authority Family", ha="center", va="top",
            fontsize=8.5, style="italic", color="#E67E22", transform=trans)

    ax.set_title(
        "Truth Override Rate by Variant and Condition\n"
        "(* p<0.05  ** p<0.01  *** p<0.001 after Holm-Bonferroni)",
        fontsize=10, pad=14,
    )
    for c_i, cond in enumerate(present_conds):
        if cond in ("asch_history_5", "authoritative_bias"):
            for r_i in range(len(present_variants)):
                ax.add_patch(
                    plt.Rectangle(
                        (c_i, r_i), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.0, zorder=6,
                    )
                )

    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure C: Tone modulation ──────────────────────────────────────────────

def plot_tone_comparison(
    df: pd.DataFrame,
    cochrans_q: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    import numpy as np
    mpl = _mpl()
    import matplotlib.pyplot as plt

    tone_conds = [c for c in TONE_CONDITIONS if c in df["condition_name"].unique()]
    if len(tone_conds) < 2:
        print("[fig C] Tone comparison: fewer than 2 tone conditions, skipping.")
        return

    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()]
    agg = (
        factual[factual["condition_name"].isin(tone_conds)]
        .groupby(["variant", "condition_name"])["agrees_wrong_answer"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate"})
    )

    variants_present = [v for v in VARIANT_ORDER if v in agg["variant"].unique()]
    n_v = len(variants_present)
    fig, axes = plt.subplots(1, n_v, figsize=(3.2 * n_v, 5), sharey=True)
    if n_v == 1:
        axes = [axes]

    tone_colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E74C3C"]
    tone_labels = [CONDITION_LABELS.get(c, c) for c in tone_conds]
    x = np.arange(len(tone_conds))

    for ax, variant in zip(axes, variants_present):
        v_agg = agg[agg["variant"] == variant]
        rates, errs_lo, errs_hi = [], [], []
        for cond in tone_conds:
            row = v_agg[v_agg["condition_name"] == cond]
            if row.empty:
                rates.append(0.0); errs_lo.append(0.0); errs_hi.append(0.0)
            else:
                p = float(row["rate"].iloc[0])
                n = int(row["count"].iloc[0])
                lo, hi = _wilson_ci(p, n)
                rates.append(p)
                errs_lo.append(p - lo)
                errs_hi.append(hi - p)

        ax.bar(x, rates, width=0.62, color=tone_colors[: len(tone_conds)], alpha=0.82, edgecolor="white")
        ax.errorbar(x, rates, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=4, linewidth=1.2)

        q_row = (
            cochrans_q[(cochrans_q["variant"] == variant) & (cochrans_q["family"] == "tone")]
            if not cochrans_q.empty else pd.DataFrame()
        )
        if not q_row.empty:
            q_val = float(q_row["Q_stat"].iloc[0])
            p_val = float(q_row["p_value"].iloc[0])
            p_str = f"Cochran Q={q_val:.1f}, p={p_val:.3f}" if not math.isnan(p_val) else f"Q={q_val:.1f}"
            ax.text(0.98, 0.97, p_str, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tone_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Wrong-Answer Agreement Rate" if ax is axes[0] else "", fontsize=9)
        ax.grid(axis="y", alpha=0.28, linestyle=":")

    fig.suptitle(
        "Tone Modulation Across Peer-Pressure Conditions\n"
        "(error bars: 95% Wilson CI; Cochran's Q tests within-variant)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure D: Authority variant comparison ─────────────────────────────────

def plot_authority_comparison(
    df: pd.DataFrame,
    cochrans_q: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    import numpy as np
    mpl = _mpl()
    import matplotlib.pyplot as plt

    auth_conds = [
        c for c in ("authoritative_bias", "authority_zhu_unbiased_trust", "authority_zhu_unbiased_trust_da")
        if c in df["condition_name"].unique()
    ]
    if len(auth_conds) < 2:
        print("[fig D] Authority comparison: fewer than 2 authority conditions, skipping.")
        return

    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()]
    agg = (
        factual[factual["condition_name"].isin(auth_conds)]
        .groupby(["variant", "condition_name"])["agrees_wrong_answer"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate"})
    )

    variants_present = [v for v in VARIANT_ORDER if v in agg["variant"].unique()]
    n_v = len(variants_present)
    fig, axes = plt.subplots(1, n_v, figsize=(3.2 * n_v, 5), sharey=True)
    if n_v == 1:
        axes = [axes]

    auth_colors = ["#E67E22", "#D35400", "#BA4A00"]
    auth_labels = [CONDITION_LABELS.get(c, c) for c in auth_conds]
    x = np.arange(len(auth_conds))

    for ax, variant in zip(axes, variants_present):
        v_agg = agg[agg["variant"] == variant]
        rates, errs_lo, errs_hi = [], [], []
        for cond in auth_conds:
            row = v_agg[v_agg["condition_name"] == cond]
            if row.empty:
                rates.append(0.0); errs_lo.append(0.0); errs_hi.append(0.0)
            else:
                p = float(row["rate"].iloc[0])
                n = int(row["count"].iloc[0])
                lo, hi = _wilson_ci(p, n)
                rates.append(p)
                errs_lo.append(p - lo)
                errs_hi.append(hi - p)

        ax.bar(x, rates, width=0.62, color=auth_colors[: len(auth_conds)], alpha=0.82, edgecolor="white")
        ax.errorbar(x, rates, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=4, linewidth=1.2)

        q_row = (
            cochrans_q[(cochrans_q["variant"] == variant) & (cochrans_q["family"] == "authority")]
            if not cochrans_q.empty else pd.DataFrame()
        )
        if not q_row.empty:
            q_val = float(q_row["Q_stat"].iloc[0])
            p_val = float(q_row["p_value"].iloc[0])
            p_str = f"Cochran Q={q_val:.1f}, p={p_val:.3f}" if not math.isnan(p_val) else f"Q={q_val:.1f}"
            ax.text(0.98, 0.97, p_str, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(auth_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Wrong-Answer Agreement Rate" if ax is axes[0] else "", fontsize=9)
        ax.grid(axis="y", alpha=0.28, linestyle=":")

    fig.suptitle(
        "Authority Condition Variants: Wrong-Answer Agreement\n"
        "(error bars: 95% Wilson CI; Cochran's Q tests within-variant)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure E: Temperature CI band plot ────────────────────────────────────

def plot_temperature_ci_bands(
    df: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    import numpy as np
    mpl = _mpl()
    import matplotlib.pyplot as plt

    featured_conds = [
        "asch_history_5",
        "asch_zhu_unbiased_unanimous_plain",
        "asch_zhu_unbiased_diverse_plain",
        "authoritative_bias",
        "authority_zhu_unbiased_trust",
    ]
    featured_conds = [c for c in featured_conds if c in df["condition_name"].unique()]
    if len(featured_conds) < 2:
        print("[fig E] Temperature CI bands: insufficient conditions, skipping.")
        return

    factual = df[df["is_factual"] & df["is_correct"].notna()].copy()
    ctrl = factual[factual["condition_name"] == "control"][
        ["temperature", "variant", "item_id", "is_correct"]
    ].rename(columns={"is_correct": "ctrl_correct"})

    variants_present = [v for v in VARIANT_ORDER if v in factual["variant"].unique()]
    n_v = len(variants_present)
    n_cols = min(2, n_v)
    n_rows = math.ceil(n_v / n_cols)
    all_temps = sorted(factual["temperature"].unique())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        sharey=True,
    )
    axes_flat = np.array(axes).flatten() if n_v > 1 else [axes]

    cond_style: Dict[str, Tuple[str, str]] = {
        "asch_history_5":                    ("-",  "#2980B9"),
        "asch_zhu_unbiased_unanimous_plain": ("--", "#5DADE2"),
        "asch_zhu_unbiased_diverse_plain":   (":",  "#85C1E9"),
        "authoritative_bias":                ("-",  "#E67E22"),
        "authority_zhu_unbiased_trust":      ("--", "#D35400"),
    }
    core_set = {"asch_history_5", "authoritative_bias"}

    for ax, variant in zip(axes_flat, variants_present):
        ctrl_v = ctrl[ctrl["variant"] == variant]
        for cond in featured_conds:
            pres = factual[
                (factual["variant"] == variant) & (factual["condition_name"] == cond)
            ]
            if pres.empty:
                continue
            temps_plot, rates_plot, lo_plot, hi_plot = [], [], [], []
            for temp in all_temps:
                ctrl_t = ctrl_v[ctrl_v["temperature"] == temp]
                pres_t = pres[pres["temperature"] == temp]
                merged = ctrl_t[["item_id", "ctrl_correct"]].merge(
                    pres_t[["item_id", "is_correct"]].rename(columns={"is_correct": "pres_correct"}),
                    on="item_id", how="inner",
                )
                cc = merged[merged["ctrl_correct"] == 1]
                if len(cc) < 5:
                    continue
                p = float((cc["pres_correct"] == 0).mean())
                lo, hi = _wilson_ci(p, len(cc))
                temps_plot.append(temp)
                rates_plot.append(p)
                lo_plot.append(lo)
                hi_plot.append(hi)
            if not temps_plot:
                continue
            ls, color = cond_style.get(cond, ("-", "#95A5A6"))
            lw = 2.2 if cond in core_set else 1.4
            ax.plot(
                temps_plot, rates_plot,
                linestyle=ls, color=color, linewidth=lw,
                label=CONDITION_LABELS.get(cond, cond),
                marker="o", markersize=4, zorder=4,
            )
            ax.fill_between(temps_plot, lo_plot, hi_plot, color=color, alpha=0.12)

        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xlabel("Temperature", fontsize=8.5)
        ax.set_ylabel("Truth Override Rate" if ax is axes_flat[0] else "", fontsize=8.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(all_temps)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.28, linestyle=":")

    for ax in axes_flat[n_v:]:
        ax.set_visible(False)

    axes_flat[0].legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.9)
    fig.suptitle(
        "Truth Override Rate vs Temperature\n(core + representative sub-conditions; shaded band: 95% Wilson CI)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure F: Mitigation effectiveness ────────────────────────────────────

def plot_mitigation_effectiveness(
    df: pd.DataFrame,
    mcnemar: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    import numpy as np
    mpl = _mpl()
    import matplotlib.pyplot as plt

    baseline = "asch_zhu_unbiased_unanimous_plain"
    mit_conds = [
        c for c in ("asch_zhu_unbiased_da", "asch_zhu_unbiased_qd", "asch_zhu_unbiased_diverse_plain")
        if c in df["condition_name"].unique()
    ]
    if baseline not in df["condition_name"].unique() or not mit_conds:
        print("[fig F] Mitigation: baseline or mitigations not available, skipping.")
        return

    compare_conds = [baseline] + mit_conds
    factual = df[df["is_factual"] & df["agrees_wrong_answer"].notna()]
    agg = (
        factual[factual["condition_name"].isin(compare_conds)]
        .groupby(["variant", "condition_name"])["agrees_wrong_answer"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate"})
    )

    sig_lookup: Dict[Tuple[str, str], str] = {}
    if not mcnemar.empty:
        for _, row in mcnemar.iterrows():
            sig_lookup[(row["variant"], row["condition_name"])] = row["sig_adjusted"]

    variants_present = [v for v in VARIANT_ORDER if v in agg["variant"].unique()]
    n_v = len(variants_present)
    fig, axes = plt.subplots(1, n_v, figsize=(3.5 * n_v, 5.5), sharey=True)
    if n_v == 1:
        axes = [axes]

    palette = {
        baseline: "#E74C3C",
        "asch_zhu_unbiased_da": "#27AE60",
        "asch_zhu_unbiased_qd": "#8E44AD",
        "asch_zhu_unbiased_diverse_plain": "#2980B9",
    }
    x_vals = np.arange(len(compare_conds))

    for ax, variant in zip(axes, variants_present):
        v_agg = agg[agg["variant"] == variant]
        rates, errs_lo, errs_hi = [], [], []
        for cond in compare_conds:
            row = v_agg[v_agg["condition_name"] == cond]
            if row.empty:
                rates.append(0.0); errs_lo.append(0.0); errs_hi.append(0.0)
            else:
                p = float(row["rate"].iloc[0])
                n = int(row["count"].iloc[0])
                lo, hi = _wilson_ci(p, n)
                rates.append(p)
                errs_lo.append(p - lo)
                errs_hi.append(hi - p)

        colors = [palette.get(c, "#95A5A6") for c in compare_conds]
        ax.bar(x_vals, rates, width=0.58, color=colors, alpha=0.83, edgecolor="white")
        ax.errorbar(x_vals, rates, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=4, linewidth=1.2)

        y_top = max(rates) if rates else 0.5
        for i, mit_cond in enumerate(mit_conds):
            sig = sig_lookup.get((variant, mit_cond), "ns")
            if sig != "ns":
                x1 = 0
                x2 = i + 1
                y_br = y_top + 0.04 + i * 0.06
                ax.plot(
                    [x1, x1, x2, x2], [y_br, y_br + 0.015, y_br + 0.015, y_br],
                    color="black", linewidth=0.9,
                )
                ax.text(
                    (x1 + x2) / 2.0, y_br + 0.018, sig,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in compare_conds],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0.0, min(1.12, y_top + 0.35))
        ax.set_ylabel("Wrong-Answer Agreement Rate" if ax is axes[0] else "", fontsize=9)
        ax.grid(axis="y", alpha=0.28, linestyle=":")
        ax.patches[0].set_edgecolor("black")
        ax.patches[0].set_linewidth(2.0)

    fig.suptitle(
        "Mitigation Effectiveness: DA / QD / Diverse vs Unanimous Plain (Baseline)\n"
        "(brackets: McNemar adjusted significance vs plain baseline; 95% Wilson CI)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure G: Domain x Variant error-rate heatmap (control) ───────────────

def plot_domain_error_heatmap(
    error_rates: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    """Heatmap: error rate per domain x variant under CONTROL condition, pooled across temperatures."""
    import seaborn as sns
    mpl = _mpl()
    import matplotlib.pyplot as plt

    ctrl = error_rates[error_rates["condition_name"] == "control"].copy()
    if ctrl.empty:
        print("[fig G] Domain error heatmap: no control data, skipping.")
        return

    # Pool across temperatures
    pooled = (
        ctrl.groupby(["variant", "dataset_category"])
        .apply(lambda g: pd.Series({
            "error_rate": (g["error_rate"] * g["n_trials"]).sum() / g["n_trials"].sum(),
            "n_trials": g["n_trials"].sum(),
        }), include_groups=False)
        .reset_index()
    )

    present_domains = [d for d in FACTUAL_CATEGORIES if d in pooled["dataset_category"].unique()]
    present_variants = [v for v in VARIANT_ORDER if v in pooled["variant"].unique()]
    if not present_domains or not present_variants:
        print("[fig G] Domain error heatmap: insufficient data, skipping.")
        return

    pivot = pooled.pivot_table(
        index="variant", columns="dataset_category", values="error_rate"
    ).reindex(index=present_variants, columns=present_domains)

    n_pivot = pooled.pivot_table(
        index="variant", columns="dataset_category", values="n_trials"
    ).reindex(index=present_variants, columns=present_domains)

    row_labels = [VARIANT_LABELS.get(v, v) for v in present_variants]
    col_labels = [DOMAIN_LABELS.get(d, d) for d in present_domains]

    fig, ax = plt.subplots(figsize=(max(8, len(present_domains) * 1.4), max(3, len(present_variants) * 0.9)))
    sns.heatmap(
        pivot.values, ax=ax,
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor="white", annot=False,
        cbar_kws={"label": "Error Rate (Control)", "shrink": 0.75},
    )
    for r_i in range(len(present_variants)):
        for c_i in range(len(present_domains)):
            val = pivot.iloc[r_i, c_i]
            n = n_pivot.iloc[r_i, c_i]
            if pd.isna(val):
                continue
            text_color = "white" if val > 0.6 else "black"
            n_str = f"\n(n={int(n)})" if not pd.isna(n) else ""
            ax.text(c_i + 0.5, r_i + 0.5, f"{val:.2f}{n_str}",
                    ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
    ax.set_title(
        "Baseline Error Rate by Domain and Model Variant (Control Condition)\n"
        "(pooled across all temperatures)",
        fontsize=10, pad=12,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure H: Domain x Variant truth-override heatmap (peer pressure) ─────

def plot_domain_override_heatmap(
    truth_override: pd.DataFrame,
    pressure_condition: str,
    title_suffix: str,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    """Heatmap: truth override rate per domain x variant for a given pressure condition."""
    import seaborn as sns
    mpl = _mpl()
    import matplotlib.pyplot as plt

    sub = truth_override[truth_override["pressure_condition"] == pressure_condition].copy()
    if sub.empty:
        print(f"[fig H/I] Domain override heatmap ({pressure_condition}): no data, skipping.")
        return

    pooled = (
        sub.groupby(["variant", "dataset_category"])
        .apply(lambda g: pd.Series({
            "truth_override_rate": (g["truth_override_rate"] * g["n_items"]).sum() / g["n_items"].sum(),
            "n_items": g["n_items"].sum(),
        }), include_groups=False)
        .reset_index()
    )

    present_domains = [d for d in FACTUAL_CATEGORIES if d in pooled["dataset_category"].unique()]
    present_variants = [v for v in VARIANT_ORDER if v in pooled["variant"].unique()]
    if not present_domains or not present_variants:
        return

    pivot = pooled.pivot_table(
        index="variant", columns="dataset_category", values="truth_override_rate"
    ).reindex(index=present_variants, columns=present_domains)

    n_pivot = pooled.pivot_table(
        index="variant", columns="dataset_category", values="n_items"
    ).reindex(index=present_variants, columns=present_domains)

    row_labels = [VARIANT_LABELS.get(v, v) for v in present_variants]
    col_labels = [DOMAIN_LABELS.get(d, d) for d in present_domains]

    fig, ax = plt.subplots(figsize=(max(8, len(present_domains) * 1.4), max(3, len(present_variants) * 0.9)))
    sns.heatmap(
        pivot.values, ax=ax,
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor="white", annot=False,
        cbar_kws={"label": "Truth Override Rate", "shrink": 0.75},
    )
    for r_i in range(len(present_variants)):
        for c_i in range(len(present_domains)):
            val = pivot.iloc[r_i, c_i]
            n = n_pivot.iloc[r_i, c_i]
            if pd.isna(val):
                continue
            text_color = "white" if val > 0.6 or val < 0.15 else "black"
            n_str = f"\n(n={int(n)})" if not pd.isna(n) else ""
            ax.text(c_i + 0.5, r_i + 0.5, f"{val:.2f}{n_str}",
                    ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
    ax.set_title(
        f"Truth Override Rate by Domain and Variant — {title_suffix}\n"
        "(pooled across all temperatures)",
        fontsize=10, pad=12,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure J: Domain-level pressure delta grouped bars ────────────────────

def plot_domain_pressure_deltas(
    effects: pd.DataFrame,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    """Grouped bar chart: pressure delta per domain, faceted by variant, for core conditions."""
    import numpy as np
    mpl = _mpl()
    import matplotlib.pyplot as plt

    core_pressure = ["asch_history_5", "authoritative_bias"]
    sub = effects[effects["condition_name"].isin(core_pressure)].copy()
    if sub.empty:
        print("[fig J] Domain pressure deltas: no data, skipping.")
        return

    # Pool across temperatures (weighted by n)
    pooled = (
        sub.groupby(["variant", "condition_name", "dataset_category"])
        .apply(lambda g: pd.Series({
            "delta_error": (
                (g["pressure_error_rate"] * g["n_pressure"]).sum() / g["n_pressure"].sum()
                - (g["control_error_rate"] * g["n_control"]).sum() / g["n_control"].sum()
            ),
        }), include_groups=False)
        .reset_index()
    )

    present_domains = [d for d in FACTUAL_CATEGORIES if d in pooled["dataset_category"].unique()]
    present_variants = [v for v in VARIANT_ORDER if v in pooled["variant"].unique()]
    if not present_domains or not present_variants:
        return

    n_v = len(present_variants)
    fig, axes = plt.subplots(1, n_v, figsize=(4.0 * n_v, 5.5), sharey=True)
    if n_v == 1:
        axes = [axes]

    x = np.arange(len(present_domains))
    width = 0.35
    cond_colors = {"asch_history_5": "#2980B9", "authoritative_bias": "#E67E22"}
    cond_labels = {"asch_history_5": "Peer (Asch-5)", "authoritative_bias": "Authority"}

    for ax, variant in zip(axes, present_variants):
        for i, cond in enumerate(core_pressure):
            vals = []
            for d in present_domains:
                row = pooled[
                    (pooled["variant"] == variant)
                    & (pooled["condition_name"] == cond)
                    & (pooled["dataset_category"] == d)
                ]
                vals.append(float(row["delta_error"].iloc[0]) if not row.empty else 0.0)
            offset = (i - 0.5) * width
            ax.bar(
                x + offset, vals, width=width,
                color=cond_colors.get(cond, "#95A5A6"),
                alpha=0.82, edgecolor="white",
                label=cond_labels.get(cond, cond),
            )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS.get(d, d) for d in present_domains], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Delta Error (pressure - control)" if ax is axes[0] else "", fontsize=9)
        ax.grid(axis="y", alpha=0.28, linestyle=":")

    axes[-1].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Pressure Effect on Error Rate by Domain (Core Conditions)\n"
        "(pooled across temperatures; positive = pressure increases errors)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure K: Temperature x Domain heatmap (pooled across variants) ──────

def plot_temp_domain_heatmap(
    truth_override: pd.DataFrame,
    pressure_condition: str,
    title_suffix: str,
    out_path_png: Path,
    out_path_pdf: Optional[Path] = None,
) -> None:
    """Heatmap: truth-override rate per temperature x domain, pooled across variants."""
    import seaborn as sns
    mpl = _mpl()
    import matplotlib.pyplot as plt

    sub = truth_override[truth_override["pressure_condition"] == pressure_condition].copy()
    if sub.empty:
        print(f"[fig K] Temp x Domain heatmap ({pressure_condition}): no data, skipping.")
        return

    pooled = (
        sub.groupby(["temperature", "dataset_category"])
        .apply(lambda g: pd.Series({
            "truth_override_rate": (g["truth_override_rate"] * g["n_items"]).sum() / g["n_items"].sum(),
            "n_items": g["n_items"].sum(),
        }), include_groups=False)
        .reset_index()
    )

    present_domains = [d for d in FACTUAL_CATEGORIES if d in pooled["dataset_category"].unique()]
    all_temps = sorted(pooled["temperature"].unique())
    if not present_domains or len(all_temps) < 2:
        return

    pivot = pooled.pivot_table(
        index="temperature", columns="dataset_category", values="truth_override_rate"
    ).reindex(index=all_temps, columns=present_domains)

    col_labels = [DOMAIN_LABELS.get(d, d) for d in present_domains]
    row_labels = [f"T={t}" for t in all_temps]

    fig, ax = plt.subplots(figsize=(max(8, len(present_domains) * 1.3), max(3, len(all_temps) * 0.7)))
    sns.heatmap(
        pivot.values, ax=ax,
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor="white", annot=True, fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Truth Override Rate", "shrink": 0.75},
    )
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
    ax.set_title(
        f"Truth Override Rate: Temperature x Domain — {title_suffix}\n"
        "(pooled across all column variants)",
        fontsize=10, pad=12,
    )
    plt.tight_layout()
    _save_fig(fig, out_path_png, out_path_pdf)
    plt.close(fig)
    print(f"    saved: {out_path_png.name}")


# ── Figure L: Per-variant Temperature x Domain heatmaps ──────────────────

def plot_per_variant_temp_domain_heatmaps(
    truth_override: pd.DataFrame,
    pressure_condition: str,
    title_suffix: str,
    figs_dir: Path,
) -> None:
    """One heatmap per variant: temperature x domain truth-override rates."""
    import seaborn as sns
    mpl = _mpl()
    import matplotlib.pyplot as plt

    sub = truth_override[truth_override["pressure_condition"] == pressure_condition].copy()
    if sub.empty:
        return

    present_variants = [v for v in VARIANT_ORDER if v in sub["variant"].unique()]
    present_domains = [d for d in FACTUAL_CATEGORIES if d in sub["dataset_category"].unique()]
    all_temps = sorted(sub["temperature"].unique())

    if not present_variants or not present_domains or len(all_temps) < 2:
        return

    for variant in present_variants:
        v_sub = sub[sub["variant"] == variant]
        pivot = v_sub.pivot_table(
            index="temperature", columns="dataset_category", values="truth_override_rate"
        ).reindex(index=all_temps, columns=present_domains)

        col_labels = [DOMAIN_LABELS.get(d, d) for d in present_domains]
        row_labels = [f"T={t}" for t in all_temps]

        fig, ax = plt.subplots(figsize=(max(7, len(present_domains) * 1.2), max(3, len(all_temps) * 0.65)))
        sns.heatmap(
            pivot.values, ax=ax,
            cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
            xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.5, linecolor="white", annot=True, fmt=".2f",
            annot_kws={"fontsize": 8},
            cbar_kws={"label": "Truth Override Rate", "shrink": 0.7},
        )
        ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
        ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
        vlabel = VARIANT_LABELS.get(variant, variant)
        ax.set_title(
            f"{vlabel}: Truth Override x Temperature x Domain — {title_suffix}",
            fontsize=10, pad=12,
        )
        plt.tight_layout()

        cond_tag = "peer" if "asch" in pressure_condition else "auth"
        fname = f"figL_{variant}_{cond_tag}_temp_domain"
        _save_fig(fig, figs_dir / f"{fname}.png", figs_dir / f"{fname}.pdf")
        plt.close(fig)
        print(f"    saved: {fname}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publication V2 Column – Non-Think Variant Analysis."
    )
    parser.add_argument("--runs-dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="Comparing_Experiments/publication_V2_column")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = (REPO_ROOT / args.runs_dir).resolve()

    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_file():
        metadata_path = (REPO_ROOT / args.metadata).resolve()
    if not metadata_path.is_file():
        print(f"Metadata not found: {args.metadata}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_conds = len(ALL_CONDITIONS)
    print("=" * 72)
    print("Publication V2 Column — Non-Think Variant Analysis")
    print(f"  runs_dir  : {runs_dir}")
    print(f"  metadata  : {metadata_path}")
    print(f"  out_dir   : {out_dir}")
    print(f"  Variants  : {', '.join(VARIANT_ORDER)}")
    print(f"  Excluded  : {', '.join(EXCLUDED_VARIANTS)}")
    print(f"  Conditions: {n_conds}  (3 core + {n_conds - 3} sub-conditions)")
    print("=" * 72)

    # ── Phase 1: Item-set generation ──────────────────────────────────────────
    runs = discover_runs(runs_dir, metadata_path)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    print(f"\n{len(runs)} run(s) found. Computing judge-label coverage ({n_conds} conditions)...")
    run_stats: List[Dict[str, Any]] = []
    all_coverage_rows: List[Dict[str, Any]] = []

    for label, run_id, db_path in runs:
        stats = query_run_judge_coverage(db_path, run_id, EXCLUDED_VARIANTS, ALL_CONDITIONS)
        stats["label"] = label
        stats["run_id"] = run_id
        stats["db_path"] = db_path
        run_stats.append(stats)
        n_full = len(stats["fully_covered_items"])
        print(
            f"  T={label}: {stats['n_included_models']} models, "
            f"{n_full} items with complete {n_conds}-condition coverage "
            f"(need {stats['expected_labels_per_item']} labels/item)"
        )
        for m in stats["per_model_coverage"]:
            all_coverage_rows.append({
                "temperature": label,
                "variant": m["variant"],
                "model_id": m["model_id"],
                "n_expected": m["n_expected"],
                "n_judge_labeled": m["n_judge_labeled"],
                "coverage_pct": m["coverage_pct"],
                "complete": m["complete"],
            })

    all_sets = [s["fully_covered_items"] for s in run_stats]
    intersection: set = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    print(f"\nItems with complete coverage in ALL {len(runs)} runs: {len(intersection)}")

    meta_rows = query_item_metadata(run_stats[0]["db_path"], intersection)
    dataset_to_items: Dict[str, set] = {}
    for r in meta_rows:
        dataset_to_items.setdefault(r["dataset_name"], set()).add(r["item_id"])

    print("\nPer-dataset counts in publication item set:")
    dataset_counts: Dict[str, int] = {}
    for ds, items in sorted(dataset_to_items.items()):
        dataset_counts[ds] = len(items)
        print(f"  {ds:<40} {len(items):>4} items")

    min_per_dataset = min(dataset_counts.values()) if dataset_counts else 0
    n_included_models = run_stats[0]["n_included_models"]
    total_pub_trials = len(intersection) * n_conds * n_included_models * len(runs)

    print(f"\n  Total items:                    {len(intersection)}")
    print(f"  Min items per dataset:          {min_per_dataset}")
    print(f"  Included models per run:        {n_included_models}")
    print(f"  Total judge-labeled trials:     {total_pub_trials:,}")

    # Write item_set.csv
    item_set_path = out_dir / "item_set.csv"
    meta_by_id = {r["item_id"]: r for r in meta_rows}
    with open(item_set_path, "w", encoding="utf-8") as f:
        f.write("item_id,dataset_name,domain,ground_truth_text\n")
        for item_id in sorted(intersection):
            r = meta_by_id.get(item_id, {})
            ds = r.get("dataset_name", "")
            domain = r.get("domain", "")
            gt = (r.get("ground_truth_text") or "").replace('"', '""')
            f.write(f'"{item_id}","{ds}","{domain}","{gt}"\n')
    print(f"\nItem set written: {item_set_path} ({len(intersection)} items)")

    # Write judge_coverage_by_model.csv
    coverage_path = out_dir / "judge_coverage_by_model.csv"
    with open(coverage_path, "w", encoding="utf-8") as f:
        f.write("temperature,variant,model_id,n_expected,n_judge_labeled,coverage_pct,complete\n")
        for row in all_coverage_rows:
            f.write(
                f"{row['temperature']},{row['variant']},{row['model_id']},"
                f"{row['n_expected']},{row['n_judge_labeled']},{row['coverage_pct']},{row['complete']}\n"
            )
    print(f"Coverage CSV:     {coverage_path}")

    if not args.run_analysis:
        print(
            "\n[Phases 2-5 skipped] Pass --run-analysis to load trial data, "
            "compute statistics, and generate figures."
        )
        print("=" * 72)
        return 0

    # ── Phase 2: Trial data loading ───────────────────────────────────────────
    print("\n-- Phase 2: Loading trial data ----------------------------------------")
    print(f"  Loading all {n_conds} conditions for {len(intersection)} publication items...")
    df_all = load_all_trial_data(runs, ALL_CONDITIONS, EXCLUDED_VARIANTS, intersection)
    if df_all.empty:
        print("No trial data loaded. Exiting.", file=sys.stderr)
        return 1

    print(
        f"  Loaded {len(df_all):,} rows  |  "
        f"{df_all['variant'].nunique()} variants  |  "
        f"{df_all['condition_name'].nunique()} conditions  |  "
        f"{df_all['temperature'].nunique()} temperatures"
    )

    # Balance check
    balance_df = check_balance(df_all)
    stats_dir = out_dir / "statistical_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)
    if not balance_df.empty:
        balance_path = stats_dir / "cell_balance_check.csv"
        balance_df.to_csv(balance_path, index=False)
        n_imb = int(balance_df["imbalanced"].sum()) if "imbalanced" in balance_df.columns else 0
        if n_imb > 0:
            print(f"  [WARN] {n_imb} cells deviate >5% from median — see {balance_path.name}")
        else:
            median_n = int(balance_df["n_trials"].median())
            print(f"  Balance: all cells within 5% of median ({median_n} trials/cell).")

    df_all["variant"] = pd.Categorical(df_all["variant"], categories=VARIANT_ORDER, ordered=True)

    # ── Phase 3: Metric computation ───────────────────────────────────────────
    print("\n-- Phase 3: Computing metrics -----------------------------------------")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    error_rates = compute_error_rates(df_all)
    if not error_rates.empty:
        error_rates.to_csv(tables_dir / "error_rates_all_conditions.csv", index=False)
        print(f"  Error rates:         {len(error_rates)} rows")

    effects = compute_pressure_effects(error_rates)
    if not effects.empty:
        effects.to_csv(tables_dir / "pressure_effects_all_conditions.csv", index=False)
        print(f"  Pressure effects:    {len(effects)} rows")

    pressure_conds = [c for c in ALL_CONDITIONS if c != "control" and c in df_all["condition_name"].unique()]
    override_dfs, rescue_dfs, flip_dfs = [], [], []
    for cond in pressure_conds:
        ov = compute_truth_override(df_all, cond)
        if not ov.empty:
            override_dfs.append(ov)
        rs = compute_truth_rescue(df_all, cond)
        if not rs.empty:
            rescue_dfs.append(rs)
        fl = compute_wrong_answer_flip(df_all, cond)
        if not fl.empty:
            flip_dfs.append(fl)

    truth_override = pd.concat(override_dfs, ignore_index=True) if override_dfs else pd.DataFrame()
    truth_rescue = pd.concat(rescue_dfs, ignore_index=True) if rescue_dfs else pd.DataFrame()
    wrong_flip = pd.concat(flip_dfs, ignore_index=True) if flip_dfs else pd.DataFrame()
    opinion = compute_opinion_agreement(df_all)

    if not truth_override.empty:
        truth_override.to_csv(tables_dir / "truth_override_all_conditions.csv", index=False)
        print(f"  Truth override:      {len(truth_override)} rows")
    if not truth_rescue.empty:
        truth_rescue.to_csv(tables_dir / "truth_rescue_all_conditions.csv", index=False)
        print(f"  Truth rescue:        {len(truth_rescue)} rows")
    if not wrong_flip.empty:
        wrong_flip.to_csv(tables_dir / "wrong_answer_flip_all_conditions.csv", index=False)
        print(f"  Wrong-answer flip:   {len(wrong_flip)} rows")
    if not opinion.empty:
        opinion.to_csv(tables_dir / "opinion_agreement_all_conditions.csv", index=False)
        print(f"  Opinion agreement:   {len(opinion)} rows")

    # ── Phase 4: Statistical tests ────────────────────────────────────────────
    print("\n-- Phase 4: Statistical tests -----------------------------------------")

    n_tests = len(pressure_conds) * len(VARIANT_ORDER)
    print(f"  McNemar tests: up to {n_tests} comparisons (Holm-Bonferroni corrected)...")
    mcnemar_df = compute_mcnemar_tests(df_all)
    if not mcnemar_df.empty:
        mcnemar_df.to_csv(stats_dir / "mcnemar_pressure_vs_control.csv", index=False)
        n_sig_adj = int((mcnemar_df["sig_adjusted"] != "ns").sum())
        n_sig_raw = int((mcnemar_df["sig_raw"] != "ns").sum())
        print(
            f"  McNemar: {len(mcnemar_df)} tests | "
            f"{n_sig_raw} significant (raw) | {n_sig_adj} after correction"
        )

    print("  Cochran's Q tests across condition families...")
    cochrans_q_df = compute_cochrans_q_families(df_all)
    if not cochrans_q_df.empty:
        cochrans_q_df.to_csv(stats_dir / "cochrans_q_condition_families.csv", index=False)
        print(f"  Cochran's Q: {len(cochrans_q_df)} tests computed")

    print(f"  BCa bootstrap CIs ({args.n_boot:,} resamples, vectorised jackknife)...")
    boot_cis = compute_bootstrap_cis(df_all, n_boot=args.n_boot)
    if not boot_cis.empty:
        boot_cis.to_csv(stats_dir / "bootstrap_cis.csv", index=False)
        print(f"  Bootstrap CIs: {len(boot_cis)} estimates written")

    # ── Phase 5: Publication figures ──────────────────────────────────────────
    print("\n-- Phase 5: Generating publication figures -----------------------------")
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # --- Figures A–F (standard V8 figures) ---
    if not boot_cis.empty and not mcnemar_df.empty:
        print("  [A] Forest plot of pressure effects with BCa CIs...")
        plot_forest_pressure_effects(
            boot_cis, mcnemar_df,
            figs_dir / "figA_forest_pressure_effects.png",
            figs_dir / "figA_forest_pressure_effects.pdf",
        )
        print("  [B] Heatmap with McNemar significance stars...")
        plot_heatmap_with_stars(
            boot_cis, mcnemar_df,
            figs_dir / "figB_heatmap_significance.png",
            figs_dir / "figB_heatmap_significance.pdf",
        )
    else:
        print("  [A/B] Skipped: bootstrap CIs or McNemar results unavailable.")

    print("  [C] Tone modulation comparison...")
    plot_tone_comparison(
        df_all, cochrans_q_df,
        figs_dir / "figC_tone_modulation.png",
        figs_dir / "figC_tone_modulation.pdf",
    )

    print("  [D] Authority variant comparison...")
    plot_authority_comparison(
        df_all, cochrans_q_df,
        figs_dir / "figD_authority_comparison.png",
        figs_dir / "figD_authority_comparison.pdf",
    )

    print("  [E] Temperature CI band plot...")
    plot_temperature_ci_bands(
        df_all,
        figs_dir / "figE_temperature_ci_bands.png",
        figs_dir / "figE_temperature_ci_bands.pdf",
    )

    print("  [F] Mitigation effectiveness...")
    plot_mitigation_effectiveness(
        df_all, mcnemar_df,
        figs_dir / "figF_mitigation_effectiveness.png",
        figs_dir / "figF_mitigation_effectiveness.pdf",
    )

    # --- Figures G–L (NEW domain/topic heatmaps for slides) ---
    print("\n  --- Domain & Topic Heatmaps (for slides) ---")

    print("  [G] Domain x Variant error-rate heatmap (control)...")
    plot_domain_error_heatmap(
        error_rates,
        figs_dir / "figG_domain_error_heatmap_control.png",
        figs_dir / "figG_domain_error_heatmap_control.pdf",
    )

    if not truth_override.empty:
        print("  [H] Domain x Variant truth-override heatmap (peer pressure)...")
        plot_domain_override_heatmap(
            truth_override, "asch_history_5", "Peer Pressure (Asch-5)",
            figs_dir / "figH_domain_override_peer.png",
            figs_dir / "figH_domain_override_peer.pdf",
        )

        print("  [I] Domain x Variant truth-override heatmap (authority)...")
        plot_domain_override_heatmap(
            truth_override, "authoritative_bias", "Authority Pressure",
            figs_dir / "figI_domain_override_authority.png",
            figs_dir / "figI_domain_override_authority.pdf",
        )

    print("  [J] Domain-level pressure delta grouped bars...")
    plot_domain_pressure_deltas(
        effects,
        figs_dir / "figJ_domain_pressure_deltas.png",
        figs_dir / "figJ_domain_pressure_deltas.pdf",
    )

    if not truth_override.empty:
        print("  [K] Temperature x Domain heatmap (peer, pooled variants)...")
        plot_temp_domain_heatmap(
            truth_override, "asch_history_5", "Peer Pressure (Asch-5)",
            figs_dir / "figK_temp_domain_peer.png",
            figs_dir / "figK_temp_domain_peer.pdf",
        )

        print("  [K2] Temperature x Domain heatmap (authority, pooled variants)...")
        plot_temp_domain_heatmap(
            truth_override, "authoritative_bias", "Authority Pressure",
            figs_dir / "figK2_temp_domain_authority.png",
            figs_dir / "figK2_temp_domain_authority.pdf",
        )

        print("  [L] Per-variant Temperature x Domain heatmaps (peer)...")
        plot_per_variant_temp_domain_heatmaps(
            truth_override, "asch_history_5", "Peer (Asch-5)", figs_dir,
        )

        print("  [L] Per-variant Temperature x Domain heatmaps (authority)...")
        plot_per_variant_temp_domain_heatmaps(
            truth_override, "authoritative_bias", "Authority", figs_dir,
        )

    print(f"\n  Figures   -> {figs_dir}")
    print(f"  Stats     -> {stats_dir}")
    print(f"  Tables    -> {tables_dir}")

    print("\n" + "=" * 72)
    print("DONE. Publication V2 Column analysis complete.")
    print(f"  Items   : {len(intersection)}  ({len(dataset_counts)} datasets, "
          f"{min_per_dataset} min items/dataset)")
    print(f"  Models  : {n_included_models}  ({', '.join(VARIANT_ORDER)})")
    print(f"  Excluded: {', '.join(EXCLUDED_VARIANTS)}")
    print(f"  Conditions: {n_conds} total (3 core + {n_conds - 3} sub-conditions)")
    print(f"  Temperatures: {len(runs)} (0.0 to 1.0)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
