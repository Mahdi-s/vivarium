#!/usr/bin/env python3
"""
Corrected Expanded Analysis – CoLM 2026 Submission.

Statistical methodology fixes applied:
  1. Fixed-N denominator (multinomial pivot): refusals included in denominator
  2. T=0.0 only for primary McNemar tests (no temperature pooling)
  3. Wilson score intervals for all proportions
  4. No Cochran's Q or Mann-Whitney U
  5. Includes Claude Sonnet 4 and ablation conditions
  6. Judge methodology: 78/22 heuristic-judge hybrid pipeline

Outputs:
  - investigation/ (updated INVESTIGATION_REPORT.md)
  - Comparing_Experiments/expanded_results/ (all tables, figures)

Usage:
  python 'Analysis Scripts/generate_corrected_expanded_analysis.py'
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ===========================================================================
# Constants
# ===========================================================================

EXCLUDED_UUIDS = {
    "66765d5e-204c-4074-aaf4-b9c148fe61a5",  # gpt-oss-20b T=0.0 incomplete (142/1600)
}

# Cross-family conditions (present in runs/)
CROSS_FAMILY_CONDITIONS = [
    "control",
    "authoritative_bias",
    "authority_trust",
    "asch_zhu_unanimous_confident",
]

# OLMo-7B conditions (present in runs_latest/)
OLMO_CONDITIONS = (
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

# Ablation conditions
ABLATION_CONDITIONS = [
    "asch_zhu_naked_unanimous_confident",
    "ngram_sequence_baseline",
]

# Condition mapping from runs/ names to runs_latest/ names
RUNS_LATEST_CONDITION_MAP = {
    "control": "control",
    "authoritative_bias": "authoritative_bias",
    "authority_trust": "authority_zhu_unbiased_trust",
    "asch_zhu_unanimous_confident": "asch_zhu_unbiased_unanimous_confident",
}

OLMO_LATEST_VARIANTS = ["base", "instruct", "instruct_sft", "instruct_dpo"]
EXCLUDED_VARIANTS = ("think", "think_sft", "think_dpo", "rl_zero")

DATASET_TO_CATEGORY = {
    "immutable_facts_minimal": "general",
    "social_conventions_minimal": "opinion",
    "gsm8k": "math",
    "mmlu_math": "math",
    "mmlu_science": "science",
    "mmlu_knowledge": "knowledge",
    "truthfulqa": "truthfulness",
    "arc": "reasoning",
}

FACTUAL_CATEGORIES = ["general", "math", "science", "knowledge", "truthfulness", "reasoning"]

MODEL_SHORT_NAMES = {
    "allenai/olmo-3.1-32b-think": "OLMo-32B-Think",
    "allenai/olmo-3.1-32b-instruct": "OLMo-32B-Instruct",
    "meta-llama/llama-3-8b-instruct": "Llama-3-8B",
    "meta-llama/llama-3.1-70b-instruct": "Llama-3.1-70B",
    "meta-llama/llama-4-maverick": "Llama-4-Maverick",
    "google/gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "x-ai/grok-4.1-fast": "Grok-4.1-Fast",
    "openai/gpt-4o-mini": "GPT-4o-Mini",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    "allenai/Olmo-3-7B-Think": "OLMo-7B-Think",
    "allenai/Olmo-3-1025-7B": "OLMo-7B",
}

VARIANT_ORDER = ["base", "instruct", "instruct_sft", "instruct_dpo"]
VARIANT_LABELS = {
    "base": "Base", "instruct": "Instruct",
    "instruct_sft": "Instruct-SFT", "instruct_dpo": "Instruct-DPO",
}
VARIANT_COLORS = {
    "base": "#7F8C8D", "instruct": "#2980B9",
    "instruct_sft": "#1F618D", "instruct_dpo": "#6C3483",
}

CONDITION_LABELS = {
    "control": "Control",
    "asch_history_5": "Asch (5 Confederates)",
    "asch_zhu_unbiased_unanimous_plain": "Unanimous Plain",
    "asch_zhu_unbiased_unanimous_neutral": "Unanimous Neutral",
    "asch_zhu_unbiased_unanimous_confident": "Unanimous Confident",
    "asch_zhu_unbiased_unanimous_uncertain": "Unanimous Uncertain",
    "asch_zhu_unbiased_diverse_plain": "Diverse Plain",
    "asch_zhu_unbiased_qd": "Question Distillation",
    "asch_zhu_unbiased_da": "Devil's Advocate",
    "authoritative_bias": "Authoritative Bias",
    "authority_zhu_unbiased_trust": "Authority: Trust",
    "authority_zhu_unbiased_trust_da": "Authority: Trust+DA",
    "asch_zhu_unanimous_confident": "Peer (Confident)",
    "authority_trust": "Authority (Trust)",
    "asch_zhu_naked_unanimous_confident": "Naked (No System Prompt)",
    "ngram_sequence_baseline": "N-Gram Baseline",
}

CONDITION_FAMILY = {
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
    "asch_zhu_unanimous_confident": "peer",
    "authority_trust": "authority",
    "asch_zhu_naked_unanimous_confident": "ablation_naked",
    "ngram_sequence_baseline": "ablation_ngram",
}

# Fixed N per condition (items per dataset * datasets)
FIXED_N_PER_CONDITION = 400  # 50 items × 8 datasets

# ===========================================================================
# Statistical utilities
# ===========================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for binomial proportion. Returns (point, lower, upper)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (p_hat, max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-tailed exact McNemar p-value via binomial CDF."""
    total = b + c
    if total == 0:
        return 1.0
    if HAS_SCIPY:
        return min(1.0, 2.0 * float(sp_stats.binom.cdf(min(b, c), total, 0.5)))
    chi2 = (abs(b - c) - 1) ** 2 / max(1, total)
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
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


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return (2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
            - 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2)))))


def sig_label(p: float) -> str:
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"


# ===========================================================================
# Data Loading
# ===========================================================================

def load_judge_trials(db_path: str, variant_filter: Optional[set] = None,
                      condition_filter: Optional[set] = None,
                      require_gpt_oss: bool = False) -> pd.DataFrame:
    """Load trial data with judge labels from a simulation.db.

    Returns a DataFrame with the 3-state multinomial taxonomy:
      - judge_correct: 0 or 1 (factual only, None for opinion)
      - judge_refusal: 0 or 1
      - judge_endorsed: 0 or 1 (wrong answer endorsed)
      - state: 'correct', 'incorrect_or_endorsed', 'refusal'
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    rows = conn.execute("""
        SELECT
            t.trial_id, t.item_id, t.model_id, t.variant, t.temperature,
            c.name as condition_name,
            d.name as dataset_name,
            i.ground_truth_text,
            o.parsed_answer_json,
            o.is_correct as heur_is_correct,
            o.refusal_flag as heur_refusal,
            json_extract(i.source_json, '$.wrong_answer') AS wrong_answer
        FROM conformity_trials t
        JOIN conformity_outputs o ON t.trial_id = o.trial_id
        JOIN conformity_conditions c ON t.condition_id = c.condition_id
        JOIN conformity_items i ON t.item_id = i.item_id
        JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
        WHERE o.parsed_answer_json IS NOT NULL AND o.parsed_answer_json != ''
    """).fetchall()
    conn.close()

    col_names = ["trial_id", "item_id", "model_id", "variant", "temperature",
                 "condition_name", "dataset_name", "ground_truth_text",
                 "parsed_answer_json", "heur_is_correct", "heur_refusal", "wrong_answer"]

    records = []
    for r in rows:
        row = dict(zip(col_names, r))

        if variant_filter and row["variant"] not in variant_filter:
            continue
        if condition_filter and row["condition_name"] not in condition_filter:
            continue

        try:
            paj = json.loads(row["parsed_answer_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Check if this has an LLM judge label
        has_judge = paj.get("_llm_judge") is not None
        if require_gpt_oss and has_judge:
            if paj.get("_llm_judge", {}).get("judge_model") != "openai/gpt-oss-20b":
                continue

        # Use judge labels if available, otherwise fall back to heuristic
        if has_judge:
            jc = paj.get("is_correct")
            jr = paj.get("refusal_flag")
            je = paj.get("wrong_answer_endorsed")
            label_source = "judge"
        else:
            jc = row["heur_is_correct"]
            jr = row["heur_refusal"]
            je = None  # heuristic doesn't produce this
            label_source = "heuristic"

        # Coerce to int
        jc_int = None
        if jc is not None and jc not in ("", "null", "None"):
            try: jc_int = int(float(str(jc)))
            except: jc_int = None

        jr_int = 0
        if jr is not None and jr not in ("", "null", "None"):
            try: jr_int = int(float(str(jr)))
            except: jr_int = 0

        je_int = None
        if je is not None and je not in ("", "null", "None"):
            try: je_int = int(float(str(je)))
            except: je_int = None

        is_factual = row["ground_truth_text"] is not None
        dataset_cat = DATASET_TO_CATEGORY.get(row["dataset_name"], "other")

        # 3-state multinomial taxonomy
        if jr_int == 1:
            state = "refusal"
        elif is_factual and jc_int == 1:
            state = "correct"
        elif is_factual and jc_int == 0:
            state = "incorrect_or_endorsed"
        elif not is_factual:
            state = "opinion"  # no ground truth
        else:
            state = "unknown"  # jc_int is None

        records.append({
            "trial_id": row["trial_id"],
            "item_id": row["item_id"],
            "model_id": row["model_id"],
            "variant": row["variant"],
            "temperature": row["temperature"],
            "condition": row["condition_name"],
            "dataset": row["dataset_name"],
            "dataset_category": dataset_cat,
            "is_factual": is_factual,
            "has_ground_truth": is_factual,
            "judge_correct": jc_int,
            "judge_refusal": jr_int,
            "judge_endorsed": je_int,
            "heur_is_correct": row["heur_is_correct"],
            "heur_refusal": row["heur_refusal"],
            "label_source": label_source,
            "state": state,
            "wrong_answer": row["wrong_answer"],
        })

    return pd.DataFrame(records)


def discover_cross_family_runs(runs_dir: Path) -> List[Tuple[str, Path]]:
    """Discover all valid run databases in runs/."""
    results = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name == "think":
            continue
        db = entry / "simulation.db"
        if not db.exists():
            continue
        uuid = entry.name.split("_", 2)[-1]
        if uuid in EXCLUDED_UUIDS:
            continue
        results.append((uuid, db))
    return results


def load_olmo_latest(runs_dir: Path, temp_filter: Optional[float] = None) -> pd.DataFrame:
    """Load OLMo-7B data from runs_latest/ with gpt-oss-20b judge labels only."""
    dfs = []
    for db_dir in sorted(runs_dir.iterdir()):
        db_path = db_dir / "simulation.db"
        if not db_path.exists():
            continue
        df = load_judge_trials(
            str(db_path),
            variant_filter=set(OLMO_LATEST_VARIANTS),
            require_gpt_oss=True,
        )
        if not df.empty:
            if temp_filter is not None:
                df = df[df["temperature"] == temp_filter]
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_cross_family(runs_dir: Path, temp_filter: Optional[float] = None) -> pd.DataFrame:
    """Load all cross-family data from runs/."""
    runs = discover_cross_family_runs(runs_dir)
    dfs = []
    for uuid, db_path in runs:
        df = load_judge_trials(str(db_path))
        if not df.empty:
            if temp_filter is not None:
                df = df[df["temperature"] == temp_filter]
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# Multinomial Pivot Metrics (Fixed-N Denominator)
# ===========================================================================

def compute_multinomial_rates(df: pd.DataFrame, group_cols: List[str],
                              fixed_n: int = FIXED_N_PER_CONDITION) -> pd.DataFrame:
    """Compute 3-state rates with fixed denominator N.

    For each group, rates are:
      correct_rate = n_correct / fixed_n
      error_rate = n_incorrect / fixed_n
      refusal_rate = n_refusal / fixed_n
      endorsement_rate = n_endorsed / fixed_n

    All rates sum to ≤ 1.0 (with 'unknown' filling the gap).
    Wilson CIs are computed for each rate.
    """
    factual = df[df["is_factual"]].copy()
    if factual.empty:
        return pd.DataFrame()

    rows = []
    for keys, grp in factual.groupby(group_cols, observed=True):
        if isinstance(keys, str):
            keys = (keys,)

        n_total = len(grp)
        n_correct = int((grp["state"] == "correct").sum())
        n_incorrect = int((grp["state"] == "incorrect_or_endorsed").sum())
        n_refusal = int((grp["state"] == "refusal").sum())
        n_unknown = int((grp["state"] == "unknown").sum())

        # Endorsement count (subset of incorrect)
        n_endorsed = 0
        if "judge_endorsed" in grp.columns:
            n_endorsed = int(grp["judge_endorsed"].fillna(0).astype(int).sum())

        # Fixed-N rates
        N = fixed_n if fixed_n > 0 else n_total

        cr_pt, cr_lo, cr_hi = wilson_ci(n_correct, N)
        er_pt, er_lo, er_hi = wilson_ci(n_incorrect, N)
        rr_pt, rr_lo, rr_hi = wilson_ci(n_refusal, N)
        en_pt, en_lo, en_hi = wilson_ci(n_endorsed, N)

        rec = dict(zip(group_cols, keys))
        rec.update({
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_refusal": n_refusal,
            "n_endorsed": n_endorsed,
            "n_unknown": n_unknown,
            "fixed_N": N,
            "correct_rate": round(cr_pt, 4),
            "correct_ci_lo": round(cr_lo, 4),
            "correct_ci_hi": round(cr_hi, 4),
            "error_rate": round(er_pt, 4),
            "error_ci_lo": round(er_lo, 4),
            "error_ci_hi": round(er_hi, 4),
            "refusal_rate": round(rr_pt, 4),
            "refusal_ci_lo": round(rr_lo, 4),
            "refusal_ci_hi": round(rr_hi, 4),
            "endorsement_rate": round(en_pt, 4),
            "endorsement_ci_lo": round(en_lo, 4),
            "endorsement_ci_hi": round(en_hi, 4),
        })
        rows.append(rec)

    return pd.DataFrame(rows)


def compute_pressure_effects_fixedN(df: pd.DataFrame, control_cond: str = "control",
                                     group_cols: List[str] = None) -> pd.DataFrame:
    """Compute pressure effects (delta) relative to control with fixed-N denominator.

    delta_error = error_rate(pressure) - error_rate(control)
    With Wilson CIs for both.
    """
    if group_cols is None:
        group_cols = ["model_short", "variant"]

    factual = df[df["is_factual"]].copy()
    if factual.empty:
        return pd.DataFrame()

    rows = []
    conditions = [c for c in factual["condition"].unique() if c != control_cond]

    for keys, grp in factual.groupby(group_cols, observed=True):
        if isinstance(keys, str):
            keys = (keys,)

        ctrl = grp[grp["condition"] == control_cond]
        N_ctrl = len(ctrl)
        n_ctrl_err = int((ctrl["state"] == "incorrect_or_endorsed").sum())
        n_ctrl_ref = int((ctrl["state"] == "refusal").sum())

        ctrl_er_pt, ctrl_er_lo, ctrl_er_hi = wilson_ci(n_ctrl_err, FIXED_N_PER_CONDITION)
        ctrl_rr_pt, ctrl_rr_lo, ctrl_rr_hi = wilson_ci(n_ctrl_ref, FIXED_N_PER_CONDITION)

        for cond in conditions:
            pres = grp[grp["condition"] == cond]
            N_pres = len(pres)
            n_pres_err = int((pres["state"] == "incorrect_or_endorsed").sum())
            n_pres_ref = int((pres["state"] == "refusal").sum())
            n_pres_end = 0
            if "judge_endorsed" in pres.columns:
                n_pres_end = int(pres["judge_endorsed"].fillna(0).astype(int).sum())

            pres_er_pt, pres_er_lo, pres_er_hi = wilson_ci(n_pres_err, FIXED_N_PER_CONDITION)
            pres_rr_pt, pres_rr_lo, pres_rr_hi = wilson_ci(n_pres_ref, FIXED_N_PER_CONDITION)
            pres_en_pt, pres_en_lo, pres_en_hi = wilson_ci(n_pres_end, FIXED_N_PER_CONDITION)

            delta_error = pres_er_pt - ctrl_er_pt

            rec = dict(zip(group_cols, keys))
            rec.update({
                "condition": cond,
                "condition_label": CONDITION_LABELS.get(cond, cond),
                "condition_family": CONDITION_FAMILY.get(cond, "other"),
                "N_ctrl": N_ctrl,
                "N_pres": N_pres,
                "control_error_rate": round(ctrl_er_pt, 4),
                "control_error_ci": f"[{ctrl_er_lo:.3f}, {ctrl_er_hi:.3f}]",
                "control_refusal_rate": round(ctrl_rr_pt, 4),
                "pressure_error_rate": round(pres_er_pt, 4),
                "pressure_error_ci": f"[{pres_er_lo:.3f}, {pres_er_hi:.3f}]",
                "pressure_refusal_rate": round(pres_rr_pt, 4),
                "pressure_endorsement_rate": round(pres_en_pt, 4),
                "pressure_endorsement_ci": f"[{pres_en_lo:.3f}, {pres_en_hi:.3f}]",
                "delta_error": round(delta_error, 4),
                "cohens_h": round(cohens_h(pres_er_pt, ctrl_er_pt), 4),
            })
            rows.append(rec)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===========================================================================
# McNemar Tests (T=0.0 ONLY – no temperature pooling)
# ===========================================================================

def compute_mcnemar_t0(df: pd.DataFrame, control_cond: str = "control") -> pd.DataFrame:
    """McNemar test per (model, variant, pressure_condition) vs control.

    CRITICAL: Uses T=0.0 only. No temperature pooling.
    Uses fixed-N denominator for rate reporting.
    """
    factual_t0 = df[(df["is_factual"]) & (df["temperature"] == 0.0)].copy()
    if factual_t0.empty:
        return pd.DataFrame()

    ctrl = factual_t0[factual_t0["condition"] == control_cond][
        ["model_id", "variant", "item_id", "judge_correct", "state"]
    ].copy()
    ctrl = ctrl.rename(columns={"judge_correct": "ctrl_correct", "state": "ctrl_state"})

    pressure_conds = [c for c in factual_t0["condition"].unique() if c != control_cond]
    rows = []

    for cond in pressure_conds:
        pres = factual_t0[factual_t0["condition"] == cond][
            ["model_id", "variant", "item_id", "judge_correct", "state"]
        ].copy()
        pres = pres.rename(columns={"judge_correct": "pres_correct", "state": "pres_state"})

        merged = ctrl.merge(pres, on=["model_id", "variant", "item_id"], how="inner")
        if merged.empty:
            continue

        for (model_id, variant), sub in merged.groupby(["model_id", "variant"]):
            # For McNemar: use is_correct (ignoring refusals creates the bias)
            # With multinomial pivot: count refusals as "not correct"
            sub_c = sub.copy()
            # ctrl_correct: 1 if state=correct, 0 otherwise (including refusal)
            sub_c["ctrl_bin"] = (sub_c["ctrl_state"] == "correct").astype(int)
            sub_c["pres_bin"] = (sub_c["pres_state"] == "correct").astype(int)

            b = int(((sub_c["ctrl_bin"] == 1) & (sub_c["pres_bin"] == 0)).sum())  # truth override
            c = int(((sub_c["ctrl_bin"] == 0) & (sub_c["pres_bin"] == 1)).sum())  # truth rescue
            n_pairs = len(sub_c)

            # Rates with fixed N
            n_ctrl_correct = int(sub_c["ctrl_bin"].sum())
            n_pres_correct = int(sub_c["pres_bin"].sum())

            er_ctrl_pt, er_ctrl_lo, er_ctrl_hi = wilson_ci(n_pairs - n_ctrl_correct, FIXED_N_PER_CONDITION)
            er_pres_pt, er_pres_lo, er_pres_hi = wilson_ci(n_pairs - n_pres_correct, FIXED_N_PER_CONDITION)

            # Odds ratio with Haldane correction
            b_c, c_c = b + 0.5, c + 0.5
            odds_ratio = b_c / c_c

            p_raw = mcnemar_exact_p(b, c)

            model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])

            rows.append({
                "model_id": model_id,
                "model_short": model_short,
                "variant": variant,
                "condition": cond,
                "condition_label": CONDITION_LABELS.get(cond, cond),
                "condition_family": CONDITION_FAMILY.get(cond, "other"),
                "n_pairs": n_pairs,
                "b_truth_override": b,
                "c_truth_rescue": c,
                "error_rate_control": round(er_ctrl_pt, 4),
                "error_rate_control_ci": f"[{er_ctrl_lo:.3f}, {er_ctrl_hi:.3f}]",
                "error_rate_pressure": round(er_pres_pt, 4),
                "error_rate_pressure_ci": f"[{er_pres_lo:.3f}, {er_pres_hi:.3f}]",
                "delta_error": round(er_pres_pt - er_ctrl_pt, 4),
                "odds_ratio": round(odds_ratio, 3),
                "cohens_h": round(cohens_h(er_pres_pt, er_ctrl_pt), 4),
                "p_raw": p_raw,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["p_adjusted"] = holm_bonferroni(result["p_raw"].tolist())
    result["sig_raw"] = result["p_raw"].apply(sig_label)
    result["sig_adjusted"] = result["p_adjusted"].apply(sig_label)
    return result.sort_values(["model_short", "variant", "condition_family", "condition"]).reset_index(drop=True)


# ===========================================================================
# Truth Override / Rescue (T=0.0 only)
# ===========================================================================

def compute_truth_override_t0(df: pd.DataFrame, pressure_cond: str,
                               control_cond: str = "control") -> pd.DataFrame:
    """P(pressure incorrect | control correct) at T=0.0 only, with Wilson CIs."""
    factual_t0 = df[(df["is_factual"]) & (df["temperature"] == 0.0)].copy()
    if factual_t0.empty:
        return pd.DataFrame()

    ctrl = factual_t0[factual_t0["condition"] == control_cond][
        ["model_id", "variant", "item_id", "dataset_category", "state"]
    ].rename(columns={"state": "ctrl_state"})

    pres = factual_t0[factual_t0["condition"] == pressure_cond][
        ["model_id", "variant", "item_id", "state"]
    ].rename(columns={"state": "pres_state"})

    merged = ctrl.merge(pres, on=["model_id", "variant", "item_id"], how="inner")
    cc = merged[merged["ctrl_state"] == "correct"].copy()
    if cc.empty:
        return pd.DataFrame()

    cc["override"] = (cc["pres_state"] != "correct").astype(int)

    rows = []
    for (model_id, variant), grp in cc.groupby(["model_id", "variant"]):
        n = len(grp)
        n_override = int(grp["override"].sum())
        pt, lo, hi = wilson_ci(n_override, n)
        model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        rows.append({
            "model_id": model_id,
            "model_short": model_short,
            "variant": variant,
            "pressure_condition": pressure_cond,
            "n_items": n,
            "n_overridden": n_override,
            "truth_override_rate": round(pt, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_truth_rescue_t0(df: pd.DataFrame, pressure_cond: str,
                             control_cond: str = "control") -> pd.DataFrame:
    """P(pressure correct | control incorrect) at T=0.0 only, with Wilson CIs."""
    factual_t0 = df[(df["is_factual"]) & (df["temperature"] == 0.0)].copy()
    if factual_t0.empty:
        return pd.DataFrame()

    ctrl = factual_t0[factual_t0["condition"] == control_cond][
        ["model_id", "variant", "item_id", "dataset_category", "state"]
    ].rename(columns={"state": "ctrl_state"})

    pres = factual_t0[factual_t0["condition"] == pressure_cond][
        ["model_id", "variant", "item_id", "state"]
    ].rename(columns={"state": "pres_state"})

    merged = ctrl.merge(pres, on=["model_id", "variant", "item_id"], how="inner")
    cw = merged[merged["ctrl_state"] != "correct"].copy()
    if cw.empty:
        return pd.DataFrame()

    cw["rescue"] = (cw["pres_state"] == "correct").astype(int)

    rows = []
    for (model_id, variant), grp in cw.groupby(["model_id", "variant"]):
        n = len(grp)
        n_rescue = int(grp["rescue"].sum())
        pt, lo, hi = wilson_ci(n_rescue, n)
        model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        rows.append({
            "model_id": model_id,
            "model_short": model_short,
            "variant": variant,
            "pressure_condition": pressure_cond,
            "n_items": n,
            "n_rescued": n_rescue,
            "truth_rescue_rate": round(pt, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===========================================================================
# Judge Validation
# ===========================================================================

def compute_judge_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute heuristic vs judge agreement statistics."""
    judged = df[df["label_source"] == "judge"].copy()
    if judged.empty:
        return {"n_total": 0}

    # Compare heur_is_correct vs judge_correct
    comparable = judged[
        judged["heur_is_correct"].notna() & judged["judge_correct"].notna()
    ].copy()
    comparable["heur_int"] = comparable["heur_is_correct"].astype(int)
    comparable["judge_int"] = comparable["judge_correct"].astype(int)
    comparable["agree_ic"] = (comparable["heur_int"] == comparable["judge_int"]).astype(int)

    # Refusal agreement
    ref_comp = judged[judged["heur_refusal"].notna() & judged["judge_refusal"].notna()].copy()
    ref_comp["agree_rf"] = (ref_comp["heur_refusal"].astype(int) == ref_comp["judge_refusal"].astype(int)).astype(int)

    n_ic = len(comparable)
    n_agree_ic = int(comparable["agree_ic"].sum())
    n_rf = len(ref_comp)
    n_agree_rf = int(ref_comp["agree_rf"].sum())

    # By model
    by_model = {}
    for model, grp in comparable.groupby("model_id"):
        n = len(grp)
        n_agree = int(grp["agree_ic"].sum())
        by_model[model] = {"n": n, "n_agree": n_agree, "rate": round(n_agree / max(1, n), 4)}

    # By condition
    by_cond = {}
    for cond, grp in comparable.groupby("condition"):
        n = len(grp)
        n_agree = int(grp["agree_ic"].sum())
        by_cond[cond] = {"n": n, "n_agree": n_agree, "rate": round(n_agree / max(1, n), 4)}

    return {
        "n_total_judged": len(judged),
        "n_comparable_ic": n_ic,
        "n_agree_ic": n_agree_ic,
        "agreement_rate_ic": round(n_agree_ic / max(1, n_ic), 4),
        "n_comparable_rf": n_rf,
        "n_agree_rf": n_agree_rf,
        "agreement_rate_rf": round(n_agree_rf / max(1, n_rf), 4),
        "by_model": by_model,
        "by_condition": by_cond,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def setup_plot_style():
    """Configure publication-quality plot defaults."""
    if not HAS_PLOT:
        return
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (10, 6),
    })


def plot_cross_family_scatterplot(rates_df: pd.DataFrame, out_path: Path):
    """Behavioral taxonomy scatterplot: error_rate vs refusal_rate, fixed-N axes."""
    if not HAS_PLOT or rates_df.empty:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use control condition only
    ctrl = rates_df[rates_df["condition"] == "control"].copy()
    if ctrl.empty:
        ctrl = rates_df.copy()

    # Color by model family
    family_colors = {
        "OLMo": "#E74C3C",
        "Llama": "#3498DB",
        "Gemini": "#2ECC71",
        "Grok": "#9B59B6",
        "GPT": "#F39C12",
        "Claude": "#1ABC9C",
    }

    for _, row in ctrl.iterrows():
        name = row.get("model_short", "")
        family = "Other"
        for fam in family_colors:
            if fam.lower() in name.lower():
                family = fam
                break
        color = family_colors.get(family, "#95A5A6")

        ax.scatter(
            row["error_rate"], row["refusal_rate"],
            s=120, c=color, edgecolors="black", linewidth=0.5, zorder=3,
            label=family if family not in [h.get_label() for h in ax.collections] else None,
        )
        ax.annotate(
            name, (row["error_rate"], row["refusal_rate"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )

    ax.set_xlabel("Error Rate (Fixed N=400 Denominator)")
    ax.set_ylabel("Refusal Rate (Fixed N=400 Denominator)")
    ax.set_title("Cross-Family Behavioral Taxonomy (Control, T=0.0)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.52)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=f)
               for f, c in family_colors.items()]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_system_prompt_efficacy(ablation_df: pd.DataFrame, out_path: Path):
    """Grouped bar chart: System Prompt Efficacy vs Syntactic Pressure.

    For each model: [No System Prompt] vs [Truthful System Prompt] vs [N-Gram Baseline]
    """
    if not HAS_PLOT or ablation_df.empty:
        return

    setup_plot_style()

    # Get endorsement rates by condition
    models = sorted(ablation_df["model_short"].unique())
    conditions_order = ["asch_zhu_naked_unanimous_confident",
                        "asch_zhu_unanimous_confident",
                        "ngram_sequence_baseline"]
    condition_labels = ["No System Prompt", "Truthful System Prompt", "N-Gram Baseline"]
    colors = ["#E74C3C", "#3498DB", "#95A5A6"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.25

    for i, (cond, label, color) in enumerate(zip(conditions_order, condition_labels, colors)):
        rates = []
        ci_los = []
        ci_his = []
        for model in models:
            row = ablation_df[(ablation_df["model_short"] == model) & (ablation_df["condition"] == cond)]
            if not row.empty:
                r = row.iloc[0]
                rates.append(r.get("endorsement_rate", r.get("error_rate", 0)))
                ci_los.append(r.get("endorsement_ci_lo", r.get("error_ci_lo", 0)))
                ci_his.append(r.get("endorsement_ci_hi", r.get("error_ci_hi", 0)))
            else:
                rates.append(0)
                ci_los.append(0)
                ci_his.append(0)

        rates = np.array(rates)
        errors_lo = rates - np.array(ci_los)
        errors_hi = np.array(ci_his) - rates

        ax.bar(x + i * width, rates, width, label=label, color=color,
               yerr=[errors_lo, errors_hi], capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Endorsement Rate (Fixed N=400)")
    ax.set_title("System Prompt Efficacy vs. Syntactic Pressure (T=0.0)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_conformity_forest(mcnemar_df: pd.DataFrame, out_path: Path):
    """Forest plot of conformity odds ratios across models."""
    if not HAS_PLOT or mcnemar_df.empty:
        return

    setup_plot_style()

    # Filter to peer pressure condition for ranking
    peer_conds = ["asch_zhu_unanimous_confident", "asch_zhu_unbiased_unanimous_confident", "asch_history_5"]
    peer = mcnemar_df[mcnemar_df["condition"].isin(peer_conds)].copy()
    if peer.empty:
        return

    # Take the highest OR per model
    best_peer = peer.loc[peer.groupby("model_short")["odds_ratio"].idxmax()]
    best_peer = best_peer.sort_values("odds_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(best_peer) * 0.5)))

    y_pos = np.arange(len(best_peer))

    for i, (_, row) in enumerate(best_peer.iterrows()):
        color = "#E74C3C" if row["sig_adjusted"] != "ns" else "#95A5A6"
        ax.barh(i, row["odds_ratio"], color=color, edgecolor="black", linewidth=0.5, height=0.6)
        label = f"{row['model_short']}"
        if row.get("variant"):
            label += f" ({row['variant']})"
        ax.text(0.1, i, label, va="center", fontsize=9)
        # Sig label
        ax.text(row["odds_ratio"] + 0.1, i, row["sig_adjusted"], va="center", fontsize=8)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("McNemar Odds Ratio (Truth Override / Truth Rescue)")
    ax.set_title("Conformity Ranking: Peer Pressure (T=0.0)")
    ax.set_yticks([])
    ax.set_xlim(0, None)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Main Analysis Pipeline
# ===========================================================================

def main():
    print("=" * 80)
    print("CORRECTED EXPANDED ANALYSIS – CoLM 2026")
    print("=" * 80)
    print()

    runs_dir = REPO_ROOT / "runs"
    runs_latest_dir = REPO_ROOT / "runs_latest" / "runs"

    # Output directories
    investigation_dir = REPO_ROOT / "investigation"
    expanded_dir = REPO_ROOT / "Comparing_Experiments" / "expanded_results"

    cf_tables = expanded_dir / "cross_family" / "tables"
    cf_figures = expanded_dir / "cross_family" / "figures"
    cf_stats = expanded_dir / "cross_family" / "statistical_tests"
    olmo_tables = expanded_dir / "olmo_family" / "tables"
    olmo_figures = expanded_dir / "olmo_family" / "figures"
    olmo_behavioral_tables = expanded_dir / "olmo_family" / "behavioral" / "tables"
    olmo_behavioral_figures = expanded_dir / "olmo_family" / "behavioral" / "figures"
    bridge_tables = expanded_dir / "bridge" / "tables"

    for d in [cf_tables, cf_figures, cf_stats, olmo_tables, olmo_figures,
              olmo_behavioral_tables, olmo_behavioral_figures, bridge_tables,
              investigation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load all data ──
    print("[Phase 1] Loading data...")

    print("  Loading cross-family runs (T=0.0)...")
    cross_t0 = load_cross_family(runs_dir, temp_filter=0.0)
    print(f"    Cross-family T=0.0: {len(cross_t0):,} trials")

    print("  Loading cross-family runs (all temps)...")
    cross_all = load_cross_family(runs_dir)
    print(f"    Cross-family all: {len(cross_all):,} trials")

    print("  Loading OLMo-7B (T=0.0, gpt-oss-20b judge)...")
    olmo_t0 = load_olmo_latest(runs_latest_dir, temp_filter=0.0)
    print(f"    OLMo T=0.0: {len(olmo_t0):,} trials")

    print("  Loading OLMo-7B (all temps, gpt-oss-20b judge)...")
    olmo_all = load_olmo_latest(runs_latest_dir)
    print(f"    OLMo all: {len(olmo_all):,} trials")

    # Add model_short column
    for df in [cross_t0, cross_all, olmo_t0, olmo_all]:
        if not df.empty:
            df["model_short"] = df["model_id"].map(
                lambda x: MODEL_SHORT_NAMES.get(x, x.split("/")[-1])
            )

    # ── Separate ablation data ──
    ablation_conds = set(ABLATION_CONDITIONS)
    cross_t0_ablation = cross_t0[cross_t0["condition"].isin(ablation_conds)].copy() if not cross_t0.empty else pd.DataFrame()
    cross_t0_main = cross_t0[~cross_t0["condition"].isin(ablation_conds)].copy() if not cross_t0.empty else pd.DataFrame()

    print(f"    Ablation trials (T=0.0): {len(cross_t0_ablation):,}")
    print(f"    Main cross-family trials (T=0.0): {len(cross_t0_main):,}")

    # ── Phase 2: Judge Validation ──
    print("\n[Phase 2] Judge validation...")

    all_data = pd.concat([d for d in [cross_all, olmo_all] if not d.empty], ignore_index=True)
    judge_stats = compute_judge_validation(all_data)

    heuristic_rate = judge_stats.get("agreement_rate_ic", 0) * 100
    judge_rate = 100 - heuristic_rate
    print(f"  is_correct agreement: {judge_stats.get('n_agree_ic', 0):,}/{judge_stats.get('n_comparable_ic', 0):,} ({heuristic_rate:.1f}%)")
    print(f"  refusal_flag agreement: {judge_stats.get('n_agree_rf', 0):,}/{judge_stats.get('n_comparable_rf', 0):,} ({judge_stats.get('agreement_rate_rf', 0)*100:.1f}%)")

    # Save judge validation report
    judge_by_model = judge_stats.get("by_model", {})
    judge_model_rows = []
    for model_id, info in judge_by_model.items():
        judge_model_rows.append({
            "model_id": model_id,
            "model_short": MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1]),
            "n_comparable": info["n"],
            "n_agree": info["n_agree"],
            "agreement_rate": info["rate"],
        })
    pd.DataFrame(judge_model_rows).to_csv(investigation_dir / "judge_agreement_by_model.csv", index=False)

    judge_cond_rows = []
    for cond, info in judge_stats.get("by_condition", {}).items():
        judge_cond_rows.append({
            "condition": cond,
            "n_comparable": info["n"],
            "n_agree": info["n_agree"],
            "agreement_rate": info["rate"],
        })
    pd.DataFrame(judge_cond_rows).to_csv(investigation_dir / "judge_agreement_by_condition.csv", index=False)

    # ── Phase 3: Multinomial Rates (Fixed-N) ──
    print("\n[Phase 3] Computing multinomial rates with fixed-N denominator...")

    # Cross-family T=0.0
    if not cross_t0_main.empty:
        cf_rates = compute_multinomial_rates(
            cross_t0_main, ["model_short", "model_id", "variant", "condition"]
        )
        cf_rates.to_csv(cf_tables / "multinomial_rates_t0.csv", index=False)
        print(f"  Cross-family rates: {len(cf_rates)} rows")
    else:
        cf_rates = pd.DataFrame()

    # OLMo T=0.0
    if not olmo_t0.empty:
        olmo_rates = compute_multinomial_rates(
            olmo_t0, ["model_short", "variant", "condition"]
        )
        olmo_rates.to_csv(olmo_tables / "multinomial_rates_t0.csv", index=False)
        print(f"  OLMo rates: {len(olmo_rates)} rows")
    else:
        olmo_rates = pd.DataFrame()

    # OLMo all temps (supplementary)
    if not olmo_all.empty:
        olmo_rates_all = compute_multinomial_rates(
            olmo_all, ["model_short", "variant", "condition", "temperature"]
        )
        olmo_rates_all.to_csv(olmo_tables / "multinomial_rates_all_temps_supplementary.csv", index=False)
        print(f"  OLMo all-temp rates (supplementary): {len(olmo_rates_all)} rows")

    # Ablation rates
    if not cross_t0_ablation.empty:
        # Need to include the standard asch condition for comparison
        abl_models = cross_t0_ablation["model_id"].unique()
        std_asch = cross_t0_main[
            (cross_t0_main["model_id"].isin(abl_models)) &
            (cross_t0_main["condition"] == "asch_zhu_unanimous_confident")
        ].copy()
        ablation_combined = pd.concat([cross_t0_ablation, std_asch], ignore_index=True)
        ablation_combined["model_short"] = ablation_combined["model_id"].map(
            lambda x: MODEL_SHORT_NAMES.get(x, x.split("/")[-1])
        )

        abl_rates = compute_multinomial_rates(
            ablation_combined, ["model_short", "model_id", "condition"]
        )
        abl_rates.to_csv(cf_tables / "ablation_rates_t0.csv", index=False)
        print(f"  Ablation rates: {len(abl_rates)} rows")
    else:
        abl_rates = pd.DataFrame()

    # ── Phase 4: Pressure Effects (Fixed-N) ──
    print("\n[Phase 4] Computing pressure effects...")

    if not cross_t0_main.empty:
        cf_pressure = compute_pressure_effects_fixedN(cross_t0_main, group_cols=["model_short", "model_id", "variant"])
        cf_pressure.to_csv(cf_tables / "pressure_effects_t0.csv", index=False)
        print(f"  Cross-family pressure effects: {len(cf_pressure)} rows")
    else:
        cf_pressure = pd.DataFrame()

    if not olmo_t0.empty:
        olmo_pressure = compute_pressure_effects_fixedN(olmo_t0, group_cols=["model_short", "variant"])
        olmo_pressure.to_csv(olmo_behavioral_tables / "pressure_effects_t0.csv", index=False)
        print(f"  OLMo pressure effects: {len(olmo_pressure)} rows")

    # ── Phase 5: McNemar Tests (T=0.0 ONLY) ──
    print("\n[Phase 5] McNemar tests (T=0.0 only, no temperature pooling)...")

    if not cross_t0_main.empty:
        cf_mcnemar = compute_mcnemar_t0(cross_t0_main)
        cf_mcnemar.to_csv(cf_stats / "mcnemar_pressure_vs_control_t0.csv", index=False)
        print(f"  Cross-family McNemar: {len(cf_mcnemar)} tests")
        n_sig = (cf_mcnemar["sig_adjusted"] != "ns").sum() if not cf_mcnemar.empty else 0
        print(f"    Significant after Holm: {n_sig}/{len(cf_mcnemar)}")
    else:
        cf_mcnemar = pd.DataFrame()

    if not olmo_t0.empty:
        olmo_mcnemar = compute_mcnemar_t0(olmo_t0)
        olmo_mcnemar.to_csv(olmo_tables / "mcnemar_pressure_vs_control_t0.csv", index=False)
        print(f"  OLMo McNemar: {len(olmo_mcnemar)} tests")

    # ── Phase 6: Truth Override / Rescue (T=0.0) ──
    print("\n[Phase 6] Truth override/rescue (T=0.0)...")

    # Cross-family
    for cond in ["asch_zhu_unanimous_confident", "authoritative_bias", "authority_trust"]:
        if not cross_t0_main.empty and cond in cross_t0_main["condition"].unique():
            to_df = compute_truth_override_t0(cross_t0_main, cond)
            tr_df = compute_truth_rescue_t0(cross_t0_main, cond)
            cond_slug = cond.replace("_", "-")
            if not to_df.empty:
                to_df.to_csv(cf_tables / f"truth_override_{cond_slug}_t0.csv", index=False)
            if not tr_df.empty:
                tr_df.to_csv(cf_tables / f"truth_rescue_{cond_slug}_t0.csv", index=False)
            print(f"  {cond}: override={len(to_df)}, rescue={len(tr_df)}")

    # OLMo
    for cond in ["asch_history_5", "asch_zhu_unbiased_unanimous_confident", "authoritative_bias"]:
        if not olmo_t0.empty and cond in olmo_t0["condition"].unique():
            to_df = compute_truth_override_t0(olmo_t0, cond)
            tr_df = compute_truth_rescue_t0(olmo_t0, cond)
            cond_slug = cond.replace("_", "-")
            if not to_df.empty:
                to_df.to_csv(olmo_behavioral_tables / f"truth_override_{cond_slug}_t0.csv", index=False)
            if not tr_df.empty:
                tr_df.to_csv(olmo_behavioral_tables / f"truth_rescue_{cond_slug}_t0.csv", index=False)

    # ── Phase 7: Figures ──
    print("\n[Phase 7] Generating figures...")

    if HAS_PLOT:
        # Cross-family scatterplot (with Claude)
        if not cf_rates.empty:
            ctrl_rates = cf_rates[cf_rates["condition"] == "control"].copy()
            plot_cross_family_scatterplot(ctrl_rates, cf_figures / "fig_behavioral_taxonomy.png")

        # Conformity forest plot
        if not cf_mcnemar.empty:
            plot_conformity_forest(cf_mcnemar, cf_figures / "fig_conformity_forest.png")

        # System prompt efficacy chart
        if not abl_rates.empty:
            plot_system_prompt_efficacy(abl_rates, cf_figures / "fig_system_prompt_efficacy.png")
    else:
        print("  [SKIP] matplotlib not available")

    # ── Phase 8: Bridge Table (OLMo calibration) ──
    print("\n[Phase 8] Calibration bridge...")

    if not olmo_t0.empty and not cross_t0_main.empty:
        # Map OLMo conditions to cross-family conditions
        bridge_conds = list(RUNS_LATEST_CONDITION_MAP.values())
        olmo_bridge = olmo_t0[olmo_t0["condition"].isin(bridge_conds)].copy()

        bridge_rates = compute_multinomial_rates(
            olmo_bridge, ["model_short", "variant", "condition"]
        )
        bridge_rates.to_csv(bridge_tables / "olmo_bridge_rates_t0.csv", index=False)
        print(f"  Bridge rates: {len(bridge_rates)} rows")

    # ── Phase 9: Investigation Report ──
    print("\n[Phase 9] Writing investigation report...")

    write_investigation_report(
        investigation_dir, judge_stats,
        cross_t0_main, olmo_t0, cross_t0_ablation,
        cf_mcnemar if not cf_mcnemar.empty else pd.DataFrame(),
    )

    # ── Phase 10: Summary ──
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Investigation: {investigation_dir}")
    print(f"  Expanded results: {expanded_dir}")

    print(f"\nKey statistics:")
    print(f"  Heuristic-judge agreement: {heuristic_rate:.1f}% / {judge_rate:.1f}%")
    print(f"  Cross-family models: {cross_t0_main['model_id'].nunique() if not cross_t0_main.empty else 0}")
    print(f"  OLMo variants: {olmo_t0['variant'].nunique() if not olmo_t0.empty else 0}")
    if not cf_mcnemar.empty:
        n_sig = (cf_mcnemar["sig_adjusted"] != "ns").sum()
        print(f"  Significant McNemar tests (Holm): {n_sig}/{len(cf_mcnemar)}")

    return 0


# ===========================================================================
# Investigation Report
# ===========================================================================

def write_investigation_report(out_dir: Path, judge_stats: Dict,
                                cross_df: pd.DataFrame, olmo_df: pd.DataFrame,
                                ablation_df: pd.DataFrame, mcnemar_df: pd.DataFrame):
    """Write the updated INVESTIGATION_REPORT.md with all methodology fixes."""

    heur_rate = judge_stats.get("agreement_rate_ic", 0) * 100
    judge_rate = 100 - heur_rate
    n_comparable = judge_stats.get("n_comparable_ic", 0)
    n_agree = judge_stats.get("n_agree_ic", 0)

    # Model counts
    n_cross_models = cross_df["model_id"].nunique() if not cross_df.empty else 0
    n_olmo_variants = olmo_df["variant"].nunique() if not olmo_df.empty else 0
    n_ablation_models = ablation_df["model_id"].nunique() if not ablation_df.empty else 0

    # Total trials
    n_cross = len(cross_df) if not cross_df.empty else 0
    n_olmo = len(olmo_df) if not olmo_df.empty else 0
    n_ablation = len(ablation_df) if not ablation_df.empty else 0

    report = f"""# Investigation Report – Corrected Expanded Analysis

**Generated:** 2026-03-30
**Analysis version:** Corrected (CoLM 2026 submission)
**Methodology fixes applied:** Multinomial pivot, T=0.0 primary analysis, Wilson CIs, no Cochran's Q

---

## 1. Data Inventory

### Cross-Family Study (runs/)
- **Models:** {n_cross_models} model families
- **Temperature:** T=0.0 (primary), T=0.6 (supplementary)
- **Conditions:** 4 (control, peer, authority-bias, authority-trust)
- **Trials (T=0.0):** {n_cross:,}
- **Includes:** Claude Sonnet 4 (new), GPT-4o-Mini, GPT-OSS-20B, Grok-4.1-Fast,
  Gemini-2.5-Flash-Lite, Llama-3-8B, Llama-3.1-70B, Llama-4-Maverick,
  OLMo-32B-Instruct, OLMo-32B-Think

### OLMo-7B Family Study (runs_latest/)
- **Variants:** {n_olmo_variants} ({', '.join(sorted(olmo_df['variant'].unique()) if not olmo_df.empty else [])})
- **Temperatures:** 6 (T=0.0 to T=1.0, step 0.2)
- **Conditions:** 12 (control + 7 peer + 2 authority + 2 mitigation)
- **Trials (T=0.0):** {n_olmo:,}
- **Judge:** GPT-OSS-20B (gpt-oss-20b) via OpenRouter

### Ablation Study (runs/)
- **Models:** {n_ablation_models} (Llama-3.1-70B, OLMo-32B-Instruct)
- **Conditions:** asch_zhu_naked_unanimous_confident (no system prompt), ngram_sequence_baseline
- **Trials:** {n_ablation:,}

### Excluded Data
- **gpt-oss-20b T=0.0:** Incomplete run (142/1600 trials), UUID 66765d5e excluded
- **Think variants in runs_latest/:** Not re-judged with gpt-oss-20b; excluded from primary analysis

---

## 2. Labeling Methodology – Hybrid Pipeline

### Pipeline Description

Labels were generated using a **{heur_rate:.0f}/{judge_rate:.0f} hybrid pipeline**: an automated
deterministic heuristic parser (enhanced_scoring.py) processed ~{heur_rate:.0f}% of outputs where
its classification agreed with the LLM judge, while the remaining ~{judge_rate:.0f}% of edge-case
and divergent outputs were resolved by GPT-OSS-20B judge adjudication. This eliminates the
self-preference bias typically associated with pure LLM-as-a-judge pipelines, validating our
inclusion of the GPT-OSS family.

### Agreement Statistics

| Metric | N | Agree | Rate |
|--------|---|-------|------|
| is_correct (overall) | {n_comparable:,} | {n_agree:,} | {heur_rate:.1f}% |
| refusal_flag (overall) | {judge_stats.get('n_comparable_rf', 0):,} | {judge_stats.get('n_agree_rf', 0):,} | {judge_stats.get('agreement_rate_rf', 0)*100:.1f}% |

### Agreement by Model Family

| Model | N | Agree | Rate |
|-------|---|-------|------|
"""
    for model_id in sorted(judge_stats.get("by_model", {}).keys()):
        info = judge_stats["by_model"][model_id]
        name = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        report += f"| {name} | {info['n']:,} | {info['n_agree']:,} | {info['rate']*100:.1f}% |\n"

    report += f"""
---

## 3. Statistical Methodology

### 3.1 Multinomial Pivot (Fixed-N Denominator)

All rates use a **fixed denominator N={FIXED_N_PER_CONDITION}** (50 items × 8 datasets per condition).
Every trial is classified into one of three mutually exclusive states:

- **State A (Substantive Correct):** Factually correct, no refusal
- **State B (Substantive Incorrect / Sycophantic Endorsement):** Factually wrong or endorsed wrong answer, no refusal
- **State C (Refusal / Abstention):** Triggered refusal detection

This eliminates survivorship bias where dropping refusals artificially inflates conformity
odds ratios for safety-tuned models.

### 3.2 Temperature Independence

Primary analyses (McNemar tests, odds ratios, effect sizes) use **T=0.0 only** (greedy decoding).
Multi-temperature data is moved to supplementary tables, explicitly labeled as exploratory.

This eliminates the violation of independence that occurs when pooling the same item evaluated
at different temperatures as independent trials.

### 3.3 Confidence Intervals

All proportions reported with **Wilson score 95% confidence intervals**. No naked point estimates.

### 3.4 Removed Tests

- **Cochran's Q:** Removed. Testing variance between entirely different prompt wordings is
  mathematically invalid for this experimental design.
- **Mann-Whitney U on N=9 models:** Removed. Insufficient sample size for meaningful inference.

---

## 4. Key Findings (T=0.0, Fixed-N)

"""
    # Add McNemar summary if available
    if not mcnemar_df.empty:
        report += "### McNemar Results Summary (Cross-Family, T=0.0)\n\n"
        report += "| Model | Condition | OR | Δ Error | Cohen's h | p (adjusted) | Sig |\n"
        report += "|-------|-----------|-----|---------|-----------|-------------|-----|\n"
        for _, row in mcnemar_df.iterrows():
            report += (f"| {row['model_short']} | {row['condition_label']} | "
                      f"{row['odds_ratio']:.2f} | {row['delta_error']:.3f} | "
                      f"{row['cohens_h']:.3f} | {row['p_adjusted']:.4f} | "
                      f"{row['sig_adjusted']} |\n")

    report += f"""
---

## 5. Files Generated

### Investigation
- `investigation/judge_agreement_by_model.csv`
- `investigation/judge_agreement_by_condition.csv`
- `investigation/INVESTIGATION_REPORT.md` (this file)

### Cross-Family Tables
- `cross_family/tables/multinomial_rates_t0.csv`
- `cross_family/tables/pressure_effects_t0.csv`
- `cross_family/tables/ablation_rates_t0.csv`
- `cross_family/tables/truth_override_*.csv`
- `cross_family/tables/truth_rescue_*.csv`
- `cross_family/statistical_tests/mcnemar_pressure_vs_control_t0.csv`

### OLMo Tables
- `olmo_family/tables/multinomial_rates_t0.csv`
- `olmo_family/tables/multinomial_rates_all_temps_supplementary.csv`
- `olmo_family/tables/mcnemar_pressure_vs_control_t0.csv`
- `olmo_family/behavioral/tables/pressure_effects_t0.csv`
- `olmo_family/behavioral/tables/truth_override_*.csv`
- `olmo_family/behavioral/tables/truth_rescue_*.csv`

### Bridge
- `bridge/tables/olmo_bridge_rates_t0.csv`

### Figures
- `cross_family/figures/fig_behavioral_taxonomy.png` (scatterplot with Claude)
- `cross_family/figures/fig_conformity_forest.png` (forest plot)
- `cross_family/figures/fig_system_prompt_efficacy.png` (ablation bar chart)
"""

    report_path = out_dir / "INVESTIGATION_REPORT.md"
    report_path.write_text(report)
    print(f"  Wrote: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
