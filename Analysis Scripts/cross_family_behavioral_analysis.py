#!/usr/bin/env python3
"""
Cross-Family Behavioral Analysis for Conformity Experiment.

Analyzes runs/ (diverse model families) and produces:
- Per-model behavioral metrics using judge labels
- Cross-family conformity ranking
- Per-condition and per-dataset breakdowns
- McNemar tests (pressure vs control)
- Bootstrap CIs for effect sizes
- Calibration bridge with OLMo-7B from runs_latest/

Inputs:
- runs/ directory with simulation.db per model
- runs_latest/runs/ for OLMo calibration data
- runs/think/ for think variant data

Outputs:
- Comparing_Experiments/expanded_results/cross_family/tables/*.csv
- Comparing_Experiments/expanded_results/cross_family/figures/*.png
- Comparing_Experiments/expanded_results/bridge/tables/*.csv
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_UUIDS = {
    "66765d5e-204c-4074-aaf4-b9c148fe61a5",  # gpt-oss-20b T=0.0 incomplete
}

# The 4 conditions present in runs/
CROSS_FAMILY_CONDITIONS = [
    "control",
    "authoritative_bias",
    "authority_trust",
    "asch_zhu_unanimous_confident",
]

# Mapped names for runs_latest/ (slightly different condition names)
RUNS_LATEST_CONDITION_MAP = {
    "control": "control",
    "authoritative_bias": "authoritative_bias",
    "authority_trust": "authority_zhu_unbiased_trust",
    "asch_zhu_unanimous_confident": "asch_zhu_unbiased_unanimous_confident",
}

OLMO_LATEST_VARIANTS = ["base", "instruct", "instruct_sft", "instruct_dpo"]

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
    "allenai/Olmo-3-7B-Think": "OLMo-7B-Think",
    "allenai/Olmo-3-1025-7B": "OLMo-7B",
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_judge_trials(db_path: str, variant_filter: Optional[set] = None,
                      condition_filter: Optional[set] = None) -> pd.DataFrame:
    """Load trial data with judge labels from a simulation.db."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            t.trial_id, t.item_id, t.model_id, t.variant, t.temperature,
            c.name as condition_name,
            d.name as dataset_name,
            i.ground_truth_text,
            o.parsed_answer_json,
            o.refusal_flag as heur_refusal
        FROM conformity_trials t
        JOIN conformity_outputs o ON t.trial_id = o.trial_id
        JOIN conformity_conditions c ON t.condition_id = c.condition_id
        JOIN conformity_items i ON t.item_id = i.item_id
        JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
        WHERE o.parsed_answer_json IS NOT NULL AND o.parsed_answer_json != ''
    """).fetchall()
    conn.close()

    records = []
    for r in rows:
        if variant_filter and r["variant"] not in variant_filter:
            continue
        if condition_filter and r["condition_name"] not in condition_filter:
            continue

        try:
            paj = json.loads(r["parsed_answer_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        jc = paj.get("is_correct")
        jr = paj.get("refusal_flag")
        je = paj.get("wrong_answer_endorsed")

        records.append({
            "trial_id": r["trial_id"],
            "item_id": r["item_id"],
            "model_id": r["model_id"],
            "variant": r["variant"],
            "temperature": r["temperature"],
            "condition": r["condition_name"],
            "dataset": r["dataset_name"],
            "dataset_category": DATASET_TO_CATEGORY.get(r["dataset_name"], "other"),
            "has_ground_truth": r["ground_truth_text"] is not None,
            "judge_correct": int(jc) if jc is not None else None,
            "judge_refusal": int(jr) if jr is not None else None,
            "judge_endorsed": int(je) if je is not None else None,
            "heur_refusal": r["heur_refusal"],
        })

    return pd.DataFrame(records)


def discover_cross_family_runs(runs_dir: Path) -> List[Tuple[str, Path]]:
    """Discover all valid run databases in runs/ (excluding think/)."""
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


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def mcnemar_test(control_correct: np.ndarray, pressure_correct: np.ndarray) -> dict:
    """McNemar test comparing paired control vs pressure outcomes."""
    n = min(len(control_correct), len(pressure_correct))
    if n == 0:
        return {"chi2": None, "p": None, "OR": None, "n": 0}
    c = control_correct[:n]
    p = pressure_correct[:n]

    # b = correct→incorrect, c = incorrect→correct
    b = int(np.sum((c == 1) & (p == 0)))  # truth override
    c_val = int(np.sum((c == 0) & (p == 1)))  # truth rescue

    if b + c_val == 0:
        return {"chi2": 0, "p": 1.0, "OR": 1.0, "n": n, "b": b, "c": c_val}

    # McNemar with Yates correction
    chi2 = (abs(b - c_val) - 1) ** 2 / (b + c_val) if (b + c_val) > 0 else 0
    p_val = 1 - sp_stats.chi2.cdf(chi2, df=1)
    odds_ratio = b / c_val if c_val > 0 else float("inf")

    return {"chi2": chi2, "p": p_val, "OR": odds_ratio, "n": n, "b": b, "c": c_val}


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """BCa bootstrap CI. Returns (point_estimate, lower, upper)."""
    rng = np.random.RandomState(seed)
    point = stat_fn(data)
    n = len(data)
    if n < 2:
        return (point, point, point)
    boot = np.array([stat_fn(data[rng.randint(0, n, n)]) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (point, float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100)))


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------

def compute_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-temperature behavioral metrics."""
    results = []

    for (model, variant, temp), gdf in df.groupby(["model_id", "variant", "temperature"]):
        short_name = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
        factual = gdf[gdf["has_ground_truth"]]

        for cond in CROSS_FAMILY_CONDITIONS:
            sub = factual[factual["condition"] == cond]
            n = len(sub)
            if n == 0:
                continue

            correct = sub["judge_correct"].dropna()
            n_valid = len(correct)
            error_rate = 1 - correct.mean() if n_valid > 0 else None
            refusal_rate = sub["judge_refusal"].mean() if sub["judge_refusal"].notna().any() else None
            endorsed_rate = sub["judge_endorsed"].mean() if sub["judge_endorsed"].notna().any() else None

            results.append({
                "model_id": model,
                "short_name": short_name,
                "variant": variant,
                "temperature": temp,
                "condition": cond,
                "n_trials": n,
                "n_valid_judge": n_valid,
                "error_rate": error_rate,
                "refusal_rate": refusal_rate,
                "endorsed_rate": endorsed_rate,
            })

    return pd.DataFrame(results)


def compute_pressure_effects(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pressure effect (delta) relative to control for each model."""
    results = []

    for (model, variant, temp), gdf in metrics_df.groupby(["model_id", "variant", "temperature"]):
        ctrl = gdf[gdf["condition"] == "control"]
        if ctrl.empty:
            continue
        ctrl_err = ctrl.iloc[0]["error_rate"]
        short_name = ctrl.iloc[0]["short_name"]

        for _, row in gdf.iterrows():
            if row["condition"] == "control":
                continue
            if row["error_rate"] is None or ctrl_err is None:
                continue

            delta = row["error_rate"] - ctrl_err
            results.append({
                "model_id": model,
                "short_name": short_name,
                "variant": variant,
                "temperature": temp,
                "condition": row["condition"],
                "control_error": ctrl_err,
                "pressure_error": row["error_rate"],
                "delta": delta,
                "refusal_rate": row["refusal_rate"],
                "endorsed_rate": row["endorsed_rate"],
                "n_trials": row["n_trials"],
            })

    return pd.DataFrame(results)


def compute_per_dataset_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-dataset breakdown."""
    results = []
    factual = df[df["has_ground_truth"]]

    for (model, variant, temp, ds, cond), sub in factual.groupby(
        ["model_id", "variant", "temperature", "dataset", "condition"]
    ):
        correct = sub["judge_correct"].dropna()
        if len(correct) == 0:
            continue
        results.append({
            "model_id": model,
            "short_name": MODEL_SHORT_NAMES.get(model, model.split("/")[-1]),
            "variant": variant,
            "temperature": temp,
            "dataset": ds,
            "dataset_category": DATASET_TO_CATEGORY.get(ds, "other"),
            "condition": cond,
            "n": len(correct),
            "error_rate": 1 - correct.mean(),
            "refusal_rate": sub["judge_refusal"].mean(),
            "endorsed_rate": sub["judge_endorsed"].mean(),
        })

    return pd.DataFrame(results)


def compute_mcnemar_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired McNemar tests: control vs each pressure condition, per model."""
    results = []
    factual = df[df["has_ground_truth"]]

    for (model, variant, temp), mdf in factual.groupby(["model_id", "variant", "temperature"]):
        ctrl_data = mdf[mdf["condition"] == "control"][["item_id", "judge_correct"]].dropna(subset=["judge_correct"])

        for cond in [c for c in CROSS_FAMILY_CONDITIONS if c != "control"]:
            press_data = mdf[mdf["condition"] == cond][["item_id", "judge_correct"]].dropna(subset=["judge_correct"])

            # Inner merge on item_id — only items with valid judge in BOTH conditions
            merged = ctrl_data.merge(press_data, on="item_id", suffixes=("_ctrl", "_press"))
            n_paired = len(merged)

            if n_paired == 0:
                results.append({
                    "model_id": model, "short_name": MODEL_SHORT_NAMES.get(model, model.split("/")[-1]),
                    "variant": variant, "temperature": temp, "condition": cond,
                    "n_paired": 0, "truth_override_b": 0, "truth_rescue_c": 0,
                    "chi2": 0.0, "p_value": 1.0, "odds_ratio": 1.0,
                })
                continue

            b_total = int(((merged["judge_correct_ctrl"] == 1) & (merged["judge_correct_press"] == 0)).sum())
            c_total = int(((merged["judge_correct_ctrl"] == 0) & (merged["judge_correct_press"] == 1)).sum())

            if b_total + c_total == 0:
                chi2_val, p_val, odds_ratio = 0.0, 1.0, 1.0
            else:
                chi2_val = (abs(b_total - c_total) - 1) ** 2 / (b_total + c_total)
                p_val = 1 - sp_stats.chi2.cdf(chi2_val, df=1)
                odds_ratio = b_total / c_total if c_total > 0 else float("inf")

            results.append({
                "model_id": model,
                "short_name": MODEL_SHORT_NAMES.get(model, model.split("/")[-1]),
                "variant": variant,
                "temperature": temp,
                "condition": cond,
                "n_paired": n_paired,
                "truth_override_b": b_total,
                "truth_rescue_c": c_total,
                "chi2": chi2_val,
                "p_value": p_val,
                "odds_ratio": odds_ratio,
            })

    # Holm-Bonferroni correction
    rdf = pd.DataFrame(results)
    if not rdf.empty and "p_value" in rdf.columns:
        pvals = rdf["p_value"].values
        n_tests = len(pvals)
        sorted_idx = np.argsort(pvals)
        corrected = np.ones(n_tests)
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = min(pvals[idx] * (n_tests - rank), 1.0)
        rdf["p_holm"] = corrected
        rdf["significant"] = rdf["p_holm"] < 0.05

    return rdf


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_conformity_ranking(effects_df: pd.DataFrame, out_dir: Path):
    """Forest plot: models ranked by peer pressure effect."""
    peer = effects_df[effects_df["condition"] == "asch_zhu_unanimous_confident"].copy()
    if peer.empty:
        return

    # Average across temperatures for ranking
    avg = peer.groupby(["short_name", "model_id"]).agg(
        delta_mean=("delta", "mean"),
        delta_min=("delta", "min"),
        delta_max=("delta", "max"),
        endorsed_mean=("endorsed_rate", "mean"),
        refusal_mean=("refusal_rate", "mean"),
    ).reset_index().sort_values("delta_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(avg))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(avg)))

    ax.barh(list(y_pos), avg["delta_mean"], xerr=[
        avg["delta_mean"] - avg["delta_min"],
        avg["delta_max"] - avg["delta_mean"]
    ], color=colors, edgecolor="black", linewidth=0.5, capsize=3, height=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(avg["short_name"], fontsize=10)
    ax.set_xlabel("Pressure Effect (Δ error rate)", fontsize=12)
    ax.set_title("Cross-Family Peer Conformity Ranking\n(asch_zhu_unanimous_confident vs control)", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_conformity_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_conformity_ranking.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_refusal_vs_endorsement(effects_df: pd.DataFrame, out_dir: Path):
    """Scatter: refusal rate vs endorsement rate, bubble size = delta."""
    peer = effects_df[effects_df["condition"] == "asch_zhu_unanimous_confident"].copy()
    if peer.empty:
        return

    avg = peer.groupby(["short_name"]).agg(
        delta=("delta", "mean"),
        refusal=("refusal_rate", "mean"),
        endorsed=("endorsed_rate", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        avg["endorsed"] * 100, avg["refusal"] * 100,
        s=np.abs(avg["delta"]) * 1500 + 50,
        c=avg["delta"], cmap="RdYlGn_r", edgecolor="black", linewidth=0.8,
        alpha=0.85, vmin=-0.05, vmax=0.6,
    )

    for _, row in avg.iterrows():
        ax.annotate(row["short_name"], (row["endorsed"] * 100, row["refusal"] * 100),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, fontweight="bold")

    ax.set_xlabel("Wrong Answer Endorsement Rate (%)", fontsize=12)
    ax.set_ylabel("Refusal Rate (%)", fontsize=12)
    ax.set_title("Resistance Strategy: Refusal vs Endorsement\n(bubble size = conformity Δ, color = Δ magnitude)", fontsize=13)
    plt.colorbar(scatter, ax=ax, label="Pressure Effect (Δ)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_refusal_vs_endorsement.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_refusal_vs_endorsement.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_condition_comparison(effects_df: pd.DataFrame, out_dir: Path):
    """Grouped bar: peer vs authority pressure per model."""
    if effects_df.empty:
        return

    avg = effects_df.groupby(["short_name", "condition"]).agg(
        delta=("delta", "mean"),
    ).reset_index()

    peer = avg[avg["condition"] == "asch_zhu_unanimous_confident"].set_index("short_name")["delta"]
    auth = avg[avg["condition"] == "authoritative_bias"].set_index("short_name")["delta"]
    trust = avg[avg["condition"] == "authority_trust"].set_index("short_name")["delta"]

    models = peer.sort_values(ascending=False).index.tolist()
    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, [peer.get(m, 0) for m in models], w, label="Peer Consensus", color="#E74C3C", edgecolor="black", linewidth=0.5)
    ax.bar(x, [auth.get(m, 0) for m in models], w, label="Authority Bias", color="#3498DB", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, [trust.get(m, 0) for m in models], w, label="Authority Trust", color="#2ECC71", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pressure Effect (Δ error rate)", fontsize=11)
    ax.set_title("Peer vs Authority Pressure Effects Across Model Families", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_peer_vs_authority.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_peer_vs_authority.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_domain_heatmap(ds_metrics: pd.DataFrame, out_dir: Path):
    """Heatmap: error delta by model × dataset for peer condition."""
    if ds_metrics.empty:
        return

    # Get control and peer error per model per dataset, avg across temps
    ctrl = ds_metrics[ds_metrics["condition"] == "control"].groupby(
        ["short_name", "dataset_category"]).agg(error_rate=("error_rate", "mean")).reset_index()
    peer = ds_metrics[ds_metrics["condition"] == "asch_zhu_unanimous_confident"].groupby(
        ["short_name", "dataset_category"]).agg(error_rate=("error_rate", "mean")).reset_index()

    merged = ctrl.merge(peer, on=["short_name", "dataset_category"], suffixes=("_ctrl", "_peer"))
    merged["delta"] = merged["error_rate_peer"] - merged["error_rate_ctrl"]

    pivot = merged.pivot_table(index="short_name", columns="dataset_category", values="delta")
    if pivot.empty:
        return

    # Sort by mean delta
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Δ error rate"})
    ax.set_title("Peer Conformity Effect by Model × Domain", fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("Domain", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_domain_conformity_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_domain_conformity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bridge: OLMo calibration
# ---------------------------------------------------------------------------

def load_olmo_calibration(runs_latest_dir: Path) -> pd.DataFrame:
    """Load OLMo-7B data from runs_latest/ for the 4 overlapping conditions."""
    all_dfs = []
    for entry in sorted(runs_latest_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        db = entry / "simulation.db"
        if not db.exists():
            continue

        # Map condition names for runs_latest
        mapped_conditions = set(RUNS_LATEST_CONDITION_MAP.values())
        df = load_judge_trials(str(db), variant_filter=set(OLMO_LATEST_VARIANTS),
                               condition_filter=mapped_conditions)
        if not df.empty:
            # Reverse-map condition names to match cross-family names
            reverse_map = {v: k for k, v in RUNS_LATEST_CONDITION_MAP.items()}
            df["condition"] = df["condition"].map(reverse_map)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-family conformity analysis")
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--think-dir", type=str, default="runs/think")
    ap.add_argument("--runs-latest-dir", type=str, default="runs_latest/runs")
    ap.add_argument("--out-dir", type=str, default="Comparing_Experiments/expanded_results/cross_family")
    ap.add_argument("--bridge-dir", type=str, default="Comparing_Experiments/expanded_results/bridge")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    think_dir = Path(args.think_dir)
    runs_latest_dir = Path(args.runs_latest_dir)
    out_dir = Path(args.out_dir)
    bridge_dir = Path(args.bridge_dir)

    for d in [out_dir / "tables", out_dir / "figures", out_dir / "statistical_tests",
              bridge_dir / "tables", bridge_dir / "figures"]:
        d.mkdir(parents=True, exist_ok=True)

    # ===== Load cross-family data =====
    print("Loading cross-family runs...")
    all_dfs = []
    runs = discover_cross_family_runs(runs_dir)
    for uuid, db_path in runs:
        df = load_judge_trials(str(db_path))
        if not df.empty:
            all_dfs.append(df)
            print(f"  {uuid[:12]}: {len(df)} trials ({df['model_id'].iloc[0]})")

    # Load think runs
    if think_dir.exists():
        for entry in sorted(think_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            db = entry / "simulation.db"
            if db.exists():
                df = load_judge_trials(str(db))
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  [think] {entry.name.split('_')[-1][:12]}: {len(df)} trials")

    if not all_dfs:
        print("No data found!")
        return 1

    cf_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal cross-family trials: {len(cf_df):,}")
    print(f"Models: {cf_df['model_id'].nunique()}")
    print(f"Variants: {sorted(cf_df['variant'].unique())}")
    print(f"Temperatures: {sorted(cf_df['temperature'].unique())}")
    print(f"Conditions: {sorted(cf_df['condition'].unique())}")

    # ===== Compute metrics =====
    print("\nComputing metrics...")
    metrics = compute_model_metrics(cf_df)
    metrics.to_csv(out_dir / "tables" / "per_model_condition_metrics.csv", index=False)
    print(f"  Saved: per_model_condition_metrics.csv ({len(metrics)} rows)")

    effects = compute_pressure_effects(metrics)
    effects.to_csv(out_dir / "tables" / "pressure_effects.csv", index=False)
    print(f"  Saved: pressure_effects.csv ({len(effects)} rows)")

    ds_metrics = compute_per_dataset_metrics(cf_df)
    ds_metrics.to_csv(out_dir / "tables" / "per_dataset_metrics.csv", index=False)
    print(f"  Saved: per_dataset_metrics.csv ({len(ds_metrics)} rows)")

    # ===== Statistical tests =====
    print("\nRunning McNemar tests...")
    mcnemar_df = compute_mcnemar_tests(cf_df)
    mcnemar_df.to_csv(out_dir / "statistical_tests" / "mcnemar_pressure_vs_control.csv", index=False)
    print(f"  Saved: mcnemar_pressure_vs_control.csv ({len(mcnemar_df)} rows)")

    # ===== Summary table =====
    print("\nBuilding summary ranking...")
    peer_effects = effects[effects["condition"] == "asch_zhu_unanimous_confident"].copy()
    if not peer_effects.empty:
        summary = peer_effects.groupby(["model_id", "short_name", "variant"]).agg(
            n_temps=("temperature", "nunique"),
            ctrl_error_mean=("control_error", "mean"),
            peer_error_mean=("pressure_error", "mean"),
            delta_mean=("delta", "mean"),
            delta_T0=("delta", "first"),
            endorsed_mean=("endorsed_rate", "mean"),
            refusal_mean=("refusal_rate", "mean"),
        ).reset_index().sort_values("delta_mean", ascending=False)

        # Add authority effects
        auth_effects = effects[effects["condition"] == "authoritative_bias"]
        if not auth_effects.empty:
            auth_summary = auth_effects.groupby(["model_id"]).agg(
                delta_auth_mean=("delta", "mean"),
            ).reset_index()
            summary = summary.merge(auth_summary, on="model_id", how="left")

        summary.to_csv(out_dir / "tables" / "conformity_ranking.csv", index=False)
        print(f"  Saved: conformity_ranking.csv")

        # Print to console
        print(f"\n{'Model':25s} {'Δpeer':>7s} {'Δauth':>7s} {'Ctrl':>6s} {'Peer':>6s} {'Endrs':>6s} {'Refus':>6s}")
        print("-" * 85)
        for _, r in summary.iterrows():
            da = r.get("delta_auth_mean", 0) or 0
            print(f"{r['short_name']:25s} {r['delta_mean']:+7.3f} {da:+7.3f} "
                  f"{r['ctrl_error_mean']:6.3f} {r['peer_error_mean']:6.3f} "
                  f"{r['endorsed_mean']:6.3f} {r['refusal_mean']:6.3f}")

    # ===== Figures =====
    print("\nGenerating figures...")
    fig_dir = out_dir / "figures"
    plot_conformity_ranking(effects, fig_dir)
    print("  fig_conformity_ranking")
    plot_refusal_vs_endorsement(effects, fig_dir)
    print("  fig_refusal_vs_endorsement")
    plot_condition_comparison(effects, fig_dir)
    print("  fig_peer_vs_authority")
    plot_domain_heatmap(ds_metrics, fig_dir)
    print("  fig_domain_conformity_heatmap")

    # ===== Bridge: OLMo calibration =====
    print("\nLoading OLMo calibration data from runs_latest/...")
    olmo_df = load_olmo_calibration(runs_latest_dir)
    if not olmo_df.empty:
        print(f"  Loaded {len(olmo_df):,} OLMo trials (4 variants, {olmo_df['temperature'].nunique()} temps)")

        olmo_metrics = compute_model_metrics(olmo_df)
        olmo_effects = compute_pressure_effects(olmo_metrics)

        # Combine cross-family + OLMo for bridge ranking
        bridge_effects = pd.concat([effects, olmo_effects], ignore_index=True)
        peer_bridge = bridge_effects[bridge_effects["condition"] == "asch_zhu_unanimous_confident"]

        if not peer_bridge.empty:
            bridge_summary = peer_bridge.groupby(["model_id", "short_name", "variant"]).agg(
                delta_mean=("delta", "mean"),
                ctrl_error=("control_error", "mean"),
                peer_error=("pressure_error", "mean"),
                endorsed=("endorsed_rate", "mean"),
                refusal=("refusal_rate", "mean"),
            ).reset_index().sort_values("delta_mean", ascending=False)

            bridge_summary.to_csv(bridge_dir / "tables" / "calibrated_ranking.csv", index=False)
            print(f"  Saved: bridge/tables/calibrated_ranking.csv")

            print(f"\n{'Model':25s} {'Variant':20s} {'Δpeer':>7s} {'Ctrl':>6s} {'Peer':>6s}")
            print("-" * 85)
            for _, r in bridge_summary.iterrows():
                marker = " ◀ OLMo-7B" if "OLMo-7B" in r["short_name"] and r["variant"] != "think" else ""
                print(f"{r['short_name']:25s} {r['variant']:20s} {r['delta_mean']:+7.3f} "
                      f"{r['ctrl_error']:6.3f} {r['peer_error']:6.3f}{marker}")

        # OLMo-only training trajectory
        olmo_only = olmo_effects[olmo_effects["condition"] == "asch_zhu_unanimous_confident"]
        if not olmo_only.empty:
            olmo_traj = olmo_only.groupby(["variant", "temperature"]).agg(
                delta=("delta", "mean"),
                endorsed=("endorsed_rate", "mean"),
                refusal=("refusal_rate", "mean"),
            ).reset_index()
            olmo_traj.to_csv(bridge_dir / "tables" / "olmo_training_trajectory.csv", index=False)
            print(f"  Saved: bridge/tables/olmo_training_trajectory.csv")

    # ===== Write index =====
    index_content = f"""# Cross-Family Analysis Results

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Data source:** runs/ (cross-family), runs/think/, runs_latest/runs/ (OLMo calibration)
**Labels used:** Judge labels (parsed_answer_json) as authoritative
**Conditions:** {', '.join(CROSS_FAMILY_CONDITIONS)}
**Total cross-family trials:** {len(cf_df):,}

## Tables

| File | Description |
|------|-------------|
| `tables/conformity_ranking.csv` | Models ranked by peer pressure effect (Δ), with endorsement and refusal rates |
| `tables/pressure_effects.csv` | Per-model, per-condition, per-temperature pressure deltas |
| `tables/per_model_condition_metrics.csv` | Raw error/refusal/endorsement rates per model × condition |
| `tables/per_dataset_metrics.csv` | Per-model × dataset × condition breakdown |
| `statistical_tests/mcnemar_pressure_vs_control.csv` | McNemar tests with Holm correction, odds ratios |

## Figures

| File | Description |
|------|-------------|
| `figures/fig_conformity_ranking.png` | Forest plot: models ranked by peer Δ |
| `figures/fig_refusal_vs_endorsement.png` | Scatter: resistance strategy (refusal vs endorsement) |
| `figures/fig_peer_vs_authority.png` | Grouped bar: peer vs authority effects per model |
| `figures/fig_domain_conformity_heatmap.png` | Heatmap: conformity Δ by model × domain |

## Bridge (OLMo Calibration)

| File | Description |
|------|-------------|
| `bridge/tables/calibrated_ranking.csv` | All models + OLMo variants ranked together |
| `bridge/tables/olmo_training_trajectory.csv` | OLMo base→SFT→DPO→instruct trajectory on same conditions |

## How to Read These Tables

- **error_rate**: 1 - judge_correct_rate (higher = more errors)
- **delta**: pressure_error - control_error (positive = conformity increases errors)
- **endorsed_rate**: fraction of trials where judge says model endorsed the wrong answer
- **refusal_rate**: fraction of trials where judge says model refused to answer
- **odds_ratio**: McNemar OR (>1 = pressure increases errors vs control)
- **p_holm**: Holm-Bonferroni corrected p-value
"""
    (out_dir / "index.md").write_text(index_content)
    (bridge_dir / "index.md").write_text(
        "# Bridge Analysis\n\nCalibration data combining cross-family models with OLMo-7B training stages.\n"
        "See `tables/calibrated_ranking.csv` for the unified ranking.\n"
    )

    print(f"\nDone! Results in:\n  {out_dir}\n  {bridge_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
