#!/usr/bin/env python3
"""
Validate Gemma 3 1B judge labels by re-evaluating a random subsample
with a stronger model (default: gpt-oss:20b).

For each of the 11 experimental conditions, randomly samples N
judge-labeled outputs from the run databases, re-labels them using a
stronger model via OllamaJudgeClient with the identical system/user
prompts used in the original judge pipeline, and writes per-condition
agreement statistics to CSV.

The databases are never modified.  Output files:
  <out-dir>/raw.csv     - one row per sampled trial with both label sets
  <out-dir>/summary.csv - per-condition x label-dimension agreement stats

Usage
-----
python scripts/validate_judge_subsample.py \\
    --runs-dir runs_latest/runs \\
    --metadata Comparing_Experiments/runs_metadata_v6.json \\
    --n-samples 10 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Import judge infrastructure — read-only, never modifies any DB
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vivarium.experiments.olmo_conformity.ollama_judge import (
    JudgeInput,
    OllamaJudgeClient,
    OllamaJudgeConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
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
]

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_runs_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = []
    for temp_str, info in meta.get("experiments", {}).items():
        if info.get("status") != "completed":
            continue
        runs.append(
            {
                "temperature": float(temp_str),
                "run_id": str(info["run_id"]),
                "run_dir": str(info["run_dir"]),
            }
        )
    return sorted(runs, key=lambda r: r["temperature"])


# ---------------------------------------------------------------------------
# DB reading (read-only)
# ---------------------------------------------------------------------------

_SQL = """
WITH first_outputs AS (
  SELECT trial_id, MIN(created_at) AS min_created_at
  FROM conformity_outputs
  GROUP BY trial_id
),
first_output_ids AS (
  SELECT MIN(o.output_id) AS output_id, o.trial_id
  FROM conformity_outputs o
  JOIN first_outputs fo
    ON fo.trial_id = o.trial_id AND fo.min_created_at = o.created_at
  GROUP BY o.trial_id
)
SELECT
  o.output_id,
  t.trial_id,
  t.temperature,
  t.variant,
  t.model_id,
  i.item_id,
  d.name  AS dataset_name,
  i.question,
  i.ground_truth_text,
  json_extract(i.source_json, '$.wrong_answer') AS wrong_answer,
  c.name  AS condition_name,
  o.raw_text,
  json_extract(o.parsed_answer_json, '$.is_correct')            AS gemma_is_correct,
  json_extract(o.parsed_answer_json, '$.refusal_flag')          AS gemma_refusal_flag,
  json_extract(o.parsed_answer_json, '$.wrong_answer_endorsed') AS gemma_wrong_answer_endorsed,
  json_extract(o.parsed_answer_json, '$._llm_judge.judge_model') AS gemma_model
FROM conformity_trials t
JOIN conformity_conditions c  ON c.condition_id = t.condition_id
JOIN conformity_items      i  ON i.item_id      = t.item_id
JOIN conformity_datasets   d  ON d.dataset_id   = i.dataset_id
JOIN first_output_ids      foi ON foi.trial_id  = t.trial_id
JOIN conformity_outputs    o   ON o.output_id   = foi.output_id
WHERE t.run_id = ?
  AND o.parsed_answer_json IS NOT NULL
  AND trim(o.parsed_answer_json) != ''
ORDER BY t.created_at ASC;
"""


def read_judged_rows(db_path: Path, run_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(_SQL, conn, params=(run_id,))
    finally:
        conn.close()
    for col in ("gemma_is_correct", "gemma_refusal_flag", "gemma_wrong_answer_endorsed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).sum()) / n
    p1 = float(a.sum()) / n
    p2 = float(b.sum()) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    b_count = int(((a == 1) & (b == 0)).sum())
    c_count = int(((a == 0) & (b == 1)).sum())
    n = b_count + c_count
    return float(binomtest(b_count, n, 0.5).pvalue) if n > 0 else 1.0


def _agreement_stats(
    gemma: np.ndarray, strong: np.ndarray, *, dim: str
) -> dict[str, Any]:
    valid = ~(np.isnan(gemma) | np.isnan(strong))
    g = gemma[valid].astype(int)
    s = strong[valid].astype(int)
    n = int(len(g))
    if n == 0:
        return {
            "label_dimension": dim,
            "n": 0,
            "n_agree": 0,
            "agreement_rate": float("nan"),
            "cohens_kappa": float("nan"),
            "mcnemar_p": float("nan"),
        }
    n_agree = int((g == s).sum())
    return {
        "label_dimension": dim,
        "n": n,
        "n_agree": n_agree,
        "agreement_rate": n_agree / n,
        "cohens_kappa": _cohens_kappa(g, s),
        "mcnemar_p": _mcnemar_p(g, s),
    }


# ---------------------------------------------------------------------------
# Re-judging
# ---------------------------------------------------------------------------


async def _rejudge_rows(
    rows: pd.DataFrame, cfg: OllamaJudgeConfig, results: list[dict[str, Any]]
) -> None:
    """Re-evaluate each sampled row with the stronger model, appending to results."""
    total = len(rows)

    async with OllamaJudgeClient(cfg) as judge:
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            ref_answer: Optional[str] = (
                str(row["ground_truth_text"])
                if pd.notna(row.get("ground_truth_text"))
                else None
            )
            # Fix A: control condition never injects a wrong answer into the prompt,
            # even though source_json always stores one for the item. Pass None so
            # _normalise_labels correctly nulls out wrong_answer_endorsed.
            injected_wrong: Optional[str] = (
                str(row["wrong_answer"])
                if pd.notna(row.get("wrong_answer")) and str(row["condition_name"]) != "control"
                else None
            )

            ji = JudgeInput(
                condition_name=str(row["condition_name"]),
                condition_type="",
                system_prompt="",
                user_prompt="",
                chat_history_json="[]",
                question=str(row["question"] or ""),
                model_output_raw=str(row["raw_text"] or ""),
                reference_answer=ref_answer,
                injected_wrong_answer=injected_wrong,
                dataset_name=str(row["dataset_name"] or ""),
                item_id=str(row["item_id"] or ""),
                variant=str(row["variant"] or ""),
            )

            labels = await judge.judge(ji)

            print(
                f"  [{idx}/{total}] {row['condition_name']:<45}"
                f"  is_correct={labels.get('is_correct')!s:<5}"
                f"  refusal={labels.get('refusal_flag')!s:<5}"
                f"  endorsed={labels.get('wrong_answer_endorsed')!s:<5}"
            )

            results.append(
                {
                    "output_id": row["output_id"],
                    "trial_id": row["trial_id"],
                    "condition_name": row["condition_name"],
                    "temperature": row["temperature"],
                    "variant": row["variant"],
                    "model_id": row["model_id"],
                    "item_id": row["item_id"],
                    "dataset_name": row["dataset_name"],
                    # Existing (Gemma 3) labels
                    "gemma_is_correct": row.get("gemma_is_correct"),
                    "gemma_refusal_flag": row.get("gemma_refusal_flag"),
                    "gemma_wrong_answer_endorsed": row.get("gemma_wrong_answer_endorsed"),
                    "gemma_model": row.get("gemma_model"),
                    # Stronger model labels
                    "strong_is_correct": labels.get("is_correct"),
                    "strong_refusal_flag": labels.get("refusal_flag"),
                    "strong_wrong_answer_endorsed": labels.get("wrong_answer_endorsed"),
                    "strong_notes": labels.get("notes", ""),
                    "strong_model": cfg.model,
                }
            )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_DIMS = [
    ("is_correct", "gemma_is_correct", "strong_is_correct"),
    ("refusal_flag", "gemma_refusal_flag", "strong_refusal_flag"),
    ("wrong_answer_endorsed", "gemma_wrong_answer_endorsed", "strong_wrong_answer_endorsed"),
]


def _build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cond in ALL_CONDITIONS:
        grp = raw[raw["condition_name"] == cond]
        if grp.empty:
            continue
        for dim, gcol, scol in _DIMS:
            g = pd.to_numeric(grp[gcol], errors="coerce").values
            s = pd.to_numeric(grp[scol], errors="coerce").values
            stat = _agreement_stats(g, s, dim=dim)
            stat["condition_name"] = cond
            rows.append(stat)
    cols = [
        "condition_name",
        "label_dimension",
        "n",
        "n_agree",
        "agreement_rate",
        "cohens_kappa",
        "mcnemar_p",
    ]
    return pd.DataFrame(rows)[cols]


def _print_summary(summary: pd.DataFrame) -> None:
    print("\n=== Judge Validation Summary ===")
    hdr = f"{'Condition':<45} {'n':>4} {'agree':>6} {'agr%':>6} {'kappa':>7} {'mcnemar_p':>10}"
    sep = "-" * len(hdr)

    for dim, _, _ in _DIMS:
        sub = summary[summary["label_dimension"] == dim]
        if sub.empty or sub["n"].sum() == 0:
            continue
        print(f"\n-- {dim} --")
        print(hdr)
        print(sep)
        for _, r in sub.iterrows():
            agr = f"{r['agreement_rate']:.1%}" if not np.isnan(r["agreement_rate"]) else "N/A"
            kap = f"{r['cohens_kappa']:.3f}" if not np.isnan(r["cohens_kappa"]) else "N/A"
            mcp = f"{r['mcnemar_p']:.3f}" if not np.isnan(r["mcnemar_p"]) else "N/A"
            print(
                f"{r['condition_name']:<45} {int(r['n']):>4} {int(r['n_agree']):>6}"
                f" {agr:>6} {kap:>7} {mcp:>10}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate judge labels via stronger-model subsample re-evaluation"
    )
    ap.add_argument(
        "--runs-dir",
        default="runs_latest/runs",
        help="Root directory containing per-run subdirectories with simulation.db",
    )
    ap.add_argument(
        "--metadata",
        default="Comparing_Experiments/runs_metadata_v6.json",
        help="JSON file mapping temperatures to run IDs and directories",
    )
    ap.add_argument(
        "--out-dir",
        default="Comparing_Experiments/judge_eval",
        help="Directory to write raw.csv and summary.csv",
    )
    ap.add_argument(
        "--strong-model",
        default="gpt-oss:20b",
        help="Ollama model name to use as the stronger judge",
    )
    ap.add_argument(
        "--ollama-base",
        default="http://localhost:11434/v1",
        help="Ollama OpenAI-compatible base URL",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of rows to sample per condition",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    ap.add_argument(
        "--conditions",
        default=",".join(ALL_CONDITIONS),
        help="Comma-separated list of conditions to include (default: all 11)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If set, restrict sampling to runs at this temperature only",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = (repo_root / args.runs_dir).resolve()
    metadata_path = (repo_root / args.metadata).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # --- Load run index ---
    runs = load_runs_metadata(metadata_path)
    if not runs:
        print(f"No completed runs found in {metadata_path}", file=sys.stderr)
        return 1

    if args.temperature is not None:
        runs = [r for r in runs if abs(r["temperature"] - args.temperature) < 1e-6]
        if not runs:
            print(
                f"No runs match temperature={args.temperature}", file=sys.stderr
            )
            return 1

    # --- Collect all judge-labeled rows from every run ---
    dfs: list[pd.DataFrame] = []
    for run in runs:
        db_path = runs_dir / run["run_dir"] / "simulation.db"
        if not db_path.exists():
            print(f"Warning: DB not found: {db_path}", file=sys.stderr)
            continue
        print(
            f"Reading T={run['temperature']}  {run['run_dir']} ...",
            end=" ",
            flush=True,
        )
        try:
            df = read_judged_rows(db_path, run["run_id"])
            print(f"{len(df):,} judged rows")
            dfs.append(df)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)

    if not dfs:
        print("No judged rows found across any run.", file=sys.stderr)
        return 1

    all_rows = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal judged rows pooled: {len(all_rows):,}")

    # --- Filter to requested conditions ---
    all_rows = all_rows[all_rows["condition_name"].isin(conditions)]
    if all_rows.empty:
        print("No rows remain after condition filter.", file=sys.stderr)
        return 1

    # --- Sample N per condition ---
    rng = np.random.default_rng(args.seed)
    parts: list[pd.DataFrame] = []
    for cond in conditions:
        grp = all_rows[all_rows["condition_name"] == cond]
        if grp.empty:
            print(f"Warning: no judged rows for condition '{cond}'", file=sys.stderr)
            continue
        n = min(args.n_samples, len(grp))
        idx = rng.choice(len(grp), size=n, replace=False)
        parts.append(grp.iloc[idx])
        print(f"  Sampled {n}/{len(grp):,} rows from '{cond}'")

    if not parts:
        print("Nothing to sample.", file=sys.stderr)
        return 1

    sampled = pd.concat(parts, ignore_index=True)
    print(f"\nTotal sampled: {len(sampled)} rows across {len(parts)} conditions\n")

    # --- Re-judge with stronger model ---
    # Fix C: raise max_tokens from 512 to 1024 to prevent JSON truncation
    cfg = OllamaJudgeConfig(
        model=args.strong_model,
        ollama_base=args.ollama_base,
        temperature=0.0,
        max_tokens=1024,
    )
    print(f"Re-judging {len(sampled)} rows with model '{args.strong_model}' ...")

    # Results list is passed by reference so partial progress is preserved on Ctrl+C
    results: list[dict[str, Any]] = []
    interrupted = False
    try:
        asyncio.run(_rejudge_rows(sampled, cfg, results))
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[interrupted] Saving {len(results)} partial results ...")

    if not results:
        print("No results to save.", file=sys.stderr)
        return 1

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(results)
    raw_path = out_dir / "raw.csv"
    raw_df.to_csv(raw_path, index=False)
    label = " (partial — interrupted)" if interrupted else ""
    print(f"\nWrote {raw_path.relative_to(repo_root)}{label}")

    summary_df = _build_summary(raw_df)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path.relative_to(repo_root)}")

    _print_summary(summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
