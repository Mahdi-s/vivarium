#!/usr/bin/env python3
"""
Compare agreement between LLM judge labels (parsed_answer_json) and manual labels
(is_correct, refusal_flag) across all runs in runs_metadata_v6.json.

Judge labels come from parsed_answer_json: {"is_correct": 0|1|null, "refusal_flag": 0|1, ...}
Manual labels come from conformity_outputs.is_correct and conformity_outputs.refusal_flag
(updated by the enhanced rescoring pipeline).

Output: Comparing_Experiments/judge_manual_agreement/
  - summary.csv: per-run and overall agreement stats
  - by_condition.csv: per-condition agreement breakdown (pooled across runs)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def _coerce_binary(val) -> int | None:
    """Coerce to 0 or 1, or None if null/missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("null", "none", ""):
            return None
        if v in ("1", "true", "yes"):
            return 1
        if v in ("0", "false", "no"):
            return 0
    try:
        i = int(float(val))
        return 1 if i else 0
    except (ValueError, TypeError):
        return None


def load_runs_metadata(metadata_path: Path) -> dict:
    meta = json.loads(metadata_path.read_text())
    out = {}
    for temp_str, info in meta.get("experiments", {}).items():
        if info.get("status") != "completed":
            continue
        out[temp_str] = {
            "temperature": float(temp_str),
            "run_id": str(info["run_id"]),
            "run_dir": str(info["run_dir"]),
        }
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def compute_agreement(db_path: Path, run_id: str, temperature: float) -> pd.DataFrame:
    """Query DB and return rows with judge + manual labels for agreement comparison."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    df = pd.read_sql_query(
        """
        SELECT
            o.output_id,
            t.trial_id,
            t.variant,
            c.name AS condition_name,
            o.is_correct AS manual_is_correct,
            o.refusal_flag AS manual_refusal_flag,
            json_extract(o.parsed_answer_json, '$.is_correct') AS judge_is_correct,
            json_extract(o.parsed_answer_json, '$.refusal_flag') AS judge_refusal_flag
        FROM conformity_outputs o
        JOIN conformity_trials t ON t.trial_id = o.trial_id
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ? AND o.parsed_answer_json IS NOT NULL
        """,
        conn,
        params=(run_id,),
    )
    conn.close()

    if df.empty:
        return df

    df["temperature"] = temperature
    df["run_id"] = run_id

    # Coerce to comparable ints
    for col in ["manual_is_correct", "manual_refusal_flag", "judge_is_correct", "judge_refusal_flag"]:
        df[col] = df[col].apply(_coerce_binary)

    return df


def main():
    ap = argparse.ArgumentParser(description="Compare judge vs manual label agreement")
    ap.add_argument("--runs-dir", type=str, default="runs_latest/runs")
    ap.add_argument("--metadata", type=str, default="Comparing_Experiments/runs_metadata_v6.json")
    ap.add_argument("--out-dir", type=str, default="Comparing_Experiments/judge_manual_agreement")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs_metadata(metadata_path)
    if not runs:
        print("No completed runs in metadata")
        return 1

    all_rows = []
    for temp_str, info in runs.items():
        db_path = runs_dir / info["run_dir"] / "simulation.db"
        if not db_path.exists():
            print(f"[SKIP] Missing DB: {db_path}")
            continue
        df = compute_agreement(db_path, info["run_id"], info["temperature"])
        if not df.empty:
            all_rows.append(df)
            print(f"T={info['temperature']:.1f} {info['run_id'][:12]}... {len(df):,} rows with judge labels")

    if not all_rows:
        print("No data found")
        return 1

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal rows with judge + manual: {len(combined):,}")

    # --- is_correct agreement ---
    ic = combined[
        combined["manual_is_correct"].notna() & combined["judge_is_correct"].notna()
    ].copy()
    ic["agree"] = (ic["manual_is_correct"] == ic["judge_is_correct"]).astype(int)

    # --- refusal_flag agreement ---
    rf = combined[
        combined["manual_refusal_flag"].notna() & combined["judge_refusal_flag"].notna()
    ].copy()
    rf["agree"] = (rf["manual_refusal_flag"] == rf["judge_refusal_flag"]).astype(int)

    # --- Summary table ---
    summary_rows = []

    if not ic.empty:
        overall_ic = {
            "label": "is_correct",
            "scope": "overall",
            "temperature": None,
            "run_id": None,
            "condition_name": None,
            "n": len(ic),
            "n_agree": int(ic["agree"].sum()),
            "agreement_rate": float(ic["agree"].mean()),
        }
        summary_rows.append(overall_ic)

        for temp_str, info in runs.items():
            sub = ic[ic["temperature"] == info["temperature"]]
            if not sub.empty:
                summary_rows.append({
                    "label": "is_correct",
                    "scope": "per_run",
                    "temperature": info["temperature"],
                    "run_id": info["run_id"],
                    "condition_name": None,
                    "n": len(sub),
                    "n_agree": int(sub["agree"].sum()),
                    "agreement_rate": float(sub["agree"].mean()),
                })

    if not rf.empty:
        overall_rf = {
            "label": "refusal_flag",
            "scope": "overall",
            "temperature": None,
            "run_id": None,
            "condition_name": None,
            "n": len(rf),
            "n_agree": int(rf["agree"].sum()),
            "agreement_rate": float(rf["agree"].mean()),
        }
        summary_rows.append(overall_rf)

        for temp_str, info in runs.items():
            sub = rf[rf["temperature"] == info["temperature"]]
            if not sub.empty:
                summary_rows.append({
                    "label": "refusal_flag",
                    "scope": "per_run",
                    "temperature": info["temperature"],
                    "run_id": info["run_id"],
                    "condition_name": None,
                    "n": len(sub),
                    "n_agree": int(sub["agree"].sum()),
                    "agreement_rate": float(sub["agree"].mean()),
                })

    # --- Per-condition breakdown (pooled) ---
    by_cond_rows = []
    if not ic.empty:
        cond_ic = (
            ic.groupby("condition_name", as_index=False)
            .agg(n=("trial_id", "count"), n_agree=("agree", "sum"))
        )
        cond_ic["agreement_rate"] = (cond_ic["n_agree"] / cond_ic["n"]).round(4)
        cond_ic["label"] = "is_correct"
        by_cond_rows.extend(cond_ic.to_dict("records"))

    if not rf.empty:
        cond_rf = (
            rf.groupby("condition_name", as_index=False)
            .agg(n=("trial_id", "count"), n_agree=("agree", "sum"))
        )
        cond_rf["agreement_rate"] = (cond_rf["n_agree"] / cond_rf["n"]).round(4)
        cond_rf["label"] = "refusal_flag"
        by_cond_rows.extend(cond_rf.to_dict("records"))

    summary_df = pd.DataFrame(summary_rows)
    by_cond_df = pd.DataFrame(by_cond_rows)

    summary_path = out_dir / "summary.csv"
    by_cond_path = out_dir / "by_condition.csv"
    summary_df.to_csv(summary_path, index=False)
    by_cond_df.to_csv(by_cond_path, index=False)

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {by_cond_path}")

    # Console summary
    print("\n=== Judge vs Manual Agreement ===\n")
    if not ic.empty:
        rate = ic["agree"].mean() * 100
        print(f"is_correct:    {ic['agree'].sum():,} / {len(ic):,} agree ({rate:.1f}%)")
    if not rf.empty:
        rate = rf["agree"].mean() * 100
        print(f"refusal_flag:  {rf['agree'].sum():,} / {len(rf):,} agree ({rate:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
