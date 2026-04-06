#!/usr/bin/env python3
"""
runs_latest_report.py — Matrix report for runs under runs_latest/runs (or any runs dir).

For each simulation.db: per variant × temperature, reports trial counts, cell completion
(all condition×dataset cells at max_items), and judge label coverage (valid is_correct in
parsed_answer_json), matching judge_report.sh semantics.

Default variant focus: base, instruct, instruct_sft, instruct_dpo (override with --variants).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def cyan(t: str) -> str:
    return _c("36", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


JUDGE_VALID_SQL = """
  o.parsed_answer_json IS NOT NULL AND trim(o.parsed_answer_json) != ''
  AND o.parsed_answer_json NOT LIKE '%[parse_error]%'
  AND json_extract(o.parsed_answer_json, '$.is_correct') IS NOT NULL
"""


def _load_run_config(cur: sqlite3.Cursor) -> dict | None:
    row = cur.execute("SELECT run_id, config_json FROM runs LIMIT 1").fetchone()
    if row is None:
        return None
    cfg = json.loads(row["config_json"])
    sc = cfg.get("suite_config", {})
    return {
        "run_id": row["run_id"],
        "suite_config": sc,
        "expected_conditions": [c["name"] for c in sc.get("conditions", [])],
        "expected_datasets": [d["name"] for d in sc.get("datasets", [])],
        "max_items": int(sc.get("run", {}).get("max_items_per_dataset", 50)),
        "suite_temperature": sc.get("run", {}).get("temperature"),
    }


def _scan_db(db_path: Path, variant_filter: set[str] | None) -> tuple[list[dict], dict]:
    """
    Returns (rows_per_variant_key, meta) where each row is one (folder, variant, temp) aggregate.
    meta includes union of conditions/temps for summary.
    """
    rows_out: list[dict] = []
    meta: dict = {"conditions": set(), "temps": set(), "error": None}

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "conformity_trials" not in tables:
            con.close()
            meta["error"] = "no conformity_trials"
            return rows_out, meta

        rc = _load_run_config(cur)
        if rc is None:
            con.close()
            meta["error"] = "no runs row"
            return rows_out, meta

        for c in rc["expected_conditions"]:
            meta["conditions"].add(c)

        folder_id = db_path.parent.name
        run_id = rc["run_id"]
        max_items = rc["max_items"]
        exp_conds = set(rc["expected_conditions"])
        exp_ds = set(rc["expected_datasets"])
        n_cells = len(exp_conds) * len(exp_ds) if exp_conds and exp_ds else 0

        # Per variant × temperature: trials + judge_valid
        q_metrics = f"""
        SELECT
          t.variant AS variant,
          t.temperature AS temperature,
          COUNT(*) AS trials,
          SUM(CASE WHEN {JUDGE_VALID_SQL} THEN 1 ELSE 0 END) AS judge_valid
        FROM conformity_trials t
        LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
        WHERE t.run_id = ?
        GROUP BY t.variant, t.temperature
        """
        metrics = cur.execute(q_metrics, (run_id,)).fetchall()

        # Cell counts: variant × temp × cond × ds  (temp must match — grouped with variant)
        q_cells = """
        SELECT
          t.variant AS variant,
          t.temperature AS temperature,
          cc.name AS cond,
          cd.name AS ds,
          COUNT(*) AS n
        FROM conformity_trials t
        JOIN conformity_conditions cc ON t.condition_id = cc.condition_id
        JOIN conformity_items ci ON t.item_id = ci.item_id
        JOIN conformity_datasets cd ON ci.dataset_id = cd.dataset_id
        WHERE t.run_id = ?
        GROUP BY t.variant, t.temperature, cc.name, cd.name
        """
        cell_rows = cur.execute(q_cells, (run_id,)).fetchall()
        cell_map: dict[tuple[str, float, str, str], int] = {}
        for r in cell_rows:
            key = (r["variant"], float(r["temperature"]), r["cond"], r["ds"])
            cell_map[key] = int(r["n"])

        con.close()

        for m in metrics:
            variant = m["variant"]
            temp = float(m["temperature"])
            meta["temps"].add(temp)

            if variant_filter is not None and variant not in variant_filter:
                continue

            trials = int(m["trials"] or 0)
            judge_valid = int(m["judge_valid"] or 0)

            cells_ok = 0
            if n_cells and exp_conds and exp_ds:
                for cond in sorted(exp_conds):
                    for ds in sorted(exp_ds):
                        n = cell_map.get((variant, temp, cond, ds), 0)
                        if n >= max_items:
                            cells_ok += 1

            expected_trials = max_items * n_cells if n_cells else 0
            complete = n_cells > 0 and cells_ok >= n_cells and trials >= expected_trials
            judge_cov = (100.0 * judge_valid / trials) if trials else 0.0

            rows_out.append(
                {
                    "folder_id": folder_id,
                    "run_id": run_id,
                    "variant": variant,
                    "temperature": temp,
                    "trials": trials,
                    "expected_trials": expected_trials,
                    "cells_ok": cells_ok,
                    "cells_total": n_cells,
                    "max_items": max_items,
                    "judge_valid": judge_valid,
                    "judge_cov_pct": judge_cov,
                    "complete": complete,
                }
            )

    except Exception as exc:
        meta["error"] = str(exc)

    return rows_out, meta


def _by_condition_one_db(db_path: Path, run_id: str, variant: str, temp: float | None) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    wh = "t.run_id = ? AND t.variant = ?"
    args: list = [run_id, variant]
    if temp is not None:
        wh += " AND t.temperature = ?"
        args.append(temp)
    q = f"""
    SELECT
      c.name AS condition_name,
      COUNT(*) AS trials,
      SUM(CASE WHEN {JUDGE_VALID_SQL} THEN 1 ELSE 0 END) AS judge_valid
    FROM conformity_trials t
    JOIN conformity_conditions c ON c.condition_id = t.condition_id
    LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
    WHERE {wh}
    GROUP BY c.name
    ORDER BY c.name
    """
    rows = cur.execute(q, tuple(args)).fetchall()
    con.close()
    out = []
    for r in rows:
        tr = int(r["trials"] or 0)
        jv = int(r["judge_valid"] or 0)
        out.append(
            {
                "condition": r["condition_name"],
                "trials": tr,
                "judge_valid": jv,
                "judge_cov_pct": (100.0 * jv / tr) if tr else 0.0,
            }
        )
    return out


def _print_table(headers: list[str], data: list[list[str]], widths: list[int]) -> None:
    def row_str(cells: list[str]) -> str:
        parts = []
        for i, (c, w) in enumerate(zip(cells, widths)):
            parts.append(c.ljust(w) if i < 2 else c.rjust(w))
        return "  ".join(parts)

    print(row_str(headers))
    print("  ".join("-" * w for w in widths))
    for cells in data:
        print(row_str(cells))


def main() -> None:
    default_variants = "base,instruct,instruct_sft,instruct_dpo"
    ap = argparse.ArgumentParser(description="runs_latest / multi-run conformity matrix report")
    ap.add_argument(
        "--runs-dir",
        default=None,
        help="Directory containing run subfolders (default: <repo>/runs_latest/runs)",
    )
    ap.add_argument(
        "--variants",
        default=default_variants,
        help=f"Comma-separated variant names to include (default: {default_variants})",
    )
    ap.add_argument(
        "--all-variants",
        action="store_true",
        help="Include every variant found (ignore --variants)",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Roll up across run folders: one row per variant × temperature",
    )
    ap.add_argument(
        "--by-condition",
        metavar="FOLDER",
        help="Print per-condition breakdown for one folder id (under --runs-dir)",
    )
    ap.add_argument(
        "--variant",
        default="base",
        help="Variant for --by-condition (default: base)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature filter for --by-condition",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir) if args.runs_dir else repo_root / "runs_latest" / "runs"
    if not runs_dir.is_dir():
        sys.exit(f"Error: runs directory not found: {runs_dir}")

    variant_filter: set[str] | None = None
    if not args.all_variants:
        variant_filter = {v.strip() for v in args.variants.split(",") if v.strip()}

    if args.by_condition:
        folder = runs_dir / args.by_condition
        db = folder / "simulation.db"
        if not db.is_file():
            sys.exit(f"Error: no simulation.db in {folder}")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        row = cur.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
        con.close()
        if row is None:
            sys.exit("Error: empty runs table")
        run_id = row["run_id"]
        cond_rows = _by_condition_one_db(db, run_id, args.variant, args.temperature)
        tnote = f" temp={args.temperature}" if args.temperature is not None else ""
        print()
        print(bold(cyan(f"Per-condition breakdown: {args.by_condition} variant={args.variant}{tnote}")))
        print()
        headers = ["condition", "trials", "judge_valid", "judge_cov%"]
        w = [40, 8, 12, 10]
        _print_table(headers, [[r["condition"], str(r["trials"]), str(r["judge_valid"]), f"{r['judge_cov_pct']:.1f}"] for r in cond_rows], w)
        print()
        return

    db_paths = sorted(runs_dir.glob("*/simulation.db"))
    if not db_paths:
        sys.exit(f"No simulation.db under {runs_dir}/*/")

    print(dim(f"\nScanning {len(db_paths)} database(s) in {runs_dir} …\n"))

    all_rows: list[dict] = []
    union_conditions: set[str] = set()
    union_temps: set[float] = set()
    errors: list[tuple[str, str]] = []

    for dbp in db_paths:
        part, meta = _scan_db(dbp, variant_filter)
        union_conditions |= meta["conditions"]
        union_temps |= meta["temps"]
        if meta.get("error") and not part:
            errors.append((dbp.parent.name, meta["error"]))
        all_rows.extend(part)

    print(bold(cyan("  DATA SHAPE (union across scanned DBs)")))
    print(f"  Conditions ({len(union_conditions)}): {', '.join(sorted(union_conditions))}")
    print(f"  Temperatures: {', '.join(f'{t:g}' for t in sorted(union_temps))}")
    print(f"  Run folders: {len(db_paths)} | variant-filter: {variant_filter if variant_filter else 'ALL'}")
    if errors:
        print(red(f"  Skipped / empty: {len(errors)}"))
        for fid, msg in errors:
            print(dim(f"    • {fid}: {msg}"))
    print()

    if not all_rows:
        print(yellow("No rows after variant filter."))
        return

    # --- main table ---
    if args.aggregate:
        agg: dict[tuple[str, float], dict] = {}
        for r in all_rows:
            key = (r["variant"], r["temperature"])
            g = agg.setdefault(
                key,
                {
                    "variant": r["variant"],
                    "temperature": r["temperature"],
                    "n_runs": 0,
                    "n_complete": 0,
                    "trials": 0,
                    "expected_trials": 0,
                    "judge_valid": 0,
                    "cells_ok_sum": 0,
                    "cells_total": r["cells_total"],
                },
            )
            g["n_runs"] += 1
            g["trials"] += r["trials"]
            g["expected_trials"] += r["expected_trials"]
            g["judge_valid"] += r["judge_valid"]
            g["cells_ok_sum"] += r["cells_ok"]
            if r["complete"]:
                g["n_complete"] += 1
            g["cells_total"] = max(g["cells_total"], r["cells_total"])

        agg_rows = sorted(agg.values(), key=lambda x: (x["variant"], x["temperature"]))
        headers = [
            "variant",
            "temp",
            "runs",
            "complete_runs",
            "trials",
            "exp_trials",
            "judge_valid",
            "judge_cov%",
            "cells_ok",
            "cells_tot",
        ]
        table_data = []
        for g in agg_rows:
            jc = (100.0 * g["judge_valid"] / g["trials"]) if g["trials"] else 0.0
            table_data.append(
                [
                    g["variant"],
                    f"{g['temperature']:.2f}",
                    str(g["n_runs"]),
                    str(g["n_complete"]),
                    str(g["trials"]),
                    str(g["expected_trials"]),
                    str(g["judge_valid"]),
                    f"{jc:.1f}",
                    str(g["cells_ok_sum"]),
                    str(g["n_runs"] * g["cells_total"]),
                ]
            )
        print(bold("  AGGREGATE (variant × temperature)"))
    else:
        headers = [
            "folder_id",
            "variant",
            "temp",
            "trials",
            "exp",
            "cells",
            "judge",
            "judge%",
            "status",
        ]
        table_data = []
        for r in sorted(all_rows, key=lambda x: (x["folder_id"], x["variant"], x["temperature"])):
            cstr = f"{r['cells_ok']}/{r['cells_total']}"
            st = green("COMPLETE") if r["complete"] else yellow("PARTIAL")
            if r["cells_total"] == 0:
                st = red("NO_SUITE")
            table_data.append(
                [
                    r["folder_id"],
                    r["variant"],
                    f"{r['temperature']:.2f}",
                    str(r["trials"]),
                    str(r["expected_trials"]),
                    cstr,
                    str(r["judge_valid"]),
                    f"{r['judge_cov_pct']:.1f}",
                    st,
                ]
            )
        print(bold("  PER RUN FOLDER × VARIANT"))

    widths = [max(len(h), max((len(row[i]) for row in table_data), default=0)) for i, h in enumerate(headers)]
    _print_table(headers, table_data, widths)
    print()

    # --- summary counts ---
    n_complete = sum(1 for r in all_rows if r["complete"])
    n_partial = len(all_rows) - n_complete
    full_judge = sum(1 for r in all_rows if r["trials"] and r["judge_valid"] >= r["trials"])
    print(bold("  SUMMARY (filtered rows)"))
    print(f"  {green('●')} Complete rows (folder×variant): {n_complete}")
    print(f"  {yellow('●')} Partial: {n_partial}")
    print(f"  Rows with 100% judge coverage: {full_judge} / {len(all_rows)}")
    print()
    print(dim("  Tip: scripts/runs_latest_report.sh --by-condition <folder> --variant instruct"))
    print()


if __name__ == "__main__":
    main()
