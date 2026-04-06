#!/usr/bin/env python3
"""
run_inventory.py — Scan a runs directory and print a formatted inventory table.

Shows per-run:  ID, model, temperature, total trials, completion status,
                and which (condition × dataset) cells are missing or short.

Usage:
    python scripts/run_inventory.py [--runs-dir runs] [--expected-per-cell 50]
                                    [--sort {id,model,temp,trials}]
                                    [--filter-model PATTERN]
                                    [--show-missing]
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path


# ── ANSI colours (disabled when not a tty) ──────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)
def cyan(t):   return _c("36", t)


# ── helpers ──────────────────────────────────────────────────────────────────
def _cell_pct(actual: int, expected: int) -> float:
    return 100.0 * actual / expected if expected else 0.0


def _short_model(model_id: str) -> str:
    """Truncate long model IDs so the table stays readable."""
    return model_id if len(model_id) <= 36 else model_id[:33] + "…"


def _extract_run_info(db_path: Path) -> dict | None:
    """Read a simulation.db and return a summary dict, or None on error."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # ── config ──────────────────────────────────────────────────────────
        row = cur.execute("SELECT run_id, config_json FROM runs LIMIT 1").fetchone()
        if row is None:
            con.close()
            return None

        run_id   = row["run_id"]
        cfg      = json.loads(row["config_json"])
        sc       = cfg.get("suite_config", {})
        models   = sc.get("models", [{}])
        model_id = models[0].get("model_id", "unknown") if models else "unknown"
        variant  = models[0].get("variant", "") if models else ""
        temp     = sc.get("run", {}).get("temperature", float("nan"))
        max_items = sc.get("run", {}).get("max_items_per_dataset", 50)

        # ── expected conditions × datasets from config ───────────────────────
        expected_conditions = {c["name"] for c in sc.get("conditions", [])}
        expected_datasets   = {d["name"] for d in sc.get("datasets",   [])}

        # ── actual trial counts ──────────────────────────────────────────────
        # Check if conformity tables exist
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        cell_counts: dict[tuple[str, str], int] = {}   # (condition, dataset) → count
        if "conformity_trials" in tables:
            rows = cur.execute("""
                SELECT cc.name AS cond, cd.name AS ds, COUNT(*) AS n
                FROM conformity_trials ct
                JOIN conformity_conditions cc ON ct.condition_id = cc.condition_id
                JOIN conformity_items      ci ON ct.item_id      = ci.item_id
                JOIN conformity_datasets   cd ON ci.dataset_id   = cd.dataset_id
                GROUP BY cc.name, cd.name
            """).fetchall()
            for r in rows:
                cell_counts[(r["cond"], r["ds"])] = r["n"]

        con.close()

        # ── compute missing / short cells ────────────────────────────────────
        total_actual   = sum(cell_counts.values())
        total_expected = max_items * len(expected_conditions) * len(expected_datasets)

        missing_cells: list[str] = []   # cells with 0 trials
        short_cells: list[str]   = []   # cells with > 0 but < max_items

        for cond in sorted(expected_conditions):
            for ds in sorted(expected_datasets):
                n = cell_counts.get((cond, ds), 0)
                if n == 0:
                    missing_cells.append(f"{cond}/{ds}")
                elif n < max_items:
                    short_cells.append(f"{cond}/{ds}({n}/{max_items})")

        # Also catch cells present in DB but not in config (orphaned)
        for (cond, ds), n in cell_counts.items():
            if cond not in expected_conditions or ds not in expected_datasets:
                short_cells.append(f"[extra]{cond}/{ds}({n})")

        # folder name = last two parts of path (e.g. runs/20260327_.../simulation.db)
        folder_id = db_path.parent.name

        return {
            "folder_id":    folder_id,
            "run_id":       run_id,
            "model_id":     model_id,
            "variant":      variant,
            "temperature":  temp,
            "total_actual": total_actual,
            "total_expected": total_expected,
            "max_items":    max_items,
            "n_conditions": len(expected_conditions),
            "n_datasets":   len(expected_datasets),
            "missing_cells": missing_cells,
            "short_cells":  short_cells,
            "cell_counts":  cell_counts,
            "expected_conditions": expected_conditions,
            "expected_datasets":   expected_datasets,
        }
    except Exception as exc:
        return {"_error": str(exc), "folder_id": db_path.parent.name}


def _status_tag(info: dict) -> str:
    if "_error" in info:
        return red("ERROR")
    tot_e = info["total_expected"]
    tot_a = info["total_actual"]
    miss  = info["missing_cells"]
    short = info["short_cells"]
    if tot_e == 0:
        return yellow("EMPTY")
    if not miss and not short:
        return green("COMPLETE")
    pct = _cell_pct(tot_a, tot_e)
    if pct >= 80:
        return yellow(f"PARTIAL {pct:.0f}%")
    return red(f"PARTIAL {pct:.0f}%")


# ── main table renderer ───────────────────────────────────────────────────────
def _render_table(runs: list[dict], show_missing: bool) -> None:
    if not runs:
        print(yellow("No simulation.db files found."))
        return

    # column widths
    col_id    = max(len("FOLDER ID"),   max(len(r.get("folder_id", "?"))  for r in runs))
    _model_lens = [len(_short_model(r.get("model_id","?"))) for r in runs if "_error" not in r]
    col_model = max(len("MODEL"), max(_model_lens) if _model_lens else 5)
    col_temp  = 5   # "TEMP "
    col_trials= 12  # "TRIALS(done/exp)"
    col_cells = 11  # "CELLS(ok/tot)"
    col_status= 12  # "STATUS"

    sep = "─"

    def hr():
        print(
            "┼".join([
                sep * (col_id     + 2),
                sep * (col_model  + 2),
                sep * (col_temp   + 2),
                sep * (col_trials + 2),
                sep * (col_cells  + 2),
                sep * (col_status + 2),
            ])
        )

    def hdr():
        print(
            "│".join([
                bold(f" {'FOLDER ID':<{col_id}} "),
                bold(f" {'MODEL':<{col_model}} "),
                bold(f" {'TEMP':>{col_temp}} "),
                bold(f" {'TRIALS(done/exp)':>{col_trials}} "),
                bold(f" {'CELLS(ok/tot)':>{col_cells}} "),
                bold(f" {'STATUS':<{col_status}} "),
            ])
        )

    print()
    print(bold(cyan("  RUN INVENTORY")))
    print(dim(f"  {len(runs)} run(s) found"))
    print()
    hr(); hdr(); hr()

    for r in runs:
        fid = r.get("folder_id", "?")
        if "_error" in r:
            print(
                "│".join([
                    f" {fid:<{col_id}} ",
                    f" {'<read error>':<{col_model}} ",
                    f" {'?':>{col_temp}} ",
                    f" {'?':>{col_trials}} ",
                    f" {'?':>{col_cells}} ",
                    f" {red('ERROR'):<{col_status}} ",
                ])
            )
            print(dim(f"   └─ {r['_error']}"))
            continue

        model  = _short_model(r["model_id"])
        temp   = f"{r['temperature']:.2f}"
        tot_a  = r["total_actual"]
        tot_e  = r["total_expected"]
        n_cond = r["n_conditions"]
        n_ds   = r["n_datasets"]
        n_ok   = sum(1 for (c,d), n in r["cell_counts"].items()
                     if c in r["expected_conditions"] and d in r["expected_datasets"]
                     and n >= r["max_items"])
        n_cells= n_cond * n_ds
        status = _status_tag(r)

        trials_str = f"{tot_a:>6}/{tot_e:<5}"
        cells_str  = f"{n_ok:>4}/{n_cells:<5}"

        # colour the model field by status
        miss = r["missing_cells"]
        short= r["short_cells"]
        if not miss and not short:
            model_str = green(f"{model:<{col_model}}")
        elif miss:
            model_str = red(f"{model:<{col_model}}")
        else:
            model_str = yellow(f"{model:<{col_model}}")

        print(
            "│".join([
                f" {fid:<{col_id}} ",
                f" {model_str} ",
                f" {temp:>{col_temp}} ",
                f" {trials_str:>{col_trials}} ",
                f" {cells_str:>{col_cells}} ",
                f" {status:<{col_status}} ",
            ])
        )

        if show_missing and (miss or short):
            if miss:
                label = red("missing")
                items = miss
                for chunk in _chunks(items, 3):
                    print(dim(f"   └─ {label}: ") + ", ".join(chunk))
            if short:
                label = yellow("short  ")
                items = short
                for chunk in _chunks(items, 3):
                    print(dim(f"   └─ {label}: ") + ", ".join(chunk))

    hr()
    print()


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ── summary section ──────────────────────────────────────────────────────────
def _render_summary(runs: list[dict]) -> None:
    ok    = [r for r in runs if "_error" not in r and not r["missing_cells"] and not r["short_cells"]]
    part  = [r for r in runs if "_error" not in r and (r["missing_cells"] or r["short_cells"])]
    err   = [r for r in runs if "_error" in r]

    print(bold("  SUMMARY"))
    print(f"  {green('●')} Complete  : {len(ok)}")
    print(f"  {yellow('●')} Partial   : {len(part)}")
    print(f"  {red('●')} Error     : {len(err)}")

    # model × temp coverage
    seen: dict[tuple[str, float], list[str]] = {}
    for r in runs:
        if "_error" in r:
            continue
        key = (r["model_id"], r["temperature"])
        seen.setdefault(key, []).append(r["folder_id"])

    if seen:
        print()
        print(bold("  MODEL × TEMPERATURE MATRIX"))
        col_m = max(len(m) for m,_ in seen) + 2
        col_t = 6
        print(f"  {'MODEL':<{col_m}} {'TEMP':>{col_t}}  RUNS")
        print("  " + "─" * (col_m + col_t + 14))
        for (model, temp), folders in sorted(seen.items()):
            status_chars = []
            for fid in folders:
                rr = next(x for x in runs if x.get("folder_id") == fid)
                if rr["missing_cells"] or rr["short_cells"]:
                    status_chars.append(yellow("◑"))
                else:
                    status_chars.append(green("●"))
            icons = " ".join(status_chars)
            n = len(folders)
            print(f"  {model:<{col_m}} {temp:>{col_t}.2f}  {icons}  ({n}×)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Inventory simulation run databases.")
    ap.add_argument("--runs-dir",  default="runs",
                    help="Directory to scan (default: runs)")
    ap.add_argument("--expected-per-cell", type=int, default=None,
                    help="Override max_items_per_dataset from config")
    ap.add_argument("--sort",      choices=["id","model","temp","trials"], default="id",
                    help="Sort order (default: id)")
    ap.add_argument("--filter-model", default=None,
                    help="Regex filter on model_id")
    ap.add_argument("--show-missing", action="store_true",
                    help="List missing/short condition×dataset cells per run")
    ap.add_argument("--recursive", action="store_true", default=True,
                    help="Recurse into sub-folders (default: true)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        sys.exit(f"Error: directory '{runs_dir}' does not exist.")

    db_paths = sorted(runs_dir.rglob("simulation.db")) if args.recursive \
               else sorted(runs_dir.glob("*/simulation.db"))

    if not db_paths:
        sys.exit(f"No simulation.db files found under '{runs_dir}'.")

    print(dim(f"\nScanning {len(db_paths)} database(s) in '{runs_dir}' …"), flush=True)

    infos: list[dict] = []
    for p in db_paths:
        info = _extract_run_info(p)
        if info is None:
            continue
        if args.expected_per_cell and "_error" not in info:
            # recompute totals with override
            info["max_items"] = args.expected_per_cell
            info["total_expected"] = (args.expected_per_cell
                                      * info["n_conditions"]
                                      * info["n_datasets"])
        if args.filter_model and "_error" not in info:
            if not re.search(args.filter_model, info["model_id"], re.I):
                continue
        infos.append(info)

    # sort
    def sort_key(r):
        if "_error" in r:
            return ("", "", 0.0, 0)
        s = args.sort
        if s == "model":  return (r["model_id"], r["temperature"], r["folder_id"], 0)
        if s == "temp":   return (r["temperature"], r["model_id"], r["folder_id"], 0)
        if s == "trials": return (-r["total_actual"], r["model_id"], r["folder_id"], 0)
        return (r["folder_id"], r["model_id"], r["temperature"], 0)

    infos.sort(key=sort_key)

    _render_table(infos, show_missing=args.show_missing)
    _render_summary(infos)


if __name__ == "__main__":
    main()
