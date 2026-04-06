#!/usr/bin/env python3
"""
Comprehensive investigation of all experiment runs.
Queries every simulation.db across runs/, runs/think/, and runs_latest/runs/
to produce a full inventory with judge label analysis.
"""

import sqlite3
import json
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
RUN_DIRS = [
    ("runs", ROOT / "runs"),
    ("runs/think", ROOT / "runs" / "think"),
    ("runs_latest/runs", ROOT / "runs_latest" / "runs"),
]

def find_all_dbs():
    """Find all simulation.db files across run directories."""
    dbs = []
    for label, base in RUN_DIRS:
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('.'):
                continue
            db_path = entry / "simulation.db"
            if db_path.exists():
                run_id = entry.name  # full directory name
                uuid_part = run_id.split('_')[-1] if '_' in run_id else run_id
                dbs.append({
                    "label": label,
                    "dir_name": run_id,
                    "uuid": uuid_part,
                    "db_path": str(db_path),
                })
    return dbs


def query_run_info(db_path):
    """Extract model, variant, temperature, and trial counts from a simulation.db."""
    info = {
        "model_id": None,
        "variant": None,
        "temperature": None,
        "total_trials": 0,
        "has_manual_label": 0,
        "has_judge_label": 0,
        "judge_match": 0,
        "judge_mismatch": 0,
        "judge_models_used": Counter(),
        "judge_label_details": [],
        "conditions": set(),
        "datasets": set(),
        "per_condition_stats": {},
        "parse_errors": 0,
        "null_judge": 0,
        "refusal_count": 0,
    }

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        info["error"] = str(e)
        return info

    try:
        # Get run config
        row = conn.execute("SELECT config_json FROM runs LIMIT 1").fetchone()
        if row and row["config_json"]:
            try:
                config = json.loads(row["config_json"])
                # Try different config structures
                if "model_id" in config:
                    info["model_id"] = config["model_id"]
                elif "models" in config and len(config["models"]) > 0:
                    info["model_id"] = config["models"][0].get("model_id", "unknown")
                if "variant" in config:
                    info["variant"] = config["variant"]
                elif "models" in config and len(config["models"]) > 0:
                    info["variant"] = config["models"][0].get("variant", "unknown")
                if "temperature" in config:
                    info["temperature"] = config["temperature"]
                elif "run" in config:
                    info["temperature"] = config["run"].get("temperature")
            except json.JSONDecodeError:
                pass

        # Check if conformity tables exist
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if "conformity_trials" not in tables or "conformity_outputs" not in tables:
            info["error"] = "No conformity tables"
            conn.close()
            return info

        # Get model/variant/temperature from trials if not in config
        trial_row = conn.execute(
            "SELECT model_id, variant, temperature FROM conformity_trials LIMIT 1"
        ).fetchone()
        if trial_row:
            if not info["model_id"]:
                info["model_id"] = trial_row["model_id"]
            if not info["variant"]:
                info["variant"] = trial_row["variant"]
            if info["temperature"] is None:
                info["temperature"] = trial_row["temperature"]

        # Total trials
        info["total_trials"] = conn.execute(
            "SELECT COUNT(*) FROM conformity_trials"
        ).fetchone()[0]

        # Get conditions
        try:
            for r in conn.execute(
                "SELECT DISTINCT c.name FROM conformity_conditions c "
                "JOIN conformity_trials t ON t.condition_id = c.condition_id"
            ).fetchall():
                info["conditions"].add(r[0])
        except:
            pass

        # Get datasets
        try:
            for r in conn.execute(
                "SELECT DISTINCT d.name FROM conformity_datasets d "
                "JOIN conformity_items i ON i.dataset_id = d.dataset_id "
                "JOIN conformity_trials t ON t.item_id = i.item_id"
            ).fetchall():
                info["datasets"].add(r[0])
        except:
            pass

        # Analyze outputs: manual label (is_correct) vs judge label (in parsed_answer_json)
        rows = conn.execute("""
            SELECT
                o.trial_id,
                o.is_correct,
                o.refusal_flag,
                o.parsed_answer_json,
                o.raw_text
            FROM conformity_outputs o
        """).fetchall()

        for r in rows:
            trial_id = r["trial_id"]
            manual_correct = r["is_correct"]
            manual_refusal = r["refusal_flag"]
            paj_raw = r["parsed_answer_json"]

            has_manual = manual_correct is not None
            if has_manual:
                info["has_manual_label"] += 1

            if manual_refusal == 1:
                info["refusal_count"] += 1

            # Parse the judge JSON
            if paj_raw:
                try:
                    paj = json.loads(paj_raw) if isinstance(paj_raw, str) else paj_raw
                except (json.JSONDecodeError, TypeError):
                    paj = None

                if paj and isinstance(paj, dict):
                    # Check for _llm_judge key (indicates LLM judge was used)
                    judge_data = paj.get("_llm_judge", None)

                    if judge_data and isinstance(judge_data, dict):
                        info["has_judge_label"] += 1

                        # Track which judge model was used
                        judge_model = judge_data.get("judge_model", "unknown")
                        info["judge_models_used"][judge_model] += 1

                        # Get judge's is_correct
                        judge_correct = judge_data.get("is_correct")

                        # Compare with manual
                        if has_manual and judge_correct is not None:
                            # Normalize to int for comparison
                            try:
                                m = int(manual_correct)
                                j = int(judge_correct)
                                if m == j:
                                    info["judge_match"] += 1
                                else:
                                    info["judge_mismatch"] += 1
                            except (ValueError, TypeError):
                                pass

                        # Check for parse errors in judge response
                        judge_notes = judge_data.get("notes", "")
                        if "[parse_error]" in str(judge_notes) or judge_data.get("parse_error"):
                            info["parse_errors"] += 1
                    else:
                        # Has parsed_answer_json but no _llm_judge
                        # Check if the top-level has is_correct (heuristic label)
                        if "is_correct" in paj:
                            # This is a heuristic label, not judge
                            pass
                        info["null_judge"] += 1
                else:
                    info["null_judge"] += 1
            else:
                info["null_judge"] += 1

        # Per-condition breakdown
        try:
            cond_rows = conn.execute("""
                SELECT
                    c.name as condition_name,
                    COUNT(DISTINCT t.trial_id) as n_trials,
                    SUM(CASE WHEN o.is_correct IS NOT NULL THEN 1 ELSE 0 END) as n_manual,
                    o.parsed_answer_json
                FROM conformity_trials t
                JOIN conformity_conditions c ON t.condition_id = c.condition_id
                LEFT JOIN conformity_outputs o ON t.trial_id = o.trial_id
                GROUP BY c.name
            """).fetchall()
            for cr in cond_rows:
                info["per_condition_stats"][cr["condition_name"]] = {
                    "n_trials": cr["n_trials"],
                    "n_manual": cr["n_manual"],
                }
        except:
            pass

    except Exception as e:
        info["error"] = str(e)
    finally:
        conn.close()

    return info


def analyze_judge_details(db_path):
    """Deep-dive into judge labels for a specific run, returning per-trial details."""
    details = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                t.trial_id,
                t.model_id,
                t.variant,
                t.temperature,
                c.name as condition_name,
                d.name as dataset_name,
                o.is_correct as manual_correct,
                o.refusal_flag as manual_refusal,
                o.parsed_answer_json,
                o.raw_text
            FROM conformity_trials t
            JOIN conformity_conditions c ON t.condition_id = c.condition_id
            JOIN conformity_items i ON t.item_id = i.item_id
            JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
            LEFT JOIN conformity_outputs o ON t.trial_id = o.trial_id
        """).fetchall()

        for r in rows:
            entry = {
                "trial_id": r["trial_id"],
                "model_id": r["model_id"],
                "variant": r["variant"],
                "temperature": r["temperature"],
                "condition": r["condition_name"],
                "dataset": r["dataset_name"],
                "manual_correct": r["manual_correct"],
                "manual_refusal": r["manual_refusal"],
                "judge_model": None,
                "judge_correct": None,
                "judge_refusal": None,
                "judge_wrong_endorsed": None,
                "judge_notes": None,
                "match": None,
                "has_raw_text": bool(r["raw_text"]),
            }

            paj_raw = r["parsed_answer_json"]
            if paj_raw:
                try:
                    paj = json.loads(paj_raw) if isinstance(paj_raw, str) else paj_raw
                    if isinstance(paj, dict):
                        jd = paj.get("_llm_judge", {})
                        if jd:
                            entry["judge_model"] = jd.get("judge_model")
                            entry["judge_correct"] = jd.get("is_correct")
                            entry["judge_refusal"] = jd.get("refusal_flag")
                            entry["judge_wrong_endorsed"] = jd.get("wrong_answer_endorsed")
                            entry["judge_notes"] = str(jd.get("notes", ""))[:200]

                            # Compare
                            if r["manual_correct"] is not None and entry["judge_correct"] is not None:
                                try:
                                    entry["match"] = int(r["manual_correct"]) == int(entry["judge_correct"])
                                except:
                                    pass
                except:
                    pass

            details.append(entry)

        conn.close()
    except Exception as e:
        pass

    return details


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 80)
    print("COMPREHENSIVE RUN INVESTIGATION")
    print("=" * 80)

    # Find all databases
    all_dbs = find_all_dbs()
    print(f"\nFound {len(all_dbs)} simulation databases total")

    # Query each
    all_results = []
    all_details = []

    for db_info in all_dbs:
        print(f"\n  Querying {db_info['label']}/{db_info['dir_name']}...")
        run_info = query_run_info(db_info["db_path"])
        run_info.update(db_info)
        all_results.append(run_info)

        # Get detailed trial-level data
        details = analyze_judge_details(db_info["db_path"])
        all_details.extend(details)

    # ==========================================
    # SUMMARY TABLE
    # ==========================================
    print("\n\n" + "=" * 80)
    print("RUN INVENTORY SUMMARY")
    print("=" * 80)

    summary_rows = []
    for r in all_results:
        comparable = r["judge_match"] + r["judge_mismatch"]
        match_pct = (r["judge_match"] / comparable * 100) if comparable > 0 else 0
        judge_cov = (r["has_judge_label"] / r["total_trials"] * 100) if r["total_trials"] > 0 else 0
        manual_cov = (r["has_manual_label"] / r["total_trials"] * 100) if r["total_trials"] > 0 else 0

        row = {
            "location": r["label"],
            "uuid": r["uuid"],
            "model_id": r.get("model_id", "?"),
            "variant": r.get("variant", "?"),
            "temperature": r.get("temperature", "?"),
            "total_trials": r["total_trials"],
            "has_manual": r["has_manual_label"],
            "manual_cov_%": f"{manual_cov:.1f}",
            "has_judge": r["has_judge_label"],
            "judge_cov_%": f"{judge_cov:.1f}",
            "judge_match": r["judge_match"],
            "judge_mismatch": r["judge_mismatch"],
            "match_%": f"{match_pct:.1f}",
            "parse_errors": r["parse_errors"],
            "refusals": r["refusal_count"],
            "n_conditions": len(r.get("conditions", set())),
            "n_datasets": len(r.get("datasets", set())),
            "error": r.get("error", ""),
        }
        summary_rows.append(row)

        # Print
        print(f"\n  [{r['label']}] {r['uuid'][:12]}...")
        print(f"    Model: {r.get('model_id', '?')} | Variant: {r.get('variant', '?')} | Temp: {r.get('temperature', '?')}")
        print(f"    Trials: {r['total_trials']} | Manual: {r['has_manual_label']} ({manual_cov:.0f}%) | Judge: {r['has_judge_label']} ({judge_cov:.0f}%)")
        print(f"    Judge Match: {r['judge_match']}/{comparable} ({match_pct:.1f}%) | Parse Errors: {r['parse_errors']}")
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        if r["judge_models_used"]:
            print(f"    Judge models: {dict(r['judge_models_used'])}")

    # Save summary CSV
    csv_path = out_dir / "run_inventory_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n  Saved: {csv_path}")

    # ==========================================
    # JUDGE MODEL DISTRIBUTION
    # ==========================================
    print("\n\n" + "=" * 80)
    print("JUDGE MODEL DISTRIBUTION")
    print("=" * 80)

    global_judge_counter = Counter()
    judge_by_run = {}
    for r in all_results:
        for model, count in r["judge_models_used"].items():
            global_judge_counter[model] += count
        if r["judge_models_used"]:
            judge_by_run[r["uuid"]] = dict(r["judge_models_used"])

    total_judged = sum(global_judge_counter.values())
    print(f"\nTotal judged trials: {total_judged}")

    judge_dist_rows = []
    for model, count in global_judge_counter.most_common():
        pct = count / total_judged * 100 if total_judged > 0 else 0
        print(f"  {model}: {count} ({pct:.1f}%)")
        judge_dist_rows.append({
            "judge_model": model,
            "count": count,
            "percentage": f"{pct:.1f}",
        })

    csv_path = out_dir / "judge_model_distribution.csv"
    if judge_dist_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=judge_dist_rows[0].keys())
            w.writeheader()
            w.writerows(judge_dist_rows)
        print(f"\n  Saved: {csv_path}")

    # ==========================================
    # JUDGE AGREEMENT BY JUDGE MODEL
    # ==========================================
    print("\n\n" + "=" * 80)
    print("JUDGE AGREEMENT BY JUDGE MODEL")
    print("=" * 80)

    judge_agreement = defaultdict(lambda: {"match": 0, "mismatch": 0, "total": 0})
    qwen_flash_issues = []

    for d in all_details:
        jm = d.get("judge_model")
        if jm and d["match"] is not None:
            judge_agreement[jm]["total"] += 1
            if d["match"]:
                judge_agreement[jm]["match"] += 1
            else:
                judge_agreement[jm]["mismatch"] += 1
                # Track Qwen/Flash mismatches for investigation
                if "qwen" in str(jm).lower() or "flash" in str(jm).lower():
                    qwen_flash_issues.append(d)

    agreement_rows = []
    for model, stats in sorted(judge_agreement.items(), key=lambda x: x[1]["total"], reverse=True):
        match_pct = stats["match"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {model}: {stats['match']}/{stats['total']} match ({match_pct:.1f}%)")
        agreement_rows.append({
            "judge_model": model,
            "matches": stats["match"],
            "mismatches": stats["mismatch"],
            "total_comparable": stats["total"],
            "agreement_%": f"{match_pct:.1f}",
        })

    csv_path = out_dir / "judge_agreement_by_model.csv"
    if agreement_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=agreement_rows[0].keys())
            w.writeheader()
            w.writerows(agreement_rows)
        print(f"\n  Saved: {csv_path}")

    # ==========================================
    # QWEN/FLASH DEEP DIVE
    # ==========================================
    print("\n\n" + "=" * 80)
    print("QWEN/FLASH JUDGE ISSUES DEEP DIVE")
    print("=" * 80)

    # Also look for ALL qwen/flash judged trials (not just mismatches)
    qwen_flash_all = []
    for d in all_details:
        jm = str(d.get("judge_model", "")).lower()
        if "qwen" in jm or "flash" in jm or "gemini" in jm:
            qwen_flash_all.append(d)

    print(f"\nTotal Qwen/Flash/Gemini judged trials: {len(qwen_flash_all)}")
    print(f"Qwen/Flash/Gemini mismatches: {len(qwen_flash_issues)}")

    # Break down by judge model
    qf_by_model = defaultdict(lambda: {"match": 0, "mismatch": 0, "total": 0, "null_match": 0})
    for d in qwen_flash_all:
        jm = d.get("judge_model", "unknown")
        qf_by_model[jm]["total"] += 1
        if d["match"] is True:
            qf_by_model[jm]["match"] += 1
        elif d["match"] is False:
            qf_by_model[jm]["mismatch"] += 1
        else:
            qf_by_model[jm]["null_match"] += 1

    for model, stats in sorted(qf_by_model.items()):
        match_pct = stats["match"] / (stats["match"] + stats["mismatch"]) * 100 if (stats["match"] + stats["mismatch"]) > 0 else 0
        print(f"\n  {model}:")
        print(f"    Total: {stats['total']} | Match: {stats['match']} | Mismatch: {stats['mismatch']} | No comparison: {stats['null_match']}")
        print(f"    Agreement: {match_pct:.1f}%")

    # Save Qwen/Flash mismatch details
    if qwen_flash_issues:
        csv_path = out_dir / "qwen_flash_mismatches.csv"
        keys = ["trial_id", "model_id", "variant", "condition", "dataset",
                "manual_correct", "judge_model", "judge_correct", "judge_refusal",
                "judge_wrong_endorsed", "judge_notes"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(qwen_flash_issues)
        print(f"\n  Saved: {csv_path}")

    # ==========================================
    # TRIALS WITHOUT JUDGE LABELS
    # ==========================================
    print("\n\n" + "=" * 80)
    print("TRIALS MISSING JUDGE LABELS")
    print("=" * 80)

    missing_judge_by_run = []
    for r in all_results:
        missing = r["total_trials"] - r["has_judge_label"]
        if missing > 0:
            missing_judge_by_run.append({
                "location": r["label"],
                "uuid": r["uuid"],
                "model_id": r.get("model_id", "?"),
                "variant": r.get("variant", "?"),
                "temperature": r.get("temperature", "?"),
                "total_trials": r["total_trials"],
                "has_judge": r["has_judge_label"],
                "missing_judge": missing,
                "missing_%": f"{missing / r['total_trials'] * 100:.1f}" if r['total_trials'] > 0 else "0",
            })

    missing_judge_by_run.sort(key=lambda x: int(x["missing_judge"]), reverse=True)
    for m in missing_judge_by_run:
        print(f"  {m['uuid'][:12]}... ({m['model_id']}/{m['variant']}): {m['missing_judge']}/{m['total_trials']} missing ({m['missing_%']}%)")

    csv_path = out_dir / "missing_judge_labels.csv"
    if missing_judge_by_run:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=missing_judge_by_run[0].keys())
            w.writeheader()
            w.writerows(missing_judge_by_run)
        print(f"\n  Saved: {csv_path}")

    # ==========================================
    # DETAILED TRIAL-LEVEL DUMP (for spot checks)
    # ==========================================
    csv_path = out_dir / "all_trial_details.csv"
    if all_details:
        keys = ["trial_id", "model_id", "variant", "temperature", "condition", "dataset",
                "manual_correct", "manual_refusal", "judge_model", "judge_correct",
                "judge_refusal", "judge_wrong_endorsed", "match", "has_raw_text", "judge_notes"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_details)
        print(f"\n  Saved full trial details: {csv_path}")

    # ==========================================
    # AGGREGATE STATS
    # ==========================================
    print("\n\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    total_trials_all = sum(r["total_trials"] for r in all_results)
    total_manual = sum(r["has_manual_label"] for r in all_results)
    total_judge = sum(r["has_judge_label"] for r in all_results)
    total_match = sum(r["judge_match"] for r in all_results)
    total_mismatch = sum(r["judge_mismatch"] for r in all_results)
    total_parse_err = sum(r["parse_errors"] for r in all_results)
    total_comparable = total_match + total_mismatch

    print(f"\n  Total runs: {len(all_results)}")
    print(f"  Total trials: {total_trials_all}")
    print(f"  With manual label: {total_manual} ({total_manual/total_trials_all*100:.1f}%)" if total_trials_all > 0 else "")
    print(f"  With judge label: {total_judge} ({total_judge/total_trials_all*100:.1f}%)" if total_trials_all > 0 else "")
    print(f"  Comparable (both): {total_comparable}")
    print(f"  Match: {total_match} ({total_match/total_comparable*100:.1f}%)" if total_comparable > 0 else "")
    print(f"  Mismatch: {total_mismatch} ({total_mismatch/total_comparable*100:.1f}%)" if total_comparable > 0 else "")
    print(f"  Parse errors: {total_parse_err}")

    # Breakdown by location
    for loc in ["runs", "runs/think", "runs_latest/runs"]:
        loc_results = [r for r in all_results if r["label"] == loc]
        if not loc_results:
            continue
        loc_trials = sum(r["total_trials"] for r in loc_results)
        loc_judge = sum(r["has_judge_label"] for r in loc_results)
        loc_match = sum(r["judge_match"] for r in loc_results)
        loc_mismatch = sum(r["judge_mismatch"] for r in loc_results)
        loc_comp = loc_match + loc_mismatch
        print(f"\n  [{loc}] Runs: {len(loc_results)} | Trials: {loc_trials} | Judged: {loc_judge} | " +
              (f"Agreement: {loc_match}/{loc_comp} ({loc_match/loc_comp*100:.1f}%)" if loc_comp > 0 else "No comparable"))

    # Per model summary
    print("\n\n" + "=" * 80)
    print("PER MODEL SUMMARY")
    print("=" * 80)

    model_summary = defaultdict(lambda: {"trials": 0, "manual": 0, "judge": 0, "match": 0, "mismatch": 0, "runs": 0, "temps": set()})
    for r in all_results:
        key = r.get("model_id", "unknown")
        model_summary[key]["trials"] += r["total_trials"]
        model_summary[key]["manual"] += r["has_manual_label"]
        model_summary[key]["judge"] += r["has_judge_label"]
        model_summary[key]["match"] += r["judge_match"]
        model_summary[key]["mismatch"] += r["judge_mismatch"]
        model_summary[key]["runs"] += 1
        if r.get("temperature") is not None:
            model_summary[key]["temps"].add(str(r.get("temperature")))

    model_rows = []
    for model, stats in sorted(model_summary.items(), key=lambda x: x[1]["trials"], reverse=True):
        comp = stats["match"] + stats["mismatch"]
        match_pct = stats["match"] / comp * 100 if comp > 0 else 0
        judge_cov = stats["judge"] / stats["trials"] * 100 if stats["trials"] > 0 else 0
        print(f"\n  {model}:")
        print(f"    Runs: {stats['runs']} | Temps: {', '.join(sorted(stats['temps']))} | Trials: {stats['trials']}")
        print(f"    Judge coverage: {stats['judge']}/{stats['trials']} ({judge_cov:.1f}%)")
        print(f"    Agreement: {stats['match']}/{comp} ({match_pct:.1f}%)" if comp > 0 else "    No comparable trials")

        model_rows.append({
            "model_id": model,
            "n_runs": stats["runs"],
            "temperatures": ", ".join(sorted(stats["temps"])),
            "total_trials": stats["trials"],
            "has_manual": stats["manual"],
            "has_judge": stats["judge"],
            "judge_cov_%": f"{judge_cov:.1f}",
            "match": stats["match"],
            "mismatch": stats["mismatch"],
            "agreement_%": f"{match_pct:.1f}" if comp > 0 else "N/A",
        })

    csv_path = out_dir / "per_model_summary.csv"
    if model_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=model_rows[0].keys())
            w.writeheader()
            w.writerows(model_rows)
        print(f"\n  Saved: {csv_path}")

    print("\n\nDone! All CSV files saved to investigation/")


if __name__ == "__main__":
    main()
