#!/usr/bin/env python3
"""
Investigation v2: Proper comparison of heuristic vs judge labels.

Key insight:
- SQL column `conformity_outputs.is_correct` = HEURISTIC/MANUAL label (text parser)
- `parsed_answer_json.is_correct` = JUDGE label (LLM judge verdict)
- `parsed_answer_json._llm_judge` = metadata (judge model, prompt version, etc.)
"""

import sqlite3
import json
import os
import csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

RUN_DIRS = [
    ("runs", ROOT / "runs"),
    ("runs/think", ROOT / "runs" / "think"),
    ("runs_latest/runs", ROOT / "runs_latest" / "runs"),
]


def find_all_dbs():
    dbs = []
    for label, base in RUN_DIRS:
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            db_path = entry / "simulation.db"
            if db_path.exists():
                uuid_part = entry.name.split('_')[-1] if '_' in entry.name else entry.name
                dbs.append({
                    "label": label,
                    "dir_name": entry.name,
                    "uuid": uuid_part,
                    "db_path": str(db_path),
                })
    return dbs


def analyze_run(db_path):
    """Full analysis of a single run's labels."""
    result = {
        "model_id": None, "variant": None, "temperature": None,
        "total_trials": 0, "total_outputs": 0,
        "heuristic_not_null": 0, "judge_not_null": 0,
        "both_not_null": 0,
        "agree_correct": 0, "disagree": 0,
        "heur_0_judge_1": 0, "heur_1_judge_0": 0,
        "heur_refusal_judge_no_refusal": 0, "judge_refusal_heur_no_refusal": 0,
        "judge_models": Counter(),
        "per_judge_model": defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0,
                                                  "h0j1": 0, "h1j0": 0}),
        "per_condition": defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0}),
        "per_dataset": defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0}),
        "conditions": set(), "datasets": set(),
        "parse_errors": 0, "null_endorsed": 0,
        "wrong_answer_endorsed_count": 0,
    }

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        result["error"] = str(e)
        return result, []

    details = []

    try:
        # Get model info from trials
        trial_row = conn.execute(
            "SELECT model_id, variant, temperature FROM conformity_trials LIMIT 1"
        ).fetchone()
        if trial_row:
            result["model_id"] = trial_row["model_id"]
            result["variant"] = trial_row["variant"]
            result["temperature"] = trial_row["temperature"]

        result["total_trials"] = conn.execute("SELECT COUNT(*) FROM conformity_trials").fetchone()[0]

        # Join trials + outputs + conditions + datasets
        rows = conn.execute("""
            SELECT
                t.trial_id,
                c.name as condition_name,
                d.name as dataset_name,
                o.is_correct as heuristic_correct,
                o.refusal_flag as heuristic_refusal,
                o.parsed_answer_json,
                o.raw_text
            FROM conformity_trials t
            LEFT JOIN conformity_outputs o ON t.trial_id = o.trial_id
            LEFT JOIN conformity_conditions c ON t.condition_id = c.condition_id
            LEFT JOIN conformity_items i ON t.item_id = i.item_id
            LEFT JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
        """).fetchall()

        for r in rows:
            result["total_outputs"] += 1
            cond = r["condition_name"] or "unknown"
            ds = r["dataset_name"] or "unknown"
            result["conditions"].add(cond)
            result["datasets"].add(ds)

            heur_correct = r["heuristic_correct"]
            heur_refusal = r["heuristic_refusal"]

            # Parse judge label from parsed_answer_json
            judge_correct = None
            judge_refusal = None
            judge_endorsed = None
            judge_model = None
            judge_notes = None

            paj_raw = r["parsed_answer_json"]
            if paj_raw:
                try:
                    paj = json.loads(paj_raw) if isinstance(paj_raw, str) else paj_raw
                    if isinstance(paj, dict):
                        judge_correct = paj.get("is_correct")
                        judge_refusal = paj.get("refusal_flag")
                        judge_endorsed = paj.get("wrong_answer_endorsed")
                        judge_notes = paj.get("notes", "")

                        jd = paj.get("_llm_judge", {})
                        if jd:
                            judge_model = jd.get("judge_model", "unknown")

                        if "[parse_error]" in str(judge_notes):
                            result["parse_errors"] += 1
                except:
                    pass

            if heur_correct is not None:
                result["heuristic_not_null"] += 1
            if judge_correct is not None:
                result["judge_not_null"] += 1

            if judge_model:
                result["judge_models"][judge_model] += 1
                result["per_judge_model"][judge_model]["total"] += 1
            if judge_endorsed == 1:
                result["wrong_answer_endorsed_count"] += 1
            if judge_endorsed is None and judge_model:
                result["null_endorsed"] += 1

            # Count per condition/dataset
            result["per_condition"][cond]["total"] += 1
            result["per_dataset"][ds]["total"] += 1

            # Compare heuristic vs judge
            comparable = (heur_correct is not None and judge_correct is not None)
            if comparable:
                result["both_not_null"] += 1
                result["per_condition"][cond]["comparable"] += 1
                result["per_dataset"][ds]["comparable"] += 1
                if judge_model:
                    result["per_judge_model"][judge_model]["comparable"] += 1

                try:
                    h = int(heur_correct)
                    j = int(judge_correct)
                    if h == j:
                        result["agree_correct"] += 1
                        result["per_condition"][cond]["agree"] += 1
                        result["per_dataset"][ds]["agree"] += 1
                        if judge_model:
                            result["per_judge_model"][judge_model]["agree"] += 1
                    else:
                        result["disagree"] += 1
                        result["per_condition"][cond]["disagree"] += 1
                        result["per_dataset"][ds]["disagree"] += 1
                        if judge_model:
                            result["per_judge_model"][judge_model]["disagree"] += 1
                        if h == 0 and j == 1:
                            result["heur_0_judge_1"] += 1
                            if judge_model:
                                result["per_judge_model"][judge_model]["h0j1"] += 1
                        elif h == 1 and j == 0:
                            result["heur_1_judge_0"] += 1
                            if judge_model:
                                result["per_judge_model"][judge_model]["h1j0"] += 1
                except:
                    pass

            # Refusal comparison
            if heur_refusal is not None and judge_refusal is not None:
                if int(heur_refusal) == 1 and int(judge_refusal) == 0:
                    result["heur_refusal_judge_no_refusal"] += 1
                elif int(heur_refusal) == 0 and int(judge_refusal) == 1:
                    result["judge_refusal_heur_no_refusal"] += 1

            # Detail row for mismatches
            if comparable and int(heur_correct) != int(judge_correct):
                details.append({
                    "trial_id": r["trial_id"],
                    "condition": cond,
                    "dataset": ds,
                    "heuristic_correct": heur_correct,
                    "judge_correct": judge_correct,
                    "heuristic_refusal": heur_refusal,
                    "judge_refusal": judge_refusal,
                    "judge_endorsed": judge_endorsed,
                    "judge_model": judge_model,
                    "judge_notes": str(judge_notes)[:200] if judge_notes else "",
                    "has_raw_text": bool(r["raw_text"]),
                })

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()

    return result, details


def main():
    print("=" * 80)
    print("INVESTIGATION v2: Heuristic vs Judge Label Comparison")
    print("=" * 80)

    all_dbs = find_all_dbs()
    print(f"\nFound {len(all_dbs)} simulation databases")

    all_results = []
    all_mismatches = []
    summary_rows = []

    for db_info in all_dbs:
        print(f"\n  Analyzing {db_info['label']}/{db_info['uuid'][:12]}...")
        result, mismatches = analyze_run(db_info["db_path"])
        result.update(db_info)
        all_results.append(result)

        for m in mismatches:
            m["run_uuid"] = db_info["uuid"]
            m["run_label"] = db_info["label"]
            m["model_id"] = result["model_id"]
            m["variant"] = result["variant"]
        all_mismatches.extend(mismatches)

        comp = result["both_not_null"]
        agree_pct = result["agree_correct"] / comp * 100 if comp > 0 else 0
        judge_cov = result["judge_not_null"] / result["total_trials"] * 100 if result["total_trials"] > 0 else 0

        row = {
            "location": db_info["label"],
            "uuid": db_info["uuid"],
            "model_id": result.get("model_id", "?"),
            "variant": result.get("variant", "?"),
            "temperature": result.get("temperature", "?"),
            "total_trials": result["total_trials"],
            "heuristic_labels": result["heuristic_not_null"],
            "judge_labels": result["judge_not_null"],
            "judge_cov_%": f"{judge_cov:.1f}",
            "comparable": comp,
            "agree": result["agree_correct"],
            "disagree": result["disagree"],
            "agreement_%": f"{agree_pct:.1f}" if comp > 0 else "N/A",
            "heur0_judge1": result["heur_0_judge_1"],
            "heur1_judge0": result["heur_1_judge_0"],
            "parse_errors": result["parse_errors"],
            "wrong_endorsed": result["wrong_answer_endorsed_count"],
            "null_endorsed": result["null_endorsed"],
            "n_conditions": len(result["conditions"]),
            "n_datasets": len(result["datasets"]),
            "judge_models": "; ".join(f"{m}:{c}" for m, c in result["judge_models"].most_common()),
        }
        summary_rows.append(row)

        print(f"    {result.get('model_id', '?')} / {result.get('variant', '?')} / temp={result.get('temperature', '?')}")
        print(f"    Trials: {result['total_trials']} | Heuristic: {result['heuristic_not_null']} | Judge: {result['judge_not_null']} ({judge_cov:.0f}%)")
        print(f"    Comparable: {comp} | Agree: {result['agree_correct']} ({agree_pct:.1f}%) | Disagree: {result['disagree']}")
        if result["disagree"] > 0:
            print(f"    Heur=0,Judge=1: {result['heur_0_judge_1']} | Heur=1,Judge=0: {result['heur_1_judge_0']}")
        if result["judge_models"]:
            print(f"    Judge models: {dict(result['judge_models'])}")

    # Save summary
    csv_path = OUT / "v2_run_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved: {csv_path}")

    # =============================================
    # GLOBAL JUDGE MODEL ANALYSIS
    # =============================================
    print("\n\n" + "=" * 80)
    print("JUDGE MODEL DISTRIBUTION & AGREEMENT")
    print("=" * 80)

    global_judge = defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0, "h0j1": 0, "h1j0": 0})
    for r in all_results:
        for model, stats in r["per_judge_model"].items():
            for k in ["total", "comparable", "agree", "disagree", "h0j1", "h1j0"]:
                global_judge[model][k] += stats[k]

    total_judged = sum(v["total"] for v in global_judge.values())
    judge_rows = []
    for model, stats in sorted(global_judge.items(), key=lambda x: x[1]["total"], reverse=True):
        pct_of_total = stats["total"] / total_judged * 100 if total_judged > 0 else 0
        agree_pct = stats["agree"] / stats["comparable"] * 100 if stats["comparable"] > 0 else 0
        print(f"\n  {model}:")
        print(f"    Judged: {stats['total']} ({pct_of_total:.1f}% of all)")
        print(f"    Comparable: {stats['comparable']} | Agree: {stats['agree']} ({agree_pct:.1f}%) | Disagree: {stats['disagree']}")
        if stats["disagree"] > 0:
            print(f"    H=0,J=1: {stats['h0j1']} | H=1,J=0: {stats['h1j0']}")

        judge_rows.append({
            "judge_model": model,
            "total_judged": stats["total"],
            "pct_of_all": f"{pct_of_total:.1f}",
            "comparable": stats["comparable"],
            "agree": stats["agree"],
            "disagree": stats["disagree"],
            "agreement_%": f"{agree_pct:.1f}" if stats["comparable"] > 0 else "N/A",
            "heur0_judge1": stats["h0j1"],
            "heur1_judge0": stats["h1j0"],
        })

    csv_path = OUT / "v2_judge_model_agreement.csv"
    if judge_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=judge_rows[0].keys())
            w.writeheader()
            w.writerows(judge_rows)
        print(f"\nSaved: {csv_path}")

    # =============================================
    # PER-CONDITION AGREEMENT (POOLED)
    # =============================================
    print("\n\n" + "=" * 80)
    print("PER-CONDITION AGREEMENT (POOLED ACROSS ALL RUNS)")
    print("=" * 80)

    global_cond = defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0})
    for r in all_results:
        for cond, stats in r["per_condition"].items():
            for k in ["total", "comparable", "agree", "disagree"]:
                global_cond[cond][k] += stats[k]

    cond_rows = []
    for cond, stats in sorted(global_cond.items(), key=lambda x: x[1]["comparable"], reverse=True):
        agree_pct = stats["agree"] / stats["comparable"] * 100 if stats["comparable"] > 0 else 0
        print(f"  {cond}: {stats['agree']}/{stats['comparable']} agree ({agree_pct:.1f}%)")
        cond_rows.append({
            "condition": cond,
            "total_trials": stats["total"],
            "comparable": stats["comparable"],
            "agree": stats["agree"],
            "disagree": stats["disagree"],
            "agreement_%": f"{agree_pct:.1f}" if stats["comparable"] > 0 else "N/A",
        })

    csv_path = OUT / "v2_per_condition_agreement.csv"
    if cond_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cond_rows[0].keys())
            w.writeheader()
            w.writerows(cond_rows)

    # =============================================
    # PER-DATASET AGREEMENT (POOLED)
    # =============================================
    print("\n\n" + "=" * 80)
    print("PER-DATASET AGREEMENT (POOLED ACROSS ALL RUNS)")
    print("=" * 80)

    global_ds = defaultdict(lambda: {"total": 0, "comparable": 0, "agree": 0, "disagree": 0})
    for r in all_results:
        for ds, stats in r["per_dataset"].items():
            for k in ["total", "comparable", "agree", "disagree"]:
                global_ds[ds][k] += stats[k]

    ds_rows = []
    for ds, stats in sorted(global_ds.items(), key=lambda x: x[1]["comparable"], reverse=True):
        agree_pct = stats["agree"] / stats["comparable"] * 100 if stats["comparable"] > 0 else 0
        print(f"  {ds}: {stats['agree']}/{stats['comparable']} agree ({agree_pct:.1f}%)")
        ds_rows.append({
            "dataset": ds,
            "total_trials": stats["total"],
            "comparable": stats["comparable"],
            "agree": stats["agree"],
            "disagree": stats["disagree"],
            "agreement_%": f"{agree_pct:.1f}" if stats["comparable"] > 0 else "N/A",
        })

    csv_path = OUT / "v2_per_dataset_agreement.csv"
    if ds_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ds_rows[0].keys())
            w.writeheader()
            w.writerows(ds_rows)

    # =============================================
    # QWEN/FLASH DEEP DIVE
    # =============================================
    print("\n\n" + "=" * 80)
    print("QWEN / FLASH JUDGE DEEP DIVE")
    print("=" * 80)

    qwen_flash_models = [m for m in global_judge.keys() if "qwen" in m.lower() or "flash" in m.lower()]
    for model in qwen_flash_models:
        stats = global_judge[model]
        agree_pct = stats["agree"] / stats["comparable"] * 100 if stats["comparable"] > 0 else 0
        print(f"\n  {model}:")
        print(f"    Total: {stats['total']} | Comparable: {stats['comparable']} | Agree: {stats['agree']} ({agree_pct:.1f}%)")
        print(f"    H=0,J=1: {stats['h0j1']} | H=1,J=0: {stats['h1j0']}")

    # Count Qwen/Flash mismatches by run
    qf_mismatches = [m for m in all_mismatches if m.get("judge_model") and
                     ("qwen" in m["judge_model"].lower() or "flash" in m["judge_model"].lower())]
    print(f"\n  Total Qwen/Flash mismatches: {len(qf_mismatches)}")

    # Save mismatches
    csv_path = OUT / "v2_qwen_flash_mismatches.csv"
    if qf_mismatches:
        keys = ["run_uuid", "model_id", "variant", "trial_id", "condition", "dataset",
                "heuristic_correct", "judge_correct", "judge_model", "judge_endorsed",
                "heuristic_refusal", "judge_refusal", "judge_notes"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(qf_mismatches)
        print(f"  Saved: {csv_path}")

    # =============================================
    # ALL MISMATCHES SUMMARY
    # =============================================
    print("\n\n" + "=" * 80)
    print(f"ALL MISMATCHES: {len(all_mismatches)} total")
    print("=" * 80)

    csv_path = OUT / "v2_all_mismatches.csv"
    if all_mismatches:
        keys = ["run_uuid", "run_label", "model_id", "variant", "trial_id", "condition", "dataset",
                "heuristic_correct", "judge_correct", "judge_model", "judge_endorsed",
                "heuristic_refusal", "judge_refusal", "judge_notes"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_mismatches)
        print(f"  Saved: {csv_path}")

    # Mismatches by judge model
    mm_by_judge = Counter(m.get("judge_model", "?") for m in all_mismatches)
    print("\n  Mismatches by judge model:")
    for model, count in mm_by_judge.most_common():
        total = global_judge[model]["comparable"]
        pct = count / total * 100 if total > 0 else 0
        print(f"    {model}: {count} mismatches out of {total} comparable ({pct:.1f}% disagree)")

    # Mismatches by condition
    mm_by_cond = Counter(m.get("condition", "?") for m in all_mismatches)
    print("\n  Mismatches by condition:")
    for cond, count in mm_by_cond.most_common(15):
        print(f"    {cond}: {count}")

    # Mismatches by dataset
    mm_by_ds = Counter(m.get("dataset", "?") for m in all_mismatches)
    print("\n  Mismatches by dataset:")
    for ds, count in mm_by_ds.most_common():
        print(f"    {ds}: {count}")

    # =============================================
    # AGGREGATE STATS
    # =============================================
    print("\n\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    total_trials = sum(r["total_trials"] for r in all_results)
    total_heur = sum(r["heuristic_not_null"] for r in all_results)
    total_judge = sum(r["judge_not_null"] for r in all_results)
    total_comp = sum(r["both_not_null"] for r in all_results)
    total_agree = sum(r["agree_correct"] for r in all_results)
    total_disagree = sum(r["disagree"] for r in all_results)

    print(f"\n  Total runs: {len(all_results)}")
    print(f"  Total trials: {total_trials}")
    print(f"  With heuristic label: {total_heur} ({total_heur/total_trials*100:.1f}%)")
    print(f"  With judge label: {total_judge} ({total_judge/total_trials*100:.1f}%)")
    print(f"  Comparable (both present): {total_comp} ({total_comp/total_trials*100:.1f}%)")
    if total_comp > 0:
        print(f"  Agree: {total_agree} ({total_agree/total_comp*100:.1f}%)")
        print(f"  Disagree: {total_disagree} ({total_disagree/total_comp*100:.1f}%)")

    # Per location
    for loc in ["runs", "runs/think", "runs_latest/runs"]:
        lr = [r for r in all_results if r["label"] == loc]
        if not lr:
            continue
        lt = sum(r["total_trials"] for r in lr)
        lj = sum(r["judge_not_null"] for r in lr)
        lc = sum(r["both_not_null"] for r in lr)
        la = sum(r["agree_correct"] for r in lr)
        ld = sum(r["disagree"] for r in lr)
        print(f"\n  [{loc}]")
        print(f"    Runs: {len(lr)} | Trials: {lt} | Judged: {lj} ({lj/lt*100:.1f}%)" if lt > 0 else f"    Runs: {len(lr)}")
        if lc > 0:
            print(f"    Comparable: {lc} | Agree: {la} ({la/lc*100:.1f}%) | Disagree: {ld} ({ld/lc*100:.1f}%)")

    # =============================================
    # RUNS NEEDING JUDGE LABELS
    # =============================================
    print("\n\n" + "=" * 80)
    print("RUNS NEEDING JUDGE LABELS (0% coverage)")
    print("=" * 80)

    for r in all_results:
        if r["judge_not_null"] == 0 and r["total_trials"] > 0:
            print(f"  {r['uuid'][:12]}... [{r['label']}] {r.get('model_id','?')} / {r.get('variant','?')} / temp={r.get('temperature','?')} — {r['total_trials']} trials")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
