#!/usr/bin/env python3
"""
Scoped investigation: Only the data we actually care about for the paper.

runs/ (cross-family): ALL models EXCEPT gpt-oss-20b T=0.0 (incomplete)
  - 4 conditions: control, authoritative_bias, authority_trust, asch_zhu_unanimous_confident
  - 8 datasets
  - 2 temperatures: 0.0 and 0.6
  - All model families

runs/think/: OLMo-3-7B-Think
  - 4 conditions, 8 datasets, T=0.0

runs_latest/runs/: OLMo-3-1025-7B
  - Variants: base, instruct, instruct_sft, instruct_dpo ONLY
  - 12 conditions, 8 datasets
  - 6 temperatures: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

Uses JUDGE labels (parsed_answer_json) as authoritative.
"""

import sqlite3
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_RUNS = {
    # gpt-oss-20b T=0.0 — incomplete (143 trials)
    "66765d5e-204c-4074-aaf4-b9c148fe61a5",
}

RUNS_LATEST_VARIANTS = {"base", "instruct", "instruct_sft", "instruct_dpo"}


def find_scoped_dbs():
    dbs = []
    # runs/
    for entry in sorted((ROOT / "runs").iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name == "think":
            continue
        db = entry / "simulation.db"
        if not db.exists():
            continue
        uuid = entry.name.split("_", 2)[-1]
        if uuid in EXCLUDED_RUNS:
            continue
        dbs.append({"label": "runs", "dir_name": entry.name, "uuid": uuid, "db_path": str(db)})

    # runs/think
    think = ROOT / "runs" / "think"
    if think.exists():
        for entry in sorted(think.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            db = entry / "simulation.db"
            if db.exists():
                uuid = entry.name.split("_", 2)[-1]
                dbs.append({"label": "runs/think", "dir_name": entry.name, "uuid": uuid, "db_path": str(db)})

    # runs_latest/runs
    for entry in sorted((ROOT / "runs_latest" / "runs").iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        db = entry / "simulation.db"
        if db.exists():
            uuid = entry.name.split("_", 2)[-1]
            dbs.append({"label": "runs_latest", "dir_name": entry.name, "uuid": uuid, "db_path": str(db)})

    return dbs


def analyze_scoped(db_info):
    """Analyze using JUDGE labels as authoritative."""
    db_path = db_info["db_path"]
    label = db_info["label"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Determine variant filter
    variant_filter = None
    if label == "runs_latest":
        variant_filter = RUNS_LATEST_VARIANTS

    # Build query with optional variant filter
    variant_clause = ""
    params = []
    if variant_filter:
        placeholders = ",".join("?" * len(variant_filter))
        variant_clause = f" AND t.variant IN ({placeholders})"
        params = list(variant_filter)

    rows = conn.execute(f"""
        SELECT
            t.trial_id, t.model_id, t.variant, t.temperature,
            c.name as cond, d.name as ds,
            o.is_correct as heur_correct, o.refusal_flag as heur_refusal,
            o.parsed_answer_json, o.raw_text
        FROM conformity_trials t
        JOIN conformity_outputs o ON t.trial_id = o.trial_id
        JOIN conformity_conditions c ON t.condition_id = c.condition_id
        JOIN conformity_items i ON t.item_id = i.item_id
        JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
        WHERE 1=1 {variant_clause}
    """, params).fetchall()

    result = {
        "total": len(rows),
        "variants": Counter(),
        "conditions": Counter(),
        "datasets": Counter(),
        "temperatures": Counter(),
        "judge_present": 0,
        "judge_is_correct_dist": Counter(),
        "judge_refusal_dist": Counter(),
        "judge_endorsed_dist": Counter(),
        "heur_correct_dist": Counter(),
        "heur_refusal_dist": Counter(),
        "is_correct_agree": 0,
        "is_correct_disagree": 0,
        "is_correct_comparable": 0,
        "refusal_agree": 0,
        "refusal_disagree": 0,
        "refusal_comparable": 0,
        "judge_models": Counter(),
    }

    for r in rows:
        result["variants"][r["variant"]] += 1
        result["conditions"][r["cond"]] += 1
        result["datasets"][r["ds"]] += 1
        result["temperatures"][r["temperature"]] += 1
        result["heur_correct_dist"][r["heur_correct"]] += 1
        result["heur_refusal_dist"][r["heur_refusal"]] += 1

        paj_raw = r["parsed_answer_json"]
        if not paj_raw:
            continue

        try:
            paj = json.loads(paj_raw)
        except:
            continue

        jc = paj.get("is_correct")
        jr = paj.get("refusal_flag")
        je = paj.get("wrong_answer_endorsed")
        jm = (paj.get("_llm_judge") or {}).get("judge_model")

        if jc is not None or jr is not None:
            result["judge_present"] += 1

        result["judge_is_correct_dist"][jc] += 1
        result["judge_refusal_dist"][jr] += 1
        result["judge_endorsed_dist"][je] += 1
        if jm:
            result["judge_models"][jm] += 1

        # is_correct agreement
        if r["heur_correct"] is not None and jc is not None:
            result["is_correct_comparable"] += 1
            if int(r["heur_correct"]) == int(jc):
                result["is_correct_agree"] += 1
            else:
                result["is_correct_disagree"] += 1

        # refusal agreement
        if r["heur_refusal"] is not None and jr is not None:
            result["refusal_comparable"] += 1
            if int(r["heur_refusal"]) == int(jr):
                result["refusal_agree"] += 1
            else:
                result["refusal_disagree"] += 1

    # Get model info
    trial = conn.execute("SELECT model_id FROM conformity_trials LIMIT 1").fetchone()
    result["model_id"] = trial[0] if trial else "?"

    conn.close()
    return result


def main():
    all_dbs = find_scoped_dbs()
    print(f"Scoped databases: {len(all_dbs)}")
    print(f"  Excluded: gpt-oss-20b T=0.0 (incomplete)")
    print(f"  runs_latest filter: {sorted(RUNS_LATEST_VARIANTS)}\n")

    grand = {
        "total": 0, "judge_present": 0,
        "ic_comp": 0, "ic_agree": 0,
        "rf_comp": 0, "rf_agree": 0,
    }
    per_loc = defaultdict(lambda: {"total": 0, "judge": 0, "ic_comp": 0, "ic_agree": 0, "rf_comp": 0, "rf_agree": 0})

    all_conditions = Counter()
    all_datasets = Counter()
    all_variants = Counter()
    all_temps = Counter()

    for db in all_dbs:
        r = analyze_scoped(db)
        loc = db["label"]

        ic_pct = r["is_correct_agree"] / r["is_correct_comparable"] * 100 if r["is_correct_comparable"] > 0 else 0
        rf_pct = r["refusal_agree"] / r["refusal_comparable"] * 100 if r["refusal_comparable"] > 0 else 0
        judge_cov = r["judge_present"] / r["total"] * 100 if r["total"] > 0 else 0

        print(f"  [{loc:12s}] {db['uuid'][:12]}  {r['model_id'][:40]:40s}  trials={r['total']:6d}  judge={judge_cov:.0f}%  is_correct={ic_pct:.1f}%  refusal={rf_pct:.1f}%")
        print(f"               variants={dict(r['variants'])}  temps={sorted(set(r['temperatures'].keys()))}  conds={len(r['conditions'])}  ds={len(r['datasets'])}")
        print(f"               judge_models={dict(r['judge_models'])}")

        # Judge label distribution
        jc = r["judge_is_correct_dist"]
        jr = r["judge_refusal_dist"]
        je = r["judge_endorsed_dist"]
        total_j = r["judge_present"]
        if total_j > 0:
            correct_1 = jc.get(1, 0) + jc.get("1", 0)
            correct_0 = jc.get(0, 0) + jc.get("0", 0)
            refusal_1 = jr.get(1, 0) + jr.get("1", 0)
            endorsed_1 = je.get(1, 0) + je.get("1", 0)
            print(f"               judge: correct=1:{correct_1}({correct_1/total_j*100:.0f}%)  correct=0:{correct_0}({correct_0/total_j*100:.0f}%)  refusal=1:{refusal_1}({refusal_1/total_j*100:.0f}%)  endorsed=1:{endorsed_1}({endorsed_1/total_j*100:.0f}%)")
        print()

        grand["total"] += r["total"]
        grand["judge_present"] += r["judge_present"]
        grand["ic_comp"] += r["is_correct_comparable"]
        grand["ic_agree"] += r["is_correct_agree"]
        grand["rf_comp"] += r["refusal_comparable"]
        grand["rf_agree"] += r["refusal_agree"]
        per_loc[loc]["total"] += r["total"]
        per_loc[loc]["judge"] += r["judge_present"]
        per_loc[loc]["ic_comp"] += r["is_correct_comparable"]
        per_loc[loc]["ic_agree"] += r["is_correct_agree"]
        per_loc[loc]["rf_comp"] += r["refusal_comparable"]
        per_loc[loc]["rf_agree"] += r["refusal_agree"]

        for k, v in r["conditions"].items():
            all_conditions[k] += v
        for k, v in r["datasets"].items():
            all_datasets[k] += v
        for k, v in r["variants"].items():
            all_variants[k] += v
        for k, v in r["temperatures"].items():
            all_temps[k] += v

    print("=" * 90)
    print("SCOPED SUMMARY")
    print("=" * 90)
    print(f"  Total trials in scope: {grand['total']:,}")
    print(f"  Judge coverage: {grand['judge_present']:,}/{grand['total']:,} ({grand['judge_present']/grand['total']*100:.1f}%)")
    print(f"  is_correct agreement: {grand['ic_agree']:,}/{grand['ic_comp']:,} ({grand['ic_agree']/grand['ic_comp']*100:.1f}%)")
    print(f"  refusal agreement: {grand['rf_agree']:,}/{grand['rf_comp']:,} ({grand['rf_agree']/grand['rf_comp']*100:.1f}%)")

    print(f"\n  Variants: {dict(all_variants)}")
    print(f"  Conditions ({len(all_conditions)}): {sorted(all_conditions.keys())}")
    print(f"  Datasets ({len(all_datasets)}): {sorted(all_datasets.keys())}")
    print(f"  Temperatures: {sorted(all_temps.keys())}")

    for loc in ["runs", "runs/think", "runs_latest"]:
        s = per_loc[loc]
        if s["total"] == 0:
            continue
        ic_pct = s["ic_agree"] / s["ic_comp"] * 100 if s["ic_comp"] > 0 else 0
        rf_pct = s["rf_agree"] / s["rf_comp"] * 100 if s["rf_comp"] > 0 else 0
        print(f"\n  [{loc}]")
        print(f"    Trials: {s['total']:,}  Judge: {s['judge']:,} ({s['judge']/s['total']*100:.1f}%)")
        print(f"    is_correct: {s['ic_agree']:,}/{s['ic_comp']:,} ({ic_pct:.1f}%)")
        print(f"    refusal: {s['rf_agree']:,}/{s['rf_comp']:,} ({rf_pct:.1f}%)")


if __name__ == "__main__":
    main()
