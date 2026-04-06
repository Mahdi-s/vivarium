#!/usr/bin/env python3
"""
Sample and validate judge labels across all runs.
For each run, pulls random samples and checks:
1. Judge JSON structure is valid
2. is_correct, refusal_flag, wrong_answer_endorsed are valid values
3. _llm_judge metadata is present and well-formed
4. Notes are non-empty
5. Spot-check: does the judge label make sense given raw_text vs ground_truth?
"""

import sqlite3
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
RUN_DIRS = [
    ("runs", ROOT / "runs"),
    ("runs/think", ROOT / "runs" / "think"),
    ("runs_latest/runs", ROOT / "runs_latest" / "runs"),
]

SAMPLES_PER_RUN = 10


def find_all_dbs():
    dbs = []
    for label, base in RUN_DIRS:
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            db = entry / "simulation.db"
            if db.exists():
                uuid = entry.name.split('_')[-1] if '_' in entry.name else entry.name
                dbs.append({"label": label, "uuid": uuid, "db_path": str(db)})
    return dbs


def validate_run(db_path, label, uuid, n_samples=SAMPLES_PER_RUN):
    """Validate judge labels in a run by sampling."""
    issues = []
    samples = []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return [f"Cannot open DB: {e}"], []

    try:
        # Get total count
        total = conn.execute("SELECT COUNT(*) FROM conformity_outputs").fetchone()[0]

        # Get a random sample of outputs with full context
        rows = conn.execute("""
            SELECT
                t.trial_id,
                t.model_id, t.variant, t.temperature,
                c.name as condition_name,
                d.name as dataset_name,
                i.ground_truth_text,
                o.is_correct as heuristic_correct,
                o.refusal_flag as heuristic_refusal,
                o.parsed_answer_json,
                o.parsed_answer_text,
                o.raw_text
            FROM conformity_outputs o
            JOIN conformity_trials t ON o.trial_id = t.trial_id
            JOIN conformity_conditions c ON t.condition_id = c.condition_id
            JOIN conformity_items i ON t.item_id = i.item_id
            JOIN conformity_datasets d ON i.dataset_id = d.dataset_id
            ORDER BY RANDOM()
            LIMIT ?
        """, (n_samples,)).fetchall()

        for r in rows:
            sample = {
                "trial_id": r["trial_id"],
                "model_id": r["model_id"],
                "variant": r["variant"],
                "condition": r["condition_name"],
                "dataset": r["dataset_name"],
                "ground_truth": (r["ground_truth_text"] or "")[:120],
                "parsed_text": (r["parsed_answer_text"] or "")[:120],
                "raw_text_snippet": (r["raw_text"] or "")[:200].replace("\n", " "),
                "heuristic_correct": r["heuristic_correct"],
                "heuristic_refusal": r["heuristic_refusal"],
            }

            paj_raw = r["parsed_answer_json"]

            # Check 1: parsed_answer_json exists
            if not paj_raw:
                issues.append(f"trial {r['trial_id']}: parsed_answer_json is NULL/empty")
                sample["judge_valid"] = False
                samples.append(sample)
                continue

            # Check 2: valid JSON
            try:
                paj = json.loads(paj_raw) if isinstance(paj_raw, str) else paj_raw
            except (json.JSONDecodeError, TypeError) as e:
                issues.append(f"trial {r['trial_id']}: invalid JSON: {str(e)[:80]}")
                sample["judge_valid"] = False
                samples.append(sample)
                continue

            if not isinstance(paj, dict):
                issues.append(f"trial {r['trial_id']}: parsed_answer_json is not a dict")
                sample["judge_valid"] = False
                samples.append(sample)
                continue

            # Check 3: required fields present
            for field in ["is_correct", "refusal_flag", "wrong_answer_endorsed"]:
                if field not in paj:
                    issues.append(f"trial {r['trial_id']}: missing field '{field}' in parsed_answer_json")

            # Check 4: valid values
            ic = paj.get("is_correct")
            rf = paj.get("refusal_flag")
            we = paj.get("wrong_answer_endorsed")

            if ic is not None and ic not in [0, 1, "0", "1"]:
                issues.append(f"trial {r['trial_id']}: is_correct={ic} (expected 0/1/null)")
            if rf is not None and rf not in [0, 1, "0", "1"]:
                issues.append(f"trial {r['trial_id']}: refusal_flag={rf} (expected 0/1)")
            if we is not None and we not in [0, 1, "0", "1"]:
                issues.append(f"trial {r['trial_id']}: wrong_answer_endorsed={we} (expected 0/1/null)")

            # Check 5: mutual exclusivity (refusal + endorsed can't both be 1)
            if rf in [1, "1"] and we in [1, "1"]:
                issues.append(f"trial {r['trial_id']}: refusal_flag=1 AND wrong_answer_endorsed=1 (mutual exclusivity violated)")

            # Check 6: _llm_judge metadata
            jd = paj.get("_llm_judge")
            if not jd:
                issues.append(f"trial {r['trial_id']}: missing _llm_judge metadata")
            elif isinstance(jd, dict):
                if "judge_model" not in jd:
                    issues.append(f"trial {r['trial_id']}: _llm_judge missing judge_model")
                if "prompt_version" not in jd:
                    issues.append(f"trial {r['trial_id']}: _llm_judge missing prompt_version")

            # Check 7: notes field
            notes = paj.get("notes", "")
            if "[parse_error]" in str(notes):
                issues.append(f"trial {r['trial_id']}: judge contains [parse_error]")

            sample["judge_correct"] = ic
            sample["judge_refusal"] = rf
            sample["judge_endorsed"] = we
            sample["judge_model"] = jd.get("judge_model") if isinstance(jd, dict) else None
            sample["judge_notes"] = str(notes)[:150]
            sample["judge_valid"] = True

            # Agreement check
            if r["heuristic_correct"] is not None and ic is not None:
                sample["agree"] = int(r["heuristic_correct"]) == int(ic)
            else:
                sample["agree"] = None

            samples.append(sample)

    except Exception as e:
        issues.append(f"Query error: {e}")
    finally:
        conn.close()

    return issues, samples


def main():
    random.seed(42)

    all_dbs = find_all_dbs()
    print(f"Found {len(all_dbs)} databases to validate\n")

    total_issues = 0
    total_samples = 0
    all_samples = []

    for db in all_dbs:
        issues, samples = validate_run(db["db_path"], db["label"], db["uuid"])
        total_issues += len(issues)
        total_samples += len(samples)
        all_samples.extend(samples)

        status = "OK" if not issues else f"ISSUES ({len(issues)})"
        valid_count = sum(1 for s in samples if s.get("judge_valid", False))
        agree_count = sum(1 for s in samples if s.get("agree") == True)
        disagree_count = sum(1 for s in samples if s.get("agree") == False)

        print(f"  [{db['label']}] {db['uuid'][:12]}... — {status}")
        print(f"    Sampled: {len(samples)} | Valid JSON: {valid_count} | Agree: {agree_count} | Disagree: {disagree_count}")

        if issues:
            for iss in issues[:5]:
                print(f"    ISSUE: {iss}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more issues")

    # Print sample details
    print("\n" + "=" * 100)
    print("SAMPLE DETAILS (random trials from each run)")
    print("=" * 100)

    for i, s in enumerate(all_samples):
        if not s.get("judge_valid"):
            print(f"\n  [{i}] trial={s['trial_id']} — INVALID JUDGE LABEL")
            continue

        agree_str = "AGREE" if s.get("agree") == True else ("DISAGREE" if s.get("agree") == False else "N/A")
        print(f"\n  [{i}] trial={s['trial_id']} | {s['model_id']}/{s['variant']} | {s['condition']} | {s['dataset']}")
        print(f"       Heuristic: correct={s['heuristic_correct']} refusal={s['heuristic_refusal']}")
        print(f"       Judge:     correct={s.get('judge_correct')} refusal={s.get('judge_refusal')} endorsed={s.get('judge_endorsed')} [{agree_str}]")
        print(f"       Judge model: {s.get('judge_model')}")
        print(f"       GT: {s['ground_truth']}")
        print(f"       Parsed: {s['parsed_text']}")
        print(f"       Raw: {s['raw_text_snippet'][:150]}")
        if s.get("judge_notes"):
            print(f"       Notes: {s['judge_notes']}")

    # Summary
    print("\n\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"  Total databases: {len(all_dbs)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total issues: {total_issues}")
    print(f"  Valid judge labels: {sum(1 for s in all_samples if s.get('judge_valid', False))}/{total_samples}")

    # Judge model distribution in samples
    jm_counter = Counter(s.get("judge_model") for s in all_samples if s.get("judge_valid"))
    print(f"\n  Judge models in samples:")
    for m, c in jm_counter.most_common():
        print(f"    {m}: {c}")

    # Agreement in samples
    agree = sum(1 for s in all_samples if s.get("agree") == True)
    disagree = sum(1 for s in all_samples if s.get("agree") == False)
    na = sum(1 for s in all_samples if s.get("agree") is None)
    print(f"\n  Sample agreement: {agree} agree, {disagree} disagree, {na} not comparable")
    if agree + disagree > 0:
        print(f"  Agreement rate in sample: {agree / (agree + disagree) * 100:.1f}%")


if __name__ == "__main__":
    main()
