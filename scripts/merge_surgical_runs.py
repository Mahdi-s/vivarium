#!/usr/bin/env python3
"""
Merge surgical think_dpo runs into full T=0.8 and T=1.0 databases.

For each (surgical → target) pair, this script:
  1. Deletes the incomplete think_dpo rows from the target DB for 9 conditions
  2. Inserts the complete think_dpo rows from the surgical DB (remapping run_id
     and condition_id to match the target DB)

Usage:
    python scripts/merge_surgical_runs.py            # dry-run (prints what would change)
    python scripts/merge_surgical_runs.py --commit   # actually writes changes
"""

import argparse
import pathlib
import sqlite3
import subprocess
import sys
from textwrap import dedent


# ── Configuration ────────────────────────────────────────────────────────────

MERGES = [
    {
        "label": "T=0.8 think_dpo",
        "surgical_run_dir": "20260307_230500_b5a2b275-a88e-4038-8e96-f8f5cfcd571b",
        "surgical_run_id": "b5a2b275-a88e-4038-8e96-f8f5cfcd571b",
        "target_run_dir": "20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814",
        "target_run_id": "9369442d-d825-4cd0-81a1-8ed276c37814",
    },
    {
        "label": "T=1.0 think_dpo",
        "surgical_run_dir": "20260308_123628_fcc469ac-5794-43d7-9389-d6668cb47d1e",
        "surgical_run_id": "fcc469ac-5794-43d7-9389-d6668cb47d1e",
        "target_run_dir": "20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e",
        "target_run_id": "9173bfae-4e8a-464f-8c9c-7ee91caa8b6e",
    },
]

# The 9 conditions that the surgical runs cover
SURGICAL_CONDITIONS = {
    "asch_zhu_unbiased_da",
    "asch_zhu_unbiased_diverse_plain",
    "asch_zhu_unbiased_qd",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_uncertain",
    "authority_zhu_unbiased_trust",
    "authority_zhu_unbiased_trust_da",
}


# ── Path resolution ─────────────────────────────────────────────────────────

def find_repo_root() -> pathlib.Path:
    """Find the repo root (handles worktree case)."""
    script_dir = pathlib.Path(__file__).resolve().parent
    return script_dir.parent


def find_main_repo_root(repo_root: pathlib.Path) -> pathlib.Path:
    """If we're in a worktree, find the main repo root."""
    try:
        git_common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        main_root = (repo_root / git_common).resolve().parent
        return main_root
    except Exception:
        return repo_root


def find_db(run_dir: str, repo_root: pathlib.Path,
            main_repo_root: pathlib.Path) -> pathlib.Path:
    """Find simulation.db for a run, checking worktree first, then main repo."""
    candidates = [
        repo_root / "runs_latest" / "runs" / run_dir / "simulation.db",
        main_repo_root / "runs_latest" / "runs" / run_dir / "simulation.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find simulation.db for {run_dir}.\n"
        f"  Checked: {[str(c) for c in candidates]}"
    )


# ── Merge logic ──────────────────────────────────────────────────────────────

def build_condition_map(
    surgical_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
) -> dict[str, str]:
    """Map surgical condition_ids → target condition_ids by name."""
    surgical_conds = dict(surgical_conn.execute(
        "SELECT condition_id, name FROM conformity_conditions"
    ).fetchall())
    target_conds = {
        name: cid for cid, name in target_conn.execute(
            "SELECT condition_id, name FROM conformity_conditions"
        ).fetchall()
    }

    cond_map = {}
    for s_cid, name in surgical_conds.items():
        if name not in SURGICAL_CONDITIONS:
            continue
        if name not in target_conds:
            raise ValueError(f"Condition '{name}' not found in target DB")
        cond_map[s_cid] = target_conds[name]

    if len(cond_map) != len(SURGICAL_CONDITIONS):
        found = {surgical_conds[k] for k in cond_map}
        missing = SURGICAL_CONDITIONS - found
        raise ValueError(f"Missing conditions in surgical DB: {missing}")

    return cond_map


def get_target_trial_ids(
    target_conn: sqlite3.Connection,
    target_run_id: str,
    target_condition_ids: list[str],
) -> list[str]:
    """Get trial_ids in the target DB for think_dpo + the 9 conditions."""
    placeholders = ",".join("?" * len(target_condition_ids))
    rows = target_conn.execute(
        f"""SELECT trial_id FROM conformity_trials
            WHERE run_id = ? AND variant = 'think_dpo'
            AND condition_id IN ({placeholders})""",
        [target_run_id] + target_condition_ids,
    ).fetchall()
    return [r[0] for r in rows]


def get_prompt_ids_for_trials(
    conn: sqlite3.Connection,
    trial_ids: list[str],
) -> list[str]:
    """Get prompt_ids linked to given trial_ids."""
    if not trial_ids:
        return []
    placeholders = ",".join("?" * len(trial_ids))
    rows = conn.execute(
        f"SELECT prompt_id FROM conformity_prompts WHERE trial_id IN ({placeholders})",
        trial_ids,
    ).fetchall()
    return [r[0] for r in rows]


def delete_rows_batch(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    ids: list[str],
    batch_size: int = 500,
) -> int:
    """Delete rows in batches (SQLite has a variable limit)."""
    total = 0
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cur = conn.execute(
            f"DELETE FROM {table} WHERE {id_column} IN ({placeholders})", batch
        )
        total += cur.rowcount
    return total


def copy_table_rows(
    surgical_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    trial_ids: list[str],
    remap: dict[str, dict[str, str]] | None = None,
    join_column: str = "trial_id",
    batch_size: int = 500,
) -> int:
    """Copy rows from surgical to target, optionally remapping columns."""
    total = 0
    cols_str = ", ".join(columns)
    placeholders_insert = ", ".join("?" * len(columns))

    for i in range(0, len(trial_ids), batch_size):
        batch = trial_ids[i : i + batch_size]
        ph = ",".join("?" * len(batch))
        rows = surgical_conn.execute(
            f"SELECT {cols_str} FROM {table} WHERE {join_column} IN ({ph})",
            batch,
        ).fetchall()

        if remap:
            new_rows = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                for col, mapping in remap.items():
                    row_dict[col] = mapping[row_dict[col]]
                new_rows.append(tuple(row_dict[c] for c in columns))
            rows = new_rows

        if rows:
            target_conn.executemany(
                f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders_insert})",
                rows,
            )
            total += len(rows)

    return total


def per_condition_counts(
    conn: sqlite3.Connection,
    run_id: str,
    condition_ids: list[str],
) -> dict[str, int]:
    """Count think_dpo trials per condition_id."""
    placeholders = ",".join("?" * len(condition_ids))
    rows = conn.execute(
        f"""SELECT condition_id, COUNT(*)
            FROM conformity_trials
            WHERE run_id = ? AND variant = 'think_dpo'
            AND condition_id IN ({placeholders})
            GROUP BY condition_id""",
        [run_id] + condition_ids,
    ).fetchall()
    return dict(rows)


def process_merge(merge: dict, commit: bool, repo_root: pathlib.Path,
                  main_repo_root: pathlib.Path) -> bool:
    """Process one surgical → target merge. Returns True on success."""
    label = merge["label"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # ── Open DBs ──
    surgical_path = find_db(merge["surgical_run_dir"], repo_root, main_repo_root)
    target_path = find_db(merge["target_run_dir"], repo_root, main_repo_root)
    print(f"  Surgical DB: {surgical_path}")
    print(f"  Target DB:   {target_path}")

    surgical_conn = sqlite3.connect(f"file:{surgical_path}?mode=ro", uri=True)
    target_conn = sqlite3.connect(str(target_path))

    # ── Build condition map ──
    cond_map = build_condition_map(surgical_conn, target_conn)
    target_cond_ids = list(cond_map.values())

    # Reverse lookup: target_cond_id → condition name
    cond_names = dict(target_conn.execute(
        "SELECT condition_id, name FROM conformity_conditions"
    ).fetchall())

    print(f"\n  Condition mapping ({len(cond_map)} conditions):")
    for s_cid, t_cid in sorted(cond_map.items(), key=lambda x: cond_names.get(x[1], "")):
        print(f"    {cond_names[t_cid]:45s}  {s_cid[:8]}… → {t_cid[:8]}…")

    # ── Safety check: no surgical trial_ids in target ──
    surgical_trial_ids = [
        r[0] for r in surgical_conn.execute(
            "SELECT trial_id FROM conformity_trials"
        ).fetchall()
    ]
    ph = ",".join("?" * len(surgical_trial_ids))
    collisions = target_conn.execute(
        f"SELECT COUNT(*) FROM conformity_trials WHERE trial_id IN ({ph})",
        surgical_trial_ids,
    ).fetchone()[0]
    if collisions:
        print(f"\n  ERROR: {collisions} surgical trial_ids already exist in target!")
        surgical_conn.close()
        target_conn.close()
        return False
    print(f"\n  Safety check: 0 trial_id collisions ✓")

    # ── Collect partial trial_ids from target ──
    partial_trial_ids = get_target_trial_ids(
        target_conn, merge["target_run_id"], target_cond_ids
    )
    partial_prompt_ids = get_prompt_ids_for_trials(target_conn, partial_trial_ids)

    # ── Count before ──
    before_counts = per_condition_counts(
        target_conn, merge["target_run_id"], target_cond_ids
    )

    # ── Surgical trial_ids (for insert) ──
    surgical_trial_ids_list = [
        r[0] for r in surgical_conn.execute(
            "SELECT trial_id FROM conformity_trials"
        ).fetchall()
    ]
    surgical_prompt_ids = get_prompt_ids_for_trials(surgical_conn, surgical_trial_ids_list)

    # ── Count surgical outputs (verify 1:1 with trials) ──
    surgical_output_count = surgical_conn.execute(
        "SELECT COUNT(*) FROM conformity_outputs"
    ).fetchone()[0]
    surgical_prompt_count = surgical_conn.execute(
        "SELECT COUNT(*) FROM conformity_prompts"
    ).fetchone()[0]
    surgical_trial_meta_count = surgical_conn.execute(
        "SELECT COUNT(*) FROM conformity_trial_metadata"
    ).fetchone()[0]
    surgical_prompt_meta_count = surgical_conn.execute(
        "SELECT COUNT(*) FROM conformity_prompt_metadata"
    ).fetchone()[0]

    # ── Total think_dpo count in target before ──
    total_think_dpo_before = target_conn.execute(
        "SELECT COUNT(*) FROM conformity_trials WHERE run_id = ? AND variant = 'think_dpo'",
        [merge["target_run_id"]],
    ).fetchone()[0]

    # ── Dry-run summary ──
    print(f"\n  DELETE from target (partial think_dpo, 9 conditions):")
    print(f"    conformity_trial_metadata:  {len(partial_trial_ids):>5} rows")
    print(f"    conformity_prompt_metadata: {len(partial_prompt_ids):>5} rows")
    print(f"    conformity_prompts:         {len(partial_prompt_ids):>5} rows")
    print(f"    conformity_outputs:         {len(partial_trial_ids):>5} rows")
    print(f"    conformity_trials:          {len(partial_trial_ids):>5} rows")

    print(f"\n  INSERT from surgical (complete think_dpo, 9 conditions):")
    print(f"    conformity_trials:          {len(surgical_trial_ids_list):>5} rows (remapped run_id + condition_id)")
    print(f"    conformity_outputs:         {surgical_output_count:>5} rows")
    print(f"    conformity_prompts:         {surgical_prompt_count:>5} rows")
    print(f"    conformity_prompt_metadata: {surgical_prompt_meta_count:>5} rows")
    print(f"    conformity_trial_metadata:  {surgical_trial_meta_count:>5} rows")

    expected_total = total_think_dpo_before - len(partial_trial_ids) + len(surgical_trial_ids_list)
    print(f"\n  Total think_dpo trials: {total_think_dpo_before} → {expected_total}")

    if not commit:
        print("\n  [DRY RUN] No changes written. Use --commit to apply.")
        surgical_conn.close()
        target_conn.close()
        return True

    # ── Execute merge ──
    print("\n  Executing merge...")
    target_conn.execute("BEGIN")

    try:
        # Delete dependent rows first
        n = delete_rows_batch(target_conn, "conformity_trial_metadata", "trial_id", partial_trial_ids)
        print(f"    Deleted {n} from conformity_trial_metadata")

        n = delete_rows_batch(target_conn, "conformity_prompt_metadata", "prompt_id", partial_prompt_ids)
        print(f"    Deleted {n} from conformity_prompt_metadata")

        n = delete_rows_batch(target_conn, "conformity_prompts", "trial_id", partial_trial_ids)
        print(f"    Deleted {n} from conformity_prompts")

        n = delete_rows_batch(target_conn, "conformity_outputs", "trial_id", partial_trial_ids)
        print(f"    Deleted {n} from conformity_outputs")

        n = delete_rows_batch(target_conn, "conformity_trials", "trial_id", partial_trial_ids)
        print(f"    Deleted {n} from conformity_trials")

        # Insert from surgical (trials need remapping)
        run_id_map = {merge["surgical_run_id"]: merge["target_run_id"]}
        trial_cols = [
            "trial_id", "run_id", "model_id", "variant", "item_id",
            "condition_id", "seed", "temperature", "created_at",
        ]
        n = copy_table_rows(
            surgical_conn, target_conn, "conformity_trials", trial_cols,
            surgical_trial_ids_list,
            remap={"run_id": run_id_map, "condition_id": cond_map},
            join_column="trial_id",
        )
        print(f"    Inserted {n} into conformity_trials")

        output_cols = [
            "output_id", "trial_id", "raw_text", "parsed_answer_text",
            "parsed_answer_json", "is_correct", "refusal_flag", "latency_ms",
            "token_usage_json", "created_at",
        ]
        n = copy_table_rows(
            surgical_conn, target_conn, "conformity_outputs", output_cols,
            surgical_trial_ids_list,
            join_column="trial_id",
        )
        print(f"    Inserted {n} into conformity_outputs")

        prompt_cols = [
            "prompt_id", "trial_id", "system_prompt", "user_prompt",
            "chat_history_json", "rendered_prompt_hash", "created_at",
        ]
        n = copy_table_rows(
            surgical_conn, target_conn, "conformity_prompts", prompt_cols,
            surgical_trial_ids_list,
            join_column="trial_id",
        )
        print(f"    Inserted {n} into conformity_prompts")

        n = copy_table_rows(
            surgical_conn, target_conn, "conformity_prompt_metadata",
            ["prompt_id", "metadata_json", "created_at"],
            surgical_prompt_ids,
            join_column="prompt_id",
        )
        print(f"    Inserted {n} into conformity_prompt_metadata")

        n = copy_table_rows(
            surgical_conn, target_conn, "conformity_trial_metadata",
            ["trial_id", "metadata_json", "created_at"],
            surgical_trial_ids_list,
            join_column="trial_id",
        )
        print(f"    Inserted {n} into conformity_trial_metadata")

        target_conn.execute("COMMIT")
        print("    COMMIT ✓")

    except Exception as e:
        target_conn.execute("ROLLBACK")
        print(f"    ROLLBACK due to error: {e}")
        surgical_conn.close()
        target_conn.close()
        raise

    # ── Verification ──
    after_counts = per_condition_counts(
        target_conn, merge["target_run_id"], target_cond_ids
    )
    total_after = target_conn.execute(
        "SELECT COUNT(*) FROM conformity_trials WHERE run_id = ? AND variant = 'think_dpo'",
        [merge["target_run_id"]],
    ).fetchone()[0]

    print(f"\n  Verification (total think_dpo: {total_think_dpo_before} → {total_after}):")
    print(f"  {'Condition':<50s} {'before':>6s} {'after':>6s} {'expected':>8s}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*8}")
    all_ok = True
    for t_cid in sorted(target_cond_ids, key=lambda c: cond_names.get(c, "")):
        name = cond_names[t_cid]
        before = before_counts.get(t_cid, 0)
        after = after_counts.get(t_cid, 0)
        ok = "✓" if after == 400 else "✗"
        if after != 400:
            all_ok = False
        print(f"  {name:<50s} {before:>6d} {after:>6d} {400:>8d}  {ok}")

    if not all_ok:
        print("\n  WARNING: Some conditions do not have 400 trials!")
    else:
        print(f"\n  All 9 conditions have 400 trials ✓")

    surgical_conn.close()
    target_conn.close()
    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge surgical think_dpo runs into full T=0.8/1.0 databases."
    )
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually write changes (default: dry-run)",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    main_repo_root = find_main_repo_root(repo_root)

    mode = "COMMIT" if args.commit else "DRY RUN"
    print(f"Merge surgical think_dpo runs  [{mode}]")
    print(f"  Repo root:      {repo_root}")
    if main_repo_root != repo_root:
        print(f"  Main repo root: {main_repo_root}")

    success = True
    for merge in MERGES:
        if not process_merge(merge, args.commit, repo_root, main_repo_root):
            success = False

    print(f"\n{'='*70}")
    if success:
        print("All merges completed successfully." if args.commit else "Dry run complete. Use --commit to apply changes.")
    else:
        print("Some merges failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
