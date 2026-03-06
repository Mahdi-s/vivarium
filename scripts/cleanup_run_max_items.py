#!/usr/bin/env python3
"""
Clean up a run that was resumed with a higher max_items_per_dataset than intended.

Removes all trials (and dependent rows) for items beyond the first N per dataset,
and updates the run's stored config to the intended max_items_per_dataset.

Example (temp1.0 run accidentally resumed with 200 instead of 50):
  python scripts/cleanup_run_max_items.py \\
    --runs-dir runs_latest/runs \\
    --run-id 9173bfae-4e8a-464f-8c9c-7ee91caa8b6e \\
    --max-items 50 \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from vivarium.persistence import TraceDb, TraceDbConfig  # noqa: E402


def _find_run_dir(runs_dir: str, run_id: str) -> Path:
    if not run_id or not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs_dir not found or run_id empty: {runs_dir!r}, {run_id!r}")
    suffix = f"_{run_id}"
    for name in os.listdir(runs_dir):
        if name.endswith(suffix) and os.path.isdir(os.path.join(runs_dir, name)):
            return Path(runs_dir) / name
    raise FileNotFoundError(f"No run directory ending with _{run_id} in {runs_dir}")


def _allowed_item_ids_from_db(trace_db: TraceDb, suite_config: dict, max_items: int) -> set[str]:
    """Build set of item_ids that are in the first max_items per dataset (by created_at)."""
    allowed: set[str] = set()
    for ds in suite_config.get("datasets", []):
        name = str(ds.get("name", ""))
        version = str(ds.get("version", "v0"))
        row = trace_db.conn.execute(
            "SELECT dataset_id FROM conformity_datasets WHERE name = ? AND version = ?;",
            (name, version),
        ).fetchone()
        if not row:
            continue
        dataset_id = str(row["dataset_id"])
        rows = trace_db.conn.execute(
            """
            SELECT item_id FROM conformity_items
            WHERE dataset_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
            """,
            (dataset_id, max_items),
        ).fetchall()
        for r in rows:
            allowed.add(str(r["item_id"]))
    return allowed


def _extra_trial_ids(trace_db: TraceDb, run_id: str, allowed_item_ids: set[str]) -> list[str]:
    rows = trace_db.conn.execute(
        "SELECT trial_id, item_id FROM conformity_trials WHERE run_id = ?;",
        (run_id,),
    ).fetchall()
    return [str(r["trial_id"]) for r in rows if str(r["item_id"]) not in allowed_item_ids]


def _run_cleanup(
    *,
    db_path: str,
    run_id: str,
    suite_config: dict,
    max_items: int,
    dry_run: bool,
) -> dict:
    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

    # Verify run exists
    run_row = trace_db.conn.execute(
        "SELECT run_id, config_json FROM runs WHERE run_id = ?;",
        (run_id,),
    ).fetchone()
    if not run_row:
        return {"ok": False, "error": f"Run {run_id} not found in database."}

    allowed = _allowed_item_ids_from_db(trace_db, suite_config, max_items)
    extra_trial_ids = _extra_trial_ids(trace_db, run_id, allowed)
    total_trials = trace_db.conn.execute(
        "SELECT COUNT(*) AS n FROM conformity_trials WHERE run_id = ?;",
        (run_id,),
    ).fetchone()["n"]
    keep_count = total_trials - len(extra_trial_ids)

    out = {
        "ok": True,
        "run_id": run_id,
        "allowed_item_ids_count": len(allowed),
        "extra_trial_count": len(extra_trial_ids),
        "total_trials_before": total_trials,
        "trials_after_cleanup": keep_count,
    }

    if dry_run:
        out["dry_run"] = True
        trace_db.close()
        return out

    if not extra_trial_ids:
        out["message"] = "No extra trials to delete."
        trace_db.close()
        return out

    extra_set = set(extra_trial_ids)
    placeholders = ",".join(["?"] * len(extra_trial_ids))

    # Get (time_step, agent_id) for extra trials (for activation_metadata, agent_states, merkle_log)
    step_rows = trace_db.conn.execute(
        f"SELECT time_step, agent_id FROM conformity_trial_steps WHERE trial_id IN ({placeholders});",
        extra_trial_ids,
    ).fetchall()
    step_keys = [(int(r["time_step"]), str(r["agent_id"])) for r in step_rows]

    with trace_db.conn:
        # 1) activation_metadata, agent_states, merkle_log (by run_id + time_step + agent_id)
        for ts, agent_id in step_keys:
            trace_db.conn.execute(
                "DELETE FROM activation_metadata WHERE run_id = ? AND time_step = ? AND agent_id = ?;",
                (run_id, ts, agent_id),
            )
            trace_db.conn.execute(
                "DELETE FROM agent_states WHERE run_id = ? AND time_step = ? AND agent_id = ?;",
                (run_id, ts, agent_id),
            )
            trace_db.conn.execute(
                "DELETE FROM merkle_log WHERE run_id = ? AND time_step = ? AND agent_id = ?;",
                (run_id, ts, agent_id),
            )

        # 2) Tables referencing conformity_trials or conformity_outputs
        trace_db.conn.execute(
            f"DELETE FROM conformity_activation_patching WHERE source_trial_id IN ({placeholders}) OR target_trial_id IN ({placeholders});",
            extra_trial_ids + extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_contrastive_steering WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_intervention_results WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_logit_lens_tug_of_war WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_answer_logprobs WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_logit_lens WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_think_tokens WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_probe_projections WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_outputs WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )

        # 3) Prompts and metadata
        prompt_rows = trace_db.conn.execute(
            f"SELECT prompt_id FROM conformity_prompts WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        ).fetchall()
        prompt_ids = [str(r["prompt_id"]) for r in prompt_rows]
        if prompt_ids:
            pm_placeholders = ",".join(["?"] * len(prompt_ids))
            trace_db.conn.execute(
                f"DELETE FROM conformity_prompt_metadata WHERE prompt_id IN ({pm_placeholders});",
                prompt_ids,
            )
        trace_db.conn.execute(
            f"DELETE FROM conformity_prompts WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_trial_metadata WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM conformity_trial_steps WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )

        # 4) Trials
        trace_db.conn.execute(
            f"DELETE FROM conformity_trials WHERE trial_id IN ({placeholders});",
            extra_trial_ids,
        )

        # 5) Update run config: set suite_config.run.max_items_per_dataset = max_items
        config = json.loads(run_row["config_json"])
        if "suite_config" in config and isinstance(config["suite_config"], dict):
            run_cfg = config["suite_config"].setdefault("run", {})
            if isinstance(run_cfg, dict):
                run_cfg["max_items_per_dataset"] = max_items
        trace_db.conn.execute(
            "UPDATE runs SET config_json = ? WHERE run_id = ?;",
            (json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False), run_id),
        )

    out["deleted_trials"] = len(extra_trial_ids)
    trace_db.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remove trials for items beyond the first N per dataset and set run config."
    )
    ap.add_argument("--runs-dir", type=str, required=True, help="Base directory containing run folders")
    ap.add_argument("--run-id", type=str, required=True, help="Run ID (UUID suffix of run directory)")
    ap.add_argument(
        "--suite-config",
        type=str,
        default=None,
        help="Path to suite JSON (default: experiments/olmo_conformity/configs/suite_expanded_temp1.0.json)",
    )
    ap.add_argument("--max-items", type=int, default=50, help="Intended max items per dataset (default 50)")
    ap.add_argument("--dry-run", action="store_true", help="Only report counts, do not modify DB")
    args = ap.parse_args()

    run_dir = _find_run_dir(args.runs_dir, args.run_id)
    db_path = str(run_dir / "simulation.db")
    if not os.path.isfile(db_path):
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        return 1

    suite_path = args.suite_config
    if not suite_path:
        suite_path = str(REPO_ROOT / "experiments" / "olmo_conformity" / "configs" / "suite_expanded_temp1.0.json")
    with open(suite_path, encoding="utf-8") as f:
        suite_config = json.load(f)

    result = _run_cleanup(
        db_path=db_path,
        run_id=args.run_id,
        suite_config=suite_config,
        max_items=args.max_items,
        dry_run=args.dry_run,
    )

    if not result.get("ok"):
        print(result.get("error", "Unknown error"), file=sys.stderr)
        return 1

    print(f"Run ID: {result['run_id']}")
    print(f"Allowed item_ids (first {args.max_items} per dataset): {result['allowed_item_ids_count']}")
    print(f"Extra trials (to remove): {result['extra_trial_count']}")
    print(f"Total trials before: {result['total_trials_before']}")
    print(f"Trials after cleanup: {result['trials_after_cleanup']}")
    if result.get("dry_run"):
        print("[DRY RUN] No changes written.")
    elif result.get("deleted_trials", 0) > 0:
        print(f"Deleted {result['deleted_trials']} trials and updated run config to max_items_per_dataset={args.max_items}.")
    elif result.get("message"):
        print(result["message"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
