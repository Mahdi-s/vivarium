from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils import build_arg_parser, ensure_artifacts_dirs, load_db_for_run, resolve_run_ref


def _latest_probe_id(*, db: Any, run_id: str, probe_kind: str) -> Optional[str]:
    row = db.conn.execute(
        """
        SELECT probe_id
        FROM conformity_probes
        WHERE run_id = ? AND probe_kind = ?
        ORDER BY created_at DESC
        LIMIT 1;
        """,
        (run_id, probe_kind),
    ).fetchone()
    if row is None:
        return None
    return str(row["probe_id"])


def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> int:
    p = build_arg_parser(description="Generate Turn Layer (Figure 3) outputs for a run.")
    args = p.parse_args()

    ref = resolve_run_ref(run_id=args.run_id, run_dir=args.run_dir, runs_dir=args.runs_dir)
    figures_dir, tables_dir = ensure_artifacts_dirs(ref.run_dir)

    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires pandas + matplotlib.") from e

    db = load_db_for_run(ref)
    try:
        truth_probe_id = _latest_probe_id(db=db, run_id=ref.run_id, probe_kind="truth")
        social_probe_id = _latest_probe_id(db=db, run_id=ref.run_id, probe_kind="social")
        if not truth_probe_id or not social_probe_id:
            raise RuntimeError(
                "Missing required probes for turn-layer analysis. "
                "Expected both 'truth' and 'social' rows in conformity_probes for this run."
            )

        # Compute first collision layer per trial (Turn proxy): min layer where social > truth.
        df = pd.read_sql_query(
            """
            WITH truth AS (
              SELECT trial_id, layer_index, value_float AS truth_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            social AS (
              SELECT trial_id, layer_index, value_float AS social_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            merged AS (
              SELECT
                t.trial_id,
                t.layer_index,
                (s.social_val - t.truth_val) AS diff
              FROM truth t
              JOIN social s
                ON s.trial_id = t.trial_id AND s.layer_index = t.layer_index
            ),
            collisions AS (
              SELECT trial_id, MIN(layer_index) AS first_collision_layer
              FROM merged
              WHERE diff > 0
              GROUP BY trial_id
            )
            SELECT
              c.trial_id,
              c.first_collision_layer,
              tr.variant,
              cond.name AS condition_name
            FROM collisions c
            JOIN conformity_trials tr ON tr.trial_id = c.trial_id
            JOIN conformity_conditions cond ON cond.condition_id = tr.condition_id
            WHERE tr.run_id = ?
            ORDER BY c.first_collision_layer ASC;
            """,
            db.conn,
            params=(truth_probe_id, social_probe_id, ref.run_id),
        )

        detection_path = os.path.join(tables_dir, "turn_layer_detection.json")
        _write_json(
            detection_path,
            {
                "run_id": ref.run_id,
                "truth_probe_id": truth_probe_id,
                "social_probe_id": social_probe_id,
                "n_collisions": int(len(df)),
                "collisions": df.to_dict("records"),
            },
        )

        stats_path = os.path.join(tables_dir, "turn_layer_statistics.csv")
        if not df.empty:
            stats = (
                df.groupby(["variant", "condition_name"], as_index=False)["first_collision_layer"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            stats.columns = ["variant", "condition_name", "n", "mean_first_collision_layer", "std_first_collision_layer"]
            stats.to_csv(stats_path, index=False)
        else:
            # still create an empty file with headers for downstream pipelines
            pd.DataFrame(
                columns=[
                    "variant",
                    "condition_name",
                    "n",
                    "mean_first_collision_layer",
                    "std_first_collision_layer",
                ]
            ).to_csv(stats_path, index=False)

        # Heatmap: mean(social - truth) by layer and condition (aggregated over variants).
        diff_df = pd.read_sql_query(
            """
            WITH truth AS (
              SELECT trial_id, layer_index, value_float AS truth_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            ),
            social AS (
              SELECT trial_id, layer_index, value_float AS social_val
              FROM conformity_probe_projections
              WHERE probe_id = ?
            )
            SELECT
              tr.variant,
              cond.name AS condition_name,
              t.layer_index,
              (s.social_val - t.truth_val) AS diff
            FROM truth t
            JOIN social s
              ON s.trial_id = t.trial_id AND s.layer_index = t.layer_index
            JOIN conformity_trials tr ON tr.trial_id = t.trial_id
            JOIN conformity_conditions cond ON cond.condition_id = tr.condition_id
            WHERE tr.run_id = ?;
            """,
            db.conn,
            params=(truth_probe_id, social_probe_id, ref.run_id),
        )

        fig_path = os.path.join(figures_dir, "turn_layer_heatmap.png")
        if not diff_df.empty:
            pivot = (
                diff_df.groupby(["layer_index", "condition_name"], as_index=False)["diff"]
                .mean()
                .pivot(index="layer_index", columns="condition_name", values="diff")
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", interpolation="nearest")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(list(pivot.index))
            ax.set_xlabel("Condition")
            ax.set_ylabel("Layer Index")
            ax.set_title("Turn Layer Heatmap (Social - Truth)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
        else:
            # If no data, don't error; just skip heatmap creation.
            fig_path = ""

    finally:
        db.close()

    if fig_path:
        print(f"turn_layer_heatmap={fig_path}")
    print(f"turn_layer_detection={detection_path}")
    print(f"turn_layer_statistics={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

