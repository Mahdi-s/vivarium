from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from utils import build_arg_parser, ensure_artifacts_dirs, load_db_for_run, resolve_run_ref


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
        # Multi-variant safe: use *all* truth/social probes for this run and
        # merge by (trial_id, layer_index). Because projection computation is
        # variant-scoped in the pipeline, each trial has exactly one truth and
        # one social projection per layer.
        probe_kinds = [r[0] for r in db.conn.execute(
            "SELECT DISTINCT probe_kind FROM conformity_probes WHERE run_id = ? ORDER BY probe_kind ASC;",
            (ref.run_id,),
        ).fetchall()]
        if "truth" not in probe_kinds or "social" not in probe_kinds:
            raise RuntimeError(
                "Missing required probes for turn-layer analysis. "
                "Expected both 'truth' and 'social' rows in conformity_probes for this run."
            )

        # Compute first collision layer per trial (Turn proxy): min layer where social > truth.
        df = pd.read_sql_query(
            """
            WITH truth AS (
              SELECT p.trial_id, p.layer_index, p.value_float AS truth_val
              FROM conformity_probe_projections p
              JOIN conformity_probes pr ON pr.probe_id = p.probe_id
              WHERE pr.run_id = ? AND pr.probe_kind = 'truth'
            ),
            social AS (
              SELECT p.trial_id, p.layer_index, p.value_float AS social_val
              FROM conformity_probe_projections p
              JOIN conformity_probes pr ON pr.probe_id = p.probe_id
              WHERE pr.run_id = ? AND pr.probe_kind = 'social'
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
              AND cond.name IN ('control','asch_history_5','authoritative_bias')
            ORDER BY c.first_collision_layer ASC;
            """,
            db.conn,
            params=(ref.run_id, ref.run_id, ref.run_id),
        )

        detection_path = os.path.join(tables_dir, "turn_layer_detection.json")
        _write_json(
            detection_path,
            {
                "run_id": ref.run_id,
                "n_collisions": int(len(df)),
                "collisions": df.to_dict("records"),
            },
        )

        stats_path = os.path.join(tables_dir, "turn_layer_statistics.csv")
        if not df.empty:
            stats = (
                df.groupby(["variant", "condition_name"])["first_collision_layer"]
                .agg(
                    n="count",
                    mean_first_collision_layer="mean",
                    std_first_collision_layer="std",
                )
                .reset_index()
            )
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
              SELECT p.trial_id, p.layer_index, p.value_float AS truth_val
              FROM conformity_probe_projections p
              JOIN conformity_probes pr ON pr.probe_id = p.probe_id
              WHERE pr.run_id = ? AND pr.probe_kind = 'truth'
            ),
            social AS (
              SELECT p.trial_id, p.layer_index, p.value_float AS social_val
              FROM conformity_probe_projections p
              JOIN conformity_probes pr ON pr.probe_id = p.probe_id
              WHERE pr.run_id = ? AND pr.probe_kind = 'social'
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
            WHERE tr.run_id = ?
              AND cond.name IN ('control','asch_history_5','authoritative_bias');
            """,
            db.conn,
            params=(ref.run_id, ref.run_id, ref.run_id),
        )

        fig_path = ""
        if not diff_df.empty:
            # Plot: mean first collision layer by (variant, condition)
            stats = (
                df.groupby(["variant", "condition_name"], as_index=False)["first_collision_layer"]
                .mean()
                .rename(columns={"first_collision_layer": "mean_first_collision_layer"})
            )
            if not stats.empty:
                fig, ax = plt.subplots(figsize=(10, 4.8))
                # Order variants as they appear in paper
                variant_order = ["base", "instruct", "instruct_sft", "think", "think_sft", "rl_zero"]
                cond_order = ["control", "asch_history_5", "authoritative_bias"]
                stats["variant"] = pd.Categorical(stats["variant"], categories=variant_order, ordered=True)
                stats["condition_name"] = pd.Categorical(stats["condition_name"], categories=cond_order, ordered=True)
                stats = stats.sort_values(["variant", "condition_name"])

                # Pivot for grouped bar plot
                pivot = stats.pivot(index="variant", columns="condition_name", values="mean_first_collision_layer")
                pivot.plot(kind="bar", ax=ax)
                ax.set_ylabel("Mean first collision layer (SVP > TVP)")
                ax.set_xlabel("Variant")
                ax.set_title("Turn Layer by Variant and Condition")
                ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                fig_path = os.path.join(figures_dir, "turn_layer_by_variant.png")
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)

    finally:
        db.close()

    if fig_path:
        print(f"turn_layer_heatmap={fig_path}")
    print(f"turn_layer_detection={detection_path}")
    print(f"turn_layer_statistics={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
