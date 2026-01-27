from __future__ import annotations

from utils import build_arg_parser, load_db_for_run, resolve_run_ref


def main() -> int:
    p = build_arg_parser(description="Generate all paper tables (CSVs/JSON logs) for a run.")
    args = p.parse_args()
    ref = resolve_run_ref(run_id=args.run_id, run_dir=args.run_dir, runs_dir=args.runs_dir)

    from aam.analytics import (
        compute_behavioral_metrics,
        export_behavioral_logs,
        compute_probe_metrics,
        export_probe_logs,
        compute_intervention_metrics,
        export_intervention_logs,
        compute_judgeval_metrics,
        export_judgeval_logs,
    )

    db = load_db_for_run(ref)
    try:
        try:
            m = compute_behavioral_metrics(db, ref.run_id, ref.run_dir)
            export_behavioral_logs(db, ref.run_id, ref.run_dir, m)
        except Exception:
            pass

        try:
            m = compute_probe_metrics(db, ref.run_id, ref.run_dir)
            export_probe_logs(db, ref.run_id, ref.run_dir, m)
        except Exception:
            pass

        try:
            m = compute_intervention_metrics(db, ref.run_id, ref.run_dir)
            export_intervention_logs(db, ref.run_id, ref.run_dir, m)
        except Exception:
            pass

        try:
            m = compute_judgeval_metrics(db, ref.run_id, ref.run_dir)
            export_judgeval_logs(db, ref.run_id, ref.run_dir, m)
        except Exception:
            pass
    finally:
        db.close()

    # Also emit the turn-layer stats JSON/CSV if possible.
    try:
        from generate_figure3_turn_layer import main as fig3_main

        import sys

        old = sys.argv
        try:
            sys.argv = [old[0], "--run-dir", ref.run_dir, "--run-id", ref.run_id]
            fig3_main()
        finally:
            sys.argv = old
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

