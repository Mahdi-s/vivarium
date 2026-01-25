from __future__ import annotations

from utils import build_arg_parser, load_db_for_run, resolve_run_ref


def main() -> int:
    p = build_arg_parser(description="Generate Figure 1 behavioral baselines for a run.")
    args = p.parse_args()

    ref = resolve_run_ref(run_id=args.run_id, run_dir=args.run_dir, runs_dir=args.runs_dir)
    from aam.analytics import compute_behavioral_metrics, export_behavioral_logs, generate_behavioral_graphs

    db = load_db_for_run(ref)
    try:
        metrics = compute_behavioral_metrics(db, ref.run_id, ref.run_dir)
        figures = generate_behavioral_graphs(db, ref.run_id, ref.run_dir, metrics)
        export_behavioral_logs(db, ref.run_id, ref.run_dir, metrics)
    finally:
        db.close()

    for k, v in figures.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

