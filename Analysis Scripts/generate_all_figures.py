from __future__ import annotations

from utils import build_arg_parser, resolve_run_ref


def main() -> int:
    p = build_arg_parser(description="Generate all paper figures for a run (Figure 1-4).")
    args = p.parse_args()
    ref = resolve_run_ref(run_id=args.run_id, run_dir=args.run_dir, runs_dir=args.runs_dir)

    # Import locally so each script keeps its own dependencies/behavior.
    from generate_figure1_behavioral import main as fig1_main
    from generate_figure2_tug_of_war import main as fig2_main
    from generate_figure3_turn_layer import main as fig3_main
    from generate_figure4_intervention import main as fig4_main

    # Re-run each script in-process by spoofing argv.
    import sys

    def _run(sub_main):
        old = sys.argv
        try:
            sys.argv = [old[0], "--run-dir", ref.run_dir, "--run-id", ref.run_id]
            return int(sub_main())
        finally:
            sys.argv = old

    rc = 0
    rc |= _run(fig1_main)
    rc |= _run(fig2_main)
    rc |= _run(fig3_main)
    rc |= _run(fig4_main)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

