"""Experiment registration and CLI dispatch for Vivarium."""

from __future__ import annotations

from typing import Any, Optional


def register_experiment_cli(subparsers: Any) -> None:
    """Register experiment subcommands with the main CLI."""
    from vivarium.experiments.olmo_conformity import cli as olmo_cli

    olmo_cli.register_subparsers(subparsers)


def handle_experiment_command(mode: str, args: Any) -> Optional[int]:
    """
    Dispatch to an experiment's command handler.
    Returns exit code if handled, None otherwise.
    """
    from vivarium.experiments.olmo_conformity import cli as olmo_cli

    return olmo_cli.handle_command(mode, args)
