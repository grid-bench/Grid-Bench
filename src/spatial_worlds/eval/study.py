"""CLI entry point for the single active formal study workflow."""

from __future__ import annotations

from typing import Any


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("study", help="configuration-driven formal evaluation")
    commands = parser.add_subparsers(dest="study_command", required=True)
    from .study_final import add_parser as add_formal_parser

    add_formal_parser(commands)


def from_args(args: Any) -> int:
    if args.study_command != "formal":
        raise ValueError("only the formal study workflow is active")
    from .study_final import from_args as run_formal

    return run_formal(args)
