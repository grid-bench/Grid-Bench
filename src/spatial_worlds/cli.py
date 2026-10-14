"""Command-line interface for the Grid-Bench benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .prompt import COMPLETION_PROMPT_VERSION, PROMPT_VERSION, REPRESENTATIONS, render_prompt
from .registry import list_families
from .schema import GoldRecord, Instance, SchemaError, parse_json_document
from .scoring import score_prediction, score_query_prediction
from .simulate import simulate


def _read_json(path: Path) -> Any:
    return parse_json_document(path.read_text(encoding="utf-8"))


def _int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _index_range(text: str) -> list[int]:
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start or 0), int(end)))
    return _int_list(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-worlds",
        description="Simulators, generators, and evaluation for the Grid-Bench benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    families = subparsers.add_parser("families", help="list the registered families")
    families.add_argument("--json", action="store_true", help="print metadata as JSON")

    validate = subparsers.add_parser("validate-instance", help="validate an instance JSON document")
    validate.add_argument("file", type=Path)

    render = subparsers.add_parser("render-prompt", help="render the model prompt for an instance")
    render.add_argument("file", type=Path)
    render.add_argument("--representation", choices=REPRESENTATIONS, default="cells")
    render.add_argument(
        "--prompt-version",
        choices=(PROMPT_VERSION, COMPLETION_PROMPT_VERSION),
        default=COMPLETION_PROMPT_VERSION,
    )

    sim = subparsers.add_parser("simulate", help="simulate an instance and print its gold record")
    sim.add_argument("file", type=Path)
    sim.add_argument("--gold", type=Path, help="write the gold record to this file")
    sim.add_argument("--trace", type=Path, help="write the per-step trace to this file")

    score = subparsers.add_parser("score", help="score a prediction against a gold record")
    score.add_argument("gold_file", type=Path)
    score.add_argument("prediction_file", type=Path)
    score.add_argument("--instance", type=Path, help="instance file (needed for query contracts)")

    generate = subparsers.add_parser("generate", help="generate a dataset from a config or preset")
    generate.add_argument("--config", required=True, help="config JSON path or preset name")
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--families", type=_str_list, help="comma-separated family ids")
    generate.add_argument("--dims", type=_int_list, help="comma-separated dimensions")
    generate.add_argument("--seeds", type=_index_range, help="rollout index range START:END")
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument(
        "--full-trace", action="store_true", help="store every intermediate state"
    )
    generate.add_argument("--queries", type=int, help="override the number of query coordinates")
    generate.add_argument(
        "--query-policy",
        choices=("uniform", "half-changed"),
        help="how query coordinates are drawn (default uniform)",
    )
    generate.add_argument("--split", help="override the split token")
    generate.add_argument("--name", help="override the dataset name")
    generate.add_argument("--quiet", action="store_true")

    run = subparsers.add_parser("run", help="query a model on every instance of a dataset")
    run.add_argument("--data", type=Path, help="dataset directory produced by generate")
    run.add_argument("--instances", type=Path, action="append", help="instances JSONL file(s)")
    run.add_argument(
        "--gold", type=Path, action="append", help="gold JSONL file(s) for the oracle provider"
    )
    run.add_argument(
        "--provider",
        required=True,
        choices=("anthropic", "openai", "copy-initial", "all-zero", "oracle", "suffix"),
        help=(
            "suffix executes every SET and core events in the last K event positions "
            "(--model suffix-K)"
        ),
    )
    run.add_argument("--model", default=None)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--base-url")
    run.add_argument("--api-key-env")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=128000)
    run.add_argument("--effort", default="high", help="requested reasoning effort (default high)")
    run.add_argument(
        "--thinking-budget", type=int, help="legacy thinking budget for pre-4.6 Claude models"
    )
    run.add_argument("--temperature", type=float)
    run.add_argument(
        "--param",
        action="append",
        default=[],
        help="extra request parameter key=value (JSON value)",
    )
    run.add_argument("--representation", choices=REPRESENTATIONS, default="cells")
    run.add_argument("--filter", help="comma-separated key=value filters on instance metadata")
    run.add_argument("--limit", type=int)
    run.add_argument("--retry-errors", action="store_true")
    run.add_argument(
        "--retry-kinds",
        help=(
            "failure kinds that --retry-errors re-runs (comma list or all); "
            "by default budget and deadline failures are kept"
        ),
    )
    run.add_argument("--max-attempts", type=int, default=1)
    run.add_argument("--timeout", type=float, default=7200.0)
    run.add_argument("--quiet", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="score predictions and aggregate results")
    evaluate.add_argument("--data", type=Path, help="dataset directory produced by generate")
    evaluate.add_argument("--instances", type=Path, action="append")
    evaluate.add_argument("--gold", type=Path, action="append")
    evaluate.add_argument("--traces", type=Path, action="append")
    evaluate.add_argument("--predictions", type=Path, action="append", required=True)
    evaluate.add_argument("--out", required=True, type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.5, help="EM threshold for T*")
    evaluate.add_argument("--bootstrap", type=int, default=200)
    evaluate.add_argument(
        "--token-budget",
        type=int,
        default=128000,
        help="output-token level at which the fitted tokens-versus-T line is reported",
    )
    evaluate.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="skip predictions whose instance id is not in the loaded dataset (counted)",
    )
    evaluate.add_argument(
        "--split-horizons",
        type=_int_list,
        default=None,
        help="horizons entering the dimension-split means (default: every T >= 8)",
    )
    evaluate.add_argument(
        "--split-dimensions",
        type=_int_list,
        default=None,
        help="dimensions entering the dimension-split means (default: every dimension)",
    )
    evaluate.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="price table (configs/prices.json) that turns observed usage into cost_usd",
    )

    cost = subparsers.add_parser(
        "cost", help="project the cost of a selection from a scored calibration run"
    )
    cost.add_argument("--data", required=True, type=Path, help="dataset to select from")
    cost.add_argument(
        "--filter",
        action="append",
        default=[],
        help="selection filter as in run (repeatable; the union is selected)",
    )
    cost.add_argument(
        "--basis", required=True, type=Path, help="results.jsonl of the evaluated basis run"
    )
    cost.add_argument("--basis-model", help="model whose rows supply the usage (default: --model)")
    cost.add_argument("--prices", required=True, type=Path)
    cost.add_argument("--model", required=True, help="model whose price applies")
    cost.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="output-token multiplier of --model relative to the basis model",
    )
    cost.add_argument("--label", help="label printed with the summary line")
    cost.add_argument("--json", action="store_true", help="print the full projection as JSON")

    report = subparsers.add_parser("report", help="write summary tables (and plots) from results")
    report.add_argument("--results", required=True, type=Path)
    report.add_argument("--plots", action="store_true")

    diagnose = subparsers.add_parser(
        "diagnose", help="history-retention diagnostics of a dataset (no model needed)"
    )
    diagnose.add_argument("--data", required=True, type=Path)
    diagnose.add_argument(
        "--out", type=Path, help="diagnostics JSONL path (default: <data>/diagnostics.jsonl)"
    )
    diagnose.add_argument(
        "--suffix",
        type=_int_list,
        default=None,
        help="suffix lengths k of the suffix baseline (default 8,16,32,64)",
    )
    diagnose.add_argument("--quiet", action="store_true")
    from .eval.study import add_parser as add_study_parser

    add_study_parser(subparsers)
    return parser


def _cmd_families(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps([family.metadata.to_dict() for family in list_families()], indent=2))
        return 0
    for family in list_families():
        metadata = family.metadata
        print(
            f"{metadata.id}\t{metadata.status}\t{metadata.support_source}\t"
            f"{metadata.update_type}\t{metadata.name}"
        )
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    instance = Instance.from_dict(_read_json(args.file))
    result = simulate(instance, keep_all_states=args.trace is not None)
    gold = result.gold().to_dict()
    if args.gold is not None:
        args.gold.write_text(json.dumps(gold, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(gold, indent=2))
    if args.trace is not None:
        trace = {
            "instance_id": instance.id,
            "steps": [step.to_dict() for step in result.steps],
            "states": {
                str(t): result.state_at(t).to_dict()["cells"] for t in range(instance.horizon + 1)
            },
        }
        args.trace.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    gold = GoldRecord.from_dict(_read_json(args.gold_file))
    prediction = args.prediction_file.read_text(encoding="utf-8")
    if args.instance is not None:
        instance = Instance.from_dict(_read_json(args.instance))
        if instance.queries:
            result = score_query_prediction(prediction, gold, instance.queries)
            print(json.dumps(result.to_dict(), sort_keys=True))
            return 0
    print(json.dumps(score_prediction(prediction, gold).to_dict(), sort_keys=True))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .generation import PRESETS, build_dataset, config_from_dict, load_config

    if args.config in PRESETS:
        data = dict(PRESETS[args.config])
    else:
        data = dict(load_config(args.config).source)
    if args.queries is not None:
        data["queries"] = args.queries
    if args.query_policy is not None:
        data["query_policy"] = args.query_policy
    if args.split is not None:
        data["split"] = args.split
    if args.name is not None:
        data["name"] = args.name
    config = config_from_dict(data)
    progress = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    manifest = build_dataset(
        config,
        args.out,
        families=args.families,
        dims=args.dims,
        indices=args.seeds,
        workers=args.workers,
        full_trace=args.full_trace,
        progress=progress,
    )
    totals = manifest["totals"]
    print(
        f"wrote {totals['instances']} instances from {totals['rollouts']} rollouts in "
        f"{totals['cells']} cells to {args.out} ({totals['infeasible_cells']} infeasible cells, "
        f"{totals['rejected_attempts']} rejected attempts)"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .eval.run import run_from_args

    return run_from_args(args)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .eval.aggregate import evaluate_from_args

    return evaluate_from_args(args)


def _cmd_report(args: argparse.Namespace) -> int:
    from .eval.report import report_from_args

    return report_from_args(args)


def _cmd_cost(args: argparse.Namespace) -> int:
    from .eval.cost import cost_from_args

    return cost_from_args(args)


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnostics import (
        DEFAULT_SUFFIX_LENGTHS,
        DIAGNOSTICS_FILE,
        diagnose_dataset,
        render_summary,
        summarize_diagnostics,
        write_diagnostics,
    )
    from .eval.run import load_dataset

    dataset = load_dataset(args.data)
    lengths = tuple(args.suffix) if args.suffix else DEFAULT_SUFFIX_LENGTHS
    rows = diagnose_dataset(dataset.instances, dataset.gold, suffix_lengths=lengths)
    out = args.out if args.out is not None else args.data / DIAGNOSTICS_FILE
    write_diagnostics(rows, out)
    if not args.quiet:
        print(render_summary(summarize_diagnostics(rows)))
    print(f"wrote {len(rows)} rollout diagnostics to {out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.command == "study":
        from .eval.study import from_args

        return from_args(args)
    if args.command == "families":
        return _cmd_families(args)
    if args.command == "validate-instance":
        instance = Instance.from_dict(_read_json(args.file))
        # Static schema/event validation is not sufficient for families whose
        # legality depends on the evolving state (for example, a compression
        # may merge beyond a finite value alphabet).  Replaying the instance
        # makes this command validate the complete transition sequence.
        simulate(instance)
        print(f"valid: {instance.id}")
        return 0
    if args.command == "render-prompt":
        instance = Instance.from_dict(_read_json(args.file))
        print(
            render_prompt(
                instance, representation=args.representation, prompt_version=args.prompt_version
            )
        )
        return 0
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    if args.command == "cost":
        return _cmd_cost(args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a conventional process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, json.JSONDecodeError, SchemaError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
